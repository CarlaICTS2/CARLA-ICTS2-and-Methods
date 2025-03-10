import time
import math
import operator
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import carla
import cv2

from agents.planner.isdespotp import ISDespotP
from agents.learner.models import A2C
from ped_path_predictor.M3PC.train import M2P3CWrapper
from ped_path_predictor.autobots.autobot_wrapper_cognitive import AutoBotWrapperCog

from utils.config import Config, Mode, Agent, Action
from utils.utils import load_model, save_model
from utils.logger import log_debug, log_info, log_learning_metrics


class HyLEAR_NavA2C_AutoBots_Cognitive(ISDespotP):

    def __init__(self, client, world, carla_map, scenario):
        # if Config.MODE is Mode.TRAIN:
        super(HyLEAR_NavA2C_AutoBots_Cognitive, self).__init__(client, world, carla_map, scenario)

        # must use cuda
        if not torch.cuda.is_available(): raise ValueError("No CUDA capable GPU found.")

        # model
        self.model: A2C = None

        # episode buffers of neural network's predictions for training
        self.episode_belief_state_value_estimates: list = []
        self.episode_log_probabilities: list = []
        self.episode_entropies: list = []
        #
        # self.episode_observations: List[Tuple[float, float, float, float, float, float]] = []
        # self.episode_rewards: List[float] = []
        # self.episode_actions: List[Action] = []
        # self.episode_agent_vehicle_speeds: List[float] = []
        # self.episode_despot_policies: List[Tuple[float, float, float]] = []
        # self.episode_controls: List[carla.VehicleControl] = []
        #
        # self.episode_agent_vehicle_past_trajectory: List[Tuple[int, int]] = []
        # # the planned path for the agent vehicle at the current time step
        # self.step_agent_vehicle_future_trajectory: List[Tuple[int, int, float]] = []

        # define optimizer
        self.optimizer = None

        # self.ped_pred = M2P3CWrapper("_out/M2P3CWrapper/obs15_pred20/2024-09-12_18-31-41/model_184.pth")
        self.ped_pred = AutoBotWrapperCog("ped_path_predictor/saved_models/15_20/autobots_cogntive.pth")
        self.ped_pred.model.eval()


        self.initialize_model()
        if Config.RESUME or Config.MODE is not Mode.TRAIN: 
            self.model = load_model(
                model=self.model, model_dir=Config.MODEL_DIR, checkpoint="latest", key="hyLear_autobots"
            )
            
        if Config.MODE is Mode.TRAIN: self.initialize_optimizer()


    # ================================================================================================= #
    #                                       HyLEAR SETUP METHODS                                        #
    # ================================================================================================= #
    def initialize_model(self):
        if self.model is not None:
            raise TypeError("Exisitng HyLEAP model cannot be overwritten.")
        log_info(f"initializing HyLEAR model...")

        Config.HyPLAN.NavA2C = True
        self.model = A2C(hidden_dim=Config.HyLEAP.HIDDEN_LAYER_SIZE, use_dropout=False).double().cuda()
        if Config.MODE is not Mode.TRAIN: self.model.eval()
        log_info(self.model)
          

    def initialize_optimizer(self):
        if self.optimizer is not None:
            raise TypeError("Exisitng optimizer cannot be overwritten.")
        if not isinstance(self.model, A2C):
            raise ValueError(f"Invalid HyLEAR model type: Expected '{A2C}', got '{type(self.model)}'.")
        log_info("initializing optimizers...")

        # initialize optimizers
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=0.0005
        )

        if self.optimizer is None:
            raise ValueError(f"Invalid HyLEAR optimizer: Not initialized.")     
        if not isinstance(self.optimizer, torch.optim.Adam):
            raise ValueError(f"Invalid HyLEAR optimizer: " 
                                f"Expected '{torch.optim.Adam}', got '{type(self.optimizer)}'.")


    # ================================================================================================= #
    #                                        HyLEAP CORE METHODS                                        #
    # ================================================================================================= #
    def run_step(self, step_counter: int):
        self.vehicle = self.world.player
        transform = self.vehicle.get_transform()
        start = (self.vehicle.get_location().x, self.vehicle.get_location().y, transform.rotation.yaw)
        end = self.scenario[2]

        obstacles = super(HyLEAR_NavA2C_AutoBots_Cognitive, self).get_obstacles(start)
        print("Obstacles: ", obstacles)
        if len(obstacles):
            (self.step_agent_vehicle_future_trajectory, risk) = self.get_path_with_reasoning(start, end, obstacles)
        else:
            (self.step_agent_vehicle_future_trajectory, risk) = self.get_path_simple(start, end, obstacles)

        agent_vehicle_control = carla.VehicleControl(
            throttle=0.0, steer=0.0, brake=0.0, hand_brake=False, reverse=False, manual_gear_shift=False, gear=0
        )

        # buggy path planner returning empty path means we skip this step and just maintain current velocity
        if not len(self.step_agent_vehicle_future_trajectory):
            step_summary = {"skipped_step": True, "control": agent_vehicle_control}
            return step_summary

        agent_vehicle_control.steer = (self.step_agent_vehicle_future_trajectory[2][2] - start[2]) / 70.

        # perceive current observation
        super(HyLEAR_NavA2C_AutoBots_Cognitive, self).get_current_observation(step_counter)

        # need a non-trivial path for IS-DESPOT
        if Config.MODE is Mode.TRAIN:
            birdview_car_intention = super(HyLEAR_NavA2C_AutoBots_Cognitive, self).get_birdview_car_intention(
                self.vehicle.get_transform(),
                self.world.walker.get_transform(),
                self.episode_agent_vehicle_past_trajectory,
                self.step_agent_vehicle_future_trajectory,
                step_counter
            )

            # we can use the previous reward, action and current car speed set by the above action
            # be aware that the last list entry is always the current reward, action and car speed!
            _ = self.inference_iteration(birdview_car_intention, step_counter)
            
            # current action determined by IS-DESPOT
            despot_action, despot_value, despot_policy = super(HyLEAR_NavA2C_AutoBots_Cognitive, self).get_speed_action(step_counter)

            # construct categorical distribution
            action_policy = torch.tensor([0.05, 0.05, 0.05], dtype=torch.float64, device="cpu") # cuda not worth it
            action_policy[despot_action.value] = 0.9
            hybrid_action = Action(Categorical(action_policy).sample().item())

            # translate received action into CARLA car control commands
            if hybrid_action is Action.DECELERATE:
                agent_vehicle_control.brake = 0.8
            elif hybrid_action is Action.ACCELERATE:
                agent_vehicle_control.throttle = 0.8

            # remember step
            self.episode_actions.append(hybrid_action)
            self.episode_controls.append(agent_vehicle_control)
            # calculate reward
            step_summary = super(HyLEAR_NavA2C_AutoBots_Cognitive, self).get_reward_akash(step_counter)
            step_summary["action"] = hybrid_action

        else:
            birdview_car_intention = super(HyLEAR_NavA2C_AutoBots_Cognitive, self).get_birdview_car_intention(
                self.vehicle.get_transform(),
                self.world.walker.get_transform(),
                self.episode_agent_vehicle_past_trajectory,
                self.step_agent_vehicle_future_trajectory,
                step_counter
            )

            learner_action = self.inference_iteration(birdview_car_intention, step_counter)

            # translate action into CARLA vehicle control commands
            if learner_action is Action.DECELERATE:
                agent_vehicle_control.brake = 0.8
            elif learner_action is Action.ACCELERATE:
                agent_vehicle_control.throttle = 0.8

            # remember step
            self.episode_actions.append(learner_action)
            self.episode_controls.append(agent_vehicle_control)
            # calculate reward
            step_summary = super(HyLEAR_NavA2C_AutoBots_Cognitive, self).get_reward_akash(step_counter)
            step_summary["action"] = learner_action

        step_summary["skipped_step"] = False
        step_summary["car_intention"] = birdview_car_intention
        step_summary["control"] = agent_vehicle_control

        return step_summary


    def get_path_with_reasoning(self, start, end, obstacles):
        
        car_velocity = self.vehicle.get_velocity()
        car_speed = np.sqrt(car_velocity.x ** 2 + car_velocity.y ** 2) * 3.6
        yaw = start[2]
        relaxed_sidewalk = self.grid_cost.copy()
        y = round(start[1])

        # Relax sidewalk
        sidewalk_cost = -1.0
        sidewalk_length = 20
        if self.scenario[0] in [1, 3, 4, 7, 8, 10]:
            relaxed_sidewalk[13:16, y - sidewalk_length: y + sidewalk_length] = sidewalk_cost
            relaxed_sidewalk[4:7, y - 10: sidewalk_length + sidewalk_length] = sidewalk_cost
        elif self.scenario[0] in [2, 5, 6, 9]:
            relaxed_sidewalk[94:97, y - sidewalk_length: y + sidewalk_length] = sidewalk_cost
            relaxed_sidewalk[103:106, y - sidewalk_length: y + sidewalk_length] = sidewalk_cost
        elif self.scenario[0] == 11:
            self.grid_cost[9:16, 13:] = 10000
            self.risk_cmp[10:13, 13:] = 10000
            relaxed_sidewalk = self.grid_cost.copy()
            relaxed_sidewalk[4:7, y - 10: sidewalk_length + sidewalk_length] = sidewalk_cost
            x, y = round(self.world.incoming_car.get_location().x), round(self.world.incoming_car.get_location().y)
            # Hard coding incoming car path prediction
            obstacles.append((x, y - 1))
            obstacles.append((x, y - 2))
            obstacles.append((x, y - 3))
            obstacles.append((x, y - 4))
            obstacles.append((x, y - 5))
            # All grid locations occupied by car added to obstacles
            for i in [-1, 0, 1]:
                for j in [-2, -1, 0, 1, 2]:
                    obstacles.append((x + i, y + j))
        elif self.scenario[0] == 12:
            relaxed_sidewalk[13:16, y - sidewalk_length: y + sidewalk_length] = sidewalk_cost
            relaxed_sidewalk[4:7, y - 10: sidewalk_length + sidewalk_length] = sidewalk_cost
            x, y = round(self.world.incoming_car.get_location().x), round(self.world.incoming_car.get_location().y)
            obstacles.append((x, y + 1))
            obstacles.append((x, y + 2))
            obstacles.append((x, y + 3))
            obstacles.append((x, y + 4))
            obstacles.append((x, y + 5))

        # if pedestrian path prediction is enabled and if there is enough data to make a prediction
        # and if a pedestrian is actually in the scene
        if Config.PREDICT_PEDESTRIAN_PATH and len(self.ped_history) >= 15 and self.is_pedestrian_observable and len(self.car_history) >= 15:
            # Use path predictor
            ped_updated_risk_cmp = self.risk_cmp.copy()

            car_path = np.array(self.car_history)
            # car_path = car_path[:, :2]
            # car_path = car_path.reshape((15, 2))

            ped_path = np.array(self.ped_history)
            # ped_path = ped_path[:, :2]
            # ped_path = ped_path.reshape((15, 2))


            log_debug("making pedestrian path prediction...")
            pedestrian_path = self.ped_pred.get_single_prediction(ped_path, car_path)
            new_obs = [obs for obs in obstacles]
            pedestrian_path_d = list()
            for node in pedestrian_path:
                if (round(node[0]), round(node[1])) not in new_obs:
                    new_obs.append((round(node[0]), round(node[1])))
                    pedestrian_path_d.append((round(node[0]), round(node[1])))
            for pos in new_obs:
                ped_updated_risk_cmp[pos[0] , pos[1] ] = 10000

            path_normal = self.risk_path_planner.find_path_with_risk(
                start, end, self.grid_cost, obstacles, car_speed, yaw, ped_updated_risk_cmp, True, self.scenario[0]
            )

            if path_normal[1] < 1000:
                return (path_normal[0], path_normal[1] / 6)
            
            paths = [
                path_normal,  
                self.risk_path_planner.find_path_with_risk(
                    start, end, self.grid_cost, new_obs, car_speed, yaw, ped_updated_risk_cmp, True, self.scenario[0]
                ),  # ped pred
                self.risk_path_planner.find_path_with_risk(
                    start, end, relaxed_sidewalk, obstacles, car_speed, yaw, ped_updated_risk_cmp, True, self.scenario[0]
                ) # sidewalk relaxed
            ]  
            path, risk = self.rulebook(paths)
            return (path, risk/6)
        
        # if there is not enough data to make a prediciton or if there is no pedestrian in the current 
        # scene simulation step
        if len(self.ped_history) < 15: log_debug("skipping pedestrian path predicition: not enough pedestrain data yet...")
        elif not self.is_pedestrian_observable: log_debug("skipping pedestrian path predicition: no pedestrain in scene...")
        elif not Config.PREDICT_PEDESTRIAN_PATH: log_debug("skipping pedestrian path predicition: not enabled")

        if self.scenario[0] == 11 and self.world.incoming_car.get_location().y + 2 < start[1] and start[0] <= -2.5:
            end = (end[0], start[1] + 6, end[2])

        if self.scenario[0] in [10, 1] and self.world.walker.get_location().y > start[1] and start[0] >= 2.5:
            end = (end[0], start[1] - 6, end[2])

        path_normal = self.risk_path_planner.find_path_with_risk(
            start, end, self.grid_cost, obstacles, car_speed, yaw, self.risk_cmp, True, self.scenario[0]
        )
        if path_normal[1] < 100 or not self.is_pedestrian_observable:
            return path_normal
        
        paths = [
            path_normal,
            self.risk_path_planner.find_path_with_risk( # Sidewalk relaxed
                start, end, relaxed_sidewalk, obstacles, car_speed, yaw, self.risk_cmp, True, self.scenario[0]
            )
        ]  
        path, risk = self.rulebook(paths)
        return (path, risk/6)


    def rulebook(self, paths):
        
        # No sidewalk
        data = []
        steer = []
        r = []
        for p in paths:
            path, risk = p
            len_path = len(path)
            if len_path == 0:
                lane = math.inf
            else:
                lane = sum([path[i][2] - path[i-1][2] for i in range(1, len_path)]) / len_path
            data.append((path, risk, lane, len_path))
            # r.append(risk)
            # steer.append((path[2][2] - start[2]) / 70.)

        # print("Rulebook!", r)
        # print("Steering angle: ", steer)
        data.sort(key=operator.itemgetter(1, 2, 3))
        path = data[0][0]
        risk = data[0][1]
        # print("Steering angle: ", (path[2][2] - start[2]) / 70.)
        return path, risk


    # recycles predictions made during inference and essentially only calculates the loss
    def training_iteration(self, episode_counter: int):
        # required for getting a proper time-reading of cuda related tasks
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        # rewards used for training: must contain the final reward received for the terminal state and must be appropriately scaled
        N = len(self.episode_rewards) - 1
        # propagate rewards backward in time
        for i in range(1, N+1):
            # the current reward is equal to itself + the discounted reward of the next steps
            self.episode_rewards[N - i] += Config.HyLEAP.GAMMA * self.episode_rewards[N - i + 1]
        # shape: [steps, 1]
        rewards = torch.tensor(self.episode_rewards, dtype=torch.float64, device=Config.HyLEAP.DEVICE)

        # rewards vector is normalised to have 0 mean and 1 std
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float64).eps.item())     
        
        actor_losses = []
        critic_losses = []

        for log_prob, value, reward in zip(self.episode_log_probabilities, 
                                           self.episode_belief_state_value_estimates, 
                                           rewards):
            advantage = reward - value.item()
            # calculate actor (policy) loss
            actor_losses.append(-log_prob * advantage)
            # calculate critic (value) loss using L1 smooth loss
            critic_losses.append(F.smooth_l1_loss(value, torch.tensor([[reward]], dtype=torch.float64, device=Config.HyLEAP.DEVICE)))

        total_loss = torch.stack(actor_losses).mean() + torch.stack(critic_losses).mean()
        
        actor_loss = sum(actor_losses) / len(actor_losses)
        critic_loss = sum(critic_losses) / len(critic_losses)
    
        log_debug(
            f"total_loss {type(total_loss)}:{total_loss.dtype}:" +\
            f"{np.shape(total_loss)}:{total_loss.requires_grad}:\n{total_loss}"
        )
        log_debug(
            f"actor_loss: {actor_loss}, critic_loss: {critic_loss}"
        )
        
        # debug information
        torch.cuda.synchronize()
        log_info(f"total loss for episode {episode_counter}: {total_loss.item():.4f} "
                 f"calculated in {(time.perf_counter() - start_time)*1000:.4f}ms")

        # model.zero_grad() and optimizer.zero_grad() are the same IF all your model parameters are in that optimizer
        # it is safer to call model.zero_grad() to make sure all grads are zero, 
        # e.g. if you have two or more optimizers for one model.
        # reset gradients
        self.model.zero_grad()

        torch.cuda.synchronize()
        start_time = time.perf_counter()
        # propagate gradient backwards
        total_loss.backward()

        # update weights
        self.optimizer.step()

        # debug information
        torch.cuda.synchronize()
        log_info(
            f"performed backward pass for episode {episode_counter} in "\
            f"{((time.perf_counter() - start_time)*1000):.4f}ms"
        )

        # track learning progress
        log_learning_metrics({
            "iteration": episode_counter,
            "actor_loss": actor_loss.cpu().detach().squeeze().item(),
            "critic_loss": critic_loss.cpu().detach().squeeze().item(),
            "reward": rewards.sum().cpu().detach().squeeze().item()
        })


    def inference_iteration(self, car_intention: np.ndarray, step_counter: int):
        # construct tensor of all car intention images generated at the current planning depth of IS-DESPOT
        observations = torch.tensor(car_intention, dtype=torch.float64, device=Config.HyLEAP.DEVICE)\
                            .transpose(-1, 0).unsqueeze(0)

        # construct feature tensor, which is fed to the LSTM layer in addition to the convolved observation
        # for each step t it consists of the car's current speed v_t, the last executed action a_t-1 
        # (as determined by IS-DESPOT) and the reward received for the previous simulation step r_t-1.

        # we must have perceived the current velocity that is associated with the provided car intention image
        if len(self.episode_agent_vehicle_speeds) != step_counter:
            raise ValueError(
                f"Invalid numer of episode agent vehicle speeds: "
                f"Expected {step_counter}, got {len(self.episode_agent_vehicle_speeds)} instead."
            )   
        # because of the dummy reward (0.0) for the initial step (= 1)
        if len(self.episode_actions) != step_counter:
            raise ValueError(
                f"Invalid number of episode actions: Expected {step_counter}, got {len(self.episode_actions)}."
            )
        # at this point we do not yet have calculated the reward associated with the provided (current) car intention image
        if len(self.episode_rewards) != step_counter:
            raise ValueError(
                f"Invalid number of episode rewards: Expected {step_counter}, got {len(self.episode_rewards)}."
            )

        features = torch.tensor((self.episode_agent_vehicle_speeds[-1], 
                                 self.episode_actions[-1].value,
                                 self.episode_rewards[-1]),
                                 dtype=torch.float64,
                                 device=Config.HyLEAP.DEVICE).unsqueeze(0)
        
        # commented because even when not printing this is incurring computational overhead
        '''        
        log_info(
            f"previous_lstm_hidden_state.shape: {self.previous_lstm_hidden_state.shape}, "
            f"previous_lstm_cell_state.shape: {self.previous_lstm_cell_state.shape}, " 
            f"observations.shape: {observations.shape}, "
            f"features.shape: {features.shape}"
        )
        log_info(f"STEP={step_counter} feature vector (speed, action, reward): {self.episode_car_speeds[-1]}, "
                 f"{self.episode_actions[previous_action_and_reward_index].value}, "
                 f"{self.episode_rewards[previous_action_and_reward_index]}")

        '''
        # forward pass without torch.no_grad() because predictions will be recycled for training later
        action_logits, value, (self.previous_lstm_hidden_state, self.previous_lstm_cell_state) = self.model(
            observations, 
            self.previous_lstm_hidden_state, 
            self.previous_lstm_cell_state,
            features
        )

        '''       
        log_info(
            f"actions.shape: {action_logits.shape}, values.shape: {value.shape}, "
            f"current_lstm_hidden_state.shape: {self.previous_lstm_hidden_state.shape}, "
            f"current_lstm_cell_state.shape: {self.previous_lstm_cell_state.shape}"
        )
        '''
    
        # sample action
        action_probabilities = F.softmax(action_logits, dim=-1)
        action_distribution = Categorical(action_probabilities)
        learner_action = action_distribution.sample()

        # remember neural network's prediction for training
        if Config.MODE is Mode.TRAIN:
            # required for actor loss calculation
            self.episode_log_probabilities.append(action_distribution.log_prob(learner_action))
            self.episode_entropies.append(-(F.log_softmax(action_logits, dim=-1) * action_probabilities).sum())
            self.episode_belief_state_value_estimates.append(value)

        return Action(int(learner_action.item()))


    def get_path_simple(self, start, end, obstacles):
        car_velocity = self.vehicle.get_velocity()
        car_speed = np.sqrt(car_velocity.x ** 2 + car_velocity.y ** 2) * 3.6
        yaw = start[2]

        updated_risk_cmp = np.copy(self.risk_cmp)
        for pos in obstacles:
            pos = (round(pos[0]), round(pos[1]))
            updated_risk_cmp[pos[0] + 10, pos[1] + 10] = 10000

        if len(self.ped_history) >= 15:
            ped_path = np.array(self.ped_history)
            ped_path = ped_path[:, :4]
            ped_path = ped_path.reshape((15, 4))


            car_path = np.array(self.car_history)

            if Config.PREDICT_PEDESTRIAN_PATH:
                pedestrian_path = self.ped_pred.get_single_prediction(ped_path, car_path)
                for node in pedestrian_path:
                    updated_risk_cmp[round(node[0]), round(node[1])] = 10000

        if self.scenario[0] == 11:
            self.grid_cost[9:16, 13:] = 10000
            self.risk_cmp[10:13, 13:] = 10000
            x, y = round(self.world.incoming_car.get_location().x), round(self.world.incoming_car.get_location().y)
            # hard coding incoming car path prediction
            obstacles.append((x, y - 1))
            obstacles.append((x, y - 2))
            obstacles.append((x, y - 3))
            obstacles.append((x, y - 4))
            obstacles.append((x, y - 5))
            # all grid locations occupied by car added to obstacles
            for i in [-1, 0, 1]:
                for j in [-2, -1, 0, 1, 2]:
                    obstacles.append((x + i, y + j))

        if self.scenario[0] in [10, 1] and self.world.walker.get_location().y > start[1] and start[0] >= 2.5:
            end = (end[0], start[1] - 6, end[2])

        if Config.RISK_AWARE_PATH:
            path, risk = self.risk_path_planner.find_path_with_risk(
                start, end, self.grid_cost, obstacles, car_speed, yaw, updated_risk_cmp, True, self.scenario[0]
            )
        else:
            path = self.find_path(start, end, self.grid_cost, obstacles)
            risk = []
        return (path, risk)


    def initialize_episode(self, episode_counter: int):
        # clear A2C specific buffers
        self.episode_belief_state_value_estimates.clear()
        self.episode_log_probabilities.clear()
        self.episode_entropies.clear()

        # setup initial inputs for LSTM Cell
        self.previous_lstm_hidden_state = torch.zeros((1, Config.HyLEAP.HIDDEN_LAYER_SIZE), 
                                                      dtype=torch.float64, 
                                                      device=Config.HyLEAP.DEVICE)
        self.previous_lstm_cell_state = torch.zeros((1, Config.HyLEAP.HIDDEN_LAYER_SIZE), 
                                                    dtype=torch.float64, 
                                                    device=Config.HyLEAP.DEVICE)

        # this will clear all relevant buffers
        return super().initialize_episode(episode_counter)


    def finalize_episode(self, episode_counter: int, terminated_early: bool):
        # regular epsiode clean-up
        # do not train on non-conclusive episodes
        if not terminated_early and Config.MODE is Mode.TRAIN:
            # delete first dummy reward entry (defaults to 0.0)
            del self.episode_rewards[0]
            # delete first dummy entry (defaults to Action.MAINTAIN)
            del self.episode_actions[0]

            # sanity check for consistent episode memory
            if len(self.episode_belief_state_value_estimates) != len(self.episode_rewards):
                raise ValueError(f"Inconsistent episode memory: Number of rewards {len(self.episode_rewards)}" 
                                 f" vs. belief state value estimates {len(self.episode_belief_state_value_estimates)}.")
            
            log_debug(
                f"self.episode_rewards: {self.episode_rewards} \n\n"
                f"self.episode_belief_state_value_estimates: {self.episode_belief_state_value_estimates}\n\n"
            )

            # train on this episode
            self.training_iteration(episode_counter)

            # model checkpointing
            if episode_counter != 0 and episode_counter % Config.MODEL_SAVE_FREQUENCY_EPISODES == 0:
                save_model(model_state_dict=self.model.state_dict(),
                        model_dir=Config.MODEL_DIR,
                        checkpoint=str(episode_counter),
                        name="hyLear_autobots")

            # always save most recent model
            save_model(model_state_dict=self.model.state_dict(),
                    model_dir=Config.MODEL_DIR,
                    checkpoint="latest",
                    name="hyLear_autobots")
            
        # IS-DESPOT C++ process is only running during training
        # this is only relevant when we terminated early as it initializes clean-up in IS-DESPOT'S C++ process
        if Config.MODE is Mode.TRAIN:
            return super().finalize_episode(episode_counter, terminated_early)