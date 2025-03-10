import time

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import carla

from agents.learner.rlagent import RLAgent
from agents.learner.models import A2C
from utils.config import Config, Mode, Action
from utils.utils import load_model, save_model
from utils.logger import log_debug, log_info, log_learning_metrics


class NavA2C(RLAgent):
    def __init__(self, client, client_world):
        
        # init RLAgent
        super(NavA2C, self).__init__(client, client_world)

        # must use cuda
        if not torch.cuda.is_available(): raise ValueError("No CUDA capable GPU found.")

        # episode buffers of neural network's predictions for training
        self.episode_belief_state_value_estimates: list = []
        self.episode_log_probabilities: list = []
        self.episode_entropies: list = []

        self.critic_loss = torch.nn.MSELoss(reduction="mean")

        # declare model
        self.model: A2C = None
        # define model
        self.initialize_model()
        # load model if necessary
        if Config.RESUME or Config.MODE is not Mode.TRAIN: 
            self.model = load_model(
                model=self.model, model_dir=Config.MODEL_DIR, checkpoint="latest", 
                key=f"{'nava2c' if Config.A2C.NavA2C else 'a2c'}" 
            )
            
        # declare optimizer
        self.optimizer = None
        # define optimizer
        if Config.MODE is Mode.TRAIN: self.initialize_optimizer()

            
    # ================================================================================================= #
    #                                       NavA2C SETUP METHODS                                        #
    # ================================================================================================= #
    def initialize_model(self):
        if self.model is not None:
            raise TypeError("Exisitng NavA2C model cannot be overwritten.")
        log_info(f"initializing NavA2C model...")

        self.model = A2C(hidden_dim=Config.A2C.HIDDEN_LAYER_SIZE, use_dropout=Config.A2C.DROPOUT).double().cuda()
        if Config.MODE is not Mode.TRAIN: self.model.eval()
        log_info(self.model)
          

    def initialize_optimizer(self):
        if self.optimizer is not None:
            raise TypeError("Exisitng optimizer cannot be overwritten.")
        if not isinstance(self.model, A2C):
            raise ValueError(f"Invalid NavA2C model type: Expected '{A2C}', got '{type(self.model)}'.")
        log_info("initializing optimizer...")

        # initialize optimizers
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=Config.A2C.LEARNING_RATE
        )

        if self.optimizer is None:
            raise ValueError(f"Invalid NavA2C optimizer: Not initialized.")     
        if not isinstance(self.optimizer, torch.optim.Adam):
            raise ValueError(
                f"Invalid NavA2C optimizer: Expected '{torch.optim.Adam}', got '{type(self.optimizer)}'."
            )


    # ================================================================================================= #
    #                                        NavA2C CORE METHODS                                        #
    # ================================================================================================= #
    def initialize_episode(self, episode_counter: int, scenario: tuple):
        # clear A2C specific buffers
        self.episode_belief_state_value_estimates.clear()
        self.episode_log_probabilities.clear()
        self.episode_entropies.clear()

        # setup initial inputs for LSTM cell
        self.previous_lstm_hidden_state = torch.zeros(
            (1, Config.A2C.HIDDEN_LAYER_SIZE if Config.A2C.NavA2C else Config.A2C.HIDDEN_LAYER_SIZE // 2), 
             dtype=torch.float64, 
             device=Config.DEVICE
        )
        self.previous_lstm_cell_state = torch.zeros(
            (1, Config.A2C.HIDDEN_LAYER_SIZE if Config.A2C.NavA2C else Config.A2C.HIDDEN_LAYER_SIZE // 2), 
            dtype=torch.float64, 
            device=Config.DEVICE
        )

        # this will clear all relevant buffers
        return super().initialize_episode(episode_counter, scenario)


    def finalize_episode(self, episode_counter: int, terminated_early: bool):
        # we even train on episode that have been terminated early (i.e. that have not reached a conclusive end, 
        # e.g. goal or collision), because otherwise purely learning based agents will never be trained
        # (as their initially random policy does not allow them to encounter either a collision or reach the goal)
        if Config.MODE is Mode.TRAIN:
            # delete first dummy reward entry (defaults to 0.0)
            del self.episode_rewards[0]
            # delete first dummy entry (defaults to Action.MAINTAIN)
            del self.episode_actions[0]

            # sanity check for consistent episode memory
            if len(self.episode_belief_state_value_estimates) != len(self.episode_rewards):
                raise ValueError(
                    f"Inconsistent episode memory: Number of rewards {len(self.episode_rewards)} " 
                    f"vs. belief state value estimates {len(self.episode_belief_state_value_estimates)}."
                )
            if len(self.episode_log_probabilities) != len(self.episode_rewards):
                raise ValueError(
                    f"Inconsistent episode memory: Number of rewards {len(self.episode_rewards)} " 
                    f"vs. log probabilities {len(self.episode_log_probabilities)}."
                )                
            if len(self.episode_entropies) != len(self.episode_rewards):
                raise ValueError(
                    f"Inconsistent episode memory: Number of rewards {len(self.episode_rewards)} " 
                    f"vs. entropies {len(self.episode_entropies)}."
                )   
            
            log_debug(
                f"self.episode_rewards: {self.episode_rewards} \n"
                f"self.episode_belief_state_value_estimates: {self.episode_belief_state_value_estimates}\n"
                f"self.episode_log_probabilities: {self.episode_log_probabilities}\n"
                f"self.episode_entropies: {self.episode_entropies}\n"
            )

            # train on this episode
            self.training_iteration(episode_counter)

            # model checkpointing
            if episode_counter != 0 and episode_counter % Config.MODEL_SAVE_FREQUENCY_EPISODES == 0:
                save_model(
                    model_state_dict=self.model.state_dict(), 
                    model_dir=Config.MODEL_DIR, 
                    checkpoint=str(episode_counter), 
                    name=f"{'nava2c' if Config.A2C.NavA2C else 'a2c'}" 
                )

            # always save most recent model
            save_model(
                model_state_dict=self.model.state_dict(),
                model_dir=Config.MODEL_DIR,
                checkpoint="latest",
                name=f"{'nava2c' if Config.A2C.NavA2C else 'a2c'}" 
            )

        return super().finalize_episode(episode_counter, terminated_early)


    def run_step(self, step_counter: int):
        self.vehicle = self.client_world.ego_vehicle
        transform = self.vehicle.get_transform()
        start = (self.vehicle.get_location().x, self.vehicle.get_location().y, transform.rotation.yaw)
        end = self.episode_ego_vehicle_goal_position

        (self.step_ego_vehicle_future_trajectory, risk) = super(NavA2C, self).get_path_simple(start, end)

        agent_vehicle_control = carla.VehicleControl(
            throttle=0.0, steer=0.0, brake=0.0, hand_brake=False, reverse=False, manual_gear_shift=False, gear=0
        )

        # buggy path planner returning empty path means we skip this step and just maintain current velocity
        if not len(self.step_ego_vehicle_future_trajectory):
            step_summary = {"skipped_step": True, "control": agent_vehicle_control}
            return step_summary
        
        # if planned future agent vehicle trajectory is non-trivial
        agent_vehicle_control.steer = \
            (self.step_ego_vehicle_future_trajectory[2][2] - start[2]) / Config.Carla.MAX_STEERING_ANGLE

        # perceive current observation
        super(NavA2C, self).get_current_observation(step_counter)

        birdview_car_intention = super(NavA2C, self).get_birdview_car_intention(
            self.vehicle.get_transform(),
            self.client_world.pedestrian.get_transform(),
            self.episode_ego_vehicle_past_trajectory,
            self.step_ego_vehicle_future_trajectory,
            step_counter
        )

        # get A2C's prediction
        learner_action = self.inference_iteration(birdview_car_intention, step_counter)

        # translate action into CARLA vehicle control commands
        if learner_action is Action.DECELERATE:
            agent_vehicle_control.brake = 0.6
        elif learner_action is Action.ACCELERATE:
            agent_vehicle_control.throttle = 0.6

        # remember action
        self.episode_controls.append(agent_vehicle_control)
        self.episode_actions.append(learner_action)

        # we need the current action before we can calculate the reward
        if Config.REWARD_FUNCTION == "akash":
            step_summary = super(NavA2C, self).get_reward_akash(step_counter)
        elif Config.REWARD_FUNCTION == "nils":
            step_summary = super(NavA2C, self).get_reward_nils(step_counter)
        elif Config.REWARD_FUNCTION == "despot":
            step_summary = super(NavA2C, self).get_reward_despot(step_counter)

        step_summary["skipped_step"] = False
        step_summary["car_intention"] = birdview_car_intention
        step_summary["control"] = agent_vehicle_control
        step_summary["action"] = learner_action

        return step_summary


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
            self.episode_rewards[N - i] += Config.A2C.DISCOUNT * self.episode_rewards[N - i + 1]
        # shape: [steps, 1]
        rewards = torch.tensor(self.episode_rewards, dtype=torch.float64, device=Config.DEVICE)

        if Config.A2C.STANDARDIZE_REWARD:
            # rewards vector is normalised to have 0 mean and 1 std
            rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float64).eps.item())     

        belief_state_value_estimates = torch.vstack(self.episode_belief_state_value_estimates).squeeze()
        advantage = torch.sub(rewards, belief_state_value_estimates)
        log_probabilities = torch.vstack(self.episode_log_probabilities).squeeze()
        entropies = torch.vstack(self.episode_entropies).squeeze()

        actor_loss = torch.mul(torch.mul(log_probabilities, -1), advantage).mean()
        critic_loss = self.critic_loss(belief_state_value_estimates, rewards)
        total_loss = actor_loss + critic_loss + Config.A2C.ENTROPY_FACTOR * entropies.mean()
        # debug information
        torch.cuda.synchronize()
        log_info(
            f"total loss for episode {episode_counter}: {total_loss.item():.4f} "
            f"calculated in {(time.perf_counter() - start_time)*1000:.4f}ms"
        )

        # model.zero_grad() and optimizer.zero_grad() are the same IF all model parameters are in that optimizer
        # it is safer to call model.zero_grad() to make sure all grads are zero, 
        # e.g. if we have two or more optimizers for one model
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
            # sum of undiscounted rewards
            "reward": sum(self.episode_rewards),
            "entropy": torch.stack(self.episode_entropies).mean().cpu().detach().numpy().tolist()
        })


    def inference_iteration(self, car_intention: np.ndarray, step_counter: int) -> Action:
        # construct tensor of all car intention images generated at the current planning depth of IS-DESPOT
        observations = torch.tensor(
            car_intention, dtype=torch.float64, device=Config.DEVICE
        ).transpose(-1, 0).unsqueeze(0)

        # we must have perceived the current velocity that is associated with the provided car intention image
        if len(self.episode_ego_vehicle_speeds) != step_counter:
            raise ValueError(
                f"Invalid numer of episode agent vehicle speeds: "
                f"Expected {step_counter}, got {len(self.episode_ego_vehicle_speeds)} instead."
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
        
        # construct feature tensor, which is fed to the LSTM layer in addition to the convolved observation
        # for each step t it consists of the car's current speed v_t, the last executed action a_t-1 
        # (as determined by IS-DESPOT) and the reward received for the previous simulation step r_t-1.
        features = torch.tensor(
            (self.episode_ego_vehicle_speeds[-1], 
            self.episode_actions[-1].value,
            self.episode_rewards[-1]),
            dtype=torch.float64,
            device=Config.DEVICE
        ).unsqueeze(0)
                        
        '''        
        log_debug(
            f"previous_lstm_hidden_state.shape: {self.previous_lstm_hidden_state.shape}, "
            f"previous_lstm_cell_state.shape: {self.previous_lstm_cell_state.shape}, " 
            f"observations.shape: {observations.shape}, "
            f"features.shape: {features.shape}"
        )
        log_debug(
            f"STEP={step_counter} feature vector (speed, action, reward): {self.episode_agent_vehicle_speeds[-1]}, "
            f"{self.episode_actions[-1].value}, "
            f"{self.episode_rewards[-1]}"
        )
        '''

        # forward pass without torch.no_grad() because predictions will be recycled for training later
        action_logits, value, (self.previous_lstm_hidden_state, self.previous_lstm_cell_state) = self.model(
            observations, 
            self.previous_lstm_hidden_state, 
            self.previous_lstm_cell_state,
            features
        )
          
        '''        
        log_debug(
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