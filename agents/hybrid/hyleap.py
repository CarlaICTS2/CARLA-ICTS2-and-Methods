import os
import sys
import time
from threading import Thread
import multiprocessing as mp

import numpy as np
import torch
import carla
import cv2

from agents.planner.isdespotp import ISDespotP
from agents.learner.models import A2C

from utils.config import Config, Mode
from utils.connector import DespotBridge, Connection
from utils.utils import load_model, save_model
from utils.logger import log_debug, log_info, log_exception, log_learning_metrics


def run_car_intention_generation_process(birdview_car_intention_producer, 
                                         master_to_slave: mp.Queue,
                                         slave_to_master: mp.Queue,
                                         display: bool):
        while True:
            # block until observation received
            observation = master_to_slave.get(True)

            # modify agent vehilce location according to simulated node in IS-DESPOT
            modified_agent_vehicle_location = carla.Location(x=observation["agent_vehicle_x"], 
                                                             y=observation["agent_vehicle_y"], 
                                                             z=observation["agent_vehicle_z"])
            modified_agent_vehicle_rotation = carla.Rotation(pitch=observation["agent_vehicle_pitch"],
                                                             yaw=observation["agent_vehicle_yaw"],
                                                             roll=observation["agent_vehicle_roll"])
            modified_agent_vehicle_transform = carla.Transform(modified_agent_vehicle_location,
                                                             modified_agent_vehicle_rotation)
            
            # modify pedestrian location according to simulated node in IS-DESPOT
            modified_pedestrian_location = carla.Location(x=observation["pedestrian_x"], 
                                                          y=observation["pedestrian_y"], 
                                                          z=observation["pedestrian_z"])
            modified_pedestrian_rotation = carla.Rotation(pitch=observation["pedestrian_pitch"], 
                                                          yaw=observation["pedestrian_yaw"], 
                                                          roll=observation["pedestrian_roll"])
            modified_pedestrian_transform = carla.Transform(modified_pedestrian_location,
                                                            modified_pedestrian_rotation) 
            
            birdview_car_intention = birdview_car_intention_producer.produce(
                modified_agent_vehicle_transform,
                modified_pedestrian_transform,
                observation["agent_vehicle_past_trajectory"],
                # angle is omitted as it is not required for drawing a line with cv2
                [waypoint[:-1] for waypoint in observation["agent_vehicle_future_trajectory"]] ,
                observation["scenario_id"] # required for loading pre-drawn parked car canvas
            )

            # produces np.ndarray of shape (height, width, 3)
            birdview_car_intention = birdview_car_intention_producer.as_rgb(birdview_car_intention)

            # print agent vehicle and pedestrian positions + angles as additional debug information
            if display:
                # determine hspace between newlines
                (_, label_height), _ = cv2.getTextSize("text", cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
                # list containing lines to be printed on car intention images
                lines = [                
                    f"agent vehicle (x: {observation['agent_vehicle_x']:.2f}, "
                    f"y: {observation['agent_vehicle_y']:.2f}, "
                    f"z: {observation['agent_vehicle_z']:.2f}, "
                    f"a: {observation['agent_vehicle_yaw']:.2f})",
                    f"pedstrian (x: {observation['pedestrian_x']:.2f}, "
                    f"y: {observation['pedestrian_y']:.2f}, "
                    f"z: {observation['pedestrian_z']:.2f}, "
                    f"a: {observation['pedestrian_yaw']:.2f})",
                    f"initial position: {observation['initial_position']}",
                    f"goal position: {observation['goal_position']}",
                    f"step: {observation['step_counter']}",
                    f"search depth: {observation['search_depth']}",
                    f"current velocity: {observation['agent_vehicle_velocity']:.2f}km/h",
                    f"previous action: {'DEC' if observation['previous_action'] == 0 else 'MAIN' if observation['previous_action'] == 1 else 'ACC'}",
                    f"previous reward: {observation['previous_reward']}"
                ]
                # cv2 can't print multiple lines in a single putText() call...
                for index, line in enumerate(lines):
                    cv2.putText(
                        birdview_car_intention, # image
                        line, # text
                        # offset line position by height of text
                        (0, birdview_car_intention_producer.target_size.height//8 + index*2*label_height), # position
                        cv2.FONT_HERSHEY_SIMPLEX, # font
                        0.3, # font scale
                        (255,255,255), # font color
                        1, # line-thickness         
                        2 # line type
                    )
            # notify master
            slave_to_master.put(birdview_car_intention)   


class HyLEAP(ISDespotP):
    def __init__(self, client, world, carla_map, scenario):
        super(HyLEAP, self).__init__(client, world, carla_map, scenario)

        # must use cuda
        if not torch.cuda.is_available(): raise ValueError("No CUDA capable GPU found.")

        # model
        self.model: A2C = None

        # used for parallelizing car intention image generation as a single image can take up to 10ms to create
        self.car_intention_generation_workers = []
        self.message_queues = []

        # car intention of the root node of each DESPOT associated with the current simulation step (used for debugging)
        self.root_node_car_intention = None

        # episode buffers of neural network's predictions for training
        self.episode_belief_state_value_estimates: list = []
        self.episode_predicted_policies: list = []

        # current planning depth of IS-DESPOT in this step 
        self.despot_search_depth: int = 0
        # because the functions of hyleap & hyplan are not directly called by the controller
        self.step_counter: int = 0

        # define optimizer
        self.optimizer = None
        
        # define losses with reduction="none" so we can check each loss individually for correctness, i.e. > 0
        self.actor_loss_function = torch.nn.CrossEntropyLoss(reduction="none")
        self.critic_loss_function = torch.nn.MSELoss(reduction="none")

        # init connections here so that controller.py waits for connections to be established
        self.establish_evaluation_connection()

        self.initialize_car_intention_generation_workers()

        # do not run as process to avoid having to communicate data/model params
        # speed up would only be marginal
        Thread(target=self.run_hyleap_server).start()


    # ================================================================================================= #
    #               METHODS FOR CONTROLLING IS-DESPOT C++ PROCESS INTERACTION                           #
    # ================================================================================================= #
    def establish_evaluation_connection(self):
        if self.despot_connection is None:
            raise TypeError("IS-DESPOT C++ process interface has not been initialized.")
        if not isinstance(self.despot_connection, DespotBridge):
            raise TypeError(f"Invalid IS-DESPOT C++ prcess interface: " 
                            f"Expected '{DespotBridge}', got '{type(self.despot_connection)}'.")
        
        # general control connection
        self.despot_connection.establish_connection(Connection.HyLEAP_EVALUATION)



    # ================================================================================================= #
    #                                       HyLEAP SETUP METHODS                                        #
    # ================================================================================================= #
    def initialize_model(self):
        if self.model is not None:
            raise TypeError("Exisitng HyLEAP model cannot be overwritten.")
        log_info(f"initializing HyLEAP model...")

        self.model = A2C(hidden_dim=Config.HyLEAP.HIDDEN_LAYER_SIZE, use_dropout=Config.HyLEAP.DROPOUT).double().cuda()
        if Config.MODE is not Mode.TRAIN: self.model.eval()
        log_info(self.model)
          

    def initialize_optimizer(self):
        if self.optimizer is not None:
            raise TypeError("Exisitng optimizer cannot be overwritten.")
        if not isinstance(self.model, A2C):
            raise ValueError(f"Invalid HyLEAP model type: Expected '{A2C}', got '{type(self.model)}'.")
        log_info("initializing optimizers...")

        # initialize optimizers
        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(), 
            lr=Config.HyLEAP.LEARNING_RATE, 
            alpha=Config.HyLEAP.DECAY,
            momentum=Config.HyLEAP.MOMENTUM, 
            eps=Config.HyLEAP.EPSILON, 
            weight_decay=Config.HyLEAP.L2_DECAY
        )


    def initialize_car_intention_generation_workers(self):
        if len(self.car_intention_generation_workers) != 0:
            raise ValueError("Car intention generation processes already initialized.")
        if os.cpu_count() < 12:
            raise NotImplementedError("Invalid number of CPU cores. Fixed implementation requires at least 12.")

        log_info("initializing car intention generation workers...")
        # create as many processes as possible - (this script process + IS-DESPOT process + CARLA simulation server process)
        number_of_workers = os.cpu_count()-3
        if number_of_workers > 12: number_of_workers = 12
        # local script execution is just for debugging
        if not Config.Carla.REMOTE: number_of_workers = 1

        for cpu_core_id in range(number_of_workers):
            # fork instead of spawning because latter does some funky stuff when threads are created (c.f. run_hyleap_server())
            master_to_slave = mp.get_context('spawn').Queue()
            slave_to_master = mp.get_context('spawn').Queue()
            process = mp.get_context('spawn').Process(
                target=run_car_intention_generation_process, 
                args=(
                    self.birdview_car_intention_producer, 
                    master_to_slave,
                    slave_to_master,
                    Config.DISPLAY,
                )
            )
            process.start()
            log_info(f"started worker process with ID: {cpu_core_id}")
            self.message_queues.append((master_to_slave, slave_to_master))
            self.car_intention_generation_workers.append(process) 


    # ================================================================================================= #
    #                                        HyLEAP CORE METHODS                                        #
    # ================================================================================================= #
    # recycles predictions made during inference and essentially only calculates the loss
    def training_iteration(self, episode_counter: int):
        # required for getting a proper time-reading of cuda related tasks
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        belief_state_value_estimates = torch.vstack(self.episode_belief_state_value_estimates).squeeze()
        # rewards used for training: must contain the final reward received for the terminal state and must be appropriately scaled
        N = len(self.episode_rewards) - 1
        # propagate rewards backward in time
        for i in range(1, N + 1):
            # the current reward is equal to itself + the discounted reward of the next steps
            self.episode_rewards[N - i] += Config.HyLEAP.GAMMA * self.episode_rewards[N - i + 1]
        # shape: [steps, 1]
        rewards = torch.tensor(self.episode_rewards, dtype=torch.float64, device=Config.HyLEAP.DEVICE)

        if Config.HyPLAN.IMPROVE_REWARD:
            # rewards vector is normalised to have 0 mean and 1 std
            rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float64).eps.item())

        # shape: [steps, 3]
        predicted_policies = torch.vstack(self.episode_predicted_policies)
        # shape: [steps, 3]
        despot_policies = torch.tensor(self.episode_despot_policies, 
                                       dtype=torch.float64, 
                                       device=Config.HyLEAP.DEVICE)         

        log_debug(
            f"belief_state_value_estimates.shape: {belief_state_value_estimates.shape}, "
            f"rewards.shape: {rewards.shape}, "
            f"predicted_policies.shape: {predicted_policies.shape}, "
            f"despot_policies.shape: {despot_policies.shape}"
        )
        
        # calculate actor loss
        actor_loss_episode = self.actor_loss_function(predicted_policies, despot_policies)
        for loss in actor_loss_episode.tolist():
            if loss < 0:
                raise ValueError(f"Invalid actor loss: By definition '{self.actor_loss_function}' " 
                                 f"can not be negative, but returned {loss:.4f}.")
        actor_loss = actor_loss_episode.mean()
        # calculate critic loss
        critic_loss_episode = self.critic_loss_function(belief_state_value_estimates, rewards)
        for loss in critic_loss_episode.tolist():
            if loss < 0:
                raise ValueError(f"Invalid critic loss: By definition '{self.critic_loss_function}' " 
                                 f"can not be negative, but returned {loss:.4f}")
        critic_loss = critic_loss_episode.mean()
    
        # get total loss
        total_loss = actor_loss + critic_loss

        log_debug(
            f"total_loss {type(total_loss)}:{total_loss.dtype}:" +\
            f"{np.shape(total_loss)}:{total_loss.requires_grad}:\n{total_loss}"
        )    
        log_debug(
            f"actor_loss_episode.shape: {actor_loss_episode.shape}, "
            f"critic_loss_episode.shape: {critic_loss_episode.shape}, "
            f"actor_loss: {actor_loss}, critic_loss: {critic_loss}"
        )
        
        # debug information
        torch.cuda.synchronize()
        log_info(f"total loss for episode {episode_counter}: {total_loss.item():.4f} "
                 f"calculated in {(time.perf_counter() - start_time)*1000:.4f}ms")

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
            "reward": sum(self.episode_rewards)
        })


    def inference_iteration(self):
        # receive newly expanded nodes from IS-DESPOT
        previous_lstm_hidden_state, \
        previous_lstm_cell_state, \
        nodes,\
        agent_vehicle_past_simulated_trajectories, \
        is_root_node = self.despot_connection.receive_expanded_nodes(Connection.HyLEAP_EVALUATION)

        # for debugging
        if is_root_node:
            self.step_counter += 1
            self.despot_search_depth = 0
        else:
            self.despot_search_depth += 1

        number_of_samples = len(nodes)
        # obtain current simulation state of ego vehicle and pedestrian in CARLA
        agent_vehicle_transform = self.world.player.get_transform() 
        pedestrian_transform = self.world.walker.get_transform()

        # list of car intentions
        observations = []

        # modify CARLA world state according to IS-DESPOT's simulation for each received observation
        # this general case also correctly captures the special case when a node is a root node
        car_intention_start = time.perf_counter()
        for index in range(number_of_samples):
            # obtain current node
            node = nodes[index]                

            # add waypoints to past trajectory that have already been visited in IS-DESPOT's planning simulation
            modified_ego_vehicle_past_trajectory = self.episode_agent_vehicle_past_trajectory.copy() + \
                                                   agent_vehicle_past_simulated_trajectories[index]
            
            # prepare message for worker process
            observation = {
                "scenario_id": self.scenario[0], # required for loading pre-drawn parked car canvas
                "agent_vehicle_x": node[0],
                "agent_vehicle_y": node[1],
                "agent_vehicle_z": agent_vehicle_transform.location.z,
                "agent_vehicle_pitch": agent_vehicle_transform.rotation.pitch,
                "agent_vehicle_yaw": node[2],
                "agent_vehicle_roll": agent_vehicle_transform.rotation.roll,
                "pedestrian_x": node[3],
                "pedestrian_y": node[4],
                "pedestrian_z": pedestrian_transform.location.z,
                "pedestrian_pitch": pedestrian_transform.rotation.pitch,
                "pedestrian_yaw": pedestrian_transform.rotation.yaw,
                "pedestrian_roll": pedestrian_transform.rotation.roll,
                "agent_vehicle_past_trajectory": modified_ego_vehicle_past_trajectory,
                "agent_vehicle_future_trajectory": self.step_agent_vehicle_future_trajectory,
                # debug information when displaying
                "initial_position": f"{self.scenario[3]}",
                "goal_position": f"{self.scenario[2]}",
                "step_counter": self.step_counter,
                "search_depth": self.despot_search_depth,
                "agent_vehicle_velocity": node[5] * 3.6, # for km/h
                "previous_reward": node[5],
                "previous_action": node[6]
            }
            # cycle through workers
            index = index % len(self.car_intention_generation_workers)
            # send observation in master_to_slave queue
            self.message_queues[index][0].put(observation)             

        # collect car intention images
        for index in range(number_of_samples):
            # cycle through workers
            index = index % len(self.car_intention_generation_workers)
            # get generated car intention from slave_to_master queue
            car_intention = self.message_queues[index][1].get(True)
            observations.append(car_intention)
            # update root node car intention (for debugging)
            if is_root_node: self.root_node_car_intention = car_intention
            # debugging
            if Config.DISPLAY and False:
                # compare modified vs root car intention
                car_intention_comparison = np.hstack((self.root_node_car_intention, car_intention))
                cv2.namedWindow(winname='CarIntention')
                cv2.imshow('CarIntention', car_intention_comparison)
                # stop simulation until manual key stroke is registered
                cv2.waitKey()  

        log_debug(
            f"constructed car-intention images for {len(nodes)} nodes in "
            f"{(time.perf_counter()-car_intention_start)*1000:.4f}ms"
        )

        # construct tensor of all car intention images generated at the current planning depth of IS-DESPOT
        observations = torch.tensor(np.array(observations), dtype=torch.float64, device=Config.HyLEAP.DEVICE)\
                            .transpose(1, -1)
        # all observation at any given depth have the same previous lstm state, because they all share the same parent
        previous_lstm_hidden_state = torch.tensor(previous_lstm_hidden_state, 
                                                  dtype=torch.float64, 
                                                  device=Config.HyLEAP.DEVICE)
        previous_lstm_cell_state = torch.tensor(previous_lstm_cell_state, 
                                                dtype=torch.float64, 
                                                device=Config.HyLEAP.DEVICE)

        # construct feature tensor, which is fed to the LSTM layer in addition to the convolved observation
        # for each step t it consists of the car's current speed v_t, the last executed action a_t-1 
        # (as determined by IS-DESPOT) and the reward received for the previous simulation step r_t-1.
        car_speeds = torch.tensor(nodes[:, 5], dtype=torch.float64, device=Config.HyLEAP.DEVICE)
        actions = torch.tensor(nodes[:, 6], dtype=torch.float64, device=Config.HyLEAP.DEVICE)
        rewards = torch.tensor(nodes[:, 7], dtype=torch.float64, device=Config.HyLEAP.DEVICE)
        features = torch.stack((car_speeds, actions, rewards), dim=0).transpose(1, 0)

        # commented because even when not printing this is incurring computational overhead
        ''' 
        log_debug(
                f"previous_lstm_hidden_state.shape: {previous_lstm_hidden_state.shape}, "
                f"previous_lstm_cell_state.shape: {previous_lstm_cell_state.shape}, " 
                f"observations.shape: {observations.shape}"
        )
        log_debug(
            f"car_speeds.shape: {car_speeds.shape}, actions.shape: {actions.shape}, "
            f"rewards.shape: {rewards.shape}, features.shape: {features.shape}"
        )
        log_debug(
            f"previous_lstm_hidden_state: {previous_lstm_hidden_state} \n"
            f"previous_lstm_cell_state: {previous_lstm_cell_state} \n" 
            f"observations: {observations} \n"
            f"car_speeds: {car_speeds} \nactions: {actions} \nrewards: {rewards} \nfeatures: {features}"
        )
        '''

        # forward pass without torch.no_grad() because predictions will be recycled for training later
        actions, values, (current_lstm_hidden_state, current_lstm_cell_state) = self.model(observations, 
                                                                                           previous_lstm_hidden_state, 
                                                                                           previous_lstm_cell_state,
                                                                                           features)
        '''
        log_debug(
            f"actions.shape: {actions.shape}, values.shape: {values.shape}, "
            f"current_lstm_hidden_state.shape: {current_lstm_hidden_state.shape}, "
            f"current_lstm_cell_state.shape: {current_lstm_cell_state.shape}"
        )
        log_debug(
            f"actions: {actions}, values: {values}, \n"
            f"current_lstm_hidden_state: {current_lstm_hidden_state}, \n"
            f"current_lstm_cell_state: {current_lstm_cell_state}"
        )
        '''
        
        # save car intention images of root nodes
        if Config.RECORD_CAR_INTENTION_IMAGES and is_root_node:
            car_intention_save_dir = os.path.join(Config.DATA_DIR, f"episode_{self.episode_counter}")
            os.makedirs(car_intention_save_dir, exist_ok=True)
            car_intention_save_path = os.path.join(car_intention_save_dir, 
                                                    f"car_intention_image_at_step_{self.step_counter}.jpg")
            cv2.imwrite(car_intention_save_path, self.root_node_car_intention)

        # remember neural network's prediction for training, 
        # i.e. the network is only trained on the root nodes of the episode (which were not simulated by IS-DESPOT)
        if is_root_node and Config.MODE is Mode.TRAIN:
            self.episode_belief_state_value_estimates.append(values)
            self.episode_predicted_policies.append(actions)
        
        # shape: [batch_size, 1]
        actions = torch.argmax(actions, dim=1).cpu().detach().numpy()
        # shape: [batch_size, 1]
        values = values.squeeze(1).cpu().detach().numpy()
        # shape: [batch_size, 128]
        current_lstm_hidden_state = current_lstm_hidden_state.cpu().detach().numpy()
        # shape: [batch_size, 128]
        current_lstm_cell_state = current_lstm_cell_state.cpu().detach().numpy()

        message = []
        # len(nodes) = batch_size
        for node in range(len(nodes)):
            # append node's lstm state
            [message.append(element) for element in np.concatenate((current_lstm_hidden_state[node], 
                                                                    current_lstm_cell_state[node]), axis=0)]
            message.append(actions[node])
            message.append(values[node])
            # dummy uncertainty value
            message.append(1.0)

        self.despot_connection.send_drla_prediction(message, Connection.HyLEAP_EVALUATION)


    def run_hyleap_server(self):
        try:    
            self.initialize_model()
            if Config.RESUME or Config.MODE is not Mode.TRAIN: 
                self.model = load_model(model=self.model, 
                                        model_dir=Config.MODEL_DIR,
                                        checkpoint="latest",
                                        key="hyleap")
            if Config.MODE is Mode.TRAIN: 
                self.initialize_optimizer()
            if Config.MODE is Mode.TRAIN and self.optimizer is None:
                raise ValueError(f"Invalid HyLEAP optimizer: Not initialized.")     
            if Config.MODE is Mode.TRAIN and not isinstance(self.optimizer, torch.optim.RMSprop):
                raise ValueError(f"Invalid HyLEAP optimizer: " 
                                 f"Expected '{torch.optim.RMSprop}', got '{type(self.optimizer)}'.")
            if self.despot_connection is None:
                raise TypeError("IS-DESPOT C++ process interface has not been initialized.")
            if not isinstance(self.despot_connection, DespotBridge):
                raise TypeError(f"Invalid IS-DESPOT C++ prcess interface: " 
                                f"Expected '{DespotBridge}', got {type(self.despot_connection)}.")
            
            while not Config.TERMINATE:
                self.inference_iteration()

        except: log_exception(*sys.exc_info())


    def initialize_episode(self, episode_counter: int):
        self.episode_belief_state_value_estimates.clear()
        self.episode_predicted_policies.clear()
        return super().initialize_episode(episode_counter)


    def finalize_episode(self, episode_counter: int, terminated_early: bool):
        # regular epsiode clean-up
        # do not train during testing...
        if Config.MODE is Mode.TRAIN:
            # delete first dummy reward entry (defaults to 0.0)
            del self.episode_rewards[0]
            # delete first dummy entry (defaults to Action.MAINTAIN)
            del self.episode_actions[0]

            # sanity check for consistent episode memory
            if len(self.episode_belief_state_value_estimates) != len(self.episode_rewards):
                raise ValueError(f"Inconsistent episode memory: Number of rewards {len(self.episode_rewards)}" 
                                 f" vs. belief state value estimates {len(self.episode_belief_state_value_estimates)}.")
            if len(self.episode_predicted_policies) != len(self.episode_despot_policies):
                raise ValueError(f"Inconsistent episode memory: Number of IS-DESPOT policies {len(self.episode_despot_policies)}" 
                                 f" vs. NN's predicted policies {len(self.episode_predicted_policies)}.")
            
            log_debug(
                f"self.episode_rewards: {self.episode_rewards} \n\n"
                f"self.episode_belief_state_value_estimates: {self.episode_belief_state_value_estimates}\n\n"
            )
            log_debug(
                f"self.episode_despot_policies: {self.episode_despot_policies}\n\n"
                f"self.episode_predicted_policies: {self.episode_predicted_policies}\n\n"
            )        

            # train on this episode
            self.training_iteration(episode_counter)

            # model checkpointing
            if episode_counter != 0 and episode_counter % Config.MODEL_SAVE_FREQUENCY_EPISODES == 0:
                save_model(
                    model_state_dict=self.model.state_dict(),
                    model_dir=Config.MODEL_DIR,
                    checkpoint=str(episode_counter),
                    name="hyleap"
                )

            # always save most recent model
            save_model(
                model_state_dict=self.model.state_dict(),
                model_dir=Config.MODEL_DIR,
                checkpoint="latest",
                name="hyleap"
            )
            
        # always called
        return super().finalize_episode(episode_counter, terminated_early)