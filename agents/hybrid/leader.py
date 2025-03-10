import sys
from itertools import zip_longest
from threading import Thread

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from agents.planner.isdespotp import ISDespotP
from agents.hybrid.models import AttentionNet, CriticNet
from agents.hybrid.buffers import ReplayBuffer

from utils.config import Config, Mode, StringEnum
from utils.connector import DespotBridge, Connection
from utils.logger import log_debug, log_info, log_exception, log_learning_metrics
from utils.utils import load_model, save_model


class ModelType(StringEnum):
    ATTENTION = AttentionNet
    CRITIC = CriticNet


# this implementation is based on the published code of the original paper: https://github.com/modanesh/LEADER
class LEADER(ISDespotP):
    
    def __init__(self, client, world, carla_map, scenario):
        super(LEADER, self).__init__(client, world, carla_map, scenario)
        # keep track of model updates
        self.training_counter = 0

        # temporary buffer
        self.trajectory_replay_buffer = []

        # primary buffer containing individual data points
        self.replay_buffer = ReplayBuffer(Config.Leader.REPLAY_MAX)

        # models
        self.attention_model: AttentionNet = None
        self.critic_model: CriticNet = None

        # memory
        self.attention_memory: torch.Tensor = None
        self.critic_memory: torch.Tensor = None

        # optimizers
        self.attention_model_optimizer: torch.optim.Adam = None
        self.critic_model_optimizer: torch.optim.Adam = None

        # loss functions
        self.mean_squared_error_loss = nn.MSELoss()
        self.elementwise_mean_squared_error = nn.L1Loss()

        # atm there are only scenarios whith one pedestrian
        Config.Leader.MAX_TRAJECTORY_LENGTH = Config.Leader.NUM_FEATURES * Config.Leader.MAX_PEDESTRIANS

        # init connections here so that the controller waits for connections to be established
        self.establish_evaluation_connection()

        # do not run as process to avoid having to communicate data/model params
        Thread(target=self.run_leader_server).start()


    # ================================================================================================= #
    #               METHODS FOR CONTROLLING IS-DESPOT C++ PROCESS INTERACTION                           #
    # ================================================================================================= #
    def establish_evaluation_connection(self):
        if self.despot_connection is None:
            raise TypeError("IS-DESPOT C++ process interface has not been initialized.")
        if not isinstance(self.despot_connection, DespotBridge):
            raise TypeError(f"Invalid IS-DESPOT C++ prcess interface: " 
                            f"Expected '{DespotBridge}', got '{type(self.despot_connection)}'.")
        
        # leader specific despot connection
        self.despot_connection.establish_connection(Connection.LEADER)


    # ================================================================================================= #
    #                                       LEADER SETUP METHODS                                        #
    # ================================================================================================= #
    def initialize_models(self):
        if self.attention_model is not None:
            raise ValueError("Exisitng attention model cannot be overwritten.")
        if self.critic_model is not None:
            raise ValueError("Exisitng critic model cannot be overwritten.")

        log_info(f"initializing models...")

        # only ever considers one pedestrian
        self.attention_model = AttentionNet(
            input_size=2*Config.Leader.MAX_FEATURE_LEN,
            gru_size=Config.Leader.MEMORY_SIZE,
            is_weight_size=Config.Leader.ATTENTION_SIZE
        ).float().to(Config.Leader.DEVICE)

        # considers input over all pedestrians
        self.critic_model = CriticNet(
            input_size=Config.Leader.MAX_TRAJECTORY_LENGTH*Config.Leader.MAX_FEATURE_LEN,
            gru_size=Config.Leader.MEMORY_SIZE
        ).float().to(Config.Leader.DEVICE)


    def initialize_memory(self):
        log_info("initializing model memory...")

        # initialize attention and critic memory
        self.attention_memory = torch.tensor(
            np.random.random((1, Config.Leader.MEMORY_SIZE)),
            device=Config.Leader.DEVICE,
            dtype=torch.float32,
            requires_grad=False,
            pin_memory=False
        )
        log_debug(
            f"initialized AttentionNet memory as {type(self.attention_memory)}:" +\
            f"{self.attention_memory.dtype}:{self.attention_memory.size()}:\n{self.attention_memory}"
        )
        
        self.critic_memory = torch.tensor(
            np.random.random((Config.Leader.BATCH_SIZE, Config.Leader.MEMORY_SIZE)), 
            device=Config.Leader.DEVICE,
            dtype=torch.float32,
            requires_grad=False,
            pin_memory=False
        )
        log_debug(
            f"initialized CriticNet memory as {type(self.critic_memory)}:" +\
            f"{self.critic_memory.dtype}:{self.critic_memory.size()}:\n{self.critic_memory}"
        )


    def initialize_optimizers(self):
        if self.attention_model_optimizer is not None:
            raise ValueError("Exisitng attention model optimizer cannot be overwritten.")
        if self.critic_model_optimizer is not None:
            raise ValueError("Exisitng attention model optimizer cannot be overwritten.")
        if not isinstance(self.attention_model, AttentionNet):
            raise ValueError(f"Invalid attention model type: Expected '{AttentionNet}', got '{type(self.attention_model)}'.")
        if not isinstance(self.critic_model, CriticNet):
            raise ValueError(f"Invalid critic model type: Expected '{CriticNet}', got '{type(self.critic_model)}'.")
        
        log_info("initializing optimizers...")
        # initialize optimizers
        self.attention_model_optimizer = optim.Adam(self.attention_model.parameters(), lr=Config.Leader.LEARNING_RATE)
        self.critic_model_optimizer = optim.Adam(self.critic_model.parameters(), lr=Config.Leader.LEARNING_RATE)


    def initialize_episode(self, episode_counter: int):
        self.initialize_memory()
        return super().initialize_episode(episode_counter)


    def finalize_episode(self, episode_counter: int, terminated_early: bool):
        # clean up non-conclusive episode termination in IS-DESPOT
        return super().finalize_episode(episode_counter, terminated_early)
    

    # ================================================================================================= #
    #                                        LEADER UTIL METHODS                                        #
    # ================================================================================================= #   
    def save_models(self, checkpoint:str = "latest"):
        # save attention model
        save_model(
            model_state_dict=self.attention_model.state_dict(), 
            checkpoint=checkpoint,
            model_dir=Config.MODEL_DIR,
            name="leader_generator",
            number=6
        )   
        # save critic model
        save_model(
            model_state_dict=self.critic_model.state_dict(), 
            checkpoint=checkpoint,
            model_dir=Config.MODEL_DIR,
            name="leader_critic",
            number=6
        )
        

    def load_models(self):
        # load attention model
        self.attention_model = load_model(
            model=self.attention_model,
            model_dir=Config.MODEL_DIR,
            checkpoint=Config.MODEL_CHECKPOINT,
            key="generator"
        )    
        # load critic model
        self.critic_model = load_model(
            model=self.critic_model,
            model_dir=Config.MODEL_DIR,
            checkpoint=Config.MODEL_CHECKPOINT,
            key="critic"
        )
   
    
    def normal_rand_attentions(self, att_len)-> list:
        weights = np.random.normal(0.5, 0.5, att_len)
        # this must not be 0, because we divide the existing belief by LEADER's attention
        # i.e. if pedestrian goal directions has probability mass == 0, we get nans/inf
        weights[weights < 0] = np.finfo(np.float).eps.item()
        weights /= sum(weights)
        return weights.tolist()


    def uniform_rand_attentions(self, att_len) -> list:
        weights = np.random.random(att_len)
        weights /= sum(weights)
        return weights.tolist()


    # ================================================================================================= #
    #                                        LEADER CORE METHODS                                        #
    # ================================================================================================= # 
    def training_iteration(self):
        if not isinstance(self.replay_buffer, ReplayBuffer):
            raise ValueError(f"Invalid replay buffer type: Expected 'ReplayBuffer', got '{type(self.replay_buffer)}'.")
        
        # too few samples for training
        executed_steps = len(self.replay_buffer)
        if executed_steps < Config.Leader.REPLAY_MIN:
            log_info(f'waiting for minimum buffer size... {executed_steps}/{Config.Leader.REPLAY_MIN}')
            return

        # shape = [BATCH_SIZE, 4*p, n] with p the number of maximum allowed pedestrians 
        # and n the number of possible goal directions (all rows are padded to n)
        sampled_data = np.array(self.replay_buffer.sample(Config.Leader.BATCH_SIZE))
        log_debug(
            f"sampled data for this batch {type(sampled_data)}:{sampled_data.dtype}:" +\
            f"{np.shape(sampled_data)}:\n{sampled_data}"
        )

        # each value corresponds to a different step
        # true_values = [v_hat_1, ...., v_hat_b]
        true_values = sampled_data[:, -1, 0]
        true_values = torch.tensor(true_values, dtype=torch.float32, device=Config.Leader.DEVICE).reshape([-1,1])
        log_debug(
            f"IS-DESPOT estimated values for this batch {type(true_values)}:" +\
            f"{true_values.dtype}:{np.shape(true_values)}:\n{true_values}"
        )

        # sampled data without labels (is-despot estimated v_hat)
        # shape: [BATCH_SIZE, 3*p, n] 
        train_data = sampled_data[:, :-1]
        log_debug(
            f"training data for this batch {type(train_data)}:{train_data.dtype}:" +\
            f"{np.shape(train_data)}\n{train_data}"
        )

        # shape: [BATCH_SIZE, 3*p*n]
        # critic_input
        # [
        #   [p(g_1), ..., p(g_n), car.x, car.y, car.v, car.theta, ped.x, ped.y, 0, ..., 0, p(g_1), ..., p(g_n)],
        #   .
        #   .
        #   .
        #   [p(g_1), ..., p(g_n), car.x, car.y, car.v, car.theta, ped.x, ped.y, 0, ..., 0, p(g_1), ..., p(g_n)]
        # ]
        critic_input = torch.tensor(train_data, dtype=torch.float32, device=Config.Leader.DEVICE) \
                            .reshape(Config.Leader.BATCH_SIZE, -1)
        log_debug(
            f"critic input for this batch {type(critic_input)}:{critic_input.dtype}:" +\
            f"{critic_input.size()}:\n{critic_input}"
        )

        # predict belief state values of provided steps
        # v_tilde = Critic(b_1, c_1, w_1, b_2, ..., b_p, c_p, w_p)
        estimated_values, self.critic_memory = self.critic_model(critic_input, self.critic_memory.detach())
        log_debug(
            f"critic pedictions for this batch {type(estimated_values)}:" +\
            f"{estimated_values.dtype}:{estimated_values.size()}:\n{estimated_values}"
        )
        
        # critic_loss = ||v_tilde-v_hat||
        critic_loss = self.mean_squared_error_loss(estimated_values, true_values)
        log_debug(
            f"critic loss for this batch {type(critic_loss)}:" +\
            f"{critic_loss.dtype}:{critic_loss.size()}:\n{critic_loss}"
        )

        # update critic network weights accordingly
        # reset gradients
        self.critic_model_optimizer.zero_grad()
        # propagate loss backwards
        critic_loss.backward()
        # apply gradient update
        self.critic_model_optimizer.step()
        
        # train the critic first for 1000 iterations with uniform attention and once the
        # critic is "reasonably" trained, start training the attention generator
        # if we are resuming, we have likely already seen more than 1000 steps...
        attention_loss = None
        if self.training_counter >= Config.Leader.CRITIC_WARM_UP_ITERATIONS or Config.RESUME:
            # dummy true values: w_i = Generator(b_i, c_i)
            true_values = torch.zeros(estimated_values.shape, dtype=torch.float32, device=Config.Leader.DEVICE).detach()

            # generator_loss = min(v)
            attention_loss = self.elementwise_mean_squared_error(
                estimated_values.detach().clone().requires_grad_(), true_values
            )
            log_debug(
                f"attention loss for this batch {type(attention_loss)}:" +\
                f"{attention_loss.dtype}:{attention_loss.size()}:\n" +\
                f"{attention_loss}"
            )
            
            # reset gradients
            self.attention_model_optimizer.zero_grad()

            # since the estimated_values are mostly negative, then it should be multiplied by -1 to make l1 loss
            # negative thus the minimization works fine
            (-attention_loss).backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(self.attention_model.parameters(), 1.0)
            self.attention_model_optimizer.step()

        self.training_counter += 1

        # log learning metrics
        log_learning_metrics({
            "iteration": self.training_counter,
            "critic_loss": critic_loss.cpu().detach().squeeze().item(),
            "generator_loss": attention_loss.cpu().detach().squeeze().item() if attention_loss is not None else "nan"
        })

        # save models each episode
        if self.training_counter != 0 and self.training_counter % Config.MAX_EPISODE_STEPS == 0:
            self.save_models("latest")
        
        # rough approximation of actually trained episodes
        num_episodes = self.training_counter // Config.MAX_EPISODE_STEPS
        # model checkpoint
        if self.training_counter != 0 and num_episodes % Config.MODEL_SAVE_FREQUENCY_EPISODES == 0:
            self.save_models(f"{num_episodes}")


    # performs one forward pass of the attention generator model
    def inference_iteration(self, belief, observation) -> np.array:
        # check for None
        if not isinstance(belief, list):
            raise ValueError(f"Invalid belief type: Expected 'list', got '{type(belief)}'.")
        if not isinstance(observation, list):
            raise ValueError(f"Invalid observation type: Expected 'list', got '{type(observation)}'.")

        # check for correct dimensions
        belief = np.array(belief, dtype=float)
        if np.shape(belief)[0] > Config.Leader.MAX_FEATURE_LEN:
            raise ValueError(
                f"Too many belief values: Expected {Config.Leader.MAX_FEATURE_LEN}, got {np.shape(belief)[0]}."
            )
        if len(np.shape(belief)) != 1:
            raise ValueError(f"Invalid number of belief dimensions: Expected 1, got {len(np.shape(belief))}.")
        
        observation = np.array(observation, dtype=float)
        if np.shape(observation)[0] > Config.Leader.MAX_FEATURE_LEN:
            raise ValueError(
                f"Too many observation values: Expected {Config.Leader.MAX_FEATURE_LEN}, got {np.shape(observation)[0]}."
            )
        if len(np.shape(observation)) != 1:
            raise ValueError(f"Invalid number of observation dimensions: Expected 1, got {len(np.shape(observation))}.")

        # during training
        if not (Config.MODE in [Mode.TEST, Mode.VAL]):
            # and attention model not sufficiently trained
            if len(self.replay_buffer) < Config.Leader.REPLAY_MIN or self.training_counter < 1000:
                log_info(
                    f"minimum number of training iterations not met ({self.training_counter}/1000), " +\
                    f"sending uniform random distribution"
                )
                is_weights = self.uniform_rand_attentions(Config.Leader.ATTENTION_SIZE)
                return is_weights

        # same steps as in handle_weight()
        datum = np.array((belief, observation), dtype=object)
        datum = zip_longest(*datum, fillvalue=0)
        datum = list(datum)
        # shape: [n, 2] with n the number of possible goal directions or Config.Leader.MAX_FEATURE_LEN
        datum = np.array(datum)
        # shape: [1, 2*n]
        # datum = [p(g_1), car.x, p(g_2), car.y, ..., p(g_n), 0], i.e. belief and obs are interleaved
        datum = datum.reshape(1, -1)
        datum = torch.tensor(datum, dtype=torch.float32, device=Config.Leader.DEVICE)
        '''
        log_debug(f"inference datum {type(datum)}:{datum.dtype}:{np.shape(datum)}:\n{datum}")
        '''
        with torch.no_grad():
            is_weights, self.attention_memory = self.attention_model(datum, self.attention_memory)
            is_weights = is_weights.squeeze(0).cpu().numpy().tolist()

        return is_weights
   

    # send by is-despot before search to query importance distribution weights
    # called once by is-despot for each pedestrian (belief)
    # atm only one pedestrian at any point in time is supported
    def handle_weight(self, belief = None, observation = None):
        # @@@ IN @@@
        # belief/is_weights = [p(g_1), ..., p(g_n)] = [n, 1] with n the number of possible goal directions g
        # observation = [car.x, car.y, car.v, car.theta, ped.x, ped.y]
        if Config.Carla.NUM_PEDESTRIANS != 1:
            raise ValueError("Implementation doesn't suppoprt multiple pedestrians (but can be extended).")
        if Config.Leader.MAX_PEDESTRIANS != 1:
            raise ValueError("Implementation doesn't suppoprt multiple pedestrians (but can be extended).")
        if not isinstance(belief, list):
            raise ValueError(f"Invalid belief type: Expected 'list', got '{type(belief)}'.")
        if len(belief) > Config.Leader.MAX_FEATURE_LEN:
            raise ValueError(f"Invalid belief length: Expected {Config.Leader.MAX_FEATURE_LEN}, got {len(belief)}.")
        if not isinstance(observation, list):
            raise ValueError(f"Invalid observation type: Expected 'list', got '{type(observation)}'.")
        if len(observation) > Config.Leader.MAX_FEATURE_LEN:
            raise ValueError(
                f"Invalid observation length: Expected {Config.Leader.MAX_FEATURE_LEN}, got {len(observation)}."
            )
        if not isinstance(self.trajectory_replay_buffer, list):
            raise ValueError(
                f"Invalid trajectory replay buffer type: Expected 'list' got '{type(self.trajectory_replay_buffer)}'."
            )

        # @@@ PROCESS @@@
        # query attention model for importance sampling weights
        # belief and observation comes from despot process
        is_weights = self.inference_iteration(belief, observation)
        if len(np.shape(is_weights)) > 1:
            raise ValueError(
                f"Invalid dimensions of importance sampling weights: Expected (1,) got {np.shape(is_weights)}."
            )
        if np.shape(is_weights)[0] != Config.Leader.ATTENTION_SIZE:
            raise ValueError(
                f"Invalid number of importance sampling weights: "
                f"Expected {Config.Leader.ATTENTION_SIZE}, got {np.shape(is_weights)[0]}."
            )

        # @@@ SAVE @@@@
        # save tuple of belief, observation and importance sampling distribution for training critic network
        datum = np.array((belief, observation, is_weights), dtype=object)
        datum = zip_longest(*datum, fillvalue=0)
        datum = list(datum)
        # shape: [3, n]
        datum = np.array(datum).T

        # shape: [p*3, n]
        # originally the trajectory replay buffer held a datum for each pedestrian for one step
        # handle_value() would then append the value after search to the whole buffer, before the
        # whole thing was transferred into the self.replac_buffer as a single training sample
        # [
        #        # first pedestrian
        #        [p(g1), ......................................... ,p(gn)], # belief from is-despot
        #        [car.x, car.y, car.v, car.theta, ped.x, ped.y, 0, ..., 0], # 0-filled observation from carla
        #        [p(g1), ......................................... ,p(gn)], # is-weights generated from attention model
        #   .
        #   .
        #   .
        #        # last pedestrian
        #        [p(g1), ......................................... ,p(gn)], 
        #        [car.x, car.y, car.v, car.theta, ped.x, ped.y, 0, ..., 0], 
        #        [p(g1), ......................................... ,p(gn)]
        # ]
        '''
        log_debug(
            f"extending self.trajectory_replay_buffer with {type(datum)}:" +\
            f"{datum.dtype}:{np.shape(datum)}:\n{datum}"
        )
        '''
        self.trajectory_replay_buffer.extend(datum)

        # @@@ OUT @@@
        # send attention weights to despot process
        '''
        log_debug(
            f"sending importance sampling distribution weights {type(is_weights)}:" +\
            f"{is_weights.dtype}:{is_weights.shape}:\n{is_weights}"
        )
        '''
        self.despot_connection.send_is_weights(is_weights, Connection.LEADER)


    # send by is-despot after search
    # integrates one "full-trajectory-padded" step into the replay buffer
    def handle_value(self, value: float = None):
        if not isinstance(value, float):
            raise ValueError(f"Invalid value type: Expected 'float', got '{type(value)}'.")
        if not isinstance(self.replay_buffer, ReplayBuffer):
            raise ValueError(f"Invalid replay buffer type: Expected 'ReplayBuffer', got '{type(self.replay_buffer)}'.")
        if len(self.trajectory_replay_buffer) == 0:
            raise ValueError(f"Invalid trajectory replay buffer length: Expected >0, got 0.")
     
        # @@@ IN @@@
        # estimated value from IS-DESPOT
        log_info(f"IS-DESPOT root state value estimate: {value}")
        
        # @@@ PROCESS @@@
        # get the current trajectory, i.e. belief, observation and is-weights of current step
        trajectory_replay_buffer = np.array(self.trajectory_replay_buffer)
        log_debug(
            f"self.trajectory_replay_buffer {type(trajectory_replay_buffer)}:" +\
            f"{trajectory_replay_buffer.dtype}:{np.shape(trajectory_replay_buffer)}:\n" +\
            f"{trajectory_replay_buffer}"
        )
 
        # each feature is a single row
        # 3 rows correspond to a single step of a single pedestrian
        # MAX_TRAJECTORY_LENGTH denotes the maximum number of rows allowed in the buffer according
        # to the maximum number pedestrians, i.e. Config.Leader.MAX_PEDESTRIANS
        # since atm (22.12.23) only one pedestrian is supported, this is irrelevant
        # but theoretically, if there were more pedestrians allowed than are currently in a scene,
        # this would ensure that a single critic training sample always has the same dimensions
        remaining_rows = Config.Leader.MAX_TRAJECTORY_LENGTH - trajectory_replay_buffer.shape[0]
        columns = trajectory_replay_buffer.shape[1]

        # remaining allowad rows
        remaining_trajectory = np.zeros((remaining_rows, columns))
        log_debug(
            f"remaining_trajectory.shape {type(remaining_trajectory)}:" +\
            f"{remaining_trajectory.dtype}:{np.shape(remaining_trajectory)}:" +\
            f"{remaining_trajectory}"
        )
        # create 1d np.array of length 'columns' filled with despot's estimated value
        values = np.full(columns, value)
        log_debug(
            f"inflated IS-DESPOT value estimate {type(values)}:{values.dtype}:" +\
            f"{np.shape(values)}:\n{values}"
        )

        # inflate the replay buffer by padding the actual trajectory to full length with 0-rows
        full_trajectory = np.vstack([trajectory_replay_buffer, remaining_trajectory, values])
        log_debug(
            f"full_trajectory {type(full_trajectory)}:{full_trajectory.dtype}:" +\
            f"{np.shape(full_trajectory)}:\n{full_trajectory}"
        )    
            
        # @@@ OUT @@@
        # add to existing replay buffer
        # shape: [p*4, n]
        # with p: the maximum number of allowed pedestrians,
        # 4: one row for each: belief, obs, is-weights and value
        # and n: number of possible goal directions (or Config.Leader.MAX_FEATURE_LEN)
        # [
        #        # first pedestrian
        #        [p(g1), ........................................., p(gn)], # belief from is-despot
        #        [car.x, car.y, car.v, car.theta, ped.x, ped.y, 0, ..., 0], # 0-filled observation from carla
        #        [p(g1), ........................................., p(gn)], # is-weights generated from attention model
        #        [v_hat, ........................................., v_hat], # state value estimate from is-despot
        #    .
        #    .
        #    .
        #        # last pedestrian
        #        [p(g1), ......................................... ,p(gn)],
        #        [car.x, car.y, car.v, car.theta, ped.x, ped.y, 0, ..., 0],
        #        [p(g1), ......................................... ,p(gn)],
        #        [v_hat, ........................................., v_hat]
        # ]
        self.replay_buffer.append(full_trajectory)
        # empty temporary replay buffer
        self.trajectory_replay_buffer.clear()


    def run_leader_server(self):
        try:
            self.initialize_models()
            if Config.RESUME or Config.MODE is not Mode.TRAIN: self.load_models()
            if Config.MODE is Mode.TRAIN: 
                self.initialize_optimizers()
                self.critic_model.train()
                self.attention_model.train()
            elif Config.MODE in [Mode.VAL, Mode.TEST]:
                self.critic_model.eval()
                self.attention_model.eval()

            if not isinstance(self.attention_model, AttentionNet):
                raise ValueError(f"Invalid attention model type: Expected '{AttentionNet}', got '{type(self.attention_model)}'.")
            if not isinstance(self.critic_model, CriticNet):
                raise ValueError(f"Invalid critic model type: Expected '{CriticNet}', got '{type(self.critic_model)}'.")
            if not isinstance(self.attention_memory, torch.Tensor):
                raise ValueError(f"Invalid attention memory type: Expected '{torch.Tensor}', got '{type(self.attention_memory)}'.")
            if not isinstance(self.critic_memory, torch.Tensor):
                raise ValueError(f"Invalid critic memory type: Expected '{torch.Tensor}', got '{type(self.critic_memory)}'.")
            if self.despot_connection is None:
                raise TypeError("IS-DESPOT C++ process interface has not been initialized.")
            if not isinstance(self.despot_connection, DespotBridge):
                raise TypeError(f"Invalid IS-DESPOT C++ prcess interface: " 
                                f"Expected '{DespotBridge}', got '{type(self.despot_connection)}'.")

            # start server
            while not Config.TERMINATE:
                # step always starts with IS-DESPOT sending current belief and observation associated with root node
                # will return once IS-DESPOT received an observation with a pedestrian
                belief, observation = self.despot_connection.receive_belief_and_observation(Connection.LEADER)           
                # calls inference_iteration(), i.e. sends LEADER's importance distribution to IS-DESPOT
                self.handle_weight(belief, observation)
                # no special partsing necessary
                despot_value = self.despot_connection.receive_exact_message(Connection.LEADER)[0]
                # update model only during training
                if Config.MODE is Mode.TRAIN:
                    # consume empirical value of IS-DESPOT (incorporate into buffer)
                    self.handle_value(despot_value)
                    # training the models takes too little time (~4ms) to justify starting a process
                    self.training_iteration()
            
        except: log_exception(*sys.exc_info())



         


