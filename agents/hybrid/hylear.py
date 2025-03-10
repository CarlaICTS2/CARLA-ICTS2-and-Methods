"""
Author: Dikshant Gupta
Time: 10.11.21 01:14
"""

import math
from collections import deque
import operator
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from multiprocessing import Process

import carla

from benchmark.environment.environment import CarlaCTS02

from agents.learner.rlagent import RLAgent
from agents.learner.models import DQNBase, TwinnedQNetwork, CateoricalPolicy
from agents.learner.buffers import LazyMultiStepMemory, LazyPrioritizedMultiStepMemory

from benchmark.risk.risk_aware_path import PathPlanner
from ped_path_predictor.m2p3 import PathPredictor

from utils.config import Config, Mode
from utils.connector import DespotBridge
from utils.utils import run_despot
from utils.logger import log_info, log_debug


class RunningMeanStats:

    def __init__(self, n=10):
        self.n = n
        self.stats = deque(maxlen=n)

    def append(self, x):
        self.stats.append(x)

    def get(self):
        return np.mean(self.stats)


class HyLEAR(RLAgent):

    def __init__(self,
                 env: CarlaCTS02, 
                 world, 
                 carla_map, 
                 scenario, 
                 conn: DespotBridge = None,
                 test_env: CarlaCTS02 = None):
        super(HyLEAR, self).__init__(world, carla_map, scenario)

        torch.manual_seed(Config.SEED)
        np.random.seed(Config.SEED)

        self.risk_cmp = np.zeros((110, 310))
        # Road Network
        self.risk_cmp[7:13, 13:] = 1.0
        self.risk_cmp[97:103, 13:] = 1.0
        self.risk_cmp[7:, 7:13] = 1.0
        # Sidewalk Network
        sidewalk_cost = 50.0
        self.risk_cmp[4:7, 4:] = sidewalk_cost
        self.risk_cmp[:, 4:7] = sidewalk_cost
        self.risk_cmp[13:16, 13:] = sidewalk_cost
        self.risk_cmp[94:97, 13:] = sidewalk_cost
        self.risk_cmp[103:106, 13:] = sidewalk_cost
        self.risk_cmp[13:16, 16:94] = sidewalk_cost

        self.conn = conn
        self.env = env
        self.test_env = test_env

        # information of a single step
        self.history: np.array = None

        self.action_count: dict = None
        self.action_count_critic: dict = None

        # segmentation image
        self.current_observation: np.array = None
        
        # information about mutliple episodes
        self.experience_replay_buffer: LazyMultiStepMemory = None

        # how manysteps has the agent seen?
        self.steps = 0

        # used to determine when training statistics are logged
        self.learning_steps = 0

        # used for saving the best model as determined on the validation episodes
        self.best_eval_score = -np.inf

        self.device = torch.device(Config.HyLEAR.DEVICE)

        if Config.PREDICT_PEDESTRIAN_PATH:
            self.ped_pred = PathPredictor(Config.MODEL_DIR + "m2p3.pth")
            self.ped_pred.model.eval()

        self.risk_path_planner = PathPlanner()

        # always init models
        self.initialize_models()

        # only use IS-DESPOT during training
        if Config.MODE is Mode.TRAIN:
            self.initialize_buffers()
            self.initialize_optimizers()

            Process(target=run_despot).start()
            self.conn.establish_connection()
            m = self.conn.receive_message()
            log_info(m)  # RESET

        # load models if resuming
        if Config.RESUME and Config.MODE is Mode.TRAIN:
            self.conv.load(Config.MODEL_DIR + "conv.pth")
            self.policy.load(Config.MODEL_DIR + "policy.pth")
            self.online_critic.load(Config.MODEL_DIR + "online_critic.pth")
            self.target_critic.load(Config.MODEL_DIR + "target_critic.pth")
            # copy parameters of the learning network to the target network
            self.update_target_critic()
            # disable gradient calculations of the target network
            self.disable_gradients(self.target_critic)
            log_info("successfully loaded HyLEAR models for resuming training")

        # load subset of models required for testing only
        if Config.MODE is not Mode.TRAIN:
            self.conv.load(Config.MODEL_DIR + "conv.pth")
            self.policy.load(Config.MODEL_DIR + "policy.pth")
            self.conv.eval()
            self.policy.eval()
            log_info("successfully loaded HyLEAR models for testing")

        # logging
        self.writer = SummaryWriter(log_dir=Config.METRICS_DIR)
        self.train_return = RunningMeanStats(Config.HyLEAR.log_interval)

    def initialize_models(self):
        log_info("initializing models...")
        # Define networks.
        self.conv = DQNBase(self.env.observation_space.shape[2]).to(self.device)

        self.policy = CateoricalPolicy(self.env.observation_space.shape[2], 
                                       self.env.action_space.n, 
                                       shared=True)\
                                       .to(self.device)
        
        if Config.MODE is not Mode.TRAIN: return
        self.online_critic = TwinnedQNetwork(self.env.observation_space.shape[2], 
                                             self.env.action_space.n,
                                             dueling_net=Config.HyLEAR.dueling_net, 
                                             shared=True)\
                                             .to(device=self.device)
        
        self.target_critic = TwinnedQNetwork(self.env.observation_space.shape[2], 
                                             self.env.action_space.n,
                                             dueling_net=Config.HyLEAR.dueling_net, 
                                             shared=True)\
                                             .to(device=self.device).eval()

    def initialize_buffers(self):
        log_info("initializing experience replace buffer...")

        if Config.HyLEAR.use_per:
            beta_steps = (Config.HyLEAR.num_steps - Config.HyLEAR.start_steps) / Config.HyLEAR.update_interval
            self.experience_replay_buffer = LazyPrioritizedMultiStepMemory(capacity=Config.HyLEAR.buffer_capacity,
                                                                           state_shape=self.env.observation_space.shape,
                                                                           device=self.device, 
                                                                           gamma=Config.HyLEAR.gamma, 
                                                                           multi_step=Config.HyLEAR.multi_step,
                                                                           beta_steps=beta_steps)
        else:
            self.experience_replay_buffer = LazyMultiStepMemory(capacity=Config.HyLEAR.buffer_capacity,
                                                                state_shape=self.env.observation_space.shape,
                                                                device=self.device, 
                                                                gamma=Config.HyLEAR.gamma, 
                                                                multi_step=Config.HyLEAR.multi_step)

    def initialize_optimizers(self):
        log_info("initializing optimizers...")
        # initialize optimizers
        self.policy_optim = Adam(self.policy.parameters(), lr=Config.HyLEAR.lr)
        self.q1_optim = Adam(list(self.conv.parameters()) + list(self.online_critic.Q1.parameters()), lr=Config.HyLEAR.lr)
        self.q2_optim = Adam(self.online_critic.Q2.parameters(), lr=Config.HyLEAR.lr)

        # We optimize log(alpha), instead of alpha.
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = Adam([self.log_alpha], lr=Config.HyLEAR.lr)

        # target entropy is -log(1/|A|) * ratio (= maximum entropy * ratio)
        self.target_entropy = -np.log(1.0 / self.env.action_space.n) * Config.HyLEAR.target_entropy_ratio

    # called at the beginning of an episode
    def initialize_history(self):
        log_info("initializing history...")
        self.history = np.zeros(6)  # reward, velocity.x, velocity.y, onehot-encoded last action
        self.history[3 + 1] = 1.0  # index = 3 + last_action(maintain)
        self.action_count = {0: 0, 1: 0, 2: 0}
        self.action_count_critic = {0: 0, 1: 0, 2: 0}

    # called every step of an episode
    def forward_pass(self, observation, step):
        
        # validation and testing 
        if Config.MODE is not Mode.TRAIN:
            return self.exploit((observation, self.history))
        
        # training
        # save observation for appending to experience replay buffer later
        self.current_observation = observation

        # random action if the model has not been trained enough
        if step < Config.HyLEAR.start_steps:
                action = self.env.action_space.sample()
                critic_action = action
                symbolic_action = action
        # read from IS-DESPOT set control object
        else:
            if self.env.control.throttle > 0: symbolic_action = 0
            elif self.env.control.brake > 0: symbolic_action = 2
            else: symbolic_action = 1

            symbolic_probs = torch.FloatTensor([0.1, 0.1, 0.1])
            symbolic_probs[symbolic_action] = 0.8

            # samples a speed action from the above defined distriubtion, i.e.
            # IS-DESPOT speed action has the highest prob and with some randomness other actions are chosen
            action, critic_action = self.explore((observation, self.history), symbolic_probs)
        
        self.action_count[action] += 1
        self.action_count_critic[critic_action] += 1
        self.steps += 1

        return action
        
    # called every step of an episode
    def update_history(self, next_observation, reward, terminal, velocity, action, step):
        # clip reward to [-1.0, 1.0]
        clipped_reward = max(min(reward, 1.0), -1.0)

        if step + 1 == Config.MAX_EPISODE_STEPS: mask = False
        else: mask = terminal

        new_history = np.zeros(6)
        new_history[0] = clipped_reward
        new_history[1] = velocity.x / Config.max_speed
        new_history[2] = velocity.y / Config.max_speed
        new_history[3 + action] = 1.0

        # fill buffer
        if Config.MODE is Mode.TRAIN:
            self.experience_replay_buffer.append((self.current_observation, self.history),
                                                action,
                                                clipped_reward,
                                                (next_observation, new_history),
                                                mask)
        # update for next step
        self.current_observation = next_observation
        self.history = new_history

    def backward_pass(self):
        # TODO parallelize with threads
        if self.steps % Config.HyLEAR.update_interval == 0 and self.steps >= Config.HyLEAR.start_steps:
            self.learn()

        if self.steps % Config.HyLEAR.target_update_interval == 0:
            self.update_target_critic()

        #if self.steps % Config.DSAC.eval_interval == 0:
        #    self.evaluate()

    def learn(self):
        assert hasattr(self, 'q1_optim') and hasattr(self, 'q2_optim') and\
               hasattr(self, 'policy_optim') and hasattr(self, 'alpha_optim')

        self.learning_steps += 1

        if Config.HyLEAR.use_per:
            batch, weights = self.experience_replay_buffer.sample(Config.HyLEAR.batch_size)
        else:
            batch = self.experience_replay_buffer.sample(Config.HyLEAR.batch_size)
            # Set priority weights to 1 when we don't use PER.
            weights = 1.

        q1_loss, q2_loss, errors, mean_q1, mean_q2 = self.calc_critic_loss(batch, weights)
        policy_loss, entropies = self.calc_policy_loss(batch, weights)
        entropy_loss = self.calc_entropy_loss(entropies, weights)

        self.update_params(self.q1_optim, q1_loss)
        self.update_params(self.q2_optim, q2_loss)
        self.update_params(self.policy_optim, policy_loss)
        self.update_params(self.alpha_optim, entropy_loss)

        self.alpha = self.log_alpha.exp()

        if Config.HyLEAR.use_per: self.experience_replay_buffer.update_priority(errors)

        if self.learning_steps % Config.HyLEAR.log_interval == 0:
            self.writer.add_scalar('loss/Q1', q1_loss.detach().item(), self.learning_steps)
            self.writer.add_scalar('loss/Q2', q2_loss.detach().item(), self.learning_steps)
            self.writer.add_scalar('loss/policy', policy_loss.detach().item(), self.learning_steps)
            self.writer.add_scalar('loss/alpha', entropy_loss.detach().item(), self.learning_steps)
            self.writer.add_scalar('stats/alpha', self.alpha.detach().item(), self.learning_steps)
            self.writer.add_scalar('stats/mean_Q1', mean_q1, self.learning_steps)
            self.writer.add_scalar('stats/mean_Q2', mean_q2, self.learning_steps)
            self.writer.add_scalar('stats/entropy', entropies.detach().mean().item(), self.learning_steps)

        self.save_models()

    def explore(self, state, probs=None):
        # Act with randomness.
        state, t = state
        state = torch.ByteTensor(state[None, ...]).to(self.device).float() / 255.
        t = torch.FloatTensor(t[None, ...]).to(self.device)
        probs = torch.FloatTensor(probs[None, ...]).to(self.device)
        with torch.no_grad():
            state = self.conv(state)
            state = torch.cat([state, t], dim=1)
            action, _, _ = self.policy.sample(state, probs, self.steps)
            curr_q1 = self.online_critic.Q1(state)
            curr_q2 = self.online_critic.Q2(state)
            q = torch.min(curr_q1, curr_q2)
            critic_action = torch.argmax(q, dim=1)
        return action.item(), critic_action.item()

    def exploit(self, state):
        # Act without randomness.
        state, t = state
        state = torch.ByteTensor(state[None, ...]).to(self.device).float() / 255.
        t = torch.FloatTensor(t[None, ...]).to(self.device)
        with torch.no_grad():
            state = self.conv(state)
            state = torch.cat([state, t], dim=1)
            action = self.policy.act(state)
        return action.item()

    def update_target_critic(self):
        self.target_critic.load_state_dict(self.online_critic.state_dict())

    def update_params(optim, loss, retain_graph=False):
        optim.zero_grad()
        loss.backward(retain_graph=retain_graph)
        optim.step()

    def disable_gradients(network):
        # Disable calculations of gradients.
        for param in network.parameters():
            param.requires_grad = False

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        states, t = states
        states = self.conv(states)
        states = torch.cat([states, t], dim=-1)
        curr_q1 = self.online_critic.Q1(states).gather(1, actions.long())
        curr_q2 = self.online_critic.Q2(states.detach()).gather(1, actions.long())
        return curr_q1, curr_q2

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_states, t_new = next_states
            next_states = self.conv(next_states)
            next_states = torch.cat([next_states, t_new], dim=1)
            _, action_probs, log_action_probs = self.policy.sample(next_states)
            next_q1, next_q2 = self.target_critic(next_states)
            next_q = (action_probs * (
                torch.min(next_q1, next_q2) - self.alpha * log_action_probs
                )).sum(dim=1, keepdim=True)

        assert rewards.shape == next_q.shape
        return rewards + (1.0 - dones) * self.gamma_n * next_q

    def calc_policy_loss(self, batch, weights):
        states, actions, rewards, next_states, dones = batch
        states, t = states

        with torch.no_grad():
            states = self.conv(states)
        states = torch.cat([states, t], dim=1)

        # (Log of) probabilities to calculate expectations of Q and entropies.
        _, action_probs, log_action_probs = self.policy.sample(states)

        with torch.no_grad():
            # Q for every actions to calculate expectations of Q.
            q1, q2 = self.online_critic(states)
            q = torch.min(q1, q2)

        # Expectations of entropies.
        entropies = -torch.sum(action_probs * log_action_probs, dim=1, keepdim=True)

        # Expectations of Q.
        q = torch.sum(torch.min(q1, q2) * action_probs, dim=1, keepdim=True)

        # Policy objective is maximization of (Q + alpha * entropy) with
        # priority weights.
        policy_loss = (weights * (- q - self.alpha * entropies)).mean()

        return policy_loss, entropies.detach()

    def calc_critic_loss(self, batch, weights):
        curr_q1, curr_q2 = self.calc_current_q(*batch)
        target_q = self.calc_target_q(*batch)

        # TD errors for updating priority weights
        errors = torch.abs(curr_q1.detach() - target_q)

        # We log means of Q to monitor training.
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()

        # Critic loss is mean squared TD errors with priority weights.
        q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
        q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)

        return q1_loss, q2_loss, errors, mean_q1, mean_q2
    
    def calc_entropy_loss(self, entropies, weights):
        assert not entropies.requires_grad

        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(self.log_alpha * 
                                   (self.target_entropy - entropies) *
                                    weights)
        return entropy_loss
    
    def save_models(self):
        log_info("saving latest models...")
        self.conv.save(Config.MODEL_DIR + 'conv.pth')
        self.policy.save(Config.MODEL_DIR + 'policy.pth')
        self.online_critic.save(Config.MODEL_DIR + 'online_critic.pth')
        self.target_critic.save(Config.MODEL_DIR + 'target_critic.pth')

    def __del__(self):
        self.env.close()
        if self.test_env: self.test_env.close()
        self.file.close()

    def get_reward_despot(self, action):
        base_reward, goal, hit, nearmiss, terminal = super(HyLEAR, self).get_reward(action)
        reward = 0
        if goal:
            reward += 1.0
        return reward, goal, hit, nearmiss, terminal

    def get_speed_action(self, path, control):
        transform = self.vehicle.get_transform()
        start = (self.vehicle.get_location().x, self.vehicle.get_location().y, transform.rotation.yaw)
        walker_x, walker_y = self.world.walker.get_location().x, self.world.walker.get_location().y

        if self.prev_action is not None:
            reward, goal, hit, near_miss, terminal = self.get_reward_despot(self.prev_speed)
            terminal = goal or hit
        else:
            # handling first instance
            reward = 0
            terminal = False
        angle = transform.rotation.yaw
        car_pos = [self.vehicle.get_location().x, self.vehicle.get_location().y]
        car_velocity = self.vehicle.get_velocity()
        car_speed = np.sqrt(car_velocity.x ** 2 + car_velocity.y ** 2)
        pedestrian_positions = [[self.world.walker.get_location().x, self.world.walker.get_location().y]]

        if len(path) == 0:
            control.brake = 0.6
            self.prev_speed = 2
        elif not self.is_pedestrian_observable:
            control.throttle = 0.6
        else:
            self.conn.send_observation(terminal, reward, angle, car_pos, car_speed, pedestrian_positions, path)
            m = self.conn.receive_message()
            if m == "START":
                self.conn.send_observation(terminal, reward, angle, car_pos, car_speed, pedestrian_positions, path)
                m = self.conn.receive_message()
            self.prev_speed = 1
            if m[0] == '0':
                control.throttle = 0.6
                self.prev_speed = 0
            elif m[0] == '2':
                control.brake = 0.6
                self.prev_speed = 2

        self.prev_action = control
        return control

    def run_step(self, debug=False):
        self.vehicle = self.world.player
        transform = self.vehicle.get_transform()
        start = (self.vehicle.get_location().x, self.vehicle.get_location().y, transform.rotation.yaw)
        end = self.scenario[2]

        # t = time.time()
        # Steering action on the basis of shortest and safest path(Hybrid A*)
        obstacles = self.get_obstacles(start)
        if len(obstacles):
            (path, risk), intention = self.get_path_with_reasoning(start, end, obstacles)
        else:
            (path, risk), intention = self.get_path_simple(start, end, obstacles)
        # print("time taken: ", time.time() - t)

        control = carla.VehicleControl()
        control.brake = 0.0
        control.hand_brake = False
        control.manual_gear_shift = False

        if len(path) == 0:
            control.steer = 0
        else:
            control.steer = (path[2][2] - start[2]) / 70.
        # print("Angle: ", control.steer)

        # Best speed action for the given path from IS-DESPOT - TRAIN ARCHITECTURE
        if Config.MODE is Mode.TRAIN:
            control = self.get_speed_action(path, control)
        self.prev_action = control
        return control, intention, risk, self.is_pedestrian_observable


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
        if Config.PREDICT_PEDESTRIAN_PATH and len(self.ped_history) >= 15 and self.is_pedestrian_observable:
            # Use path predictor
            ped_updated_risk_cmp = self.risk_cmp.copy()
            ped_path = np.array(self.ped_history)
            ped_path = ped_path.reshape((15, 2))
            log_debug("making pedestrian path prediction...")
            pedestrian_path = self.ped_pred.get_single_prediction(ped_path)
            new_obs = [obs for obs in obstacles]
            pedestrian_path_d = list()
            for node in pedestrian_path:
                if (round(node[0]), round(node[1])) not in new_obs:
                    new_obs.append((round(node[0]), round(node[1])))
                    pedestrian_path_d.append((round(node[0]), round(node[1])))
            for pos in new_obs:
                ped_updated_risk_cmp[pos[0] + 10, pos[1] + 10] = 10000

            path_normal = self.risk_path_planner.find_path_with_risk(start, end, self.grid_cost, obstacles, car_speed,
                                                                     yaw, ped_updated_risk_cmp, True, self.scenario[0])
            # print(len(new_obs), path_normal[1])
            if path_normal[1] < 1000:
                # print("normal!", path_normal[1], (path_normal[0][2][2] - path_normal[0][1][2]) / 70.0)
                return (path_normal[0], path_normal[1] / 6), self.get_car_intention(pedestrian_path_d, path_normal[0], start)
            # print(start, end, obstacles)
            paths = [path_normal,  # Normal
                     self.risk_path_planner.find_path_with_risk(start, end, self.grid_cost, new_obs, car_speed,
                                                                yaw, ped_updated_risk_cmp, True, self.scenario[0]),  # ped pred
                     self.risk_path_planner.find_path_with_risk(start, end, relaxed_sidewalk, obstacles, car_speed,
                                                                yaw, ped_updated_risk_cmp, True, self.scenario[0])]  # Sidewalk relaxed
                     # self.risk_path_planner.find_path_with_risk(start, end, relaxed_sidewalk, new_obs, car_speed,
                     #                                            yaw, ped_updated_risk_cmp, True)]  # Sidewalk relaxed + ped pred
            path, risk = self.rulebook(paths, start)
            return (path, risk/6), self.get_car_intention(pedestrian_path_d, path, start)
        
        # if there is not enough data to make a prediciton or if there is no pedestrian in the current 
        # scene simulation step
        if len(self.ped_history) < 15: log_debug("skipping pedestrian path predicition: not enough pedestrain data yet...")
        elif not self.is_pedestrian_observable: log_debug("skipping pedestrian path predicition: no pedestrain in scene...")
        elif not Config.PREDICT_PEDESTRIAN_PATH: log_debug("skipping pedestrian path predicition: not enabled")

        if self.scenario[0] == 11 and self.world.incoming_car.get_location().y + 2 < start[1] and start[0] <= -2.5:
            end = (end[0], start[1] + 6, end[2])

        if self.scenario[0] in [10, 1] and self.world.walker.get_location().y > start[1] and start[0] >= 2.5:
            end = (end[0], start[1] - 6, end[2])
        path_normal = self.risk_path_planner.find_path_with_risk(start, end, self.grid_cost, obstacles, car_speed,
                                                                    yaw, self.risk_cmp, True, self.scenario[0])
        if path_normal[1] < 100 or not self.is_pedestrian_observable:
            return path_normal, self.get_car_intention([], path_normal[0], start)
        paths = [path_normal,
                self.risk_path_planner.find_path_with_risk(start, end, relaxed_sidewalk, obstacles, car_speed,
                                                            yaw, self.risk_cmp, True, self.scenario[0])]  # Sidewalk relaxed
        path, risk = self.rulebook(paths, start)
        intention = self.get_car_intention([], path, start)
        #log_info(f"path: {path} \n risk: {risk} \n intention: {intention}")
        return (path, risk/6), intention

    @staticmethod
    def rulebook(paths, start):
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