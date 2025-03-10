from collections import deque
import numpy as np
import torch
from torch.optim import Adam

from agents.learner.models import DQNBase, TwinnedQNetwork, CateoricalPolicy
from agents.learner.buffers import LazyMultiStepMemory, LazyPrioritizedMultiStepMemory

from utils.config import Config, Mode
from utils.logger import log_info


class RunningMeanStats:

    def __init__(self, n=10):
        self.n = n
        self.stats = deque(maxlen=n)

    def append(self, x):
        self.stats.append(x)

    def get(self):
        return np.mean(self.stats)


class SharedDiscreteSoftActorCritic():

    def __init__(self, env, test_env):
        self.env = env
        self.test_env = test_env

        torch.manual_seed(Config.SEED)
        np.random.seed(Config.SEED)

        self.file = open(Config.METRICS_DIR + "eval_results.log", "w")
        #self.writer = SummaryWriter(log_dir=Config.METRICS_DIR)
        self.train_return = RunningMeanStats(Config.HyLEAR.log_interval)

        # information of a single step
        self.history: np.array = None

        self.action_count: dict = None
        self.action_count_critic: dict = None

        # segmentation image
        self.current_observation: np.array = None
        
        # information about mutliple episodes
        self.experience_replay_buffer: LazyMultiStepMemory = None

        # how many episodes and steps has the agent seen?
        self.episodes = 0
        self.steps = 0

        # used to determine when training statistics are logged
        self.learning_steps = 0

        # used for saving the best model as determined on the validation episodes
        self.best_eval_score = -np.inf

        # harms performance
        # torch.backends.cudnn.deterministic = True  
        # torch.backends.cudnn.benchmark = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.initialize_models()

        # load models if resuming or testing
        if Config.RESUME or Config.MODE is not Mode.TRAIN:
            self.conv.load(Config.MODEL_DIR + "conv.pth")
            self.policy.load(Config.MODEL_DIR + "policy.pth")
            self.online_critic.load(Config.MODEL_DIR + "online_critic.pth")
            self.target_critic.load(Config.MODEL_DIR + "target_critic.pth")
            log_info("successfully loaded HyLEAR models for resuming taining/testing")
       
        # copy parameters of the learning network to the target network
        self.target_critic.load_state_dict(self.online_critic.state_dict()) 
       
        # disable gradient calculations of the target network
        self.disable_gradients(self.target_critic)

        # initialize optimizers
        self.policy_optim = Adam(self.policy.parameters(), lr=Config.HyLEAR.lr)
        self.q1_optim = Adam(list(self.conv.parameters()) + list(self.online_critic.Q1.parameters()), lr=Config.HyLEAR.lr)
        self.q2_optim = Adam(self.online_critic.Q2.parameters(), lr=Config.HyLEAR.lr)

        # target entropy is -log(1/|A|) * ratio (= maximum entropy * ratio)
        self.target_entropy = -np.log(1.0 / self.env.action_space.n) * Config.HyLEAR.target_entropy_ratio

        # We optimize log(alpha), instead of alpha.
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = Adam([self.log_alpha], lr=Config.HyLEAR.lr)

    def initialize_models(self):
        # Define networks.
        self.conv = DQNBase(self.env.observation_space.shape[2]).to(self.device)

        self.policy = CateoricalPolicy(self.env.observation_space.shape[2], 
                                       self.env.action_space.n, 
                                       shared=True)\
                                       .to(self.device)
        
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

    # called at the beginning of an episode
    def initialize_history(self):
        self.history = np.zeros(6)  # reward, vx, vt, onehot last action
        self.history[3 + 1] = 1.0  # index = 3 + last_action(maintain)
        self.action_count = {0: 0, 1: 0, 2: 0}
        self.action_count_critic = {0: 0, 1: 0, 2: 0}

    # called every step of an episode
    def get_speed_action(self, observation, step):
        self.current_observation = observation
        # random action if the model has not been trained enough
        if step < Config.HyLEAR.start_steps:
                action = self.env.action_space.sample()
                critic_action = action
                symbolic_action = action
        else:
            if self.env.control.throttle > 0:
                symbolic_action = 0
            elif self.env.control.brake > 0:
                symbolic_action = 2
            else:
                symbolic_action = 1

            symbolic_probs = torch.FloatTensor([0.1, 0.1, 0.1])
            symbolic_probs[symbolic_action] = 0.8

            action, critic_action = self.explore((observation, self.history), symbolic_probs)
            self.action_count[action] += 1
            self.action_count_critic[critic_action] += 1

            return action
        
    # called every step of an episode
    def update_history(self, next_observation, reward, terminal, velocity, action, step):
        # clip reward to [-1.0, 1.0]
        clipped_reward = max(min(reward, 1.0), -1.0)

        if step + 1 == Config.MAX_EPISODE_STEPS:
            mask = False
        else:
            mask = terminal

        new_history = np.zeros(6)
        new_history[0] = clipped_reward
        new_history[1] = velocity.x / Config.max_speed
        new_history[2] = velocity.y / Config.max_speed
        new_history[3 + action] = 1.0

        # To calculate efficiently, set priority=max_priority here.
        self.experience_replay_buffer.append((self.current_observation, self.history),
                                             action,
                                             clipped_reward,
                                             (next_observation, new_history),
                                             mask)
        
        self.current_observation = next_observation
        self.history = new_history

        # TODO parallelize with threads
        if self.steps % Config.HyLEAR.update_interval == 0 and self.steps >= Config.HyLEAR.start_steps:
            self.learn()

        if self.steps % Config.HyLEAR.target_update_interval == 0:
            self.update_target()

        if self.steps % Config.HyLEAR.eval_interval == 0:
            self.evaluate()

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

    def train_episode(self):
        self.episodes += 1
        episode_return = 0.
        episode_steps = 0

        done = False
        nearmiss = False
        accident = False
        goal = False
        # initial step of episode, calls is-despot
        # state = control, intention, risk, self.pedestrian_observable
        state = self.env.reset()
        action_count = {0: 0, 1: 0, 2: 0}
        action_count_critic = {0: 0, 1: 0, 2: 0}

        t = np.zeros(6)  # reward, vx, vt, onehot last action
        t[3 + 1] = 1.0  # index = 3 + last_action(maintain)

        while (not done) and episode_steps < self.max_episode_steps:
            if self.display:
                self.env.render()
            # if self.steps > Config.pre_train_steps:
            #     self.env.planner_agent.eval_mode = True
            if self.start_steps > self.steps:
                action = self.env.action_space.sample()
                critic_action = action
                symbolic_action = action
            else:
                if self.env.control.throttle > 0:
                    symbolic_action = 0
                elif self.env.control.brake > 0:
                    symbolic_action = 2
                else:
                    symbolic_action = 1
                symbolic_probs = torch.FloatTensor([0.1, 0.1, 0.1])
                symbolic_probs[symbolic_action] = 0.8
                action, critic_action = self.explore((state, t), symbolic_probs)

            # uses IS-DESPOT's velocity action
            next_state, reward, done, info = self.env.step(action)
            action_count[action] += 1
            action_count_critic[critic_action] += 1

            # Clip reward to [-1.0, 1.0].
            clipped_reward = max(min(reward, 1.0), -1.0)
            if episode_steps + 1 == self.max_episode_steps:
                mask = False
            else:
                mask = done
            # mask = False if episode_steps + 1 == self.max_episode_steps else done

            t_new = np.zeros(6)
            t_new[0] = clipped_reward
            t_new[1] = info['velocity'].x / Config.max_speed
            t_new[2] = info['velocity'].y / Config.max_speed
            t_new[3 + action] = 1.0

            # To calculate efficiently, set priority=max_priority here.
            self.memory.append((state, t), action, clipped_reward, (next_state, t_new), mask)

            self.steps += 1
            episode_steps += 1
            episode_return += reward
            state = next_state
            t = t_new
            nearmiss = nearmiss or info['near miss']
            accident = accident or info['accident']
            goal = info['goal']
            done = done or accident

            if self.is_update():
                self.learn()

            if self.steps % self.target_update_interval == 0:
                self.update_target()

            if self.steps % self.eval_interval == 0:
                self.evaluate()

            if self.steps % self.save_interval == 0:
                self.save_models()

        # log running mean of training rewards
        self.train_return.append(episode_return)

        if self.episodes % self.log_interval == 0:
            self.writer.add_scalar('reward/train', self.train_return.get(), self.steps)

        print("Episode: {}, Scenario: {}, Pedestrian Speed: {:.2f}m/s, Ped_distance: {:.2f}m".format(
            self.episodes, info['scenario'], info['ped_speed'], info['ped_distance']))
        print('Goal reached: {}, Accident: {}, Nearmiss: {}'.format(goal, accident, nearmiss))
        print('Total steps: {}, Episode steps: {}, Reward: {:.4f}'.format(self.steps, episode_steps, episode_return))
        print("Policy; ", action_count, "Critic: ", action_count_critic, "Alpha: {:.4f}".format(self.alpha.item()))

    def evaluate(self):
        num_episodes = self.test_env.number_of_episode
        log_info(f"number of validation epsidoes: {num_episodes}")
        log_info(f"val iter: {type(self.test_env.test_episodes)}")
        num_steps = 0
        total_return = 0.0
        total_goal = 0
        log_info('-' * 60)

        for episode in range(num_episodes):
            state = self.test_env.reset()
            episode_steps = 0
            episode_return = 0.0
            done = False
            action_count = {0: 0, 1: 0, 2: 0}
            t = np.zeros(6)  # reward, vx, vt, onehot last action
            t[3 + 1] = 1.0  # index = 3 + last_action(maintain)

            while (not done) and episode_steps < self.max_episode_steps:
                action = self.exploit((state, t))
                next_state, reward, done, info = self.test_env.step(action)
                action_count[action] += 1
                num_steps += 1
                episode_steps += 1
                episode_return += reward
                state = next_state
                t = np.zeros(6)
                t[0] = max(min(reward, 2.0), -2.0)
                t[1] = info['velocity'].x / Config.max_speed
                t[2] = info['velocity'].y / Config.max_speed
                t[3 + action] = 1.0
                done = done or info["accident"]

            episode += 1
            total_return += episode_return
            total_goal += int(info['goal'])
            log_info(f"@@@@@@@@@@ evaluating current model @@@@@@@@@@ \n" +\
                     f"current episode vs total episodes: {episode}/{num_episodes}\n"+\
                     f"current step vs total allowed steps: {episode}/{num_episodes}\n"+\
                     f"Speed: {info['ped_speed']:.2f}m/s, Dist.: {info['ped_distance']:.2f}m, Return: {episode_return:.4f}\n" +\
                     f"Goal: {info['goal']}, Accident: {info['accident']}, Act Dist.: {action_count}")

            self.file.write("Speed: {:.2f}m/s, Dist.: {:.2f}m, Return: {:.4f}".format(
                info['ped_speed'], info['ped_distance'], episode_return))
            self.file.write("Goal: {}, Accident: {}, Act Dist.: {}".format(
                info['goal'], info['accident'], action_count))

            if num_steps > self.num_eval_steps:
                break

        mean_return = total_return / num_episodes

        # if mean_return > self.best_eval_score:
        if total_goal > self.best_eval_score:
            self.best_eval_score = total_goal
            self.save_models()

        self.writer.add_scalar('reward/test', mean_return, self.steps)
        self.writer.add_scalar('reward/goal', total_goal, self.steps)

        self.test_env.prepare_validation_episodes()

        log_info(f"steps: {self.steps:<5}, mean return: {mean_return:<5f}, total goal {total_goal}")
        log_info('-' * 60)

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

    def update_target(self):
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
        self.conv.save(Config.MODEL_DIR + 'conv.pth')
        self.policy.save(Config.MODEL_DIR + 'policy.pth')
        self.online_critic.save(Config.MODEL_DIR + 'online_critic.pth')
        self.target_critic.save(Config.MODEL_DIR + 'target_critic.pth')

    def __del__(self):
        self.env.close()
        self.test_env.close()
        self.file.close()