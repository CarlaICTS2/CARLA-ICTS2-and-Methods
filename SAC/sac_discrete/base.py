from abc import ABC, abstractmethod
import os
import time
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import psutil
from SAC.sac_discrete.sacd.memory import LazyMultiStepMemory, LazyPrioritizedMultiStepMemory
from SAC.sac_discrete.sacd.utils import update_params, RunningMeanStats
from benchmark.environment.ped_controller import l2_distance
from config import Config


class BaseAgent(ABC):

    def __init__(self, env, test_env, log_dir, num_steps=100000, batch_size=64,
                 memory_size=1000000, gamma=0.99, multi_step=1,
                 target_entropy_ratio=0.98, start_steps=20000,
                 update_interval=4, target_update_interval=8000, 
                 use_per=False, num_eval_steps=125000, max_episode_steps=27000, save_interval=100000,
                 log_interval=10, eval_interval=1000, cuda=True, seed=0, display=False, resume=False,lr=0.005,max_grad_norm=None):
        super().__init__()
        self.env = env
        self.test_env = test_env # not used
        self.resume = resume
        self.max_grad_norm = max_grad_norm

        # Set seed.
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)
#        self.test_env.seed(2**31-1-seed)
        # torch.backends.cudnn.deterministic = True  # It harms a performance.
        # torch.backends.cudnn.benchmark = False  # It harms a performance.

        self.device = torch.device(
            "cuda" if cuda and torch.cuda.is_available() else "cpu")
        print(self.device)
        # LazyMemory efficiently stores FrameStacked states.
        if use_per:
            beta_steps = (num_steps - start_steps) / update_interval
            self.memory = LazyPrioritizedMultiStepMemory(
                capacity=memory_size,
                state_shape=self.env.observation_space.shape,
                device=self.device, gamma=gamma, multi_step=multi_step,
                beta_steps=beta_steps)
        else:
            self.memory = LazyMultiStepMemory(
                capacity=memory_size,
                state_shape=self.env.observation_space.shape,
                device=self.device, gamma=gamma, multi_step=multi_step)

        self.log_dir = log_dir + "_ter" + str(target_entropy_ratio) + "_lr" + str(lr) + "_bs" + str(batch_size) + "_start"+str(start_steps) + ("_gn" + str(self.max_grad_norm) if not max_grad_norm is None else "")
        print(self.log_dir)
        self.model_dir = os.path.join(self.log_dir, 'model')
        self.summary_dir = os.path.join(self.log_dir, 'summary')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
        self.file = open(self.summary_dir + "eval_results.log", "w")

        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.train_return = RunningMeanStats(log_interval)
        self.train_distance = RunningMeanStats(100)

        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.best_eval_score = -np.inf
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.gamma_n = gamma ** multi_step
        self.start_steps = start_steps
        self.update_interval = update_interval
        self.target_update_interval = target_update_interval
        self.use_per = use_per
        self.num_eval_steps = num_eval_steps
        self.max_episode_steps = max_episode_steps
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.display = display
        self.save_interval = save_interval

        self.best_crash_rate = 20
        self.best_goal_rate = 0
        self.best_near_miss = 20

    def run(self):
        print(self.num_steps)
        self.best_dist = 0
        while True:
            print(self.eval_interval)
            self.train_episode()
            if self.episodes % self.eval_interval == 0 and self.steps > 120000:
                self.save_models(os.path.join(self.model_dir, str(self.episodes)))
                # self.env.mode = "VALIDATION"
                # self.evaluate()
                self.env.mode = "TRAINING"

            if self.steps > self.num_steps:
                print("Done")
                break

    def is_update(self):
        return self.steps % self.update_interval == 0 and self.steps >= self.start_steps

    @abstractmethod
    def explore(self, state):
        pass

    @abstractmethod
    def exploit(self, state):
        pass

    @abstractmethod
    def update_target(self):
        pass

    @abstractmethod
    def calc_current_q(self, states, actions, rewards, next_states, dones):
        pass

    @abstractmethod
    def calc_target_q(self, states, actions, rewards, next_states, dones):
        pass

    @abstractmethod
    def calc_critic_loss(self, batch, weights):
        pass

    @abstractmethod
    def calc_policy_loss(self, batch, weights):
        pass

    @abstractmethod
    def calc_entropy_loss(self, entropies, weights):
        pass

    def train_episode(self):
        self.episodes += 1
        episode_return = 0.
        episode_steps = 0

        done = False
        nearmiss = False
        accident = False
        goal = False

        state = self.env.reset()
        action_count = {0: 0, 1: 0, 2: 0}
        action_count_critic = {0: 0, 1: 0, 2: 0}

        t = np.zeros(6)  # reward, vx, vt, onehot last action
        t[3 + 1] = 1.0  # index = 3 + last_action(maintain)
        begin_pos = self.env.world.player.get_location()
        while (not done) and episode_steps < self.max_episode_steps:
            if self.display:
                self.env.render()
            if self.start_steps > self.steps:

                if self.resume:
                    action, critic_action = self.explore((state, t))
                else:
                    action = self.env.action_space.sample()
                    action = np.random.choice([0,1,2],p=[0.7,0.2,0.1])
                    critic_action = action
            else:
                action, critic_action = self.explore((state, t))
            
            velocity = self.env.world.player.get_velocity()
            speed = (velocity.x * velocity.x + velocity.y * velocity.y) ** 0.5
            speed *= 3.6

            # TODO remove
            #if speed > Config.max_speed_kmh-1:
            #    action = 1
            #    critic_action = 1

            next_state, reward, done, info = self.env.step(action)
            action_count[action] += 1
            action_count_critic[critic_action] += 1
            #print(reward)

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
            if self.steps%1000==0:
                print(self.steps)
            
            if self.is_update():
                #print("Update")
                self.learn()

            if self.steps % self.target_update_interval == 0:
                self.update_target()

            if False:#self.plot_intention:
                #self.pc +=1
                if self.steps % 20 == 0:
                    fig=plt.figure()
                    plt.imshow(state)
                    fig.savefig("cp_debug/hylear_cp_%d.png"%self.steps,dpi=400)


            # if self.steps % self.save_interval == 0:
            #     self.save_models(os.path.join(self.model_dir, str(self.steps)))

        # We log running mean of training rewards.

        end_pos = self.env.world.player.get_location()
        dist = l2_distance(begin_pos, end_pos)
        self.train_return.append(episode_return)
        self.train_distance.append(dist)

        average_dist = self.train_distance.get()
        #if self.episodes % self.log_interval == 0:
        self.writer.add_scalar(
                'reward/train', self.train_return.get(), self.steps)
        self.writer.add_scalar(
                'stats/dist', dist, self.steps)
        self.writer.add_scalar(
            'stats/avg_dist', average_dist, self.steps
        )

        if average_dist > self.best_dist and self.steps > 2*self.start_steps:
            self.best_dist = average_dist
            self.save_models(os.path.join(self.model_dir, 'best'))
            print(30*"*")
            print("Saved with average dist", average_dist)
            print(30*"*")
        
        

            
            
            


        print("Episode: {}, Scenario: {}, Pedestrian Speed: {:.2f}m/s, Ped_distance: {:.2f}m".format(
            self.episodes, info['scenario'], info['ped_speed'], info['ped_distance']))
        print('Goal reached: {}, Accident: {}, Nearmiss: {}'.format(goal, accident, nearmiss))
        print('Total steps: {}, Episode steps: {}, Reward: {:.4f}'.format(self.steps, episode_steps, episode_return))
        print("Policy; ", action_count, "Critic: ", action_count_critic, "Alpha: {:.4f}".format(self.alpha.item()))
        print("Distance: ", dist, "\n")

    def learn(self):
        assert hasattr(self, 'q1_optim') and hasattr(self, 'q2_optim') and\
            hasattr(self, 'policy_optim') and hasattr(self, 'alpha_optim')

        for it in range(1):
            self.learning_steps += 1
            if self.use_per:
                batch, weights = self.memory.sample(self.batch_size)
            else:
                batch = self.memory.sample(self.batch_size)
                # Set priority weights to 1 when we don't use PER.
                weights = 1.

            q1_loss, q2_loss, errors, mean_q1, mean_q2 = \
                self.calc_critic_loss(batch, weights)
            policy_loss, entropies = self.calc_policy_loss(batch, weights)
            entropy_loss = self.calc_entropy_loss(entropies, weights)

            update_params(self.q1_optim, q1_loss)
            update_params(self.q2_optim, q2_loss)

            self.policy_optim.zero_grad()
            policy_loss.backward(retain_graph=False)
            #before = self.compute_policy_grad_norm()
            #if before > self.max_grad_norm:
            #    print("Before", before)
            #torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            #if before > self.max_grad_norm:
            #    after = self.compute_policy_grad_norm()
            #    print("After", after)
            self.policy_optim.step()
            #if self.steps < self.start_steps + 5000:

            #with torch.no_grad():
            #    torch.clamp(self.log_alpha, max=0.2)
            update_params(self.alpha_optim, entropy_loss)


            self.alpha = self.log_alpha.exp()
            #if self.steps < self.start_steps + 5000:
            #    print(self.log_alpha)
            #    print(self.alpha)

            if self.use_per:
                self.memory.update_priority(errors)

            if self.learning_steps % self.log_interval == 0:
                self.writer.add_scalar(
                    'loss/Q1', q1_loss.detach().item(),
                    self.learning_steps)
                self.writer.add_scalar(
                    'loss/Q2', q2_loss.detach().item(),
                    self.learning_steps)
                self.writer.add_scalar(
                    'loss/policy', policy_loss.detach().item(),
                    self.learning_steps)
                self.writer.add_scalar(
                    'loss/entropy_loss', entropy_loss.detach().item(),
                    self.learning_steps)
                self.writer.add_scalar(
                    'stats/alpha', self.alpha.detach().item(),
                    self.learning_steps)
                self.writer.add_scalar(
                    'stats/log_alpha', self.log_alpha.detach().item(),
                    self.learning_steps)        
                self.writer.add_scalar(
                    'stats/mean_Q1', mean_q1, self.learning_steps)
                self.writer.add_scalar(
                    'stats/mean_Q2', mean_q2, self.learning_steps)
                self.writer.add_scalar(
                    'stats/entropy', entropies.detach().mean().item(),
                    self.learning_steps)
                with torch.no_grad():
                    total_norm = 0
                    for p in self.policy.parameters():
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1. / 2)
                    self.writer.add_scalar(
                        'loss/policy_loss_gradient', total_norm,
                        self.learning_steps)
            
    def compute_policy_grad_norm(self):
        total_norm = 0
        for p in self.policy.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

    def evaluate(self, mode="VALIDATION"):

        self.env.mode = mode
        if mode=="VALIDATION":
            num_episodes = len(self.env.val_episodes)
        else:
            num_episodes = len(self.env.test_episodes)
        print(self.env.mode, num_episodes)
        episodes = 0
        num_steps = 0
        total_return = 0.0
        total_goal = 0
        time_to_goal_list = []
        exec_time_list = []
        num_accidents = 0
        num_nearmisses = 0
        num_goals = 0
        velocity_goal = 0
        print('-' * 60)
        #self.env.reset_eval_iterator()
        print(self.env.mode, num_episodes)
        for current_episode in range(1,num_episodes+1):
            state = self.env.reset()
            episode_steps = 0
            episode_return = 0.0
            done = False
            action_count = {0: 0, 1: 0, 2: 0}
            nearmiss = False
            acccident = False
            reward = 0
            t = np.zeros(6)  # reward, vx, vt, onehot last action
            t[3 + 1] = 1.0  # index = 3 + last_action(maintain)
            begin_pos = self.env.world.player.get_location()
            #print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
            while (not done) and episode_steps < self.max_episode_steps:
                start_time = time.time()
                action = self.exploit((state, t))
                next_state, reward, done, info = self.env.step(action)
                time_taken = time.time() - start_time
                exec_time_list.append(time_taken*1000)
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
                nearmiss = nearmiss or info["near miss"]

            episodes += 1
            velocity = info['velocity']
            speed = (velocity.x * velocity.x + velocity.y * velocity.y) ** 0.5
            speed *= 3.6
            time_to_goal = num_steps * Config.simulation_step
            time_to_goal_list.append(time_to_goal)
            num_accidents += 1 if info["accident"] else 0
            num_nearmisses += 1 if nearmiss and not info["accident"] else 0
            num_goals += 1 if info['goal'] else 0
            velocity_goal +=  speed if info['goal'] else 0
            end_pos = self.env.world.player.get_location()
            dist = l2_distance(begin_pos, end_pos)

            print("Done:", episodes)
            total_return += episode_return
            total_goal += int(info['goal'])
            print("Speed: {:.2f}m/s, Dist.: {:.2f}m, Return: {:.4f}".format(
                info['ped_speed'], info['ped_distance'], episode_return))
            print("Goal: {}, Accident: {}, Act Dist.: {}".format(info['goal'], info['accident'], action_count))
            print("Policy; ", action_count, "Distance: ", dist)
            print("Crash Rate: {:.2f}, Nearmiss Rate: {:.2f}, Goal Rate: {:.2f}".format(
                num_accidents/current_episode, num_nearmisses/current_episode, num_goals/current_episode))
            print("Velocity Goal: {:.4f}, Exec time: {:.4f},TTG: {:.4f}".format(velocity_goal/current_episode, np.mean(exec_time_list), np.mean(time_to_goal)))
            self.file.write("Speed: {:.2f}m/s, Dist.: {:.2f}m, Return: {:.4f}".format(
                info['ped_speed'], info['ped_distance'], episode_return))
            self.file.write("Goal: {}, Accident: {}, Act Dist.: {}".format(
                info['goal'], info['accident'], action_count))

            # if num_steps > self.num_eval_steps:
            #     break
        crash_rate = num_accidents/current_episode
        nearmiss_rate = num_nearmisses/current_episode
        goal_rate = num_goals/current_episode
        mean_return = total_return / episodes

        # if mean_return > self.best_eval_score:
        if self.env.mode == "VALIDATION":
            if crash_rate < self.best_crash_rate and total_goal > self.best_eval_score:
                self.best_eval_score = total_goal
                self.best_crash_rate = crash_rate
                self.save_models(os.path.join(self.model_dir, 'best'))
        self.save_models(os.path.join(self.model_dir, str(self.steps)))
        self.writer.add_scalar(
            'reward/test', mean_return, self.steps)
        self.writer.add_scalar(
            'reward/goal', total_goal, self.steps)

        self.env.val_episodes_iterator = iter(self.env.val_episodes)

        print(f'Num steps: {self.steps:<5}  '
              f'return: {mean_return:<5.1f}')
        print(f'Num steps: {self.steps:<5}  '
              f'goal return: {total_goal}')
        print('-' * 60)

    @abstractmethod
    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def __del__(self):
        self.env.close()
        #self.env.close()
        self.file.close()
