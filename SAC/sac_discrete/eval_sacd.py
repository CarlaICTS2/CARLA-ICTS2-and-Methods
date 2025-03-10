"""
Author: Dikshant Gupta
Time: 13.12.21 11:31
"""

import os
import numpy as np
import torch
from datetime import datetime
import time
import pickle as pkl

from benchmark.environment.ped_controller import l2_distance

from .base import BaseAgent
from config import Config
from SAC.sac_discrete.sacd.model import DQNBase, TwinnedQNetwork, CateoricalPolicy


class EvalSacdAgent(BaseAgent):

    def __init__(self, env, test_env, log_dir, path, num_steps=100000, batch_size=64,
                 lr=0.0003, memory_size=1000000, gamma=0.99, multi_step=1,
                 target_entropy_ratio=0.98, start_steps=20000,
                 update_interval=4, target_update_interval=8000,
                 use_per=False, dueling_net=False, num_eval_steps=125000, save_interval=100000,
                 max_episode_steps=27000, log_interval=10, eval_interval=1000,
                 cuda=True, seed=0, current_episode=0, agent="hypal", max_grad_norm=0):
        super().__init__(
            env, test_env, log_dir, num_steps, batch_size, memory_size, gamma,
            multi_step, target_entropy_ratio, start_steps, update_interval,
            target_update_interval, use_per, num_eval_steps, max_episode_steps, save_interval,
            log_interval, eval_interval, cuda, seed)

        # Define networks.
        self.conv = DQNBase(
            self.env.observation_space.shape[2]).to(self.device)
        self.policy = CateoricalPolicy(
            self.env.observation_space.shape[2], self.env.action_space.n,
            shared=True).to(self.device)
        self.online_critic = TwinnedQNetwork(
            self.env.observation_space.shape[2], self.env.action_space.n,
            dueling_net=dueling_net, shared=True).to(device=self.device)
        self.target_critic = TwinnedQNetwork(
            self.env.observation_space.shape[2], self.env.action_space.n,
            dueling_net=dueling_net, shared=True).to(device=self.device).eval()

        # Copy parameters of the learning network to the target network.
        self.target_critic.load_state_dict(self.online_critic.state_dict())

        path = path
        self.conv.load_state_dict(torch.load(path + "conv.pth"))
        self.policy.load_state_dict(torch.load(path + "policy.pth"))
        self.conv.eval()
        self.policy.eval()

        self.filename = path+"test_res.pkl"
        print(self.filename)
        self.current_episode = current_episode

    def evaluate(self):
        self.env.test_episodes = list(self.env.test_episodes)
        num_episodes = len(self.env.test_episodes)
        print(self.env.mode, num_episodes)
        episodes = 0
        num_steps = 0
        total_return = 0.0
        total_goal = 0
        time_to_goal_list = []
        exec_time_list = []
        ttg_list = []
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
            if info['goal']:
                ttg_list.append(time_to_goal)
            print("Done:", episodes)
            total_return += episode_return
            total_goal += int(info['goal'])
            print("Speed: {:.2f}m/s, Dist.: {:.2f}m, Return: {:.4f}".format(
                info['ped_speed'], info['ped_distance'], episode_return))
            print("Goal: {}, Accident: {}, Act Dist.: {}".format(info['goal'], info['accident'], action_count))
            print("Policy; ", action_count, "Distance: ", dist)
            print("Crash Rate: {:.2f}, Nearmiss Rate: {:.2f}, Goal Rate: {:.2f}".format(
                num_accidents/current_episode, num_nearmisses/current_episode, num_goals/current_episode))
            print("Velocity Goal: {:.4f}, Exec time: {:.4f}".format(velocity_goal/current_episode, np.mean(exec_time_list)))
            self.file.write("Speed: {:.2f}m/s, Dist.: {:.2f}m, Return: {:.4f}".format(
                info['ped_speed'], info['ped_distance'], episode_return))
            self.file.write("Goal: {}, Accident: {}, Act Dist.: {}".format(
                info['goal'], info['accident'], action_count))

            # if num_steps > self.num_eval_steps:
            #     break

        mean_return = total_return / episodes

        # if mean_return > self.best_eval_score:
        #if total_goal > self.best_eval_score:
        #    self.best_eval_score = total_goal
        #    self.save_models(os.path.join(self.model_dir, 'best'))
        #self.save_models(os.path.join(self.model_dir, str(self.steps)))
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

        #with open(self.filename, "wb") as write_file:
        #    pkl.dump(data_log, write_file)
        print("Log file written here: {}".format(self.filename))
        print('-' * 60)
        print(round(num_accidents/current_episode,2), " % ",
            round(num_nearmisses/current_episode,2), " % ",
            round(num_goals/current_episode,2)," % ",
            round(np.mean(ttg_list),2), " % ",
            round(np.mean(exec_time_list),2), " % ",
            )

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

    def explore(self, state):
        pass

    def update_target(self):
        pass

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        pass

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        pass

    def calc_critic_loss(self, batch, weights):
        pass

    def calc_policy_loss(self, batch, weights):
        pass

    def calc_entropy_loss(self, entropies, weights):
        pass

    def save_models(self, save_dir):
        pass

    # def __del__(self):
    #     self.file.close()
