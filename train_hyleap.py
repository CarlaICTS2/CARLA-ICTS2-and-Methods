"""
Author: Dikshant Gupta
Time: 11.06.22 14:02
"""

import pygame
import subprocess
import argparse
import time
import pickle as pkl
import numpy as np
from multiprocessing import Process
from datetime import datetime
from benchmark.environment.ped_controller import l2_distance

import os
import signal

from config import Config
from hyleap.despot import HyLEAP
from utils.connector import Connector
from benchmark.environment import GIDASBenchmark


def train_hyleap(arg):
    ##############################################################
    t0 = time.time()
    # Logging file
    filename = "_out/hyleap/{}.pkl".format(datetime.now().strftime("%m%d%Y_%H%M%S"))
    print(filename)

    # Setting up environment
    print("Environment port: {}".format(Config.port))
    env = GIDASBenchmark(port=Config.port)
    conn = Connector(Config.despot_port)
    agent = HyLEAP(env.world, env.map, env.scene, conn)
    env.reset_agent(agent)
    env.retarded_agent = "hyleap"
    ##############################################################

    ##############################################################
    # Simulation loop
    current_episode = 0
    max_episodes = len(env.episodes)
    print("Total training episodes: {}".format(Config.hyleap_training_epidsodes))
    pedestrian_path = {}
    data_log = {}
    num_accidents = 0
    num_nearmisses = 0
    num_goals = 0
    velocity_goal = 0
    exec_time_list = []
    training_epidsodes = Config.hyleap_training_epidsodes
    while current_episode < training_epidsodes:
        # Get the scenario id, parameters and instantiate the world
        total_episode_reward = 0
        observation = env.reset()
        nearmiss = False
        accident = False
        count = 0
        ped_data = []
        step_num = 0

        episode_log = {}
        exec_time = []
        actions_list = []
        risk = []
        ped_obs = []
        impact_speed = []
        trajectory = []
        action_count = {0: 0, 1: 0, 2: 0}
        begin_pos = env.world.player.get_location()
        trajectory = []
        env.planner_agent.conn.send_message(True, 0, 0, [0, 0], 0, [[0, 0]], [[0, 0, 0]])  # Initial message
        env.planner_agent.conn.receive_message()

        for step_num in range(Config.num_steps):
            ped_data.append((env.world.walker.get_location().x, env.world.walker.get_location().y))
            trajectory.append((env.world.player.get_location().x, env.world.player.get_location().y))
            if Config.display:
                env.render()

            start_time = time.time()
            observation, reward, done, info = env.step(action=None)
            if env.planner_agent.pedestrian_observable:
                time_taken = (time.time() - start_time)
                exec_time_list.append(time_taken)
                count += 1
            else:
                time_taken = 0
            exec_time.append(time_taken)

            if env.control.throttle != 0:
                action = 0
            elif env.control.brake != 0:
                action = 2
            else:
                action = 1
            actions_list.append(action)
            action_count[action] += 1
            velocity = info['velocity']
            velocity_x = velocity.x
            velocity_y = velocity.y
            speed = np.sqrt(velocity_x ** 2 + velocity_y ** 2)
            impact_speed.append(speed)

            nearmiss_current = info['near miss']
            nearmiss = nearmiss_current or (nearmiss and speed > 0)
            accident_current = info['accident']
            accident = accident_current or (accident and speed > 0)
            total_episode_reward += reward
            risk.append(info['risk'])

            if done or accident:
                break
        current_episode += 1
        pedestrian_path[current_episode] = ped_data

        # Evaluate episode statistics(Crash rate, nearmiss rate, time to goal, smoothness, execution time, violations)
        time_to_goal = (step_num + 1) * Config.simulation_step
        episode_log['ttg'] = time_to_goal
        episode_log['risk'] = risk
        episode_log['actions'] = actions_list
        episode_log['exec'] = exec_time
        episode_log['impact_speed'] = impact_speed
        episode_log['trajectory'] = trajectory
        episode_log['ped_dist'] = info['ped_distance']
        episode_log['scenario'] = info['scenario']
        episode_log['ped_speed'] = info['ped_speed']
        episode_log['crash'] = accident
        episode_log['nearmiss'] = nearmiss
        data_log[current_episode] = episode_log

        speed = (velocity.x * velocity.x + velocity.y * velocity.y) ** 0.5
        speed *= 3.6
        num_accidents += 1 if accident else 0
        num_nearmisses += 1 if nearmiss and not accident else 0
        num_goals += 1 if info['goal'] else 0
        end_pos = env.world.player.get_location()
        dist = l2_distance(begin_pos, end_pos)
        velocity_goal +=  speed if info['goal'] else 0
        
        print("Episode: {}, Scenario: {}, Pedestrian Speed: {:.2f}m/s, Ped_distance: {:.2f}m".format(
            current_episode, info['scenario'], info['ped_speed'], info['ped_distance']))
        print('Goal reached: {}, Accident: {}, Nearmiss: {}, Reward: {:.4f}, Risk: {:.4f}'.format(
            info['goal'], accident, nearmiss, total_episode_reward, sum(risk) / step_num))
        print('Time to goal: {:.4f}s, Execution time: {:.4f}ms'.format(
            time_to_goal, sum(exec_time) * 1000 / (count + 1)))
        print("Policy; ", action_count, "Distance: ", dist)
        print("Crash Rate: {:.2f}, Nearmiss Rate: {:.2f}, Goal Rate: {:.2f}".format(
            num_accidents/current_episode, num_nearmisses/current_episode, num_goals/current_episode))
        print("Velocity Goal: {:.4f}, Exec time: {:.4f}".format(velocity_goal/current_episode, np.mean(exec_time_list)*1000))

        print()
        env.planner_agent.conn.send_message(True, reward, 0, [0, 0], 0, [[0, 0]], [[0, 0, 0]])
        env.planner_agent.conn.receive_message()
        ##############################################################

    env.close()
    print("Testing time: {:.4f}hrs".format((time.time() - t0) / 3600))
    with open(filename, "wb") as write_file:
        pkl.dump(data_log, write_file)
    with open('_out/pedestrian_data.pkl', 'wb') as file:
        pkl.dump(pedestrian_path, file)


def main(arg):
    print(__doc__)

    try:
        train_hyleap(arg)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
        pygame.quit()


def run_server():
        port = "-carla-port={}".format(Config.port)
        carla_p = "your path to carla"
        command = "unset SDL_VIDEODRIVER && ./CarlaUE4.sh  -quality-level="+ Config.qw  +" your arguments" + port # -quality-level=Low 
        p = subprocess.run(['cd '+carla_p+' && ' + command ], shell=True)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    arg_parser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2300)')
    arg_parser.add_argument(
        '-ep', '--episode',
        default=0,
        type=int,
        help='episode number to resume from')
    arg_parser.add_argument('--test', type=str, default='')
    arg_parser.add_argument('--agent', type=str, default='hyleap')
    arg_parser.add_argument('--despot_port', type=int, default=1245)
    args = arg_parser.parse_args()
    Config.port = args.port
    Config.despot_port = args.despot_port
    Config.qw = "Low"
    if args.test:
        Config.test_scenarios = [args.test]
    #Config.scenarios = ["01_int"]
    print(Config.scenarios)
    # p = Process(target=run_server)
    # p.start()
    # time.sleep(20)  # wait for the server to start

    main(args)
    os.kill(os.getppid(), signal.SIGHUP)
