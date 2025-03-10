"""
Author: Dikshant Gupta
Time: 28.09.22 07:31
"""

import os
import yaml
import argparse
import subprocess
import time
from datetime import datetime
from multiprocessing import Process


from SAC.sac_discrete import EvalSacdAgent
from benchmark.environment import GIDASBenchmark
from utils.connector import DespotBridge
from hylear.hylear_controller import HyLEAR
from config import Config


def run(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Create environments.
    env = GIDASBenchmark(port=Config.port, mode="TESTING")
    env.eval(current_episode=args.episode)
    conn = DespotBridge()
    eval_mode = True
    agent = HyLEAR(env.world, env.map, env.scene, conn, eval_mode)
    env.reset_agent(agent)

    # Specify the directory to log.
    name = args.config.split('/')[-1].rstrip('.yaml')
    if args.shared:
        name = 'shared-' + name
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        '_out', args.env_id, 'eval', f'{name}-seed{args.seed}-{time}')

    # Create the agent.
    agent = EvalSacdAgent(
        env=env, test_env=env, log_dir=log_dir, cuda=args.cuda, current_episode=args.episode,
        seed=args.seed, agent=args.agent, path= "./_out/hyLear_plain/hylear-seed0-20241106-2334_ter0.6_lr0.005_bs128_start25000/model/1250/")
    agent.evaluate()


def run_server():
    # train environment
    port = "-carla-port={}".format(Config.port)
    subprocess.run(['cd /home/carla && SDL_VIDEODRIVER=offscreen ./CarlaUE4.sh -opengl ' + port], shell=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('./SAC/sac_discrete/config', 'sacd.yaml'))
    parser.add_argument('--shared', action='store_true')
    parser.add_argument('--env_id', type=str, default='GIDASBenchmark')
    parser.add_argument('--agent', type=str, default='hylear')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--test', type=str, default='')
    args = parser.parse_args()

    Config.port = args.port
    print('Env. port: {}'.format(Config.port))
    if args.test:
        Config.test_scenarios = [args.test]
    Config.test_scenarios = "01_int"

    p = Process(target=run_server)
    p.start()
    time.sleep(5)

    run(args)
