import signal
import sys

from utils.logger import initialize_logging

sys.path.append("/workspace/data/CARLA-ICTS")
import os
import yaml
import argparse
import subprocess
import time
from datetime import datetime
from multiprocessing import Process


from SAC.sac_discrete import SacdAgent
from hylear.hylear_agent import SharedSacdAgent
from benchmark.environment import GIDASBenchmark
from utils.connector import DespotBridge
from hylear.hylear_controller_c import HyLEARC
from config import Config


def run(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    config['num_steps'] = 3e6
    initialize_logging()

    # Create environments.
    env = GIDASBenchmark(port=Config.port)
    conn = DespotBridge()
    eval_mode = False
    agent = HyLEARC(env.world, env.map, env.scene, conn, eval_mode)
    env.reset_agent(agent)
    #test_env = GIDASBenchmark(port=Config.port + 100, setting="special")

    # Specify the directory to log.
    name = config["name"]
    config.pop("name",None)
    name = args.config.split('/')[-1].rstrip('.yaml')
    if args.shared:
        name = 'shared-' + name
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        '_out', args.env_id, f'{name}-seed{args.seed}-{time}')
    log_dir= os.path.join(
        '_out', "hyLear_c", f'{name}-seed{args.seed}-{time}')

    # Create the agent.
    Agent = SharedSacdAgent
    agent = Agent(
        env=env, test_env=None, log_dir=log_dir, cuda=args.cuda,
        seed=args.seed, **config)
    agent.run()


def run_server():
    # train environment
    # port = "-carla-port={}".format(Config.port)
    # carla_p = "/home/carla"
    # if not Config.server:
    #     p = subprocess.run(['cd '+carla_p+' && ./CarlaUE4.sh -RenderOffScreen -carla-server -benchmark -fps=50' + port], shell=True)
    #     #cmd = 'cd '+carla_p+' && ./CarlaUE4.sh -quality-level=Low -RenderOffScreen -carla-server -benchmark -fps=50' + port
    #     #pro = subprocess.Popen(cmd, stdout=subprocess.PIPE,
    #     #                   shell=True, preexec_fn=os.setsid)
    # else:
    # command = "./CarlaUE4.sh -RenderOffscreen" # -quality-level=Low
    p = subprocess.run(["/home/carla/CarlaUE4.sh -RenderOffscreen -carla-port=56600"], shell=True)
    return p

def run_despot_server():
    p = subprocess.run(["./hyleap/smart-car-sim-master/is-despot/problems/hybridVisual_car/car "], shell=True)
    return p


def run_test_server():
    # test environment
    port = "-carla-port={}".format(Config.port + 100)
    carla_p = "your path to carla"
    command = "unset SDL_VIDEODRIVER && ./CarlaUE4.sh  -quality-level="+ Config.qw  +" your arguments" + port # -quality-level=Low 
    p = subprocess.run(['cd '+carla_p+' && ' + command ], shell=True)
    return p


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('hylear'))
    parser.add_argument('--shared', action='store_true')
    parser.add_argument('--env_id', type=str, default='GIDASBenchmark')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--agent', type=str, default='isdespot')
    parser.add_argument('--server', action='store_true')
    parser.add_argument("--qw", type=str, default="Low")
    parser.add_argument('--despot_port', type=int, default=1255)
    args = parser.parse_args()
    Config.server = args.server
    args.config = os.path.join('SAC/sac_discrete/config', args.config+".yaml")
    Config.port = args.port
    Config.qw = args.qw
    Config.despot_port = args.despot_port
    print('Env. port: {}'.format(Config.port))
    
    p = Process(target=run_server)
    p.start()
    time.sleep(20)

    # d = Process(target=run_despot_server)
    # d.start()

    #if Config.server:
    #    p2 = Process(target=run_test_server)
    #    p2.start()
    #    t.sleep(20)
    
    run(args)
    os.kill(os.getppid(), signal.SIGHUP)
