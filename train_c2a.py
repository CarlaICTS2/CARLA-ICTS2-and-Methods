import sys

from A2C.a2c.a2ccadrl import A2CCadrl
from utils.connector import Connector
sys.path.append("your path to a2c code")
import os
import yaml
import argparse
import subprocess
import time as t
from datetime import datetime
from multiprocessing import Process
from SAC.sac_controller import SAC

import os
import signal

from SAC.sac_discrete import SacdAgent
from SAC.sac_discrete.shared_sacd import SharedSacdAgent
from benchmark.environment import GIDASBenchmark
from config import Config
#from benchmark.rl.a2c.a2ctrainer import A2CTrainer
from A2C.a2c.a2ctrainer import A2CTrainer
#python train_c2a.py --server --cuda --port=2567
def run(args):

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    # Create environments.
    env = GIDASBenchmark(port=Config.port)
    env.world.camera = False
    #conn = Connector(Config.despot_port)
    agent = A2CCadrl(env.world, env.map, env.scene,conn=None)
    env.reset_agent(agent)

    name = config["name"]
    config.pop("name",None)
    if args.shared:
        name = 'shared-' + name
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        '_out', name, f'{name}-seed{args.seed}-{time}')

    Agent = A2CTrainer #SacdAgent if not args.shared else 
    agent = Agent(
        env=env, log_dir=log_dir, **config)
    print("Agent run")
    agent.run()


def run_server():
    # train environment
    port = "-carla-port={}".format(Config.port)
    if not Config.server:
        carla_p = "your path to carla"
        p = subprocess.run(['cd '+carla_p+' && ./CarlaUE4.sh your arguments' + port], shell=True)
        #cmd = 'cd '+carla_p+' && ./CarlaUE4.sh -quality-level=Low -RenderOffScreen -carla-server -benchmark -fps=50' + port
        #pro = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
        #                   shell=True, preexec_fn=os.setsid)
    else:
        carla_p = "your path to carla"
        command = "unset SDL_VIDEODRIVER && ./CarlaUE4.sh  -quality-level="+ Config.qw  +" your arguments" + port # -quality-level=Low 
        p = subprocess.run(['cd '+carla_p+' && ' + command ], shell=True)
        
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
        '--config', type=str, default='a2c.yaml')
    parser.add_argument('--shared', action='store_true')
    parser.add_argument('--env_id', type=str, default='GIDASBenchmark')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--server', action='store_true')
    parser.add_argument("--qw", type=str, default="Low")
    #parser.add_argument('--despot_port', type=int, default=1255)
    args = parser.parse_args()
    globals()["server"] = args.server
    Config.server = args.server
    args.config = os.path.join('your path to configs', args.config)
    Config.port = args.port
    Config.qw = args.qw
    #Config.despot_port = args.despot_port
    print('Env. port: {}'.format(Config.port))

    p = Process(target=run_server)
    p.start()
    t.sleep(20)
    #if Config.server:
    #    p2 = Process(target=run_test_server)
    #    p2.start()
    #    t.sleep(20)
    
    run(args)
    print("Terminating")
    os.kill(os.getppid(), signal.SIGHUP)