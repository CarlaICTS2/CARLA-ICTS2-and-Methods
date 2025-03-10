
import sys



sys.path.append("/workspace/data/CARLA-ICTS")
from A2C.a2c.hyreal_a2c_AUTOBOTS import HyREALA2C_AUTOBOTS
import os
import yaml
import argparse
import subprocess
import time
from datetime import datetime
from multiprocessing import Process
from A2C.a2c.a2ccadrl import A2CCadrl
from A2C.a2c.a2ctrainer import A2CTrainer
from A2C.a2c.hyreal_a2c import HyREALA2C
import os
import signal

from SAC.sac_discrete import EvalSacdAgent
from benchmark.environment import GIDASBenchmark
from config import Config


def run(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Create environments.
    env = GIDASBenchmark(port=Config.port,mode="TESTING")
    if args.agent == "a2c":
        path = path = "your model for eval"
        agent = A2CCadrl(env.world, env.map, env.scene,conn=None)
        env.reset_agent(agent)
    else:
        path = "./_out/hyreal_cog/hyreal_cog-seed0-20241002-1407_ec0.005_lr5e-05_vc1.0/model/model_3000.pth"
        # path = None
        # agent = HyREALA2C(env.world, env.map, env.scene,conn=None)
        agent = HyREALA2C_AUTOBOTS(env.world, env.map, env.scene,conn=None)
        env.reset_agent(agent)

    # Specify the directory to log.
    name = config["name"]
    config.pop("name",None)
    if args.shared:
        name = 'shared-' + name
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        '_out', args.env_id, 'eval', f'{name}-seed{args.seed}-{time}')
    # Create the agent.
    Agent = A2CTrainer #SacdAgent if not args.shared else 
    agent = Agent(
        env=env, log_dir=log_dir, path=path, **config)
    print("Agent run")
    agent.evaluate(mode="TESTING")



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
    p = subprocess.run([f"/home/carla/CarlaUE4.sh -RenderOffscreen -carla-port={Config.port}"], shell=True)
    return p


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default='sacd')
    parser.add_argument('--shared', action='store_true')
    parser.add_argument('--env_id', type=str, default='GIDASBenchmark')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--server', action='store_true')
    parser.add_argument("--qw", type=str, default="Low")
    parser.add_argument("--agent", type=str,default="a2c")
    parser.add_argument("--test",type=str, default=None)

    args = parser.parse_args()
    globals()["server"] = args.server
    Config.server = args.server
    args.config = os.path.join('SAC/sac_discrete/config', args.config+".yaml")
    Config.port = args.port
    Config.qw = args.qw

    if args.test:
        if args.test == "all":
            # TODO PAGI: ADD SCENARIO HERE
            Config.scenarios = ['01_int', '02_int', '03_int', '04_int', '05_int', '06_int',
                                '01_non_int', '02_non_int', '03_non_int', '04_non_int', '05_non_int', '06_non_int']
        else:
            Config.scenarios = [args.test]
    print(args.test)
    print('Env. port: {}'.format(Config.port))

    p = Process(target=run_server)
    p.start()
    time.sleep(20)
    #if Config.server:
    #    p2 = Process(target=run_test_server)
    #    p2.start()
    #    t.sleep(20)
    
    run(args)
    os.kill(os.getppid(), signal.SIGHUP)
