from argparse import ArgumentParser, SUPPRESS, RawTextHelpFormatter
import textwrap
import time
import json
from collections import Counter
from multiprocessing import Process, Queue

import numpy as np

from agents.hybrid.hylear_autobots_cognitive import HyLEAR_NavA2C_AutoBots_Cognitive
from agents.hybrid.hylear_cognitive import HyLEAR_NavA2C_Cognitive
from agents.hybrid.hylear_autobots import HyLEAR_NavA2C_AutoBots
from benchmark.environment.environment import GIDASBenchmark
from agents.planner.isdespotp import ISDespotP
from agents.learner.nava2c import NavA2C
from agents.hybrid.leader import LEADER
from agents.hybrid.hyleap import HyLEAP
from agents.hybrid.hyplan import HyPLAN
from agents.hybrid.hylear_refactored import HyLEAR_NavA2C
from utils.config import Config, Address, Agent, DespotVariant, Mode, Action, HYBRID_AGENTS
from config import Config as oldConfig
from utils.logger import initialize_logging, log_debug, log_info, log_performance_metrics, log_data
from utils.utils import run_carla_server, l2_distance, has_agent_stopped, find_free_port, kill_all_processes, run_despot

if __name__ == '__main__':

    arg_parser = ArgumentParser(
        description='CARLA CTS02 Benchmark Script',
        argument_default=SUPPRESS,
        prog=__file__, 
        usage='%(prog)s [options]',
        allow_abbrev=False,
        add_help=True,
        formatter_class=RawTextHelpFormatter
    )   

    meta_level_group = arg_parser.add_argument_group(title="META arguments", 
                                                     description="arguments that affect script execution as a whole")
    meta_level_group.add_argument(
        '--agent',
        dest="agent",
        metavar=(", ".join([str(agent) for agent in Agent])),
        action="store",
        type=str, 
        choices=[str(agent) for agent in Agent],
        nargs="?", 
        default="is_despot",
        help=textwrap.dedent(
            ''' 
            The agent to be evaluated over the Carla-CTS02 benchmark (default: %(default)s).
            
            ''')
    )
    meta_level_group.add_argument(
        '--mode',
        dest="mode",
        metavar="train, validate, test",
        action="store",
        type=str, 
        choices=["train", "validate", "test"],
        nargs="?", 
        default="train",
        help=textwrap.dedent(
            '''
            Whether to train, validate or test the specified agent (default: %(default)s).

            ''')
    )   
    meta_level_group.add_argument(
        '--scenario',
        dest="scenario",
        metavar="[0, 12]",
        action="store",
        type=str, 
        choices=["01_int", "02_int", "03_int", "04_int", "05_int", "06_int",
                 "01_non_int", "02_non_int", "03_non_int", "04_non_int", "05_non_int", "06_non_int"],
        nargs="+", 
        default=["00"],
        help=textwrap.dedent(
            '''
            Scenario(s) to evaluate the agent on (default: %(default)s = all). 
            Use like --scenario 1 2 3 4 to select the first four scenarios.
            Only mode --test can be run in conjunction with scenario 11 & 12.

            ''')
    ) 
    meta_level_group.add_argument(
        '--resume_from_episode',
        dest="resume_from_episode",
        metavar="1, 12626",
        action="store",
        type=int, 
        nargs="?", 
        default=1,
        help=textwrap.dedent(
            '''
            Episode number to resume from with the given episode included (default: %(default)s).

            ''')
    ) 
    meta_level_group.add_argument(
        '--remote',
        dest="remote",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Required if the script is run on SLURM cluster (default: %(default)s).
            
            ''')
    ) 
    meta_level_group.add_argument(
        '--predict_pedestrian_path',
        dest="predict_pedestrian_path",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Enables pedestrian path prediction using M2P3 (default: %(default)s).
            This requires the existence of a trained M2P3 model in the 'input' directory.
            
            ''')
    )
    meta_level_group.add_argument(
        '--plan_path_with_risk',
        dest="plan_path_with_risk",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Enables risk aware path planning (default: %(default)s).
            WARNING: This will influence execution time and performance.

            ''')
    ) 
    meta_level_group.add_argument(
        '--load_checkpoint',
        dest="load_checkpoint",
        metavar="DIRECTORY NAME",
        action="store",
        type=str, 
        nargs="?", 
        default="latest",
        help=textwrap.dedent(
            '''
            Specify model checkpoint directory to load for testing (default: %(default)s).
            
            ''')
    )
    meta_level_group.add_argument(
        '--output_directory',
        dest="output_directory",
        metavar="DIRECTORY NAME",
        action="store",
        type=str, 
        nargs="?", 
        default=None,
        help=textwrap.dedent(
            '''
            Directory under which all script outputs will be centralized (default: %(default)s).
            If none is specified, it will be constructed based on the arguments provided. 
            
            ''')
    )
    meta_level_group.add_argument(
        '--epochs',
        dest="epochs",
        metavar="1, 2, 3",
        action="store",
        type=int,
        choices=[1, 2, 3],
        nargs="?", 
        default=1,
        help=textwrap.dedent(
            '''
            Specifies the number of iterations over the entire training set (default: %(default)s).
            
            ''')
    )
    meta_level_group.add_argument(
        '--skip_no_pedestrian_steps',
        dest="skip_no_pedestrian_steps",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Skips agent calls if no pedestrian is in the scene and acellerates instead (default: %(default)s).
            
            ''')
    )
    meta_level_group.add_argument(
        '--max_episode_steps',
        dest="max_episode_steps",
        metavar="500, 4000",
        action="store",
        type=int, 
        choices=range(500, 4000+1, 1),
        nargs="?", 
        default=500,
        help=textwrap.dedent(
            '''
            Enforces a maximum number of steps for each simulated episode,
            after which any episode will be forcefully terminated even when 
            no conclusive result (collision or goal) has been reached yet (default: %(default)s).
            Prematurely terminated episodes will not be used for training.
            
            ''')
    )
    meta_level_group.add_argument(
        '--seed',
        dest="seed",
        metavar="0, 1024",
        action="store",
        type=int, 
        choices=range(0, 1024, 1),
        nargs="?", 
        default=42,
        help=textwrap.dedent(
            '''
            Random number seed (default: %(default)s).
            
            ''')
    ) 


    is_despot_group = arg_parser.add_argument_group("IS-DESPOT arguments")
    is_despot_group.add_argument(
        '--aggressive_belief_updates',
        dest="aggressive_belief_updates",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Belief is only influenced by current step (default: %(default)s).

            ''')
    )
    is_despot_group.add_argument(
        '--minimal_noise',
        dest="minimal_noise",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Removes as much (artificially injected) noise as possible during planning simulation (default: %(default)s).

            ''')
    )
    is_despot_group.add_argument(
        '--no_importance_sampling',
        dest="no_importance_sampling",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            RDo not use importance sampling (default: %(default)s).

            ''')
    )
    is_despot_group.add_argument(
        '--no_normalization',
        dest="no_normalization",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Disable normalization for importance distribution (default: %(default)s).

            ''')
    )
    is_despot_group.add_argument(
        '--noise',
        dest="noise",
        metavar="0, 1",
        action="store",
        type=float, 
        choices=np.arange(0, 1+0.01, 0.01),
        nargs="?", 
        default=0.1,
        help=textwrap.dedent(
            '''
            Noise level for transitions in belief update (default: %(default)s).

            ''')
    )  
    is_despot_group.add_argument(
        '--timeout',
        dest="timeout",
        metavar="0.01, 1",
        action="store",
        type=float, 
        choices=np.arange(0.01, 1.0, 0.01),
        nargs="?", 
        default=0.25,
        help=textwrap.dedent(
            '''
            Belief tree construction time per scene simulation step in seconds (default: %(default)s).

            ''')
    ) 
    is_despot_group.add_argument(
        '--time_per_planning_step',
        dest="time_per_planning_step",
        metavar="0.01, 1",
        action="store",
        type=float, 
        choices=np.arange(0.01, 1.0, 0.01),
        nargs="?", 
        default=0.25,
        help=textwrap.dedent(
            '''
            Time between planning simulation steps during belief tree construction in seconds (default: %(default)s).

            ''')
    ) 
    is_despot_group.add_argument(
        '--max_search_depth',
        dest="max_search_depth",
        metavar="1, 1000",
        action="store",
        type=int, 
        choices=range(1, 1000+1, 1),
        nargs="?", 
        default=20,
        help=textwrap.dedent(
            '''
            Maximum search depth during belief tree construction (default: %(default)s).

            ''')
    )  
    is_despot_group.add_argument(
        '--discount_factor',
        dest="discount_factor",
        metavar="0, 0.99",
        action="store",
        type=float, 
        choices=np.arange(0.0, 1.0, 0.01),
        nargs="?", 
        default=0.99,
        help=textwrap.dedent(
            '''
            Factor to discount future rewards (default: %(default)s).

            ''')
    )  
    is_despot_group.add_argument(
        '--particle_number',
        dest="particle_number",
        metavar="1, 10000",
        action="store",
        type=int, 
        choices=range(1, 10000+1, 1),
        nargs="?", 
        default=500,
        help=textwrap.dedent(
            '''
            Number of particles used to approximate belief nodes (default: %(default)s).

            ''')
    ) 
    is_despot_group.add_argument(
        '--gap_reduction_rate',
        dest="gap_reduction_rate",
        metavar="0, 1",
        action="store",
        type=float, 
        choices=np.arange(0.0, 1.0+0.01, 0.01),
        nargs="?", 
        default=0.95,
        help=textwrap.dedent(
            '''
            Required gap reduction rate of each trial (default: %(default)s).

            ''')
    ) 
    is_despot_group.add_argument(
        '--max_policy_simulation_length',
        dest="max_policy_simulation_length",
        metavar="0, 1000",
        action="store",
        type=int, 
        choices=range(0, 1000, 1),
        nargs="?", 
        default=90,
        help=textwrap.dedent(
            '''
            Number of steps to simulate the reactive controller at leaf nodes (default: %(default)s).

            ''')
    ) 
    is_despot_group.add_argument(
    '--pruning_constant',
    dest="pruning_constant",
    metavar="0.01, 1",
    action="store",
    type=float, 
    choices=np.arange(0.0, 1.0, 0.0001),
    nargs="?", 
    default=0,
    help=textwrap.dedent(
        '''
        Pruning constant for regularization (default: %(default)s).

        ''')
    ) 

    hyp_despot_group = arg_parser.add_argument_group("HYP-DESPOT arguments")
    hyp_despot_group.add_argument(
        '--GPU',
        dest="GPU",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Enable GPU parallelization (default: %(default)s).

            ''')
    )
    hyp_despot_group.add_argument(
        '--GPU_id',
        dest="GPU_id",
        metavar="0, 100",
        action="store",
        type=int, 
        choices=range(0, 100, 1),
        nargs="?", 
        default=0,
        help=textwrap.dedent(
            '''
            GPU used for parallelization (default: %(default)s).

            ''')
    )
    hyp_despot_group.add_argument(
        '--CPU',
        dest="CPU",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Enable CPU multithreading (default: %(default)s).

            ''')
    )
    hyp_despot_group.add_argument(
        '--num_threads',
        dest="num_threads",
        metavar="2, 100",
        action="store",
        type=int, 
        choices=range(2, 100, 1),
        nargs="?", 
        default=1,
        help=textwrap.dedent(
            '''
            Number of parallel CPU threads (default: %(default)s).

            ''')
    )
    hyp_despot_group.add_argument(
        '--exploration_scheme',
        dest="exploration_scheme",
        metavar="UCT | Vloss",
        action="store",
        type=str, 
        choices=["UCT", "Vloss"],
        nargs="?", 
        default="Vloss",
        help=textwrap.dedent(
            '''
            Scheme for guiding parallel simulation trajectory exploration (default: %(default)s).

            ''')
    )
    hyp_despot_group.add_argument(
        '--action_exploration_constant',
        dest="action_exploration_constant",
        metavar="0, 1",
        action="store",
        type=float, 
        choices=np.arange(0, 1+0.01, 0.01),
        nargs="?", 
        default=0.95,
        help=textwrap.dedent(
            '''
            Exploration constant for action branches (default: %(default)s).

            ''')
    )
    hyp_despot_group.add_argument(
        '--observation_exploration_const',
        dest="observation_exploration_const",
        metavar="0, 1",
        action="store",
        type=float, 
        choices=np.arange(0, 1+0.01, 0.01),
        nargs="?", 
        default=0.05,
        help=textwrap.dedent(
            '''
            Exploration constant for observation branches (default: %(default)s).

            ''')
    )
    

    hyleap_group = arg_parser.add_argument_group("HyLEAP arguments")
    hyleap_group.add_argument(
        '--hacky_hyleap',
        dest="hacky_hyleap",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Enables Florian's hacky HyLEAP conde snippets (default: %(default)s).

            ''')
    )
    hyleap_group.add_argument(
        '--decouple_hyleap',
        dest="decouple_hyleap",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Trains HyLEAP network without interferring in belief tree construction (default: %(default)s).

            ''')
    )
    hyleap_group.add_argument(
        '--use_dropout',
        dest="use_dropout",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Trains HyLEAP network with a dropout layer in the critic's NN architecture (default: %(default)s).
            This is used for fair comparability between HyLEAP and HyPLAN as both NN architectures msut be identical.

            ''')
    )
        

    hyplan_group = arg_parser.add_argument_group("HyPLAN arguments")
    hyplan_group.add_argument(
        '--improve_reward',
        dest="improve_reward",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Clips rewards to [-1,INF] and standardizes across steps of a given episode (default: %(default)s).

            ''')
    )
    hyplan_group.add_argument(
        '--hyplan_train_integration',
        dest="hyplan_train_integration",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Enables vertical pruning during HyPLAN training (default: %(default)s).

            ''')
    )
    hyplan_group.add_argument(
        '--hyplan_num_forward_passes',
        dest="hyplan_num_forward_passes",
        metavar="10, 100",
        action="store",
        type=int, 
        choices=[10, 100],
        nargs="?", 
        default=None,
        help=textwrap.dedent(
            '''
            Specifies the number of forward passes used for uncertainty calculation (default: %(default)s).

            ''')
    )
    hyplan_group.add_argument(
        '--nava2c',
        dest="nava2c",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Uses Akash Sinha's NavA2C NN model architecture for HyLEAP or HyPLAN (default: %(default)s).
            For more details view: https://arxiv.org/abs/2311.12875 or https://github.com/roboak/Nav-Q

            ''')
    )  
    hyplan_group.add_argument(
        '--hidden_layer_size',
        dest="hidden_layer_size",
        metavar="128, 256",
        action="store",
        type=int, 
        choices=[128, 256],
        nargs="?", 
        default=256,
        help=textwrap.dedent(
            '''
            Specifies the size of the hidden layer of the neural network (128: A2C, 256: NavA2C) (default: %(default)s).

            ''')
    )


    leader_group = arg_parser.add_argument_group("LEADER arguments")
    leader_group.add_argument(
        '--attention_sampling',
        dest="attention_sampling",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Samples pedestrian goal directions from attention distribution generated by LEADER (default: %(default)s).
            For more details view: https://arxiv.org/abs/2209.11422 or https://github.com/modanesh/LEADER

            ''')
    )  

    debug_group = arg_parser.add_argument_group("DEBUG arguments")
    debug_group.add_argument(
        '--carla_port', # optional command-line argument
        dest="carla_port", # name of the stored variable in the argument parser
        metavar="1024-65535", # name of the value-placeholder (displayed when --help is provided)
        action="store", # simply store provided value 
        type=int, # type conversion
        choices=range(1024, 65535+1, 1), # allowed non-system ports
        nargs="?", # extract as single item
        default=None, # default value if command-line option is not provided
        help=textwrap.dedent(
            '''
            TCP port to start the CARLA server with (default: None).
            Used exclusively when running the script locally.

            ''')
    )
    debug_group.add_argument(
        '--despot_port',
        dest="despot_port",
        metavar="1024, 65535",
        action="store",
        type=int, 
        choices=range(1024, 65535+1, 1),
        nargs="?", 
        default=None,
        help=textwrap.dedent(
            '''
            TCP port to start the C++ process running is-despot with (default: None).
            Used exclusively when running the script locally.
            
            ''')
    )
    debug_group.add_argument(
        '--track_planning_effort',
        dest="track_planning_effort",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Enables internal performance measure tracking in IS-DESPOT (default: %(default)s).
            
            ''')
    )
    debug_group.add_argument(
        '--record_pedestrian_path',
        dest="record_pedestrian_path",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Records the paths of all pedestrians of each simulated scene (default: %(default)s).
            
            ''')
    )
    debug_group.add_argument(
        '--record_car_intention_images',
        dest="record_car_intention_images",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Records the birdeye-view car intention image of each simulated scene step (default: %(default)s).
            
            ''')
    )
    debug_group.add_argument(
        '--verbose',
        dest="verbose",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Print debug output (default: %(default)s).
            WARNING: This will clutter the console massively!
            ''')
    )
    debug_group.add_argument(
        '--display',
        dest="display",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Renders the execution of each scene in the CARLA simulator (default: %(default)s).
            WARNING: This will influence execution time detrimentally and is only available locally!
            ''')
    )
    
    # print usage
    arg_parser.print_help()
    print("\n")
    # process inputs
    args = vars(arg_parser.parse_args())


    # ========================================
    #           META LEVEL ARGUMENTS
    # *affecting script execution as a whole*
    # ========================================  
    Config.AGENT = Agent(args["agent"])
    Config.MODE = Mode(args["mode"])
    Config.Carla.NUM_PEDESTRIANS = 1
    Config.PREDICT_PEDESTRIAN_PATH = args["predict_pedestrian_path"]
    Config.RISK_AWARE_PATH = args["plan_path_with_risk"]
    Config.SKIP_NO_PEDESTRIAN_STEPS = args["skip_no_pedestrian_steps"]
    Config.MAX_EPISODE_STEPS = args["max_episode_steps"]
    Config.SEED = args["seed"]

    if Config.SKIP_NO_PEDESTRIAN_STEPS:
        raise NotImplementedError("This breaks training for all learners.")

    if Config.RECORD_PEDESTRIAN_DATA and Config.PREDICT_PEDESTRIAN_PATH:
        raise ValueError("Can't record pedestrian path data while also predicting them.")
    
    if Config.RISK_AWARE_PATH and Config.AGENT is not Agent.HyLEAR:
        raise ValueError("Considering risk has only been designed for HyLEAR.")

    # determine scenarios constituting the benchmark
    if args["scenario"] == ["00"]:
        if Config.MODE is Mode.TRAIN:
            Config.TRAIN_SCENARIOS = ['01', '02', '03', '04', '05', '06', '07', '08', '09']
        elif Config.MODE is Mode.VAL:
            Config.VAL_SCENARIOS = ['01', '02', '03', '04', '05', '06', '07', '08', '09']
        elif Config.MODE is Mode.TEST:
            Config.TEST_SCENARIOS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', "11", "12"]
    else:
        if Config.MODE is Mode.TRAIN:
            Config.TRAIN_SCENARIOS = args["scenario"]
            Config.scenarios = args["scenario"]
        elif Config.MODE is Mode.VAL:
            Config.VAL_SCENARIOS = args["scenario"]
            Config.scenarios = args["scenario"]
        elif Config.MODE is Mode.TEST:
            Config.TEST_SCENARIOS = args["scenario"]
            Config.scenarios = args["scenario"]
            print("After setting the scenarios: ", Config.scenarios)
        oldConfig.scenarios = args["scenario"]
   

    # ========================================
    #    ARGUMENTS CONFLICTING WITH SLURM
    # ========================================  
    Config.Carla.REMOTE = args["remote"]
    Config.Carla.PORT = args["carla_port"]
    Config.Despot.PORT = args["despot_port"]
    Config.DISPLAY = args["display"]

    if Config.Carla.REMOTE and (Config.Carla.PORT is not None or Config.Despot.PORT is not None):
        raise ValueError("Ports for CARLA server and IS-DESPOT sub-processes are determined automatically on SLURM.")

    if Config.Carla.REMOTE and Config.DISPLAY: 
        raise ValueError("Can't render CARLA simulation on SLURM cluster.")  
    
    if not Config.Carla.REMOTE and Config.Carla.PORT is None:
        raise ValueError("Local script execution requires specifying a TCP port for the CARLA simulation server.")


    # ========================================
    #         ARGUMENTS FOR RESUMING 
    # ========================================    
    Config.EPOCHS = args["epochs"]
    Config.INITIAL_EPISODE = args["resume_from_episode"]
    Config.RESUME = True if Config.INITIAL_EPISODE != 1 else False 

    if not Config.MODE is Mode.TRAIN and Config.EPOCHS != 1:
        raise ValueError("Iterating multiple times over the test set doesn't make sense.")


    # ========================================
    #         ARGUMENTS FOR LOGGING 
    # ========================================    
    Config.VERBOSE = args["verbose"]
    Config.OUTPUT_DIR = args["output_directory"]
    Config.MODEL_CHECKPOINT = args["load_checkpoint"]
    Config.RECORD_PEDESTRIAN_DATA = args["record_pedestrian_path"]
    Config.RECORD_CAR_INTENTION_IMAGES = args["record_car_intention_images"]
    Config.Despot.TRACKING = args["track_planning_effort"]

    if Config.RECORD_PEDESTRIAN_DATA and Config.MODE is not Mode.TRAIN:
        raise ValueError("Can't record pedestrian path data of test or validation set as this leaks information.")
    
    if Config.AGENT in [Agent.IS_DESPOT, Agent.HYP_DESPOT] and Config.MODE is Mode.TRAIN and not Config.RECORD_PEDESTRIAN_DATA:
        raise ValueError("Planner can't be trained (this only makes sense if you want to record pedestrian path data).")


    #====================================================================
    #                 ARGUMENTS MODIFYING AGENT BEHAVIOUR
    #====================================================================
    # IS-DESPOT
    if Config.Despot.PORT is None: Config.Despot.PORT = find_free_port()
    Config.Despot.AGGRESSIVE_BELIEF_UPDATES = args["aggressive_belief_updates"]
    Config.Despot.MINIMAL_NOISE = args["minimal_noise"]
    Config.Despot.NO_IMPORTANCE_SAMPLING = args["no_importance_sampling"]
    Config.Despot.NO_NORMALIZATION = args["no_normalization"]
    Config.Despot.TIMEOUT = args["timeout"]
    Config.Despot.TIME_PER_PLANNING_STEP = args["time_per_planning_step"]
    Config.Despot.NOISE = args["noise"]
    Config.Despot.MAX_SEARCH_DEPTH = args["max_search_depth"]
    Config.Despot.DISCOUNT = args["discount_factor"]
    Config.Despot.PARTICLE_NUMBER = args["particle_number"]
    Config.Despot.GAP = args["gap_reduction_rate"]
    Config.Despot.MAX_POLICY_SIM_LEN = args["max_policy_simulation_length"]
    Config.Despot.PRUNING_CONSTANT = args["pruning_constant"]
    Config.Despot.PARTICLE_NUMBER = args["particle_number"]

    if Config.AGENT is Agent.HYP_DESPOT: 
        Config.Despot.VARIANT = DespotVariant.HYP_DESPOT
        Config.Despot.CPU_MULTITHREADING = args["CPU"]
        Config.Despot.NUM_THREADS = args["num_threads"]
    elif Config.AGENT is Agent.IS_DESPOT or Config.AGENT in HYBRID_AGENTS: 
        Config.Despot.VARIANT = DespotVariant.IS_DESPOT
    
    # HyLEAP
    Config.HyLEAP.HACKY = args["hacky_hyleap"]
    Config.HyLEAP.DECOUPLE = args["decouple_hyleap"]
    Config.HyLEAP.DROPOUT = args["use_dropout"]
    Config.HyLEAP.HIDDEN_LAYER_SIZE = args["hidden_layer_size"]

    if Config.HyLEAP.HACKY and Config.AGENT not in [Agent.HyLEAP, Agent.HyPLAN]:
        raise ValueError("Florian's 'hacky' HyLEAP variant is only available for HyLEAP and HyPLAN.")

    if Config.HyLEAP.DECOUPLE and not Config.AGENT in [Agent.HyLEAP, Agent.HyPLAN]:
        raise ValueError("Decoupling HyLEAP during training is only available for HyLEAP and HyPLAN.")

    if Config.HyLEAP.DECOUPLE and not Config.MODE is Mode.TRAIN: 
        raise ValueError("Can't decouple HyLEAP architecture during testing.")
    
    if Config.HyLEAP.DECOUPLE and Config.HyLEAP.HACKY:
        raise ValueError("Florian's 'hacky' HyLEAP variant is already a partial decoupling.")
    
    # HyPLAN
    Config.HyPLAN.NUM_FORWARD_PASSES = args["hyplan_num_forward_passes"]
    Config.HyPLAN.TRAIN_INTEGRATION = args["hyplan_train_integration"]
    Config.HyPLAN.IMPROVE_REWARD = args["improve_reward"]
    Config.HyPLAN.NavA2C = args["nava2c"]

    if Config.HyPLAN.NUM_FORWARD_PASSES is not None and Config.AGENT is not Agent.HyPLAN:
        raise ValueError(message="Altering the number of forward passes is only available for HyPLAN.")
    
    if Config.HyPLAN.NUM_FORWARD_PASSES is None and Config.AGENT is Agent.HyPLAN:
        raise ValueError("The number of forward passes has to be specified for HyPLAN.")
    
    if Config.HyPLAN.TRAIN_INTEGRATION and Config.AGENT is not Agent.HyPLAN:
        raise ValueError("Activating vertical pruning during training is only available for HyPLAN.")

    if Config.HyPLAN.TRAIN_INTEGRATION and Config.MODE is not Mode.TRAIN:
        raise ValueError("HyPLAN train modification arguments given although script mode is not train.")

    if Config.HyPLAN.IMPROVE_REWARD and Config.AGENT not in [Agent.HyLEAP, Agent.HyPLAN, Agent.NavA2C]:
        raise ValueError("Clipping & standardizing rewards is only available for HyLEAP, HyPLAN or NavA2C.")

    if Config.HyPLAN.IMPROVE_REWARD and Config.MODE is not Mode.TRAIN:
        raise ValueError("Clipping & standardizing rewards is only available during training.")
      
    if Config.HyPLAN.NavA2C and Config.AGENT not in [Agent.HyLEAP, Agent.HyPLAN, Agent.NavA2C]:
        raise ValueError("Swapping NN architecture is only available for HyLEAP, HyPLAN or NavA2C.")
        
    # LEADER
    Config.Leader.MAX_PEDESTRIANS = 1


    # =============
    # setup logging
    # =============  
    initialize_logging()
    log_debug(f"program executed with following arguments: {args}")


    # =================================================
    # START CARLA SERVER OR CONNECT TO RUNNING INSTANCE
    # =================================================
    # only start carla server when on slurm cluster
    if Config.Carla.REMOTE:
        log_info("running CARLA server on SLURM cluster")
        Config.Carla.HOST = Address.REMOTE
        # communication with process probing for CARLA server port
        carla_server_queue = Queue()
        # until server is actually running (often fails due to port already being used)
        while True:
            # start server in separate process
            Process(target=run_carla_server, args=(carla_server_queue,)).start()
            # default message publishing current Carla server port
            carla_server_port = carla_server_queue.get(True, None)
            try:
                # second optional message sent when Carla server crashes
                carla_server_port = carla_server_queue.get(True, 100)
                continue
            except:
                # if we haven't heard from the process after 100s we assume that the Carla server is running
                Config.Carla.PORT = carla_server_port
                log_info(f"CARLA server successfuly started at {Config.Carla.HOST}:{Config.Carla.PORT}")
                break
    else:
        Config.Carla.HOST = Address.LOCAL
        log_info("executing script on local machine: connecting to existing CARLA server instance")
    

    # ==============================================================
    #                   PYTHON CARLA INTERFACE
    # ==============================================================
    if Config.MODE is Mode.TRAIN:
        benchmark = GIDASBenchmark(port=Config.Carla.PORT, mode="TRAINING")
    elif Config.MODE is Mode.VAL:
        benchmark = GIDASBenchmark(port=Config.Carla.PORT, mode="VALIDATION")
    elif Config.MODE is Mode.TEST:
        benchmark = GIDASBenchmark(port=Config.Carla.PORT, mode="TESTING")

    # print('After creating: ', Config.scenarios)


    # ==============================================================
    #                       CREATE AGENT
    # ==============================================================
    if Config.AGENT is Agent.IS_DESPOT:
        agent = ISDespotP(benchmark.client, benchmark.world, benchmark.map, benchmark.scenario)
    elif Config.AGENT is Agent.HYP_DESPOT:
        agent = ISDespotP(benchmark.client, benchmark.world, benchmark.map, benchmark.scenario)
    elif Config.AGENT is Agent.NavA2C:
        agent = NavA2C(benchmark.client, benchmark.world, benchmark.map, benchmark.scenario)
    elif Config.AGENT is Agent.HyLEAP:
        agent = HyLEAP(benchmark.client, benchmark.world, benchmark.map, benchmark.scenario)
    elif Config.AGENT is Agent.HyLEAR:
        agent = HyLEAR_NavA2C(benchmark.client, benchmark.world, benchmark.map, benchmark.scene)
    elif Config.AGENT is Agent.HyLEAR_C:
        agent = HyLEAR_NavA2C_Cognitive(benchmark.client, benchmark.world, benchmark.map, benchmark.scene)
    elif Config.AGENT is Agent.HyLEAR_AUTOBOTS:
        agent = HyLEAR_NavA2C_AutoBots(benchmark.client, benchmark.world, benchmark.map, benchmark.scene)
    elif Config.AGENT is Agent.HyLEAR_AUTOBOTS_C:
        agent = HyLEAR_NavA2C_AutoBots_Cognitive(benchmark.client, benchmark.world, benchmark.map, benchmark.scene)

    elif Config.AGENT is Agent.HyPLAN:
        agent = HyPLAN(benchmark.client, benchmark.world, benchmark.map, benchmark.scenario)
    elif Config.AGENT is Agent.LEADER:
        agent = LEADER(benchmark.client, benchmark.world, benchmark.map, benchmark.scenario)

    benchmark.assign_agent(agent)




    # ===========================================================
    #                       PREPARE BENCHMARK
    # ===========================================================
    if Config.MODE is Mode.TRAIN:
        benchmark.prepare_train_episodes()
    if Config.MODE is Mode.VAL:
        benchmark.prepare_validation_episodes()
    elif Config.MODE is Mode.TEST:
        benchmark.prepare_test_episodes()


    # =======================
    # EXECUTE SIMULATION LOOP
    # =======================
    # determine from which episode to start (this happens when a job was interrupted due to reaching its time limit)
    episode_counter = Config.INITIAL_EPISODE
    log_info(f"{Config.MODE} on a total number of {benchmark.number_of_episode} episodes")
    
    # metrics that span all episodes
    goal_episodes = []
    ttg_episodes = []
    nearmiss_episodes = []
    crash_episodes = []
    rewards_episode = []
    execution_times_episodes = []
    steps_episodes = []
    skipped_steps_episodes = 0
    terminated_early_episodes = 0
    while episode_counter <= benchmark.number_of_episode:

        # metrics that are episode specific
        performance_log = {}
        episode_rewards = []
        episode_pedestrian_trajectory = []
        episode_actions = []
        episode_execution_times = []
        episode_pedestrian_visibility = []
        episode_car_trajectory = []
        episode_car_speeds = []
        episode_risks = []
        nearmiss = False

        agent.initialize_episode(episode_counter)
        # set step counter
        step_counter = 1
        # get the scenario id, parameters and instantiate the world
        benchmark.reset(step_counter)
        # initial car position
        car_start_position = benchmark.client_world.player.get_location()
        pedestrian_start_position = benchmark.client_world.walker.get_location()
        # episode simulation loop (step-wise incrementation)
        while True:
            if step_counter > 500: break
            # check whether episode has come to an irregular end
            terminated_early = has_agent_stopped(episode_car_speeds, past_steps=Config.MAX_EPISODE_STEPS)
            # terminated_early flag is used in finalize_episode() for proper irregular clean-up
            if terminated_early:
                terminated_early_episodes += 1
                break

            # render scene simulation in CARLA
            #if Config.DISPLAY: benchmark.render()

            # track pedestrian position (for training pedestrian path predictor if enabled)
            episode_pedestrian_trajectory.append((benchmark.client_world.walker.get_location().x, benchmark.client_world.walker.get_location().y))
            #log_info(f"pedestrian position: {benchmark.client_world.walker.get_location()}")

            # start of episode 
            start_time = time.perf_counter()
            # ACTUALLY PERFORM THE SIMULATION STEP
            step_summary = benchmark.step(step_counter)
            time_taken = (time.perf_counter() - start_time)

            # did we skip this step because of buggy path planning?
            # do not remember step if so
            if step_summary["skipped_step"]: 
                skipped_steps_episodes += 1
                continue

            # track pedestrian visiblity
            episode_pedestrian_visibility.append(1 if step_summary['ped_observable'] else 0)
            # track step execution time
            episode_execution_times.append(round(time_taken*1000, 4))
            execution_times_episodes.append(round(time_taken*1000, 4))
            # track step reward
            episode_rewards.append(step_summary["reward"])
            # track speed
            speed = float(np.sqrt(step_summary['velocity.x'] ** 2 + step_summary['velocity.y'] ** 2)) * 3.6
            episode_car_speeds.append(round(speed, 4))
            # track risk (if applicable)
            if Config.RISK_AWARE_PATH: episode_risks.append(step_summary['risk'])
            # track agent actions
            if benchmark.control.throttle > 0.0:
                action = Action.ACCELERATE
            elif benchmark.control.brake > 0.0:
                action = Action.DECELERATE
            else: action = Action.MAINTAIN
            episode_actions.append(action.value)
            agent_action = step_summary["action"]
            # if agent_action is not action:
            #     raise ValueError(
            #         f"Incongruent environment action: {action} and agent action: {agent_action}"
            #     )

            # at most one nearmiss per episode
            # preserve nearmiss until end of episode
            nearmiss = nearmiss or (step_summary['nearmiss'] and speed > 0.0)
     
            log_info(
                f"STEP {step_counter}: execution time {round(time_taken*1000, 4)}, "
                f"car speed {speed:.4f}km/h, " 
                f"action taken {'ACC' if int(action.value) == 2 else 'MAIN' if int(action.value) == 1 else 'DEC'}, "
                #f"reward received {step_summary['reward']} "
                f"pedestrian distance travelled: {l2_distance(pedestrian_start_position, benchmark.client_world.walker.get_location()):.4f}, "
                f"terminal {step_summary['terminal']}"
            )

            
            pedestrian_start_position = benchmark.client_world.walker.get_location()
            if step_summary["terminal"]: break
            else: step_counter += 1
            # @@@ end step-loop @@@ #

        agent.finalize_episode(episode_counter, terminated_early)
        # do not consider episode if it was terminated early    
        if terminated_early: 
            episode_counter += 1
            continue

        # update running averages across episodes
        steps_episodes.append(step_counter)
        goal_episodes.append(1 if step_summary['goal'] else 0)
        nearmiss_episodes.append(1 if nearmiss else 0)
        crash_episodes.append(1 if step_summary['collision'] else 0)
        rewards_episode.append(sum(episode_rewards))
        if step_summary['goal']: ttg_episodes.append(step_counter * Config.Carla.SIMULATION_STEP)

        # show running averages during execution on console
        console_log = {
            "episode": episode_counter,
            "terminated_early_episodes": terminated_early_episodes,
            "running_steps_avg": round(np.mean(steps_episodes)),
            "skipped_steps": skipped_steps_episodes,
            "running_crash_avg": round(np.mean(crash_episodes), 4),
            "runing_nearmiss_avg": round(np.mean(nearmiss_episodes), 4),
            "running_goal_avg": round(np.mean(goal_episodes), 4),
            "running_ttg_avg": round(np.mean(ttg_episodes), 4),
            "running_reward_avg": round(np.mean(rewards_episode), 4),
            "running_execution_time_avg": round(np.mean(execution_times_episodes), 4)
        }
        log_info(json.dumps(console_log, indent=2))
        print(json.dumps(console_log, indent=2))

        # calculate averages for this particular episode
        # travelled car distance this episode
        travelled_distance = round(l2_distance(car_start_position, benchmark.client_world.player.get_location()), 2)
        avg_ped_obs = round(np.mean(episode_pedestrian_visibility), 2)
        avg_car_speed = round(np.mean(episode_car_speeds), 2)

        episode_action_distribution = {}
        for key, val in Counter(episode_actions).items():
            if key == 0: verbose_key = "decelerate"
            elif key == 1: verbose_key = "maintain"
            else: verbose_key = "accelerate"
            percentage = round((val/len(episode_actions))*100, 2)
            episode_action_distribution.update({verbose_key:percentage})

        # additional debug information for each episode
        console_log = {
            "episode_pedestrian_visibility_avg": avg_ped_obs,
            "episode_travelled_distance": travelled_distance,
            "episode_car_speed_avg": avg_car_speed,
            "episode_action_distribution": episode_action_distribution,
        }
        if terminated_early: log_info(json.dumps(console_log, indent=2))

        # evaluate episode statistics (crash rate, nearmiss rate, time to goal, smoothness, execution time, violations)
        performance_log["episode"] = episode_counter
        performance_log['scenario'] = step_summary['scenario']
        performance_log['ped_distance'] = step_summary['ped_distance']
        performance_log['ped_speed'] = step_summary['ped_speed']
        performance_log['crash'] = 1 if step_summary['collision'] else 0
        performance_log['nearmiss'] = 1 if nearmiss else 0
        performance_log['goal'] = 1 if step_summary['goal'] else 0
        performance_log['ttg'] = round(step_counter * Config.Carla.SIMULATION_STEP, 4) if step_summary['goal'] else "nan"
        performance_log['total_episode_reward'] = sum(episode_rewards)
        performance_log['travelled_distance'] = travelled_distance
        performance_log['execution_times'] = episode_execution_times
        performance_log['is_ped_observable'] = episode_pedestrian_visibility
        performance_log['car_speeds'] = episode_car_speeds
        performance_log['actions'] = episode_actions
        performance_log["terminated_early"] = terminated_early
        if Config.RISK_AWARE_PATH: performance_log['risk'] = episode_risks

        # log classical performance metrics
        log_performance_metrics(performance_log)

        # log pedestrian data
        log_data({"pedestrian_trajectory": episode_pedestrian_trajectory})

        episode_counter += 1
        # @@@ end episode-loop @@@ #

    benchmark.close()
    Config.TERMINATE = True
    kill_all_processes(kill_parent=False)


