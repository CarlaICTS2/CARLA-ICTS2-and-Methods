from enum import Enum
from datetime import datetime


class StringEnum(Enum):
    def __str__(self):
        return str(self.value)


class Address(StringEnum):
    LOCAL = "172.31.144.1"
    REMOTE = "127.0.0.1"


class Agent(StringEnum):
    IS_DESPOT = "IS-DESPOT"
    HYP_DESPOT = "HyP-DESPOT"
    NavA2C = "NavA2C"
    HyLEAP = "HyLEAP"
    HyLEAR = "HyLEAR"
    HyPLAN = "HyPLAN"
    LEADER = "LEADER"
    HyLEAR_C = "HyLEAR-COGNITIVE"
    HyLEAR_AUTOBOTS = "HyLEAR-AUTOBOTS"
    HyLEAR_AUTOBOTS_C = "HyLEAR-AUTOBOTS-COGNITIVE"


class DespotVariant(StringEnum):
    IS_DESPOT = Agent.IS_DESPOT
    HYP_DESPOT = Agent.HYP_DESPOT


HYBRID_AGENTS = [Agent.HyLEAP, Agent.HyLEAR, Agent.HyPLAN, Agent.LEADER, Agent.HyLEAR_C, Agent.HyLEAR_AUTOBOTS,
                 Agent.HyLEAR_AUTOBOTS_C]


class Mode(StringEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class Action(Enum):
    DECELERATE: int = 0
    MAINTAIN: int = 1
    ACCELERATE: int = 2


class Config:
    class Carla:
        NUM_PEDESTRIANS: int = 1
        NUM_ACTIONS: int = 3
        HOST: Address = None
        PORT: int = None
        REMOTE: bool = None

        SIMULATION_STEP = 0.05  # each scene simulation step is 50 milliseconds
        SENSOR_SIMULATION_STEP = '0.5'
        SYNCHRONONOUS = True  # wait for client to tick the server before initiating new simulation step

        SEGCAM_FOV = '90'
        SEGCAM_IMAGE_WIDTH = '400'
        SEGCAM_IMAGE_HEIGHT = '400'

        GOAL_TOLERANCE = 3.0
        GRID_SIZE = 2  # grid size in meters
        MAX_SPEED = ((50 / 60) / 60) * 1000  # 50 km/h = (50/60)/60 = 50/3600 m/s
        MAX_STEERING_ANGLE = 1.22173  # 70 degrees in radians
        OCCUPANCY_GRID_WIDTH = '1920'
        OCCUPANCY_GRID_HEIGHT = '1080'
        LOCATION_THRESHOLD = 1.0
        WIDTH = 1280
        HEIGHT = 720

        FILTER = 'vehicle.audi.tt'
        AGENT_VEHICLE_ROLENAME = 'agent_vehicle'
        PARKED_VEHICLE_ROLENAME = 'parked_vehicle'
        INCOMING_VEHICLE_ROLENAME = 'incoming_vehicle'

        EGO_VEHICLE_LENGTH = 4.182
        EGO_VEHICLE_WIDTH = 1.994

        NEARMISS_FRONT_MARGIN = 1.5
        NEARMISS_SIDE_MARGIN = NEARMISS_BACK_MARGIN = 0.5

        EXO_PEDESTRIAN_LENGTH = 0.375
        EXO_PEDESTRIAN_WIDTH = 0.375

        GAMMA = 1.7
        HIT_PENALTY = 1000
        GOAL_REWARD = 1000
        BRAKING_PENALTY = 1
        TIME_PER_STEP = 1  # length of each step used in gif creation
        OBSERVATION_SIZE = 8  # Car: (x, y, angle, speed), Pedestrian: (x, y), reward, previous action

    class Despot:
        VARIANT: DespotVariant = None
        PORT: int = 1245
        TRACKING: bool = None

        AGGRESSIVE_BELIEF_UPDATES: bool = None
        MINIMAL_NOISE: bool = None
        NO_IMPORTANCE_SAMPLING: bool = None
        NO_NORMALIZATION: bool = None
        TIMEOUT: float = None
        TIME_PER_PLANNING_STEP: float = None
        NOISE: float = None
        MAX_SEARCH_DEPTH: int = None
        DISCOUNT: float = None
        PARTICLE_NUMBER: int = None
        GAP: float = None
        MAX_POLICY_SIM_LEN: int = None
        PRUNING_CONSTANT: float = None

        CPU_MULTITHREADING: bool = None
        NUM_THREADS: int = None

    class NavA2C:
        hidden_dim = 256
        exp_win_discount = 0.999
        avg_threshold = 0.01
        std_threshold = 0.05
        entropy_discount_factor = 0.9

        # A2C training parameters
        a2c_lr = 0.0005
        a2c_lr_initial = 1.0e-4
        a2c_lr_final = 5.0e-5
        a2c_gamma = 0.99
        a2c_gae_lambda = 1.0
        a2c_entropy_coef = 0.0
        start_entropy_coeff = 0.0025
        end_entropy_coeff = 0.00001

    class Leader:
        REPLAY_MIN: int = 5
        REPLAY_MAX: int = 100000
        BATCH_SIZE: int = 2
        MEMORY_SIZE: int = 1024
        MAX_FEATURE_LEN: int = 181
        NUM_FEATURES: int = 3
        MAX_PEDESTRIANS: int = 1
        MAX_TRAJECTORY_LENGTH: int = None
        LEARNING_RATE: int = 1e-4
        DEVICE: str = "cuda"
        HANDCRAFTED_ATT: bool = False
        ATTENTION_SIZE: int = 181
        CRITIC_WARM_UP_ITERATIONS = 1000
        OBSERVATION_SIZE = 6  # Car: (x, y, angle, speed), Pedestrian: (x, y)

    class HyLEAP:
        CAR_INTENTION_IMAGE_WIDTH = 310
        CAR_INTENTION_IMAGE_HEIGHT = 110
        GAMMA = 0.99  # discount factor
        HIDDEN_LAYER_SIZE = 256  # size of final convolutional layer before splitting into Advantage and Value streams
        LSTM_STATE_SIZE = 2 * 128
        LEARNING_RATE = 1e-4
        DECAY = 0.99
        MOMENTUM = 0.0
        EPSILON = 0.1
        L2_DECAY = 0.0005
        HACKY: bool = None
        DECOUPLE: bool = None
        DEVICE: str = "cuda"
        DROPOUT: bool = None

    class HyLEAR:
        DEVICE = "cuda"
        num_steps = 3000000
        start_steps = 25000
        num_eval_steps = 3000
        update_interval = 4
        target_update_interval = 3000
        log_interval = 10
        eval_interval = 10000
        batch_size = 128
        lr = 0.00005
        buffer_capacity = 60000
        gamma = 0.99
        multi_step = 1
        target_entropy_ratio = 0.6
        use_per = True
        dueling_net = True

    class HyPLAN:
        # discount factor
        GAMMA = 0.99
        # size of final convolutional layer before splitting into Advantage and Value streams
        HIDDEN_LAYER_SIZE = 256  #
        MEMORY_SIZE = 2 * 128
        OBSERVATION_SIZE = 6
        LEARNING_RATE = 1e-4
        DECAY = 0.99
        MOMENTUM = 0.0
        EPSILON = 0.1
        L2_DECAY = 0.0005
        IMPROVE_REWARD: bool = None
        NUM_FORWARD_PASSES: bool = None
        TRAIN_INTEGRATION: bool = None
        NavA2C: bool = None
        DEVICE: str = "cuda"

    # kill script
    TERMINATE: bool = False
    PI = 3.14159
    SEED: int = None

    # ===============================================
    #             META CONFIGURATION
    #     (affects script behaviour as a whole)
    # ===============================================
    AGENT: Agent = Agent.HyLEAR
    PREDICT_PEDESTRIAN_PATH: bool = None
    RISK_AWARE_PATH: bool = None
    SKIP_NO_PEDESTRIAN_STEPS: bool = None
    DISPLAY = True

    # ===============================================
    #             LOGGING CONFIGURATION
    # ===============================================
    VERBOSE: bool = None

    # logging output directories
    OUTPUT_DIR: str = None
    METRICS_DIR: str = None
    MODEL_DIR: str = None
    DEBUG_DIR: str = None
    DATA_DIR: str = None

    MODEL_SAVE_FREQUENCY_EPISODES: int = 250
    MODEL_CHECKPOINT: str = None
    RECORD_PEDESTRIAN_DATA: bool = None
    RECORD_CAR_INTENTION_IMAGES: bool = None
    GLOBAL_START_TIME: str = datetime.now().strftime('%d.%m.%Y_%H.%M.%S')

    # ===============================================
    #   SCENARIOS & EPISODES & STEPS CONFIGURATION
    # ===============================================
    MODE: Mode = Mode.TRAIN
    RESUME: bool = False
    INITIAL_EPISODE: int = None
    EPOCHS: int = None
    MAX_EPISODE_STEPS: int = None

    # bnechmark specifications
    TRAIN_SCENARIOS = []
    TRAIN_PED_SPEED_RANGE = [0.6, 2.0]  # m/s
    TRAIN_PED_DIST_RANGE = [0, 40]

    VAL_SCENARIOS = None
    VAL_PED_SPEED_RANGE = [[0.2, 0.5], [2.1, 2.8]]  # m/s
    VAL_PED_DIST_RANGE = [4.25, 49.25]

    TEST_SCENARIOS = None
    TEST_PED_SPEED_RANGE = [0.25, 2.85]  # m/s
    TEST_PED_DIST_RANGE = [4.75, 49.75]
    TEST_CAR_SPEED_RANGE = [10, 20]

