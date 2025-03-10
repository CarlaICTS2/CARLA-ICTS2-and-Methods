
import numpy as np
## Needed to put it here to do some weired circular import problem
## Same class in ped_controller.py
class ControllerConfig():
    def __init__(self, ped_speed=1.0, ped_distance=30.0):
        self.ped_speed = ped_speed
        self.ped_distance = ped_distance
        # Has to be initialized due weired initial call
        self.spawning_distance = 0
        self.walking_distance = 0
        self.looking_distance = 0
        self.crossing_distance = 0
        self.reenter_distance = 0
        self.op_reenter_distance = 0
        self.char = "yielding"


class ScenarioConfig():
    
    def __init__(self):
        rng = np.random.default_rng(seed=42)
        self.scenes = rng.permutation(self.get_scenes())
        self.n_scenes = len(self.scenes)
        #self.split = [0.18,0.35,0.55] # 0.2, 0.2, 0.6
        self.split = [0.5,0.6,1.0]
    def get_scenes(self):
        return []
    
    def get_training(self):
        lower = 0
        upper = int(self.split[0] * self.n_scenes)
        return self.scenes[lower:upper]
    
    def get_validation(self):
        lower = int(self.split[0] * self.n_scenes)
        upper = int(self.split[1] * self.n_scenes)
        return self.scenes[lower:upper]
    
    def get_test(self):
        lower = int(self.split[1] * self.n_scenes)
        upper = int(self.split[2] * self.n_scenes)
        return self.scenes[lower:upper]    


class IConfig01(ScenarioConfig):
    def __init__(self):
        #self.ped_speed_range = [1.2,1.6]
        #self.spwaning_distances = [25,35]
        self.ped_speed_range = [1.6,2.0] # 1.8
        self.spwaning_distances = [47.5,55] # 28
        self.walking_distances = [5,8] # 5
        self.looking_distances = [0.85,0.85] # 0.85
        self.crossing_distances = [3,5] # 3
        self.reenter_distances = [3,4] #8
        self.op_reenter_distances = [5,5] #5
        self.character = ["forcing", "yielding"]#, "yielding"]
        #self.character = ["yielding", "forcing"]
        super(IConfig01,self).__init__()

    def get_scenes(self):
        scenes = []
        for speed in np.arange(self.ped_speed_range[0], self.ped_speed_range[1]+0.1,0.1):
            for spawning_distance in np.arange(self.spwaning_distances[0], self.spwaning_distances[1]+2.5,2.5):
                for walking_distance in np.arange(self.walking_distances[0], self.walking_distances[1]+1,1):
                    for looking_distance in np.arange(self.looking_distances[0], self.looking_distances[1]+0.01,0.01):
                        for crossing_distance in np.arange(self.crossing_distances[0], self.crossing_distances[1]+1,1):
                            for reenter_distance in np.arange(self.reenter_distances[0], self.reenter_distances[1]+1,1):
                                #for op_reenter_distance in np.arange(self.op_reenter_distances[0], self.op_reenter_distances[1]+1,1):
                                    for char in self.character:
                                        conf = ControllerConfig(speed)
                                        conf.spawning_distance = int(spawning_distance)
                                        conf.walking_distance = walking_distance
                                        conf.looking_distance = looking_distance
                                        conf.crossing_distance = crossing_distance
                                        conf.reenter_distance = crossing_distance + reenter_distance
                                        conf.op_reenter_distance = crossing_distance
                                        conf.char = char
                                        scenes.append(("01_int", conf))
        print('Total scenes count (int-1): ', len(scenes))
        return scenes
    


class IConfig02(ScenarioConfig):
    def __init__(self):
        self.ped_speed_range = [1.6,2.0] #1.5
        self.spwaning_distances = [52.5,60] # 33.5
        self.walking_distances = [5,8] #5
        self.looking_distances = [0.85,0.86] #0.85
        self.crossing_distances = [1,3] #1
        self.op_reenter_distances = [3,4] #1
        self.character = ["forcing", "yielding"] #forcing
        #character = ["yielding", "forcing"]
        super(IConfig02,self).__init__()

    def get_scenes(self):
        scenes = []
        for speed in np.arange(self.ped_speed_range[0], self.ped_speed_range[1]+0.1,0.1):
            for spawning_distance in np.arange(self.spwaning_distances[0], self.spwaning_distances[1]+2.5,2.5):
                for walking_distance in np.arange(self.walking_distances[0], self.walking_distances[1]+1,1):
                    for looking_distance in np.arange(self.looking_distances[0], self.looking_distances[1]+0.1,0.1):
                        for crossing_distance in np.arange(self.crossing_distances[0], self.crossing_distances[1]+1,1):
                            for op_reenter_distance in np.arange(self.op_reenter_distances[0], self.op_reenter_distances[1]+1,1):
                                for char in self.character:
                                    conf = ControllerConfig(speed)
                                    conf.spawning_distance = int(spawning_distance)
                                    conf.walking_distance = walking_distance
                                    conf.looking_distance = looking_distance
                                    conf.crossing_distance = crossing_distance
                                    conf.op_reenter_distance = op_reenter_distance
                                    conf.char = char
                                    scenes.append(("02_int", conf))
        print('Total scenes count (int-2): ', len(scenes))
        return scenes


class IConfig03(ScenarioConfig):
    def __init__(self):
        self.ped_speed_range = [1.5,2.1] #2
        self.spwaning_distances = [45,59] #40
        self.looking_distances = [0.07,0.13]
        #self.character = ["yielding"]#, "yielding"]
        self.character = ["forcing", "yielding"]
        super(IConfig03,self).__init__()

    def get_scenes(self):
        scenes = []
        for speed in np.arange(self.ped_speed_range[0], self.ped_speed_range[1]+0.1,0.1):
            for spawning_distance in np.arange(self.spwaning_distances[0], self.spwaning_distances[1]+1,1):
                for looking_distance in np.arange(self.looking_distances[0], self.looking_distances[1]+0.01,0.01):
                    for char in self.character:
                        conf = ControllerConfig(speed)
                        conf.spawning_distance = int(spawning_distance)
                        conf.looking_distance = looking_distance
                        conf.char = char
                        scenes.append(("03_int", conf))
        print('Total scenes count (int-3): ', len(scenes))
        return scenes





class IConfig04(ScenarioConfig):
    def __init__(self):
        # self.ped_speed_range = [1.2,1.6]
        # self.spwaning_distances = [25,35]
        self.ped_speed_range = [1.6, 2.0]  # 1.8
        self.spawning_distances = [52.5, 60.0]  # 28
        self.walking_distances = [5, 8]  # 5

        self.walk_back_distances = [0.5, 1.5]
        self.crossing_distanceX = [5.0, 7.0]
        self.crossing_distanceY = [3.0, 3.0]

        self.looking_distance1 = [1.0, 1.0]  # 0.85
        self.looking_distance2 = [1.0, 1.0]

        self.character = ["forcing", "yielding"]  # , "yielding"]
        # self.character = ["yielding", "forcing"]
        super(IConfig04, self).__init__()

    def get_scenes(self):
        scenes = []
        for speed in np.arange(self.ped_speed_range[0], self.ped_speed_range[1]+0.1,0.1):
            for spawning_distance in np.arange(self.spawning_distances[0], self.spawning_distances[1]+2.5,2.5):
                for walking_distance in np.arange(self.walking_distances[0], self.walking_distances[1]+1, 1):
                    # for looking_distance1 in np.arange(self.looking_distance1[0], self.looking_distance1[1]+0.01, 0.01):
                    #     print("ld1")
                    #     print(looking_distance1)
                    #     for looking_distance2 in np.arange(self.looking_distance2[0], self.looking_distance2[1] + 0.01, 0.01):
                            for walk_back_distance in np.arange(self.walk_back_distances[0], self.walk_back_distances[1]+1, 0.5):
                                for crossing_distanceX in np.arange(self.crossing_distanceX[0], self.crossing_distanceX[1]+1,1):
                                    for crossing_distanceY in np.arange(self.crossing_distanceY[0], self.crossing_distanceY[1] + 1, 1):
                                        for char in self.character:
                                            conf = ControllerConfig(speed)
                                            conf.spawning_distance = int(spawning_distance)
                                            conf.walking_distance = int(walking_distance)
                                            conf.looking_distance1 = 1.5
                                            conf.looking_distance2 = 1.0
                                            conf.walk_back_distance = walk_back_distance
                                            conf.crossing_distanceX = crossing_distanceX
                                            conf.crossing_distanceY = crossing_distanceY
                                            conf.char = char
                                            scenes.append(("04_int", conf))
        print('Total scenes count (int-4): ', len(scenes))
        return scenes

class IConfig05(ScenarioConfig):
    def __init__(self):
        # self.ped_speed_range = [1.2,1.6]
        # self.spwaning_distances = [25,35]
        self.ped_speed_range = [0.6, 0.9]  # 1.8
        self.spwaning_distances = [52.5, 60.0]  # 28

        self.walking_distances_X = [1, 2]
        self.walking_distances_Y = [1, 4]
        self.crossing_distances = [2, 3]

        self.uncertain_steps = [2, 3]

        #
        # self.walking_distances = [5, 5]  # 5
        # self.looking_distances = [0.85, 0.85]  # 0.85
        # self.reenter_distances = [8, 8]  # 8
        # self.op_reenter_distances = [5, 5]  # 5
        self.character = ["forcing", "yielding"]  # , "yielding"]
        # self.character = ["yielding", "forcing"]
        super(IConfig05, self).__init__()

    def get_scenes(self):
        scenes = []
        for speed in np.arange(self.ped_speed_range[0], self.ped_speed_range[1]+0.1,0.1):
            for spawning_distance in np.arange(self.spwaning_distances[0], self.spwaning_distances[1]+2.5,2.5):
                for walking_distance_X in np.arange(self.walking_distances_X[0], self.walking_distances_X[1]+1,0.5):
                    for walking_distance_Y in np.arange(self.walking_distances_Y[0], self.walking_distances_Y[1]+1,1):
                        for crossing_distance in np.arange(self.crossing_distances[0], self.crossing_distances[1]+1,1):
                            for uncertain_steps in np.arange(self.uncertain_steps[0], self.uncertain_steps[1]+1,1):
                                for char in self.character:
                                    conf = ControllerConfig(speed)
                                    conf.spawning_distance = int(spawning_distance)
                                    conf.walking_distance_X = walking_distance_X
                                    conf.walking_distance_Y = walking_distance_Y
                                    conf.crossing_distance = crossing_distance
                                    conf.uncertain_steps = uncertain_steps
                                    conf.char = char
                                    scenes.append(("05_int", conf))
        print('Total scenes count (int-5): ', len(scenes))
        return scenes

class IConfig06(ScenarioConfig):
    def __init__(self):
        # self.ped_speed_range = [1.2,1.6]
        # self.spwaning_distances = [25,35]
        self.ped_speed_range = [0.7, 1.0]  # 1.8
        self.spwaning_distances = [47.5, 55.0]  # 28
        self.crossing_distances = [3, 5]  # 3
        self.car_avoid_X = [0.4, 1]
        self.car_avoid_Y = [7, 8.5]

        self.character = ["forcing", "yielding"]  # , "yielding"]
        # self.character = ["yielding"]  # , "yielding"]
        super(IConfig06, self).__init__()

    def get_scenes(self):
        scenes = []
        for speed in np.arange(self.ped_speed_range[0], self.ped_speed_range[1]+0.1,0.1):
            for spawning_distance in np.arange(self.spwaning_distances[0], self.spwaning_distances[1]+2.5,2.5):
                for car_avoid_X in np.arange(self.car_avoid_X[0], self.car_avoid_X[1]+0.2,0.2):
                    for car_avoid_Y in np.arange(self.car_avoid_Y[0], self.car_avoid_Y[1]+0.5,0.5):
                        for crossing_distance in np.arange(self.crossing_distances[0], self.crossing_distances[1]+1,1):
                            for char in self.character:
                                conf = ControllerConfig(speed)
                                conf.spawning_distance = int(spawning_distance)
                                conf.crossing_distance = crossing_distance
                                conf.car_avoid_X = car_avoid_X
                                conf.car_avoid_Y = car_avoid_Y
                                conf.char = char
                                scenes.append(("06_int", conf))
        print('Total scenes count (int-6): ', len(scenes))
        return scenes


class Config01(ScenarioConfig):
    def __init__(self):
        self.ped_speed_range = [1.6,2.0]
        self.spwaning_distances = [72.5,85]
        self.walking_distances = [8,12]
        self.looking_distances = [0.85,0.89]
        self.op_reenter_distances = [9.5,12]
        self.cross_walk_delta = []
        self.character = ["forcing"]
        #character = ["yielding", "forcing"]
        super(Config01,self).__init__()

    def get_scenes(self):
        scenes = []
        for speed in np.arange(self.ped_speed_range[0], self.ped_speed_range[1]+0.1,0.1):
            for spawning_distance in np.arange(self.spwaning_distances[0], self.spwaning_distances[1]+2.5,2.5):
                for looking_distance in np.arange(self.looking_distances[0], self.looking_distances[1]+0.01,0.01):
                    for op_reenter_distance in np.arange(self.op_reenter_distances[0], self.op_reenter_distances[1]+2.5,2.5):
                        for walking_distance in np.arange(self.walking_distances[0], self.walking_distances[1]+1,1):
                            for char in self.character:
                                conf = ControllerConfig(speed)
                                conf.spawning_distance = int(spawning_distance)
                                conf.looking_distance = looking_distance
                                conf.op_reenter_distance = op_reenter_distance
                                conf.walking_distance = walking_distance
                                conf.char = char
                                scenes.append(("01_non_int", conf))
        print('Total scenes count (non-int-1): ', len(scenes))
        return scenes

class Config02(ScenarioConfig):
    def __init__(self):
        self.ped_speed_range = [1.6,2.0] #6
        self.spwaning_distances = [82.5,95] #5
        self.walking_distances = [8,12] #4
        self.looking_distances = [0.85,0.89] #6
        self.op_reenter_distances = [9.5,12] #0
        self.cross_walk_delta = []
        self.character = ["forcing"]
        #character = ["yielding", "forcing"]
        super(Config02,self).__init__()

    def get_scenes(self):
        scenes = []
        for speed in np.arange(self.ped_speed_range[0], self.ped_speed_range[1]+0.1,0.1):
            for spawning_distance in np.arange(self.spwaning_distances[0], self.spwaning_distances[1]+2.5,2.5):
                for looking_distance in np.arange(self.looking_distances[0], self.looking_distances[1]+0.01,0.01):
                    for op_reenter_distance in np.arange(self.op_reenter_distances[0], self.op_reenter_distances[1]+2.5,2.5):
                        for walking_distance in np.arange(self.walking_distances[0], self.walking_distances[1]+1,1):
                            for char in self.character:
                                conf = ControllerConfig(speed)
                                conf.spawning_distance = int(spawning_distance)
                                conf.looking_distance = looking_distance
                                conf.op_reenter_distance = op_reenter_distance
                                conf.walking_distance = walking_distance
                                conf.char = char
                                scenes.append(("02_non_int", conf))
        print('Total scenes count (non-int-2): ', len(scenes))
        return scenes

class Config03(ScenarioConfig):
    def __init__(self):
        self.ped_speed_range =  [1.6,2.0] #6
        self.spwaning_distances = [47.5,55] #3
        self.walking_distances = [8,12] #4
        self.looking_distances = [0.70,0.77] #9
        self.op_reenter_distances = [9.5,12] #0
        self.cross_walk_delta = []
        self.character = ["forcing"]
        #character = ["yielding", "forcing"]
        super(Config03,self).__init__()

    def get_scenes(self):
        scenes = []
        for speed in np.arange(self.ped_speed_range[0], self.ped_speed_range[1]+0.1,0.1):
            for spawning_distance in np.arange(self.spwaning_distances[0], self.spwaning_distances[1]+2.5,2.5):
                for looking_distance in np.arange(self.looking_distances[0], self.looking_distances[1]+0.01,0.01):
                    for op_reenter_distance in np.arange(self.op_reenter_distances[0], self.op_reenter_distances[1]+2.5,2.5):
                        for walking_distance in np.arange(self.walking_distances[0], self.walking_distances[1]+1,1):
                            for char in self.character:
                                conf = ControllerConfig(speed)
                                conf.spawning_distance = int(spawning_distance)
                                conf.looking_distance = looking_distance
                                conf.op_reenter_distance = op_reenter_distance
                                conf.walking_distance = walking_distance
                                conf.char = char
                                scenes.append(("03_non_int", conf))
        print('Total scenes count (non-int-3): ', len(scenes))
        return scenes

class Config04(ScenarioConfig):
    def __init__(self):
        self.ped_speed_range =  [1.6,2.0] #6
        self.spwaning_distances = [1,8.5] #3
        self.walking_distances = [-1,+1] #4
        self.looking_distances = [0.70,0.77] #9
        self.op_reenter_distances = [0,10] #0
        self.cross_walk_delta = []
        self.character = ["forcing"]
        #character = ["yielding", "forcing"]
        super(Config04,self).__init__()

    def get_scenes(self):
        scenes = []
        for speed in np.arange(self.ped_speed_range[0], self.ped_speed_range[1]+0.1,0.1):
            for spawning_distance in np.arange(self.spwaning_distances[0], self.spwaning_distances[1]+2.5,2.5):
                for looking_distance in np.arange(self.looking_distances[0], self.looking_distances[1]+0.01,0.01):
                    for op_reenter_distance in np.arange(self.op_reenter_distances[0], self.op_reenter_distances[1]+2.5,2.5):
                        for walking_distance in np.arange(self.walking_distances[0], self.walking_distances[1]+1,2):
                            for char in self.character:
                                conf = ControllerConfig(speed)
                                conf.spawning_distance = int(spawning_distance)
                                conf.looking_distance = looking_distance
                                conf.op_reenter_distance = op_reenter_distance
                                conf.walking_distance = walking_distance
                                conf.char = char
                                scenes.append(("04_non_int", conf))
        print('Total scenes count (non-int-4): ', len(scenes))
        return scenes

class Config05(ScenarioConfig):
    def __init__(self):
        self.ped_speed_range =  [1.6,2.0] #6
        self.spwaning_distances = [1,8.5] #3
        self.walking_distances = [-1,+1] #4
        self.looking_distances = [0.70,0.77] #9
        self.op_reenter_distances = [0,10] #0
        self.cross_walk_delta = []
        self.character = ["forcing"]
        #character = ["yielding", "forcing"]
        super(Config05,self).__init__()

    def get_scenes(self):
        scenes = []
        for speed in np.arange(self.ped_speed_range[0], self.ped_speed_range[1]+0.1,0.1):
            for spawning_distance in np.arange(self.spwaning_distances[0], self.spwaning_distances[1]+2.5,2.5):
                for looking_distance in np.arange(self.looking_distances[0], self.looking_distances[1]+0.01,0.01):
                    for op_reenter_distance in np.arange(self.op_reenter_distances[0], self.op_reenter_distances[1]+2.5,2.5):
                        for walking_distance in np.arange(self.walking_distances[0], self.walking_distances[1]+1,2):
                            for char in self.character:
                                conf = ControllerConfig(speed)
                                conf.spawning_distance = int(spawning_distance)
                                conf.looking_distance = looking_distance
                                conf.op_reenter_distance = op_reenter_distance
                                conf.walking_distance = walking_distance
                                conf.char = char
                                scenes.append(("05_non_int", conf))
        print('Total scenes count (non-int-5): ', len(scenes))
        return scenes

class Config06(ScenarioConfig):
    def __init__(self):
        self.ped_speed_range =  [1.2,1.5] #6
        self.spwaning_distances = [1,8.5] #3
        self.walking_distances = [-1,+1] #4
        self.looking_distances = [0.70,0.77] #9
        self.op_reenter_distances = [0,10] #0
        self.cross_walk_delta = []
        self.character = ["forcing"]
        #character = ["yielding", "forcing"]
        super(Config06,self).__init__()

        # self.ped_speed_range =  [1.5,1.5] #6
        # self.spwaning_distances = [1,1] #3
        # self.walking_distances = [-1,-1] #4
        # self.looking_distances = [0.70,0.77] #9
        # self.op_reenter_distances = [10,10] #0
        # self.cross_walk_delta = []
        # self.character = ["forcing"]
        # #character = ["yielding", "forcing"]
        # super(Config06,self).__init__()

    def get_scenes(self):
        scenes = []
        for speed in np.arange(self.ped_speed_range[0], self.ped_speed_range[1]+0.1,0.1):
            for spawning_distance in np.arange(self.spwaning_distances[0], self.spwaning_distances[1]+2.5,2.5):
                for looking_distance in np.arange(self.looking_distances[0], self.looking_distances[1]+0.01,0.01):
                    for op_reenter_distance in np.arange(self.op_reenter_distances[0], self.op_reenter_distances[1]+2.5,2.5):
                        for walking_distance in np.arange(self.walking_distances[0], self.walking_distances[1]+1,2):
                            for char in self.character:
                                conf = ControllerConfig(speed)
                                conf.spawning_distance = int(spawning_distance)
                                conf.looking_distance = looking_distance
                                conf.op_reenter_distance = op_reenter_distance
                                conf.walking_distance = walking_distance
                                conf.char = char
                                scenes.append(("06_non_int", conf))
        print('Total scenes count (non-int-6): ', len(scenes))
        return scenes


class Config:
    PI = 3.14159

    simulation_step = 0.05  # 0.008
    sensor_simulation_step = '0.5'
    synchronous = True
    segcam_fov = '90'
    segcam_image_x = '400'  # '1280'
    segcam_image_y = '400'  # '720'

    grid_size = 2  # grid size in meters
    speed_limit = 50
    max_steering_angle = 1.22173  # 70 degrees in radians
    occupancy_grid_width = '1920'
    occupancy_grid_height = '1080'

    location_threshold = 1.0

    ped_speed_range = [1.8, 2.2]
    ped_distance_range = [25, 30]
    # car_speed_range = [6, 9]
    scenarios = ["01_int", "02_int", "03_int", "04_int", "05_int", "06_int", "01_non_int", "02_non_int", "03_non_int", "04_non_int", "05_non_int", "06_non_int"]#,'02_non_int','03_non_int']#,'02_non_int']#,"02_int", "03_int"  #, '02', '03', '04', '05', '06', '07', '08', '09']
    #scenarios = ['01_int','02_int','03_int']
    #scenarios = ['01','02', '03', '04', '05', '06', '07', '08', '09']
    val_scenarios = ['06'],# '02', '03', '04', '05', '06', '07', '08', '09']
    val_ped_speed_range = ([0.2, 0.5], [2.1, 2.8])
    val_ped_distance_range = [4.25, 49.25]
    # val_car_speed_range = [6, 9]

    test_scenarios = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    test_ped_speed_range = [0.25, 2.85]
    test_ped_distance_range = [4.75, 49.75]
    # test_car_speed_range = [6, 9]

    save_freq = 100

    # Setting the SAC training parameters
    batch_size = 2  # 32  # How many experience traces to use for each training step.
    update_freq = 4  # How often to perform a training step after each episode.
    load_model = True  # Whether to load a saved model.
    # path = "_out/sac/"  # The path to save our model to.
    total_training_steps = 1000001
    automatic_entropy_tuning = False
    target_update_interval = 1
    hidden_size = 256
    max_epLength = 500  # The max allowed length of our episode.
    sac_gamma = 0.99
    sac_tau = 0.005
    sac_lr = 0.00001
    sac_alpha = 0.1
    num_pedestrians = 4
    num_angles = 5
    num_actions = 3  # num_angles * 3  # acceleration_type
    EPS_START = 0.9
    EPS_END = 0.1
    EPS_DECAY = 500
    episode_buffer = 80
    adrqn_entropy_coef = 0.005
    grad_norm = 0.1

    # angle + 4 car related statistics + 2*num_pedestrians related statistics + one-hot encoded last_action
    input_size = 1 + 4 + 2 * num_pedestrians + num_actions
    image_input_size = 100 * 100 * 3
    tau = 1
    targetUpdateInterval = 10000

    use_dueling = False

    # Simulator Parameters
    host = '127.0.0.1'
    port = 2000
    width = 1920
    height = 1080
    display = False
    filter = 'vehicle.audi.tt'
    rolename = 'hero'
    gama = 1.7
    # despot_port = 1255
    N_DISCRETE_ACTIONS = 3
    max_speed = 30 * 0.27778  # in m/s was 50 in dikshants experiments
    max_speed_kmh = 30
    hit_penalty = 50
    goal_reward = 25
    nearmiss_penalty = 50
    too_fast = 10
    braking_penalty = 1
    record = True

    pre_train_steps = 500000

    # A2C training parameters
    a2c_lr = 0.0001
    a2c_gamma = 0.99
    a2c_gae_lambda = 1.0
    a2c_entropy_coef = 0.01
    a2c_value_loss_coef = 0.5
    a2c_eval = 500
    max_grad_norm = 50
    num_steps = 500
    train_episodes = 5000

    # PP extractor parameters
    episodes = 200
    max_episode_length = 500
    no_rendering = False

    # hyleap
    hyleap_training_epidsodes = 5000