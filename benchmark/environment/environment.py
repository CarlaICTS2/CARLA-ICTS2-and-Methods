import gym
import numpy as np
from benchmark.environment.ped_controller import ControllerConfig
import carla
import pygame
import random
from PIL import Image
import time
import matplotlib.pyplot as plt
from benchmark.environment.world import World
from benchmark.environment.hud import HUD
from benchmark.learner_example import Learner
import timeit

from config import (Config, Config01, Config02, Config03, Config04, Config05, Config06,
                    IConfig01, IConfig02, IConfig03, IConfig04, IConfig05, IConfig06, )
from utils.config import Config as newConfig
from benchmark.scenarios.scenario import Scenario


class GIDASBenchmark(gym.Env):
    def __init__(self, port=Config.port, mode = "TRAINING", setting="normal", record=False):
        # Config.scenarios = newConfig.scenarios
        super(GIDASBenchmark, self).__init__()
        random.seed(100)
        self.action_space = gym.spaces.Discrete(Config.N_DISCRETE_ACTIONS)
        height = int(Config.segcam_image_x)
        width = int(Config.segcam_image_y)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(height, width, 3), dtype=np.uint8)
        self.fig = plt.figure()
        pygame.init()
        pygame.font.init()

        self.client = carla.Client(Config.host, port)
        self.client.set_timeout(40.0)
        self.retarded_agent = None
        self.plot_intention = False
        self.pc = 0
        self.control = None
        self.display = None
        self.scenario = None
        self.speed = None
        self.distance = None
        self.record = record
        self.mode = mode
        self.setting = setting
        self._max_episode_steps = 500
        self.clock = pygame.time.Clock()
        print("Load World")
        hud = HUD(Config.width, Config.height)
        with open("./assets/Town01_my.xodr") as odr:
            self.world = self.client.generate_opendrive_world(odr.read(),
                                                              carla.OpendriveGenerationParameters(
                                                                  2.0, 50.0, 0.0, 200.0, False, True))

        # self.client.load_world('Town01_Opt')
        self.first_sleep = True 
        wld = self.client.get_world()
        self.extract = False
        self.prev_vel = 20
        print("Loaded ")
        #time.sleep(5

        for actor in wld.get_actors():
            if 'static' in actor.type_id:
                actor.destroy()

        # print("Unloading map layers")
        # print("unloading map StreetLights")
        # wld.unload_map_layer(carla.MapLayer.StreetLights)
        # wld.tick()
        # print("unloading map Buildings")
        # wld.unload_map_layer(carla.MapLayer.Buildings)
        # wld.tick()
        # print("unloading map Props")
        # wld.unload_map_layer(carla.MapLayer.Props)
        # wld.tick()
        # print("unloading map Particles")
        # wld.unload_map_layer(carla.MapLayer.Particles)
        # wld.tick()
        # print("unloading map Foliage")
        # wld.unload_map_layer(carla.MapLayer.Foliage)
        # wld.tick()
        # print("unloading map Walls")
        # wld.unload_map_layer(carla.MapLayer.Walls)
        # wld.tick()
        # print("unloading map ParkedVehicles")
        # wld.unload_map_layer(carla.MapLayer.ParkedVehicles)

        self.map = wld.get_map()
        settings = wld.get_settings()
        settings.fixed_delta_seconds = Config.simulation_step
        settings.synchronous_mode = Config.synchronous
        settings.no_rendering_mode = Config.no_rendering
        wld.apply_settings(settings)

        self.scene_generator = Scenario(wld)
        self.scene = self.scene_generator.scenario01()
        self.world = World(wld, hud, self.scene, Config)
        # self.planner_agent = RLAgent(self.world, self.map, self.scene)
        self.planner_agent = Learner(self.world, self.map, self.scene)

        wld_map = wld.get_map()
        print(wld_map.name)
        wld.tick()

        self.episodes = list()
        self.val_episodes = list()
        self.test_episodes = list()

        i=0
        print("scenarios:", Config.scenarios)
        if self.mode == "TRAINING":
            selector = lambda x: x.get_training()
        elif self.mode == "VALIDATION":
            selector = lambda x: x.get_validation()
        else:
            selector = lambda x: x.get_test()
        for scenario in Config.scenarios:
            if self.setting == "special":
                # Used for backwards compatibility
                self._get_special_scenes()
                self.mode = "TESTING"
                self.test_episodes = iter(self.episodes)
            elif scenario == "01_int":
                self.episodes.extend(IConfig01().get_training())
                self.val_episodes.extend(IConfig01().get_validation())
                self.test_episodes.extend(IConfig01().get_test())
            elif scenario == "02_int":
                self.episodes.extend(IConfig02().get_training())
                self.val_episodes.extend(IConfig02().get_validation())
                self.test_episodes.extend(IConfig02().get_test())
            elif scenario == "03_int":
                self.episodes.extend(IConfig03().get_training())
                self.val_episodes.extend(IConfig03().get_validation())
                self.test_episodes.extend(IConfig03().get_test())
            elif scenario == "01_non_int":
                self.episodes.extend(Config01().get_training())
                self.val_episodes.extend(Config01().get_validation())
                self.test_episodes.extend(Config01().get_test())
            elif scenario == "02_non_int":
                self.episodes.extend(Config02().get_training())
                self.val_episodes.extend(Config02().get_validation())
                self.test_episodes.extend(Config02().get_test())
            elif scenario == "03_non_int":
                self.episodes.extend(Config03().get_training())
                self.val_episodes.extend(Config03().get_validation())
                self.test_episodes.extend(Config03().get_test())

            elif scenario == "04_non_int":
                self.episodes.extend(Config04().get_training())
                self.val_episodes.extend(Config04().get_validation())
                self.test_episodes.extend(Config04().get_test())

            elif scenario == "05_non_int":
                self.episodes.extend(Config05().get_training())
                self.val_episodes.extend(Config05().get_validation())
                self.test_episodes.extend(Config05().get_test())

            elif scenario == "06_non_int":
                self.episodes.extend(Config06().get_training())
                self.val_episodes.extend(Config06().get_validation())
                self.test_episodes.extend(Config06().get_test())

            elif scenario == "04_int":
                self.episodes.extend(IConfig04().get_training())
                self.val_episodes.extend(IConfig04().get_validation())
                self.test_episodes.extend(IConfig04().get_test())

            elif scenario == "05_int":
                self.episodes.extend(IConfig05().get_training())
                self.val_episodes.extend(IConfig05().get_validation())
                self.test_episodes.extend(IConfig05().get_test())

            elif scenario == "06_int":
                self.episodes.extend(IConfig06().get_training())
                self.val_episodes.extend(IConfig06().get_validation())
                self.test_episodes.extend(IConfig06().get_test())


            else:
                # Used for backwards compatibility
                for speed in np.arange(Config.ped_speed_range[0], Config.ped_speed_range[1] + 0.1, 0.1):
                    for distance in np.arange(Config.ped_distance_range[0], Config.ped_distance_range[1] + 1, 1):
                        conf = ControllerConfig(speed, distance)
                        self.episodes.append((scenario, conf))
                        i=i+1

        print("TRAINING"," Number scences:", len(self.episodes))
        print("VALIDATION"," Number scences:", len(self.val_episodes))
        print("TESTING"," Number scences:", len(self.test_episodes))
        #if self.mode == "TESTING" or self.mode=="VALIDATION":
        self.val_episodes_iterator = iter(self.val_episodes)
        self.test_episodes_iterator = iter(self.test_episodes)
        self.ds = 0
        self.world.set_parked_vehicle_transforms_across_all_scenarios({})

        if self.mode == "TESTING":
            self.number_of_episode = len(self.test_episodes)
        if self.mode == "VALIDATION":
            self.number_of_episode = len(self.val_episodes)
        if self.mode == "TRAINING":
            self.number_of_episode = len(self.episodes)

        self.client_world = self.world

    def _get_special_scenes(self):
        for scenario in Config.val_scenarios:
            for speed in np.arange(Config.val_ped_speed_range[0][0], Config.val_ped_speed_range[0][1] + 0.1, 0.1):
                for distance in np.arange(Config.val_ped_distance_range[0], Config.ped_distance_range[1] + 1, 1):
                    self.episodes.append((scenario, speed, distance))
                    #TODO has to be adapted for new config interface
            for speed in np.arange(Config.val_ped_speed_range[1][0], Config.val_ped_speed_range[1][1] + 0.1, 0.1):
                for distance in np.arange(Config.val_ped_distance_range[0], Config.ped_distance_range[1] + 1, 1):
                    self.episodes.append((scenario, speed, distance))
        # episodes = [(scenario, 1.3, 40.0), (scenario, 1.5, 40.0), (scenario, 1.7, 36.0), (scenario, 2.0, 32.0),
        #         #             (scenario, 1.6, 36.0), (scenario, 2.0, 25.0), (scenario, 2.8, 18.0)]
        # self.episodes += episodes



    def reset(self, step_counter: int):
        if True:
            scenario_id, conf = self.next_scene()
            # ped_speed = 1.25  # Debug Settings
            # ped_distance = 10.75
            # scenario_id = "10"
            self.scenario = scenario_id
            self.speed = conf.ped_speed
            self.distance = conf.ped_distance
        else:
            scenario_id, self.speed, self.distance = self.next_scene() 
            conf=ControllerConfig()
            conf.ped_speed = self.speed
            conf.ped_distance = self.distance
        func = 'self.scene_generator.scenario' + scenario_id
        scenario = eval(func + '()')
        self.world.restart(scenario, conf)
        self.planner_agent.update_scenario(scenario)

        self.world.world.tick()
        i=0
        #print("Is none", self.world.semseg_sensor.array is None)
        while self.world.semseg_sensor.array is None:
            i+=1
            self.world.world.tick()
            if i > 100:
                # exit(1)
                print(i)
        #print("Is none", self.world.semseg_sensor.array is None)
        # observation, risk, ped_observable = self._get_observation(step_counter)
        self.ds = 0
        # return observation

    def reset_extract(self):
        scenario_id, conf = self.next_scene()
        self.scenario = scenario_id
        self.speed = conf.ped_speed
        self.distance = conf.ped_distance
        func = 'self.scene_generator.scenario' + scenario_id
        scenario = eval(func + '()')
        self.world.restart(scenario, conf)
        self.planner_agent.update_scenario(scenario)

        self.world.world.tick()
        # print("Is none", self.world.semseg_sensor.array is None)
        i=0
        while self.world.semseg_sensor.array is None:
            self.world.world.tick()
            if i > 100:
                print(i)
        return self.world.get_walker_state()


    # def _get_observation(self, step_counter):
    #     control, observation, risk, ped_observable = self.planner_agent.run_step(step_counter)
    #     x,y,icr,son = self.world.get_walker_state()
    #     #print(x,y,icr,son)
    #     self.control = control
    #     return observation, risk, ped_observable

    def step(self, step_counter):
        self.world.tick(self.clock)
        self.client.get_world().tick()
        step_summary = self.planner_agent.run_step(step_counter)
        step_summary["step"] = step_counter
        step_summary["velocity.x"] = self.planner_agent.vehicle.get_velocity().x
        step_summary["velocity.y"] = self.planner_agent.vehicle.get_velocity().y
        step_summary["scenario"] = self.scenario
        step_summary["ped_speed"] = float(self.speed)
        step_summary["ped_distance"] = int(self.distance)

        # update control
        self.control = step_summary["control"]

        '''        
        log_info(
            f"STEP={step_counter}: "
            f"control.throttle: {self.control.throttle:.4f}, "
            f"control.brake: {self.control.brake:.4f}, "
            f"control.steer: {self.control.steer:.4f}"
        )
        '''

        # set agent vehicle control
        print("when applied: ", self.control)
        print(self.world.player.get_acceleration())
        self.world.player.apply_control(self.control)


        # apply walker control


        # this ticks the server for a single simulation step
        # if Config.Carla.SYNCHRONONOUS: self.server_world.tick()
        # else: raise ValueError("Invalid CARLA server simulation mode: Only synchrononous simulation supported.")
        #
        # if Config.DISPLAY:
        #     # display the image in an OpenCV display window
        #     cv2.namedWindow('ImageWindowName', cv2.WINDOW_AUTOSIZE)
        #     cv2.imshow('ImageWindowName', self.client_world.render_rgb_camera_with_bounding_boxes())
        #     if cv2.waitKey(1) == ord('q'):
        #         cv2.destroyAllWindows()

        return step_summary
        # self.world.tick(self.clock)
        # velocity = self.world.player.get_velocity()
        # speed = (velocity.x * velocity.x + velocity.y * velocity.y) ** 0.5
        # speed *= 3.6
        #if speed > Config.max_speed_kmh:
        #    action = 1
        #print(speed)
        #action = 1 if self.ds%2==0 else 0
        #action = 0
        #print(speed)
        #print(velocity)
        #self.control = self.world.player.get_control()
        #print("Before",self.control)
        #acc_vec = carla.Vector3D(0,-10,0)
        #self.world.player.add_force(acc_vec)
        #if self.ds < 30:
        #    action = 0
        #else:
        #    action = 2
        #action = 1

        #if speed > 35:
            #print("Violation")
        # if action == 0:
        #     self.control.throttle = 0.6
        #     #print("Acc")
        # elif action == 2:
        #     self.control.brake = 0.6
        #     #print("Brake")
        #     #pass
        # elif action == 1:
        #     # print("Keep")
        #     self.control.throttle = 0.0
        #     self.control.brake = 0.0
        # self.control.gear=2
        # self.control.manual_gear_shift=True

        # self.world.player.apply_control(self.control)
        # if Config.synchronous:
        #     frame_num = self.client.get_world().tick()
        #     if self.record:
        #         im = Image.fromarray(self.world.camera_manager.array.copy())
        #         im.save("_out/recordings/frame_{:03d}.png".format(frame_num))
        # if Config.display:
        #     self.render()
        # #print(self.world.player.get_acceleration())
        #
        # observation, risk, ped_observable = self._get_observation()
        # if self.retarded_agent=="hyleap":
        #     if self.control.throttle == 0.6:
        #         action = 0
        #     elif self.control.brake == 0.6:
        #         action = 2
        #     else:
        #         action = 1
        # reward, goal, accident, near_miss, terminal = self.planner_agent.get_reward(action)
        # info = {"goal": goal, "accident": accident, "near miss": near_miss,
        #         "velocity": self.planner_agent.vehicle.get_velocity(), "risk": risk, 'ped_observable': ped_observable,
        #         "scenario": self.scenario, "ped_speed": self.speed, "ped_distance": self.distance}
        # self.ds+=1
        # if self.mode == "TESTING":
        #     terminal = goal or accident
        # self.plot_intention = False
        # if self.plot_intention:
        #     self.pc +=1
        #     if self.pc % 5 == 0:
        #         fig=plt.figure()
        #         plt.imshow(observation)
        #         fig.savefig("cp_debug/cp_%d.png"%self.pc,dpi=400)
        # #print("The time difference is :", timeit.default_timer() - starttime)
        # return observation, reward, terminal, info

    def extract_step(self):
        self.world.tick(self.clock)
        if Config.synchronous:
            frame_num = self.client.get_world().tick()

        x,y,icr,son = self.world.get_walker_state()

        return x,y,icr,son

    def extract_car_pos(self):
        self.world.tick(self.clock)
        if Config.synchronous:
            frame_num = self.client.get_world().tick()

        x,y= self.world.get_car_state()

        return x,y
    
    def record_step(self,):
        self.world.tick(self.clock)

        self.control = carla.VehicleControl()
        self.control.brake = 0.0
        self.control.hand_brake = False
        self.control.manual_gear_shift = False

        velocity = self.world.player.get_velocity()
        speed = (velocity.x * velocity.x + velocity.y * velocity.y) ** 0.5
        speed *= 3.6
        action = 0
        if self.scenario == '10':
            # Maintain a minimum speed of 20kmph
            if speed < 20:
                action = 0
            elif speed > 50:
                action = 2
        if self.scenario == '11' or self.scenario == '12':
            # Maintain a maximum speed of 20kmph
            if speed > 20:
                action = 2
        #if self.scenario in ["01_int", "02_int","01","01_non_int",]:
        #    action = 0
        #    if speed > 30:
        #        action = 2
        #if self.scenario in ["03_non_int"]:
        #    action = 0
        #    if speed > 25:
        #        action = 2
        #if self.scenario in ["03_int"]:
        #    if speed > 25:
        #        action = 2

        if action == 0:
            vel_target = velocity  + carla.Vector3D(0,-0.075*self.pc,0)
            self.world.player.set_target_velocity(vel_target)
            #self.control.throttle = 0.6
            #print("Acc")
        #elif action == 2:
        #    vel_target = velocity + carla.Vector3D(0,0.01,0)
        #    self.world.player.set_target_velocity(vel_target)
            #self.control.brake = 0.6
            #print("Brake")
        #elif action == 1:
        #    #print("Keep")
        #    self.control.throttle = 0
        ds = speed - self.prev_vel
        dsdt = ds*20
        self.prev_vel = speed
        print("Speed", speed)
        print("Action", action)
        print("dsdt",dsdt)
        print(vel_target)
        print(self.world.player.get_velocity())
        print(self.world.player.get_acceleration())
        self.pc+=1
        """
        #scenario01
        print(self.world.player.get_location().y)
        if self.world.player.get_location().y < 200: #200 pedestrian stops #271 pedestrian walks 
            self.control.brake = 0.6
            self.control.throttle = 0.0
        
        print(self.world.player.get_location().y)
        if self.world.player.get_location().y < 272.5: #200 pedestrian stops #285 pedestrian walks #276 pedestrian first stops then walks
            self.control.brake = 0.2
            self.control.throttle = 0.0
        
        #scenario02
        if Config.scenarios[0] == "02_int":
            if self.world.player.get_location().y < 261: #200 pedestrian stops #285 pedestrian walks #276 pedestrian first stops then walks
                self.control.brake = 0.35
                self.control.throttle = 0.0
        

        
        #scenario03
        if self.world.player.get_location().y < 200:  #262 pedestrian walks # pedestrian still stops #200 pedestrian stops 
            self.control.brake = 0.6
            self.control.throttle = 0.0
        """

        #self.world.player.apply_control(self.control)

        if Config.synchronous:
            frame_num = self.client.get_world().tick()

        x,y,icr,son = self.world.get_walker_state()

        observation, risk, ped_observable = self._get_observation()
        self.plot_intention = False
        if self.plot_intention:
            self.pc +=1
            if self.pc % 5 == 0:
                fig=plt.figure()
                plt.imshow(observation)
                fig.savefig("cp_debug/cp_%d.png"%self.pc,dpi=800)
                print("Saved")
        return x,y,icr,son

    def prepare_train_episodes(self):
        pass

    def prepare_test_episodes(self):
        pass


    def render(self, mode="human"):
        if self.display is None:
            self.display = pygame.display.set_mode(
                (Config.width, Config.height),
                pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.display.fill((0, 0, 0))
            pygame.display.flip()
        self.world.render(self.display)
        pygame.display.flip()

    def close(self):
        self.world.destroy()
        pygame.quit()

    def reset_agent(self, agent):
        # if agent == 'reactive':
        #     self.planner_agent = ReactiveController(self.world, self.map, self.scene)
        # if agent == 'isdespot':
        #     conn = Connector(Config.despot_port)
        #     self.planner_agent = ISDespotP(self.world, self.map, self.scene, conn)
        # if agent == 'hyleap':
        #     conn = Connector(Config.despot_port)
        #     self.planner_agent = HyLEAP(self.world, self.map, self.scene, conn)
        # if agent == 'isdespot*':
        #     conn = Connector(Config.despot_port)
        #     self.planner_agent = ISDespotPStar(self.world, self.map, self.scene, conn)
        # if agent == 'cadrl':
        #     self.planner_agent = A2CCadrl(self.world, self.map, self.scene)
        # if agent == 'hylear' or agent == 'hypal':
        #     conn = Connector(Config.despot_port)
        #     eval_mode = False
        #     if self.mode == "TESTING":
        #         eval_mode = True
        #     self.planner_agent = HyLEAR(self.world, self.map, self.scene, conn, eval_mode, agent)
        self.planner_agent = agent

    def eval(self, current_episode=0):
        self.mode = "TESTING"
        episodes = list()
        for scenario in Config.test_scenarios:
            if scenario in ['11', '12']:
                for speed in np.arange(10, 20 + 0.1, 0.1):
                    episodes.append((scenario, speed, 0))
            else:
                for speed in np.arange(Config.test_ped_speed_range[0], Config.test_ped_speed_range[1] + 0.1, 0.1):
                    for distance in np.arange(Config.test_ped_distance_range[0], Config.test_ped_distance_range[1] + 1, 1):
                        episodes.append((scenario, speed, distance))
        self.episodes = episodes[current_episode:]
        print("Episodes: ", len(self.episodes))
        self.test_episodes = iter(episodes[current_episode:])

    def reset_iterator(self):
        self.val_episodes_iterator = iter(self.val_episodes)
        self.test_episodes_iterator = iter(self.test_episodes)

    def next_scene(self):
        #return random.choice(self.episodes)
        if self.mode == "VALIDATION":
            return next(self.val_episodes_iterator)
        elif self.mode == "TESTING":
            return next(self.test_episodes_iterator)
        else:
            #return self.episodes[0]
            i = np.random.randint(0,len(self.episodes))
            return self.episodes[i]

    def assign_agent(self, agent):
        self.planner_agent = agent
        

    def seed(self,seed):
        pass
"""     
    def next_scene(self):
        if self.mode == "TRAINING":
            #return random.choice(self.episodes)
            return self.episodes[0]
        elif self.mode == "TESTING":
            scene_config = next(self.test_episodes)
            #return self.episodes[0]
            return scene_config
""" 

