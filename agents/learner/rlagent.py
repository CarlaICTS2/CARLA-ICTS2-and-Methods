import math
import time
from collections import deque
from typing import List, Tuple

import cv2
import carla
import numpy as np

from utils.config import Config, Action, Agent
from utils.logger import log_debug, log_info
from utils.utils import l2_distance, get_corners, degrees_to_radians
from utils.carla_birdeye_view import BirdViewProducer, BirdViewCropType, PixelDimensions

from assets.occupancy_grid import OccupancyGrid
from path_planner.hybridastar import HybridAStar
from benchmark.risk.risk_aware_path import PathPlanner
from ped_path_predictor.m2p3 import PathPredictor


# defines current car intention and reward
class RLAgent(object):

    def __init__(self, client, world, carla_map, scenario):
        self.client = client
        self.world = world
        self.vehicle = world.player
        self.wmap = carla_map
        self.scenario = scenario
        self.occupancy_grid = OccupancyGrid()

        # all of the below are only relevant for a single episode
        self.episode_observations: List[Tuple[float, float, float, float, float, float]] = []
        self.episode_rewards: List[float] = []
        self.episode_actions: List[Action] = []
        self.episode_agent_vehicle_speeds: List[float] = []
        self.episode_despot_policies: List[Tuple[float, float, float]] = []
        self.episode_controls: List[carla.VehicleControl] = []

        # required for reconstructing the already driven path which is then drawn on the car intention image
        # angle is omitted as it is not required for drawing a line with cv2
        self.episode_agent_vehicle_past_trajectory: List[Tuple[int, int]] = []
        # the planned path for the agent vehicle at the current time step
        self.step_agent_vehicle_future_trajectory: List[Tuple[int, int, float]] = []

        self.episode_pedestrian_past_trajectory: List[Tuple[float, float]] = []

        # car intention images
        self.episode_birdview_car_intentions: List[np.ndarray] = []

        # episode flags
        self.is_terminal_state: bool = False
        self.has_agent_vehicle_reached_goal: bool = False
        self.is_agent_vehicle_in_collision: bool = False
        self.is_pedestrian_observable: bool = False
        self.is_incoming_car_observable: bool = False

        # pedestrian history keepas track of the most recent 15 simulation steps
        self.ped_history = deque(list(), maxlen=15)
        self.car_history = deque(list(), maxlen=15)
        # how does the history differ from the trajectory?
        self.past_trajectory = list()

        obstacle = []

        self.grid_cost = np.ones((110, 310)) * 1000.0
        # Road Network
        self.grid_cost[7:13, 13:] = 1.0
        self.grid_cost[97:103, 13:] = 1.0
        self.grid_cost[7:, 7:13] = 1.0
        # Sidewalk Network
        self.grid_cost[4:7, 4:] = 50.0
        self.grid_cost[:, 4:7] = 50.0
        self.grid_cost[13:16, 13:] = 50.0
        self.grid_cost[94:97, 13:] = 50.0
        self.grid_cost[103:106, 13:] = 50.0
        self.grid_cost[13:16, 16:94] = 50.0

        self.min_x = -10
        self.max_x = 100
        self.min_y = -10
        self.max_y = 300
        self.path_planner = HybridAStar(self.min_x, self.max_x, self.min_y, self.max_y, obstacle,
                                        Config.Carla.EGO_VEHICLE_LENGTH)

        self.risk_cmp = np.zeros((110, 310))
        # Road Network
        self.risk_cmp[7:13, 13:] = 1.0
        self.risk_cmp[97:103, 13:] = 1.0
        self.risk_cmp[7:, 7:13] = 1.0
        # Sidewalk Network
        sidewalk_cost = 50.0
        self.risk_cmp[4:7, 4:] = sidewalk_cost
        self.risk_cmp[:, 4:7] = sidewalk_cost
        self.risk_cmp[13:16, 13:] = sidewalk_cost
        self.risk_cmp[94:97, 13:] = sidewalk_cost
        self.risk_cmp[103:106, 13:] = sidewalk_cost
        self.risk_cmp[13:16, 16:94] = sidewalk_cost

        # what is the difference between this and HybridAStar path planner?
        self.risk_path_planner = PathPlanner()

        if Config.PREDICT_PEDESTRIAN_PATH:
            self.ped_pred = PathPredictor(Config.MODEL_DIR + "m2p3.pth")
            self.ped_pred.model.eval()

        # used for producing car intention images
        self.birdview_car_intention_producer = BirdViewProducer(
            self.client,  # carla.client
            target_size=PixelDimensions(
                width=int(Config.Carla.SEGCAM_IMAGE_WIDTH),
                height=int(Config.Carla.SEGCAM_IMAGE_HEIGHT)),
            # ~50m cropping
            pixels_per_meter=(int(Config.Carla.SEGCAM_IMAGE_WIDTH) + int(Config.Carla.SEGCAM_IMAGE_HEIGHT)) / 2 / 50,
            crop_type=BirdViewCropType.FRONT_AND_REAR_AREA,
            all_parked_vehicle_transforms=self.world.all_parked_vehicle_transforms)

    # isn't this rather resetting it?
    def update_scenario(self, scenario):
        self.scenario = scenario
        self.scenario_id = scenario[0]
        self.episode_agent_vehicle_goal_position = scenario[2]
        self.episode_agent_vehicle_start_position = scenario[3]
        self.ped_history = deque(list(), maxlen=15)
        self.past_trajectory = list()

    def initialize_episode(self, episode_counter: int):
        # reset episode counters & flags
        self.is_terminal_state = False
        self.is_pedestrian_observable = False
        self.is_incoming_car_observable = False
        self.has_agent_vehicle_reached_goal = False
        self.is_agent_vehicle_in_collision = False

        # reset episode buffers
        self.episode_observations.clear()
        self.episode_rewards.clear()
        self.episode_actions.clear()
        self.episode_agent_vehicle_speeds.clear()
        self.episode_despot_policies.clear()
        self.episode_controls.clear()

        self.episode_agent_vehicle_past_trajectory.clear()
        self.episode_birdview_car_intentions.clear()

        # first dummy reward for timestep -1 (previous reward of initial state)
        self.episode_rewards.append(0.0)
        # default action: maintain
        self.episode_actions.append(Action.MAINTAIN)
        # default car speed: 0.0
        # self.episode_car_speeds.append(0.0)

    def finalize_episode(self, episode_counter: int, terminated_early: bool):
        return

    def get_current_observation(self, step_counter: int):
        # create observation components
        agent_vehicle_position = [self.vehicle.get_location().x, self.vehicle.get_location().y]
        # in m/s
        car_speed = np.sqrt(self.vehicle.get_velocity().x ** 2 + self.vehicle.get_velocity().y ** 2)
        car_angle = self.vehicle.get_transform().rotation.yaw

        # pedestrian
        if self.is_pedestrian_observable:
            pedestrian_position = [self.world.walker.get_location().x, self.world.walker.get_location().y]
        else:
            pedestrian_position = [None, None]

        # check for terminal state
        agent_vehicle_goal_distance = np.linalg.norm(
            [agent_vehicle_position[0] - self.episode_agent_vehicle_goal_position[0],
             agent_vehicle_position[1] - self.episode_agent_vehicle_goal_position[1]]
        )
        self.has_agent_vehicle_reached_goal = agent_vehicle_goal_distance <= Config.Carla.GOAL_TOLERANCE

        self.is_agent_vehicle_in_collision = self.world.collision_sensor.flag

        self.is_terminal_state = self.has_agent_vehicle_reached_goal or self.is_agent_vehicle_in_collision

        '''
        log_debug(
            f"agent_vehicle_goal_distance: {agent_vehicle_goal_distance:.2f}, "
            f"self.has_agent_vehicle_reached_goal: {self.has_agent_vehicle_reached_goal}, "
            f"self.is_agent_vehicle_in_collision: {self.is_agent_vehicle_in_collision }, "
            f"self.is_terminal_state: {self.is_terminal_state}"
        )

        log_info(
            f"STEP: {step_counter}, car_angle: {car_angle:.2f}, ped_angle: {self.world.walker.get_transform().rotation.yaw:.2f}"
        )
        '''

        self.episode_observations.append((*agent_vehicle_position, car_speed, car_angle, *pedestrian_position))
        self.episode_agent_vehicle_speeds.append(car_speed)
        self.episode_agent_vehicle_past_trajectory.append(agent_vehicle_position)

        # sanity checks
        if len(self.episode_observations) != step_counter:
            raise ValueError(
                f"Invalid numer of episode observations: "
                f"Expected {step_counter}, got {len(self.episode_observations)} instead."
            )
        if len(self.episode_agent_vehicle_past_trajectory) != step_counter:
            raise ValueError(
                f"Invalid length of agent-vehicle's past trajectory: "
                f"Expected {step_counter}, got {len(self.episode_agent_vehicle_past_trajectory)} instead."
            )
        if len(self.episode_agent_vehicle_speeds) != step_counter:
            raise ValueError(
                f"Invalid numer of episode agent vehicle speeds: "
                f"Expected {step_counter + 1}, got {len(self.episode_agent_vehicle_speeds)} instead."
            )

            # call this function when having to adjust the current state of the simulation in CARLA according to

    # IS-DESPOT's simulated planning state, i.e. this is required for HyLEAP and HyPLAN since
    # birdview car intention images have to be created for all simulated/expanded nodes during search
    def get_birdview_car_intention(
            self,
            agent_vehicle_transform: carla.Transform,
            pedestrian_transform: carla.Transform,
            episode_agent_vehicle_past_trajectory: List[Tuple[float, float]],
            step_agent_vehicle_future_trajectory: List[Tuple[float, float, float]],
            step_counter: int
    ) -> np.ndarray:

        if np.shape(step_agent_vehicle_future_trajectory)[1] != 3:
            raise ValueError(
                f"Invalid dimensions of agent-vehicle's future trajectory: "
                f"Expected (n, 3), got {np.shape(step_agent_vehicle_future_trajectory)}."
            )

        # returned result is np.ndarray with ones and zeros of shape (8, height, width)
        birdview_car_intention = self.birdview_car_intention_producer.produce(
            agent_vehicle_transform,
            pedestrian_transform,
            episode_agent_vehicle_past_trajectory.copy(),
            # angle is omitted as it is not required for drawing a line with cv2
            [waypoint[:-1] for waypoint in step_agent_vehicle_future_trajectory.copy()],
            self.scenario_id  # scenario id: required for loading parked cars
        )

        # produces np.ndarray of shape (height, width, 3)
        birdview_car_intention = self.birdview_car_intention_producer.as_rgb(birdview_car_intention)
        self.episode_birdview_car_intentions.append(birdview_car_intention)

        # print agent vehicle and pedestrian positions + angles as additional debug information
        if Config.DISPLAY:
            # determine hspace between newlines
            (_, label_height), _ = cv2.getTextSize("text", cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
            # list containing lines to be printed on car intention images
            lines = [
                f"agent vehicle (x: {agent_vehicle_transform.location.x:.2f}, "
                f"y: {agent_vehicle_transform.location.y:.2f}, "
                f"z: {agent_vehicle_transform.location.z:.2f}, "
                f"a: {agent_vehicle_transform.rotation.yaw:.2f})",
                f"pedstrian (x: {pedestrian_transform.location.x:.2f}, "
                f"y: {pedestrian_transform.location.y:.2f}, "
                f"z: {pedestrian_transform.location.z:.2f}, "
                f"a: {pedestrian_transform.rotation.yaw:.2f})",
                f"initial position: {self.episode_agent_vehicle_start_position}",
                f"goal position: {self.episode_agent_vehicle_goal_position}",
                f"step number: {step_counter}",
                f"current velocity: {self.episode_agent_vehicle_speeds[-1] * 3.6:.2f}km/h",
                f"previous action: {'DEC' if self.episode_actions[-1] is Action.DECELERATE else 'MAIN' if self.episode_actions[-1] is Action.MAINTAIN else 'ACC'}",
                f"previous reward: {self.episode_rewards[-1]}"
            ]
            # cv2 can't print multiple lines in a single putText() call...
            for index, line in enumerate(lines):
                cv2.putText(
                    birdview_car_intention,  # image
                    line,  # text
                    # offset line position by height of text
                    (0, self.birdview_car_intention_producer.target_size.height // 8 + index * 2 * label_height),
                    # position
                    cv2.FONT_HERSHEY_SIMPLEX,  # font
                    0.3,  # font scale
                    (255, 255, 255),  # font color
                    1,  # line-thickness
                    2  # line type
                )
            # display car inttention image
            cv2.namedWindow(winname='CarIntention')
            cv2.imshow('CarIntention', birdview_car_intention)
            # stop simulation until manual key stroke is registered
            cv2.waitKey()

        return birdview_car_intention

    # define hitbox of the car; used for reward calculation
    def in_rectangle(self, x, y, degrees, ped_x, ped_y, front_margin=1.5, side_margin=0.5, back_margin=0.5):
        # pedestrian position is none if the pedestrian is not observable,
        # i.e. outside a circle with a 50m radius centered around the agent
        if ped_x is None or ped_y is None:
            return False
        radians = degrees_to_radians(degrees)

        # TOP LEFT VERTEX:
        top_left_x = x + ((side_margin + Config.Carla.EGO_VEHICLE_WIDTH / 2) * np.sin(radians)) + \
                     ((front_margin + Config.Carla.EGO_VEHICLE_LENGTH / 2) * np.cos(radians))
        top_left_y = y - ((side_margin + Config.Carla.EGO_VEHICLE_WIDTH / 2) * np.cos(radians)) + \
                     ((front_margin + Config.Carla.EGO_VEHICLE_LENGTH / 2) * np.sin(radians))

        # TOP RIGHT VERTEX:
        top_right_x = x - ((side_margin + Config.Carla.EGO_VEHICLE_WIDTH / 2) * np.sin(radians)) + \
                      ((front_margin + Config.Carla.EGO_VEHICLE_LENGTH / 2) * np.cos(radians))
        top_right_y = y + ((side_margin + Config.Carla.EGO_VEHICLE_WIDTH / 2) * np.cos(radians)) + \
                      ((front_margin + Config.Carla.EGO_VEHICLE_LENGTH / 2) * np.sin(radians))

        # BOTTOM LEFT VERTEX:
        bot_left_x = x + ((side_margin + Config.Carla.EGO_VEHICLE_WIDTH / 2) * np.sin(radians)) - \
                     ((back_margin + Config.Carla.EGO_VEHICLE_LENGTH / 2) * np.cos(radians))
        bot_left_y = y - ((side_margin + Config.Carla.EGO_VEHICLE_WIDTH / 2) * np.cos(radians)) - \
                     ((back_margin + Config.Carla.EGO_VEHICLE_LENGTH / 2) * np.sin(radians))

        # BOTTOM RIGHT VERTEX:
        bot_right_x = x - ((side_margin + Config.Carla.EGO_VEHICLE_WIDTH / 2) * np.sin(radians)) - \
                      ((back_margin + Config.Carla.EGO_VEHICLE_LENGTH / 2) * np.cos(radians))
        bot_right_y = y + ((side_margin + Config.Carla.EGO_VEHICLE_WIDTH / 2) * np.cos(radians)) - \
                      ((back_margin + Config.Carla.EGO_VEHICLE_LENGTH / 2) * np.sin(radians))

        ab = [top_right_x - top_left_x, top_right_y - top_left_y]
        am = [ped_x - top_left_x, ped_y - top_left_y]
        bc = [bot_right_x - top_right_x, bot_right_y - top_right_y]
        bm = [ped_x - top_right_x, ped_y - top_right_y]

        is_in_rectangle = 0 <= np.dot(ab, am) <= np.dot(ab, ab) and 0 <= np.dot(bc, bm) <= np.dot(bc, bc)

        log_debug(
            f"agent vehicle corner positions:\n"
            f"\t- center: ({x:.2f},{y:.2f})\n"
            f"\t- theta: {degrees:.2f}\n"
            f"\t- front-margin: {front_margin:.2f}, side-margin: {side_margin:.2f}, back-margin: {back_margin:.2f}\n"
            f"\t- top-left & top-right: ({top_left_x:.2f},{top_left_y:.2f}) & ({top_right_x:.2f},{top_right_y:.2f})\n"
            f"\t- bottom-left & bottom-right: ({bot_left_x:.2f},{bot_left_y:.2f}) & ({bot_right_x:.2f},{bot_right_y:.2f})\n"
            f"pedestrian position:\n"
            f"\t- (x,y): ({ped_x:.2f},{ped_y:.2f})\n"
            f"is_in_rectangle: {'TRUE' if is_in_rectangle else 'FALSE'}"
        )

        return is_in_rectangle

    def linmap(self, a, b, c, d, x):
        return (x - a) / (b - a) * (d - c) + c

    def get_reward_akash(self, step_counter: int):
        # episode actions
        if not isinstance(self.episode_actions, list):
            raise TypeError(
                f"Invalid episode actions buffer type: Expected 'list', got '{type(self.episode_actions)}'."
            )
        if not isinstance(self.episode_actions[-1], Action):
            raise TypeError(
                f"Invalid previous action type: Expected '{Action}', got '{type(self.episode_actions[-1])}'."
            )
        # + 1 because of first dummy action (maintain) and current action is needed for calculating reward
        if len(self.episode_actions) != step_counter + 1:
            raise ValueError(
                f"Invalid number of episode actions: Expected {step_counter + 1}, got {len(self.episode_actions)}."
            )

        # episode car speeds
        if not isinstance(self.episode_agent_vehicle_speeds, list):
            raise TypeError(
                f"Invalid episode car speeds buffer type: Expected 'list', got '{type(self.episode_agent_vehicle_speeds)}'."
            )
        if not isinstance(self.episode_agent_vehicle_speeds[-1], float):
            raise TypeError(
                f"Invalid previous car speed type: Expected '{float}', got '{type(self.episode_agent_vehicle_speeds[-1])}'."
            )
        # because self.get_current_observation() must be called before self.get_reward()
        if len(self.episode_agent_vehicle_speeds) != step_counter:
            raise ValueError(
                f"Invalid numer of episode agent vehicle speeds: "
                f"Expected {step_counter}, got {len(self.episode_agent_vehicle_speeds)} instead."
            )

            # episode carla vehicle controls
        if not isinstance(self.episode_controls, list):
            raise TypeError(
                f"Invalid episode vehicle controls buffer type: Expected 'list', got '{type(self.episode_controls)}'."
            )
        if not isinstance(self.episode_controls[-1], type(carla.VehicleControl())):
            raise TypeError(
                f"Invalid previous vehicle control type: "
                f"Expected '{carla.VehicleControl()}', got '{type(self.episode_controls[-1])}'."
            )
            # because control will be set before calling reward function
        if len(self.episode_controls) != step_counter:
            raise ValueError(
                f"Invalid numer of episode vehicle control objects: "
                f"Expected {step_counter}, got {len(self.episode_controls)} instead."
            )

        # because of first dummy reward (0.0)
        if len(self.episode_rewards) != step_counter:
            raise ValueError(
                f"Invalid number of episode rewards: Expected {step_counter}, got {len(self.episode_rewards)}."
            )

        terminal = False
        goal = False
        reward = 0
        max_speed = 50  # in kmph
        hit_penalty = 100
        near_miss_penalty = 10
        goal_reward = 200
        braking_penalty = 1
        over_speeding_penalty = 10

        current_observation = self.episode_observations[-1]

        agent_vehicle_speed = current_observation[2] * 3.6

        agent_vehicle_goal_distance = np.linalg.norm(
            [current_observation[0] - self.episode_agent_vehicle_goal_position[0],
             current_observation[1] - self.episode_agent_vehicle_goal_position[1]]
        )

        if agent_vehicle_speed > 1.0:
            if agent_vehicle_speed <= 20:
                ped_hit = self.in_rectangle(
                    current_observation[0], current_observation[1],
                    current_observation[3],
                    current_observation[4], current_observation[5],
                    front_margin=1, side_margin=0.5
                )
            else:
                ped_hit = self.in_rectangle(
                    current_observation[0], current_observation[1],
                    current_observation[3],
                    current_observation[4], current_observation[5],
                    front_margin=2, side_margin=0.5)
            if ped_hit:
                # scale penalty by impact speed
                scaling = self.linmap(0, max_speed, 0, 1, min(agent_vehicle_speed, max_speed))  # in kmph
                # Different penal;ties for near miss and actual collisision.
                if self.world.collision_sensor.flag:
                    collision_reward = hit_penalty * (scaling + 0.1)
                else:
                    collision_reward = near_miss_penalty * (scaling + 0.1)
                reward -= collision_reward

        # TODO: How are these numbers calculated?
        # reward = -goal_dist / 1000
        reward -= pow(agent_vehicle_goal_distance / 4935.0, 0.8) * 1.2

        # All grid positions of incoming_car in player rectangle
        # Cost of collision with obstacles
        grid = self.grid_cost.copy()
        exo_agent_vehicle_positions = []
        # scenarios with parked or incoming cars
        if self.scenario[0] in [3, 7, 8, 10, 11, 12]:
            # each of the above scenarios has at least one parked car
            for parked_car_index in range(len(self.world.parked_cars)):
                exo_agent_vehicle_positions.append(
                    (self.world.parked_cars[parked_car_index].get_location().x,
                     self.world.parked_cars[parked_car_index].get_location().y)
                )
            # scenarios with incoming car
            if self.scenario[0] in [10, 11, 12]:
                exo_agent_vehicle_positions.append(
                    (self.world.incoming_car.get_location().x, self.world.incoming_car.get_location().y)
                )
            for (car_x, car_y) in exo_agent_vehicle_positions:
                # calculate hitbox of exo agent vehicle
                xmin = round(car_x - Config.Carla.EGO_VEHICLE_WIDTH / 2)
                xmax = round(car_x + Config.Carla.EGO_VEHICLE_WIDTH / 2)
                ymin = round(car_y - Config.Carla.EGO_VEHICLE_LENGTH / 2)
                ymax = round(car_y + Config.Carla.EGO_VEHICLE_LENGTH / 2)
                # all grid positions of incoming car
                for x in range(xmin, xmax):
                    for y in range(ymin, ymax):
                        grid[round(x), round(y)] = 100

        # cost of occupying road/non-road tile
        # Penalizing for hitting an obstacle
        location = [min(round(current_observation[0] - self.min_x), self.grid_cost.shape[0] - 1),
                    min(round(current_observation[1] - self.min_y), self.grid_cost.shape[1] - 1)]
        obstacle_cost = grid[location[0], location[1]]
        if obstacle_cost <= 100:
            reward -= (obstacle_cost / 20.0)
        elif obstacle_cost <= 150:
            reward -= (obstacle_cost / 15.0)
        elif obstacle_cost <= 200:
            reward -= (obstacle_cost / 10.0)
        else:
            reward -= (obstacle_cost / 0.22)

        # except for first state
        if len(self.episode_actions) > 1:
            # "heavily" penalize braking if you are already standing still
            if self.episode_actions[-1] is not Action.ACCELERATE and self.episode_agent_vehicle_speeds[-1] < 0.28:
                reward -= braking_penalty

        # limit maximum agent vehicle speed to 50 km/h == 13.88 m/s
        if self.episode_actions[-1] is Action.ACCELERATE and self.episode_agent_vehicle_speeds[
            -1] > Config.Carla.MAX_SPEED:
            reward -= over_speeding_penalty

        # if at least two actions have already been executed
        if len(self.episode_actions) > 1:
            if self.episode_actions[-1] is not Action.MAINTAIN:
                # penalize indecisive behaviour (if the last two actions are different)
                if self.episode_actions[-2] is not self.episode_actions[-1]:
                    reward -= 0.05

        reward -= pow(abs(self.episode_controls[-1].steer), 1.3) / 2.0

        if agent_vehicle_goal_distance < 3:
            reward += goal_reward
            goal = True

        # Normalize reward
        reward = reward / 1000.0

        collision = self.world.collision_sensor.flag or obstacle_cost > 50.0

        self.is_terminal_state = collision or goal

        nearmiss = self.in_rectangle(
            current_observation[0], current_observation[1],
            current_observation[3],
            current_observation[4], current_observation[5],
            front_margin=1.5, side_margin=0.5, back_margin=0.5)

        self.episode_rewards.append(reward)

        step_summary = {
            "reward": reward,
            "goal": goal,
            "collision": collision,
            "nearmiss": nearmiss,
            "terminal": self.is_terminal_state,
            "ped_observable": self.is_pedestrian_observable
        }
        log_debug(step_summary)
        return step_summary

    '''   
    # current observation but previous action
    # calculates the reward by detecting speeing, excessive steering, nearmisses and crashes
    def get_reward(self, step_counter: int):
        # episode actions
        if not isinstance(self.episode_actions, list):
            raise TypeError(
                f"Invalid episode actions buffer type: Expected 'list', got '{type(self.episode_actions)}'."
            )
        if not isinstance(self.episode_actions[-1], Action):
            raise TypeError(
                f"Invalid previous action type: Expected '{Action}', got '{type(self.episode_actions[-1])}'."
            )
        # + 1 because of first dummy action (maintain) and current action is needed for calculating reward
        if len(self.episode_actions) != step_counter + 1:
            raise ValueError(
                f"Invalid number of episode actions: Expected {step_counter + 1}, got {len(self.episode_actions)}."
            )

        # episode car speeds
        if not isinstance(self.episode_agent_vehicle_speeds, list):
            raise TypeError(
                f"Invalid episode car speeds buffer type: Expected 'list', got '{type(self.episode_agent_vehicle_speeds)}'."
            )
        if not isinstance(self.episode_agent_vehicle_speeds[-1], float):
            raise TypeError(
                f"Invalid previous car speed type: Expected '{float}', got '{type(self.episode_agent_vehicle_speeds[-1])}'."
            )
        # because self.get_current_observation() must be called before self.get_reward()
        if len(self.episode_agent_vehicle_speeds) != step_counter:
            raise ValueError(
                f"Invalid numer of episode agent vehicle speeds: "
                f"Expected {step_counter}, got {len(self.episode_agent_vehicle_speeds)} instead."
            )    

        # episode carla vehicle controls
        if not isinstance(self.episode_controls, list):
            raise TypeError(
                f"Invalid episode vehicle controls buffer type: Expected 'list', got '{type(self.episode_controls)}'."
            )
        if not isinstance(self.episode_controls[-1], type(carla.VehicleControl())):
            raise TypeError(
                f"Invalid previous vehicle control type: "
                f"Expected '{carla.VehicleControl()}', got '{type(self.episode_controls[-1])}'."
            )      
        # because control will be set before calling reward function
        if len(self.episode_controls) != step_counter:
            raise ValueError(
                f"Invalid numer of episode vehicle control objects: "
                f"Expected {step_counter}, got {len(self.episode_controls)} instead."
            )

        # because of first dummy reward (0.0) 
        if len(self.episode_rewards) != step_counter:
            raise ValueError(
                f"Invalid number of episode rewards: Expected {step_counter}, got {len(self.episode_rewards)}."
            )

        reward = 0
        # total distance to the goal
        goal_dist = np.sqrt(
            (car_position[0] - scenario_goal_position[0])**2 + (car_position[1] - scenario_goal_position[1])**2
        )

        # debug code for testing correctness of in_rectangle()            
        walker_x = [90.99, 91.00, 91.01, # coming from left
                    92.00, 92.00, 92.00, # critical x values for testing coming from front
                    93.01, 93.00, 92.99, # coming fron right
                    92.00, 92.00, 92.00 # critical x values for testing coming from bottom
                    ] 
        walker_y = [160.00, 160.00, 160.00, # critical y values for testing coming from left 
                    157.90, 157.91, 157.92, # coming from front
                    160.00, 160.00, 160.00, # critical y values for testing coming from right 
                    162.10, 162.09, 162.08  # coming from bottom
                    ]
        for ped_x, ped_y in zip(walker_x, walker_y):
            self.in_rectangle(start[0], start[1], start[2], ped_x, ped_y)

        # was a pedestrian hit? nearmisses count as well
        nearmiss = self.in_rectangle(
            car_position[0], car_position[1], car_angle, pedestrian_position[0], pedestrian_position[1]
        )
        if nearmiss:
            # scale penalty by impact speed
            scaling = self.linmap(0, Config.Carla.MAX_SPEED, 0, 1, min(car_speed * 0.27778, Config.Carla.MAX_SPEED))  # in m/s
            collision_reward = Config.Carla.HIT_PENALTY * scaling
            # apply collision reward
            reward -= collision_reward

        # apply reward in proportion to goal distance?
        reward -= pow(goal_dist / 4935.0, 0.8) * 1.2

        # find grid location of vehicles
        grid = self.grid_cost.copy()
        exo_agent_vehicle_positions = []
        # scenarios with parked or incoming cars
        if self.scenario[0] in [3, 7, 8, 10, 11, 12]:
            # each of the above scenarios has at least one parked car
            for parked_car_index in range(len(self.world.parked_cars)):
                exo_agent_vehicle_positions.append(
                    (self.world.parked_cars[parked_car_index].get_location().x, 
                     self.world.parked_cars[parked_car_index].get_location().y)
                )
            # scenarios with incoming car
            if self.scenario[0] in [10, 11, 12]:
                exo_agent_vehicle_positions.append(
                    (self.world.incoming_car.get_location().x, self.world.incoming_car.get_location().y)
                )
            for (car_x, car_y) in exo_agent_vehicle_positions:
                # calculate hitbox of exo agent vehicle
                xmin = round(car_x - Config.Carla.EGO_VEHICLE_WIDTH / 2)
                xmax = round(car_x + Config.Carla.EGO_VEHICLE_WIDTH / 2)
                ymin = round(car_y - Config.Carla.EGO_VEHICLE_LENGTH / 2)
                ymax = round(car_y + Config.Carla.EGO_VEHICLE_LENGTH / 2)
                # all grid positions of incoming car
                for x in range(xmin, xmax):
                    for y in range(ymin, ymax):
                        grid[round(x), round(y)] = 100

        # cost of occupying road/non-road tile
        # penalizing for hitting an obstacle
        location = [min(round(car_position[0] - self.min_x), self.grid_cost.shape[0] - 1),
                    min(round(car_position[1] - self.min_y), self.grid_cost.shape[1] - 1)]
        obstacle_cost = grid[location[0], location[1]]
        if obstacle_cost <= 100:
            reward -= (obstacle_cost / 20.0)
        elif obstacle_cost <= 150:
            reward -= (obstacle_cost / 15.0)
        elif obstacle_cost <= 200:
            reward -= (obstacle_cost / 10.0)
        else:
            reward -= (obstacle_cost / 0.22)

        # except for first state
        if len(self.episode_actions) > 1:
            # "heavily" penalize braking if you are already standing still
            if self.episode_actions[-1] is not Action.ACCELERATE and self.episode_agent_vehicle_speeds[-1] == 0.0:
                reward -= Config.Carla.BRAKING_PENALTY

        # limit maximum agent vehicle speed to 50 km/h == 13.88 m/s
        if self.episode_actions[-1] is Action.ACCELERATE and self.episode_agent_vehicle_speeds[-1] > Config.Carla.MAX_SPEED:
            reward -= Config.Carla.BRAKING_PENALTY

        # if at least two actions have already been executed
        if len(self.episode_actions) >= 2:
            # penalize indecisive behaviour (if the last two actions are different)
            if self.episode_actions[-2] is not self.episode_actions[-1]:
                reward -= 0.05

        # penalize steering
        reward -= pow(abs(self.episode_controls[-1].steer), 1.3) / 2.0

        # tolerance threshold for reaching goal position
        if goal_dist < 3:
            # apply goal reward
            reward += Config.Carla.GOAL_REWARD
            goal = True

        # normalize reward
        reward = reward / 1000.0

        # remember reward for purposes of debugging
        self.step_reward = reward

        # was a collision or nearmiss registered?
        collision = self.world.collision_sensor.flag or obstacle_cost > 50.0
        self.is_terminal_state = goal or collision

        self.episode_rewards.append(reward)

        step_summary = {
            "reward": reward,
            "goal": goal,
            "collision": collision,
            "nearmiss": nearmiss,
            "terminal": self.is_terminal_state,
            "ped_observable": self.is_pedestrian_observable
        }
        return step_summary
    '''

    def get_local_coordinates(self, path):
        world_to_camera = np.array(self.world.semseg_sensor.sensor.get_transform().get_inverse_matrix())
        world_points = np.array(path)
        world_points = world_points[:, :2]
        world_points = np.c_[world_points, np.zeros(world_points.shape[0]), np.ones(world_points.shape[0])].T
        sensor_points = np.dot(world_to_camera, world_points)
        point_in_camera_coords = np.array([
            sensor_points[1],
            sensor_points[2] * -1,
            sensor_points[0]])

        segcam_fov = float(Config.Carla.SEGCAM_FOV)
        focal = int(Config.Carla.SEGCAM_IMAGE_WIDTH) / (2.0 * np.tan(segcam_fov * np.pi / 360.0))
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = int(Config.Carla.SEGCAM_IMAGE_WIDTH) / 2.0
        K[1, 2] = int(Config.Carla.SEGCAM_IMAGE_HEIGHT) / 2.0
        points_2d = np.dot(K, point_in_camera_coords)
        points_2d = np.array([
            points_2d[0, :] / points_2d[2, :],
            points_2d[1, :] / points_2d[2, :],
            points_2d[2, :]])
        points_2d = points_2d.T
        points_in_canvas_mask = \
            (points_2d[:, 0] > 0.0) & (points_2d[:, 0] < int(Config.Carla.SEGCAM_IMAGE_WIDTH)) & \
            (points_2d[:, 1] > 0.0) & (points_2d[:, 1] < int(Config.Carla.SEGCAM_IMAGE_HEIGHT)) & \
            (points_2d[:, 2] > 0.0)
        points_2d = points_2d[points_in_canvas_mask]
        u_coord = points_2d[:, 0].astype(int)
        v_coord = points_2d[:, 1].astype(int)
        return u_coord, v_coord

    # path predicition with hard coding incoming car path prediction
    def get_path_simple(self, start, end, obstacles):
        car_velocity = self.vehicle.get_velocity()
        car_speed = np.sqrt(car_velocity.x ** 2 + car_velocity.y ** 2) * 3.6
        yaw = start[2]

        updated_risk_cmp = np.copy(self.risk_cmp)
        for pos in obstacles:
            pos = (round(pos[0]), round(pos[1]))
            updated_risk_cmp[pos[0] + 10, pos[1] + 10] = 10000

        if len(self.ped_history) >= 15:
            ped_path = np.array(self.ped_history)
            ped_path = ped_path[:, :2]
            ped_path = ped_path.reshape((15, 2))

            if Config.PREDICT_PEDESTRIAN_PATH:
                pedestrian_path = self.ped_pred.get_single_prediction(ped_path)
                for node in pedestrian_path:
                    updated_risk_cmp[round(node[0]), round(node[1])] = 10000

        if self.scenario[0] == 11:
            self.grid_cost[9:16, 13:] = 10000
            self.risk_cmp[10:13, 13:] = 10000
            x, y = round(self.world.incoming_car.get_location().x), round(self.world.incoming_car.get_location().y)
            # hard coding incoming car path prediction
            obstacles.append((x, y - 1))
            obstacles.append((x, y - 2))
            obstacles.append((x, y - 3))
            obstacles.append((x, y - 4))
            obstacles.append((x, y - 5))
            # all grid locations occupied by car added to obstacles
            for i in [-1, 0, 1]:
                for j in [-2, -1, 0, 1, 2]:
                    obstacles.append((x + i, y + j))

        if self.scenario[0] in [10, 1] and self.world.walker.get_location().y > start[1] and start[0] >= 2.5:
            end = (end[0], start[1] - 6, end[2])

        if Config.RISK_AWARE_PATH:
            path, risk = self.risk_path_planner.find_path_with_risk(
                start, end, self.grid_cost, obstacles, car_speed, yaw, updated_risk_cmp, True, self.scenario[0]
            )
        else:
            path = self.find_path(start, end, self.grid_cost, obstacles)
            risk = []
        return (path, risk)

    # wrapper function for HybridA* path planner
    # defines different starting points depending on scenario
    def find_path(self, start, end, costmap, obstacles):
        checkpoint = (92, 14, -90)
        # path computation is done in one go
        if self.scenario[0] != 9 or start[1] <= checkpoint[1]:
            t = time.time()
            paths = self.path_planner.find_path(start, end, costmap, obstacles)
            if len(paths):
                path = paths[0]
            else:
                path = []
            path.reverse()
        else:
            # path computation is split into two parts: start -> checkpoint -> end
            path_segemnt_1 = self.path_planner.find_path(start, checkpoint, costmap, obstacles)[0]
            path_segemnt_2 = self.path_planner.find_path(checkpoint, end, costmap, obstacles)[0]
            path_segemnt_2.reverse()
            path_segemnt_1.reverse()
            path = path_segemnt_1[:-1] + path_segemnt_2[1:]

        return path

    # returns all obstacles in the range of the car's sensors
    def get_obstacles(self, start):
        # update vehicle history
        self.car_history.append((self.vehicle.get_location().x, self.vehicle.get_location().y))


        obstacles = list()
        walker_x, walker_y = self.world.walker.get_location().x, self.world.walker.get_location().y
        walker_flag = False
        # walker_flag = True
        # is the pedestrian visible to the agent?
        if self.scenario[0] == 6:
            if walker_y > start[1]:
                walker_flag = True
        elif walker_y < start[1]:
            walker_flag = True
        # euclidean distance of pedestrian less than 50m to start?
        if np.sqrt((start[0] - walker_x) ** 2 + (start[1] - walker_y) ** 2) <= 120.0 and walker_flag:
            self.ped_history.append([walker_x, walker_y, self.world.walker.icr.value, self.world.walker.son.value])
            # pedestrain further away than incoming car
            if self.scenario[0] in [3, "04_non_int"] and walker_x >= self.world.incoming_car.get_location().x:
                obstacles.append((int(walker_x), int(walker_y)))
                self.is_pedestrian_observable = True
            # pedestrian is closer than incoming car
            elif self.scenario[0] in [7, 8, "05_non_int", "06_non_int"] and walker_x <= self.world.incoming_car.get_location().x:
                obstacles.append((int(walker_x), int(walker_y)))
                self.is_pedestrian_observable = True
            # any pedestrian is recognized as an obstacle
            elif self.scenario[0] in [1, 2, 4, 5, 6, 9, 10, "01_int", "02_int", "03_int", "04_int", "05_int", "06_int",
                                      "01_non_int", "02_non_int", "03_non_int", "04_non_int", "05_non_int",
                                      "06_non_int"]:
                obstacles.append((int(walker_x), int(walker_y)))
                self.pedestrian_observable = True
                obstacles.append((int(walker_x), int(walker_y)))
                self.is_pedestrian_observable = True
        if not walker_flag:
            self.is_pedestrian_observable = False
        if self.scenario[0] in [3, 7, 8, 10, "04_non_int", "05_non_int", "06_non_int"]:
            car_x, car_y = self.world.incoming_car.get_location().x, self.world.incoming_car.get_location().y
            # incoming car closer than 50m?
            if np.sqrt((start[0] - car_x) ** 2 + (start[1] - car_y) ** 2) <= 50.0:
                buffer = 0
                xmin = math.ceil(car_x - Config.Carla.EGO_VEHICLE_WIDTH / 2) - buffer
                xmax = math.ceil(car_x + Config.Carla.EGO_VEHICLE_WIDTH / 2) + buffer
                ymin = math.ceil(car_y - Config.Carla.EGO_VEHICLE_LENGTH / 2) - buffer
                ymax = math.ceil(car_y + Config.Carla.EGO_VEHICLE_LENGTH / 2) + buffer
                # discretize hitbox by iterating over length and width of the car in steps of 1
                # each step is counted as an obstacle
                for x in range(xmin, xmax + 1):
                    for y in range(ymin, ymax + 1):
                        obstacles.append((int(x), int(y)))
        # special case for scenario 11
        if self.scenario[0] == 11:
            self.is_pedestrian_observable = False
            car_x, car_y = self.world.incoming_car.get_location().x, self.world.incoming_car.get_location().y
            if np.sqrt((start[0] - car_x) ** 2 + (start[1] - car_y) ** 2) <= 50.0:
                buffer = 0
                xmin = math.ceil(car_x - Config.Carla.EGO_VEHICLE_WIDTH / 2) - buffer
                xmax = math.ceil(car_x + Config.Carla.EGO_VEHICLE_WIDTH / 2) + buffer
                ymin = math.ceil(car_y - Config.Carla.EGO_VEHICLE_LENGTH / 2) - buffer
                ymax = math.ceil(car_y + Config.Carla.EGO_VEHICLE_LENGTH / 2) + buffer
                for x in range(xmin, xmax + 1):
                    for y in range(ymin, ymax + 1):
                        obstacles.append((int(x), int(y)))
            # do the same for parked cars if there are any
            parked_cars = self.world.parked_cars
            if parked_cars is not None:
                for car in parked_cars:
                    car_x, car_y = car.get_location().x, self.world.incoming_car.get_location().y
                    if np.sqrt((start[0] - car_x) ** 2 + (start[1] - car_y) ** 2) <= 50.0:
                        buffer = 0
                        xmin = math.ceil(car_x - Config.Carla.EGO_VEHICLE_WIDTH / 2) - buffer
                        xmax = math.ceil(car_x + Config.Carla.EGO_VEHICLE_WIDTH / 2) + buffer
                        ymin = math.ceil(car_y - Config.Carla.EGO_VEHICLE_LENGTH / 2) - buffer
                        ymax = math.ceil(car_y + Config.Carla.EGO_VEHICLE_LENGTH / 2) + buffer
                        for x in range(xmin, xmax + 1):
                            for y in range(ymin, ymax + 1):
                                obstacles.append((int(x), int(y)))
        # special case for scenario 12
        if self.scenario[0] == 12:
            self.is_pedestrian_observable = False
            parked_car = self.world.parked_cars[0]
            px, py = round(parked_car.get_location().x), round(parked_car.get_location().y)
            for i in [-1, 0, 1]:
                for j in [-2, -1, 0, 1, 2]:
                    obstacles.append((px + i, py + j))

            car_x, car_y = self.world.incoming_car.get_location().x, self.world.incoming_car.get_location().y
            if np.sqrt((start[0] - car_x) ** 2 + (start[1] - car_y) ** 2) <= 50.0:
                for i in [-1, 0, 1]:
                    for j in [-2, -1, 0, 1, 2]:
                        obstacles.append((car_x + i, car_y + j))
        return obstacles