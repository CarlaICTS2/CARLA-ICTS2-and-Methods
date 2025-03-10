"""
Author: Dikshant Gupta
Time: 10.11.21 01:14
"""
import math
import operator
import carla
import numpy as np
from multiprocessing import Process
import subprocess
from config import Config

from benchmark.rlagent import RLAgent
from ped_path_predictor.M3PC.train import M2P3CWrapper
from benchmark.risk.risk_aware_path import PathPlanner


def run_server():
    subprocess.run([f'./agents/planner/isdespot/build/carla_car/is_despot_carla_car --port {Config.despot_port} --scenario=blah2  --start_timestamp=12345'],
                   shell=True)

class HyLEARC(RLAgent):
    def __init__(self, world, carla_map, scenario, conn=None, eval_mode=False, agent='hylear'):
        super(HyLEARC, self).__init__(world, carla_map, scenario)

        self.conn = conn
        self.eval_mode = eval_mode
        self.agent = agent
        if not self.eval_mode:
            p = Process(target=run_server)
            p.start()
            self.conn.establish_connection()
            # m = self.conn.receive_message()
            # print(m)  # RESET

        self.ped_pred = M2P3CWrapper("_out/M2P3CWrapper/obs15_pred20/2024-09-12_18-31-41/model_184.pth")
        self.ped_pred.model.eval()
        self.observed_frame_num = 15
        self.predicting_frame_num = 20

        self.risk_path_planner = PathPlanner()

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

    def get_reward_despot(self, action):
        base_reward, goal, hit, nearmiss, terminal = super(HyLEARC, self).get_reward(action)
        reward = 0
        if goal:
            reward += 1.0
        return reward, goal, hit, nearmiss, terminal

    def get_speed_action(self, path, control):
        transform = self.vehicle.get_transform()
        start = (self.vehicle.get_location().x, self.vehicle.get_location().y, transform.rotation.yaw)
        walker_x, walker_y = self.world.walker.get_location().x, self.world.walker.get_location().y

        if self.prev_action is not None:
            reward, goal, hit, near_miss, terminal = self.get_reward_despot(self.prev_speed)
            terminal = goal or hit
        else:
            # handling first instance
            reward = 0
            terminal = False
        angle = transform.rotation.yaw
        car_pos = [self.vehicle.get_location().x, self.vehicle.get_location().y]
        car_velocity = self.vehicle.get_velocity()
        car_speed = np.sqrt(car_velocity.x ** 2 + car_velocity.y ** 2)
        pedestrian_positions = [[self.world.walker.get_location().x, self.world.walker.get_location().y]]

        if len(path) == 0:
            control.brake = 0.6
            self.prev_speed = 2
        elif not self.pedestrian_observable:
            control.throttle = 0.6
        else:
            reward = float(reward)
            self.conn.send_observation(terminal,
                                       reward,
                                       car_pos,
                                       car_speed,
                                       angle,
                                       path,
                                       self.pedestrian_observable,
                                       pedestrian_positions[-1],
                                       )
            m = self.conn.receive_despot_simulation_result()
            if m == "START":
                self.conn.send_message(terminal, reward, angle, car_pos, car_speed, pedestrian_positions, path)
                m = self.conn.receive_message()
            self.prev_speed = 1
            if m[0] == '0':
                control.throttle = 0.6
                self.prev_speed = 0
            elif m[0] == '2':
                control.brake = 0.6
                self.prev_speed = 2

        self.prev_action = control
        return control

    def run_step(self, debug=False):
        self.vehicle = self.world.player
        transform = self.vehicle.get_transform()
        start = (self.vehicle.get_location().x, self.vehicle.get_location().y, transform.rotation.yaw)
        end = self.scenario[2]

        # t = time.time()
        # Steering action on the basis of shortest and safest path(Hybrid A*)
        obstacles = self.get_obstacles(start)
        #if len(obstacles):
        #    (path, risk), intention = self.get_path_with_reasoning(start, end, obstacles)
        #else:
        #    (path, risk), intention = self.get_path_simple(start, end, obstacles)
        # print("time taken: ", time.time() - t)
        path = self.find_path(start, end, self.grid_cost, obstacles)
        _, intention = self.get_path_with_reasoning(start, end, obstacles)
        # intention = self.get_car_intention(obstacles, path, start)
        control = carla.VehicleControl()
        control.brake = 0.0
        control.hand_brake = False
        control.manual_gear_shift = False

        if len(path) == 0:
            control.steer = 0
        else:
            control.steer = (path[2][2] - start[2]) / 70.
        # print("Angle: ", control.steer)

        # Best speed action for the given path
        if not self.eval_mode:
            control = self.get_speed_action(path, control)
        self.prev_action = control
        return control, intention, 0, self.pedestrian_observable

    def get_path_with_reasoning(self, start, end, obstacles):
        car_velocity = self.vehicle.get_velocity()
        car_speed = np.sqrt(car_velocity.x ** 2 + car_velocity.y ** 2) * 3.6
        yaw = start[2]
        relaxed_sidewalk = self.grid_cost.copy()
        y = round(start[1])

        # Relax sidewalk
        sidewalk_cost = -1.0
        sidewalk_length = 20
        if self.scenario[0] in [1, 3, 4, 7, 8, 10]:
            relaxed_sidewalk[13:16, y - sidewalk_length: y + sidewalk_length] = sidewalk_cost
            relaxed_sidewalk[4:7, y - 10: sidewalk_length + sidewalk_length] = sidewalk_cost
            # TODO PAGI: ADD SCENARIO HERE
        elif self.scenario[0] in [2, 5, 6, 9, "01_int", "02_int", "03_int", "04_int", "05_int", "06_int", "01_non_int", "02_non_int", "03_non_int", "04_non_int", "05_non_int", "06_non_int"]:
            relaxed_sidewalk[94:97, y - sidewalk_length: y + sidewalk_length] = sidewalk_cost
            relaxed_sidewalk[103:106, y - sidewalk_length: y + sidewalk_length] = sidewalk_cost
        elif self.scenario[0] == 11:
            self.grid_cost[9:16, 13:] = 10000
            self.risk_cmp[10:13, 13:] = 10000
            relaxed_sidewalk = self.grid_cost.copy()
            relaxed_sidewalk[4:7, y - 10: sidewalk_length + sidewalk_length] = sidewalk_cost
            x, y = round(self.world.incoming_car.get_location().x), round(self.world.incoming_car.get_location().y)
            # Hard coding incoming car path prediction
            obstacles.append((x, y - 1))
            obstacles.append((x, y - 2))
            obstacles.append((x, y - 3))
            obstacles.append((x, y - 4))
            obstacles.append((x, y - 5))
            # All grid locations occupied by car added to obstacles
            for i in [-1, 0, 1]:
                for j in [-2, -1, 0, 1, 2]:
                    obstacles.append((x + i, y + j))
        elif self.scenario[0] == 12:
            relaxed_sidewalk[13:16, y - sidewalk_length: y + sidewalk_length] = sidewalk_cost
            relaxed_sidewalk[4:7, y - 10: sidewalk_length + sidewalk_length] = sidewalk_cost
            x, y = round(self.world.incoming_car.get_location().x), round(self.world.incoming_car.get_location().y)
            obstacles.append((x, y + 1))
            obstacles.append((x, y + 2))
            obstacles.append((x, y + 3))
            obstacles.append((x, y + 4))
            obstacles.append((x, y + 5))

        if len(self.ped_history) < 15 or not self.pedestrian_observable:
            if self.scenario[0] == 11 and self.world.incoming_car.get_location().y + 2 < start[1] and start[0] <= -2.5:
                end = (end[0], start[1] + 6, end[2])

            if self.scenario[0] in [10, 1] and self.world.walker.get_location().y > start[1] and start[0] >= 2.5:
                end = (end[0], start[1] - 6, end[2])
            path_normal = self.risk_path_planner.find_path_with_risk(start, end, self.grid_cost, obstacles, car_speed,
                                                                     yaw, self.risk_cmp, True, self.scenario[0])
            if path_normal[1] < 100 or not self.pedestrian_observable:
                return path_normal, self.get_car_intention([], path_normal[0], start)
            paths = [path_normal,
                     self.risk_path_planner.find_path_with_risk(start, end, relaxed_sidewalk, obstacles, car_speed,
                                                                yaw, self.risk_cmp, True, self.scenario[0])]  # Sidewalk relaxed
            path, risk = self.rulebook(paths, start)
            # print(path)
            return (path, risk/6), self.get_car_intention([], path, start)
        else:
            # Use path predictor
            ped_updated_risk_cmp = self.risk_cmp.copy()
            ped_path = np.array(self.ped_history)

            # slice cognitive information:
            ped_path = ped_path[:, :4]

            ped_path = ped_path.reshape((15, 4))
            pedestrian_path = self.ped_pred.get_single_prediction(ped_path)
            new_obs = [obs for obs in obstacles]
            pedestrian_path_d = list()
            for node in pedestrian_path:
                if (round(node[0]), round(node[1])) not in new_obs:
                    new_obs.append((round(node[0]), round(node[1])))
                    pedestrian_path_d.append((round(node[0]), round(node[1])))
            for pos in new_obs:
                ped_updated_risk_cmp[pos[0], pos[1]] = 10000

            path_normal = self.risk_path_planner.find_path_with_risk(start, end, self.grid_cost, obstacles, car_speed,
                                                                     yaw, ped_updated_risk_cmp, True, self.scenario[0])
            # print(len(new_obs), path_normal[1])
            if path_normal[1] < 1000:
                # print("normal!", path_normal[1], (path_normal[0][2][2] - path_normal[0][1][2]) / 70.0)
                return (path_normal[0], path_normal[1] / 6), self.get_car_intention(pedestrian_path_d, path_normal[0], start)
            # print(start, end, obstacles)
            paths = [path_normal,  # Normal
                     self.risk_path_planner.find_path_with_risk(start, end, self.grid_cost, new_obs, car_speed,
                                                                yaw, ped_updated_risk_cmp, True, self.scenario[0]),  # ped pred
                     self.risk_path_planner.find_path_with_risk(start, end, relaxed_sidewalk, obstacles, car_speed,
                                                                yaw, ped_updated_risk_cmp, True, self.scenario[0])]  # Sidewalk relaxed
                     # self.risk_path_planner.find_path_with_risk(start, end, relaxed_sidewalk, new_obs, car_speed,
                     #                                            yaw, ped_updated_risk_cmp, True)]  # Sidewalk relaxed + ped pred
            path, risk = self.rulebook(paths, start)
            # print(path[2][2] - start[2], path[2][2], start[2])
            return (path, risk/6), self.get_car_intention(pedestrian_path_d, path, start)

    @staticmethod
    def rulebook(paths, start):
        # No sidewalk
        data = []
        steer = []
        r = []
        for p in paths:
            path, risk = p
            len_path = len(path)
            if len_path == 0:
                lane = math.inf
            else:
                lane = sum([path[i][2] - path[i-1][2] for i in range(1, len_path)]) / len_path
            data.append((path, risk, lane, len_path))
            # r.append(risk)
            # steer.append((path[2][2] - start[2]) / 70.)

        # print("Rulebook!", r)
        # print("Steering angle: ", steer)
        data.sort(key=operator.itemgetter(1, 2, 3))
        path = data[0][0]
        risk = data[0][1]
        # print("Steering angle: ", (path[2][2] - start[2]) / 70.)
        return path, risk

    def get_reward(self, action):
        # print("HyLear")
        reward = 0
        goal = False
        terminal = False
        velocity = self.vehicle.get_velocity()
        speed = pow(velocity.x * velocity.x + velocity.y * velocity.y, 0.5) * 3.6  # in kmph
        transform = self.vehicle.get_transform()
        start = (self.vehicle.get_location().x, self.vehicle.get_location().y, transform.rotation.yaw)
        walker_x, walker_y = self.world.walker.get_location().x, self.world.walker.get_location().y
        end = self.scenario[2]
        goal_dist = np.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)

        if speed > 1.0:
            if speed <= 20:
                ped_hit = self.in_rectangle(start[0], start[1], start[2], walker_x, walker_y,
                                            front_margin=1, side_margin=0.5)
            else:
                ped_hit = self.in_rectangle(start[0], start[1], start[2], walker_x, walker_y,
                                            front_margin=2, side_margin=0.5)
            if ped_hit:
                # scale penalty by impact speed
                # hit = True
                scaling = self.linmap(0, Config.max_speed, 0, 1, min(speed * 0.27778, Config.max_speed))  # in m/s
                collision_reward = Config.hit_penalty * (scaling + 0.1)
                # if collision_reward >= 700:
                #     terminal = True
                reward -= collision_reward

        reward -= pow(goal_dist / 4935.0, 0.8) * 1.2
        #print("Goal",pow(goal_dist / 4935.0, 0.8) * 1.2)
        # All grid positions of incoming_car in player rectangle
        # Cost of collision with obstacles
        grid = self.grid_cost.copy()
        if self.scenario[0] in [3, 7, 8, 10]:
            car_x, car_y = self.world.incoming_car.get_location().x, self.world.incoming_car.get_location().y
            xmin = round(car_x - self.vehicle_width / 2)
            xmax = round(car_x + self.vehicle_width / 2)
            ymin = round(car_y - self.vehicle_length / 2)
            ymax = round(car_y + self.vehicle_length / 2)
            for x in range(xmin, xmax):
                for y in range(ymin, ymax):
                    grid[round(x), round(y)] = 100
            # print(xmin, xmax, ymin, ymax)
            # x = self.world.incoming_car.get_location().x
            # y = self.world.incoming_car.get_location().y
            # grid[round(x), round(y)] = 100

        # cost of occupying road/non-road tile
        # Penalizing for hitting an obstacle
        location = [min(round(start[0] - self.min_x), self.grid_cost.shape[0] - 1),
                    min(round(start[1] - self.min_y), self.grid_cost.shape[1] - 1)]
        location = self.path_planner.loc(start, grid)
        obstacle_cost = grid[location[0], location[1]]
        if obstacle_cost <= 100:
            reward -= (obstacle_cost / 20.0)
        elif obstacle_cost <= 150:
            reward -= (obstacle_cost / 15.0)
        elif obstacle_cost <= 200:
            reward -= (obstacle_cost / 10.0)
        else:
            reward -= (obstacle_cost / 0.22)

        # "Heavily" penalize braking if you are already standing still
        if self.prev_speed is not None:
            if action != 0 and self.prev_speed < 0.28:
                reward -= Config.braking_penalty

        # Limit max speed 
        if self.prev_speed is not None:
            if action == 0 and self.prev_speed > Config.max_speed:
                reward -= Config.braking_penalty

        # Penalize braking/acceleration actions to get a smoother ride
        if self.prev_action.brake > 0: last_action = 2
        elif self.prev_action.throttle > 0: last_action = 0
        else: last_action = 1
        if last_action != 1 and last_action != action:
            reward -= 0.05

        reward -= pow(abs(self.prev_action.steer), 1.3) / 2.0

        if goal_dist < 3:
            reward += Config.goal_reward
            goal = True
            terminal = True

        # Normalize reward
        reward = reward / 1000.0
        #print("Reward",reward)
        # hit = self.in_rectangle(start[0], start[1], start[2], walker_x, walker_y,
        #                         front_margin=0.2, side_margin=0.2, back_margin=0.1) or obstacle_cost > 50.0
        # hit = self.in_rectangle(start[0], start[1], start[2], walker_x, walker_y,
        #                         front_margin=0.01, side_margin=0.01, back_margin=0.01) or obstacle_cost > 50.0
        hit = self.world.collision_sensor.flag #or obstacle_cost > 50.0 # TODO removed since no obstacles
        nearmiss = self.in_rectangle(start[0], start[1], start[2], walker_x, walker_y,
                                     front_margin=1.5, side_margin=0.5, back_margin=0.5)
        return reward, goal, hit, nearmiss, terminal