import subprocess
from collections import deque
from multiprocessing import Process
import carla
import numpy as np

from config import Config
from benchmark.rlagent import RLAgent
from ped_path_predictor.autobots.autobot_wrapper import AutoBotWrapperNew
from ped_path_predictor.autobots.autobot_wrapper_cognitive import AutoBotWrapperCog


def run_server():
    subprocess.run(['cd your path to isdespot && ./car {}'.format(
        Config.despot_port)], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    
class HyREALA2C_AUTOBOTS_COG(RLAgent):
    def __init__(self, world, carla_map, scenario, conn=None, eval_mode=False):
        super(HyREALA2C_AUTOBOTS_COG, self).__init__(world, carla_map, scenario)
        self.conn = conn
        self.eval_mode = eval_mode
        #self.agent = agent
        self.imitate = False
        self.vehicle_past = deque(list(), maxlen=60)


        if not self.eval_mode and not conn is None:
            p = Process(target=run_server)
            p.start()
            self.conn.establish_connection()
            m = self.conn.receive_message()
            print(m)  # RESET
        self.ped_pred = AutoBotWrapperCog(path="./ped_path_predictor/saved_models/60_80/autobots_cognitive.pth")


    def get_reward(self, action):
        #print("Hi A2C")
        reward = 0
        goal = False
        terminal = False

        velocity = self.vehicle.get_velocity()
        speed = pow(velocity.x * velocity.x + velocity.y * velocity.y, 0.5) * 3.6  # in kmph
        transform = self.vehicle.get_transform()
        start = (self.vehicle.get_location().x, self.vehicle.get_location().y, transform.rotation.yaw)
        walker_x, walker_y = self.world.walker.get_location().x, self.world.walker.get_location().y

        self.vehicle_past.append((self.vehicle.get_location().x, self.vehicle.get_location().y))

        end = self.scenario[2]
        initial = self.scenario[3]
        #print(initial)
        dist = np.sqrt((start[0] - initial[0]) ** 2 + (start[1] - initial[1]) ** 2)
        goal_dist = np.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)
        #print(dist)
        if speed > 1.0:
            other_agents = list()
            walker_x, walker_y = self.world.walker.get_location().x, self.world.walker.get_location().y
            other_agents.append((walker_x, walker_y))

            _, goal, hit, nearmiss, terminal = super(HyREALA2C_AUTOBOTS_COG, self).get_reward(action)
            dmin = min([np.sqrt((start[0] - x[0]) ** 2 + (start[1] - x[1]) ** 2) for x in other_agents])
            if dmin < 3.0:
                collision_reward = -0.1 - ((3.0-dmin))
                reward = collision_reward

        # "Heavily" penalize braking if you are already standing still
        if self.prev_speed is not None:
            if action != 0 and self.prev_speed < 0.28:
                reward -= Config.braking_penalty

        # Limit max speed to 30
        if self.prev_speed is not None:
            #print("1",self.prev_speed, Config.max_speed)
            if action == 0 and self.prev_speed > (Config.max_speed + 3*0.2778):
                # print("2",self.prev_speed, Config.max_speed)
                reward -= Config.too_fast

        if goal_dist < 3 or dist > 100:
            reward += Config.goal_reward
            goal = True
            terminal = True
            print("Dist in reward",dist)

        hit = self.in_rectangle(start[0], start[1], start[2], walker_x, walker_y,
                                front_margin=0, side_margin=0, back_margin=0)
        hit = self.world.collision_sensor.flag or hit #carla can be buggy sometimes
        
        if hit:
            reward -= Config.hit_penalty
            goal = False
            terminal = True

        too_close = self.in_rectangle(start[0], start[1], start[2], walker_x, walker_y,
                                     front_margin=2.0, side_margin=0.5, back_margin=0.5)
        
        if too_close and not hit:
            reward -= Config.nearmiss_penalty
        nearmiss = self.in_rectangle(start[0], start[1], start[2], walker_x, walker_y,
                                     front_margin=1.5, side_margin=0.5, back_margin=0.5)

        reward -= 0.1 #for not reaching the goal
        # Normalize reward
        reward = reward / 2000.0 #  reward scaling for gradients
        #print(reward)

        return reward, goal, hit, nearmiss, terminal


    def get_reward_for_exps_in_current_table(self, action):
        #print("Hyreal")
        reward = 0
        goal = False
        terminal = False

        velocity = self.vehicle.get_velocity()
        speed = pow(velocity.x * velocity.x + velocity.y * velocity.y, 0.5) * 3.6  # in kmph
        transform = self.vehicle.get_transform()
        start = (self.vehicle.get_location().x, self.vehicle.get_location().y, transform.rotation.yaw)
        walker_x, walker_y = self.world.walker.get_location().x, self.world.walker.get_location().y
        end = self.scenario[2]
        initial = self.scenario[3]
        #print(initial)
        dist = np.sqrt((start[0] - initial[0]) ** 2 + (start[1] - initial[1]) ** 2)
        goal_dist = np.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)
        #print(dist)
        if speed > 1.0:
            other_agents = list()
            walker_x, walker_y = self.world.walker.get_location().x, self.world.walker.get_location().y
            other_agents.append((walker_x, walker_y))
            #if self.scenario[0] in [3, 7, 8, 10]:
            #    car_x, car_y = self.world.incoming_car.get_location().x, self.world.incoming_car.get_location().y
            #    other_agents.append((car_x, car_y))

            #reward = -goal_dist / 1000
            _, goal, hit, nearmiss, terminal = super(HyREALA2C_AUTOBOTS_COG, self).get_reward(action)
            dmin = min([np.sqrt((start[0] - x[0]) ** 2 + (start[1] - x[1]) ** 2) for x in other_agents])
            if dmin < 3.0:
                collision_reward = -0.1 - ((3.0-dmin) / 2.0)
                reward -= collision_reward

        # "Heavily" penalize braking if you are already standing still
        #print(pow(goal_dist / 4935.0, 0.8) * 1.2)
        #print(Config.braking_penalty)
        if self.prev_speed is not None:
            if action != 0 and self.prev_speed < 0.28:
                reward -= Config.braking_penalty

        # Limit max speed to 30
        if self.prev_speed is not None:
            if action == 0 and self.prev_speed > (Config.max_speed + 3*0.2778):
                reward -= Config.too_fast

        # Penalize braking/acceleration actions to get a smoother ride
        #if self.prev_action.brake > 0:
        #    last_action = 2
        #elif self.prev_action.throttle > 0:
        #    last_action = 0
        #else:
        #    last_action = 1
        #if last_action != 1 and last_action != action:
        #    reward -= 0.05

        #reward -= pow(abs(self.prev_action.steer), 1.3) / 2.0

        if goal_dist < 3 or dist > 100:
            reward += Config.goal_reward
            goal = True
            terminal = True
            print("Dist in reward",dist)
        ped_hit = self.in_rectangle(start[0], start[1], start[2], walker_x, walker_y,
                                            front_margin=2, side_margin=0.5)
        if ped_hit:
            reward-=Config.hit_penalty
        reward -= 0.1 #for not reaching the goal
        # Normalize reward
        reward = reward / 2000.0 #  reward scaling for gradients
        #print(reward)
        hit = self.world.collision_sensor.flag #or obstacle_cost > 50.0
        nearmiss = self.in_rectangle(start[0], start[1], start[2], walker_x, walker_y,
                                     front_margin=1.5, side_margin=0.5, back_margin=0.5)
        return reward, goal, hit, nearmiss, terminal

    def run_step(self, debug=False):
        self.vehicle = self.world.player
        transform = self.vehicle.get_transform()
        start = (self.vehicle.get_location().x, self.vehicle.get_location().y, transform.rotation.yaw)
        end = self.scenario[2]

        # Steering action on the basis of shortest and safest path(Hybrid A*)
        obstacles = self.get_obstacles(start)
        #(path, risk), intention = self.get_path_simple(start, end, obstacles)
        path = self.find_path(start, end, self.grid_cost, obstacles)
        risk = 0
        intention = self.get_car_intention(obstacles, path, start)
        control = carla.VehicleControl()
        control.brake = 0.0
        control.hand_brake = False
        control.manual_gear_shift = False

        if len(path) == 0:
            control.steer = 0
        else:
            control.steer = (path[2][2] - start[2]) / 70.
        if self.imitate:
            control = self.get_speed_action(path, control)
        self.prev_action = control
        velocity = self.vehicle.get_velocity()
        self.prev_speed = pow(velocity.x * velocity.x + velocity.y * velocity.y, 0.5)
        return control, intention, risk, self.pedestrian_observable
    
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
            self.prev_planner_action = 2
        elif not self.pedestrian_observable:
            control.throttle = 0.6
            #self.prev_speed = 0
        else:
            self.conn.send_message(terminal, reward, angle, car_pos, car_speed, pedestrian_positions, path)
            m = self.conn.receive_message()
            if m == "START":
                self.conn.send_message(terminal, reward, angle, car_pos, car_speed, pedestrian_positions, path)
                m = self.conn.receive_message()
            self.prev_planner_action = 1
            if m[0] == '0':
                control.throttle = 0.6
                self.prev_planner_action = 0
            elif m[0] == '2':
                control.brake = 0.6
                self.prev_planner_action = 2
        self.prev_action = control
        return control
    
    def get_reward_despot(self, action):
        base_reward, goal, hit, nearmiss, terminal = super(HyREALA2C_AUTOBOTS_COG, self).get_reward(action)
        reward = 0
        if goal:
            reward += 1.0
        return reward, goal, hit, nearmiss, terminal
    

    def get_car_intention(self, obstacles, path, start):
        #print(path)
        #print(start)
        #print(self.world.semseg_sensor.array)
        #print(self.ped_history)
        ped = self.world.walker
        x,y = int(ped.get_location().x), int(ped.get_location().y)
        #print(x,y)
        self.past_trajectory.append(start)
        car_intention = self.world.semseg_sensor.array.copy()
        #print(car_intention[car_intention[x-2:x+2, y-2:y+2,0]])
        if len(path) == 0:
            return car_intention
        x, y = self.get_local_coordinates(path)
        car_intention[y, x, :] = 255.0  # overlay planned path on input with while line, color of pedestrian

        x, y = self.get_local_coordinates(self.past_trajectory)
        car_intention[y, x, :] = 0.0  # overlay past trajectory on input with black line
        if len(self.ped_history) >= self.observed_frame_num and len(self.vehicle_past) >= self.observed_frame_num:
            ped_path = np.array(self.ped_history)

            veh_past = np.array(self.vehicle_past)
            pedestrian_path = self.ped_pred.get_single_prediction(ped_path, veh_past)
            x,y = self.get_local_coordinates(pedestrian_path)
            car_intention[y,x,:] = np.array([220, 20, 60]) 
        # with open("_out/costmap_{}.pkl".format(start[1]), "wb") as file:
        #     pkl.dump(car_intention, file)
        # car_intention = np.transpose(car_intention, (2, 0, 1))
        # assert car_intention.shape[0] == 3
        #if self.plot_intention:
        #    plt.imshow(car_intention)
        #    self.fig.savefig("cp.png",dpi=1200)
        #    self.fig.clear()
        #    print("Done")
        #    time.sleep(1)
        return car_intention
