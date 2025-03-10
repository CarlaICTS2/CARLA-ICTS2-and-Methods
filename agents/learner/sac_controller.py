"""
Author: Dikshant Gupta
Time: 28.09.22 08:27
"""

import carla
from agents.learner.rlagent import RLAgent


class SAC(RLAgent):
    def __init__(self, world, carla_map, scenario, conn=None, eval_mode=False, agent='sac'):
        super(SAC, self).__init__(world, carla_map, scenario)
        self.agent = agent

    def run_step(self, debug=False):
        self.vehicle = self.client_world.player
        transform = self.vehicle.get_transform()
        start = (self.vehicle.get_location().x, self.vehicle.get_location().y, transform.rotation.yaw)
        end = self.scenario[2]

        # t = time.time()
        # Steering action on the basis of shortest and safest path(Hybrid A*)
        obstacles = self.get_obstacles(start)
        (path, risk), intention = self.get_path_simple(start, end, obstacles)
        # print("time taken: ", time.time() - t)

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
        self.prev_action = control
        return control, intention, risk, self.is_pedestrian_observable
