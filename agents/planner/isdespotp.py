from multiprocessing import Process

import carla
import psutil

from utils.logger import log_debug, log_info
from utils.utils import run_despot
from utils.config import Config, Action
from utils.connector import DespotBridge, Connection

from agents.learner.rlagent import RLAgent


class ISDespotP(RLAgent):
    def __init__(self, client, world, carla_map, scenario):
        super(ISDespotP, self).__init__(client, world, carla_map, scenario)

        # IS-DESPOT C++ process
        self.despot_process = None
        self.start_despot_process()

        # general control connection
        self.despot_connection = DespotBridge()
        self.establish_control_connection()


    # ================================================================================================= #
    #                                        IS-DESPOT SETUP METHODS                                    #
    # ================================================================================================= #
    def start_despot_process(self):
        if self.despot_process is not None:
            raise ValueError("Existing despot process must first be terminated before new invocation.")
        log_info(f"starting {Config.Despot.VARIANT} C++ process...")

        self.despot_process = Process(target=run_despot)      
        self.despot_process.start()


    def kill_despot_process(self):
        if self.despot_process == None:
            raise ValueError("No despot process to terminate.")
        if not isinstance(self.despot_process, Process):
            raise TypeError("Not a process.")
        log_info(f"killing {Config.Despot.VARIANT} C++ process...")

        parent_process = psutil.Process(self.despot_process.pid)
        for child_process in parent_process.children(recursive=True):
            log_info(f"killing C++ process {child_process.pid}")
            child_process.kill()
        parent_process.kill()
        log_info(f"killing python process {parent_process.pid}")
        self.despot_process = None


    def establish_control_connection(self):
        if self.despot_connection is None:
            raise TypeError("IS-DESPOT C++ process interface has not been initialized.")
        if not isinstance(self.despot_connection, DespotBridge):
            raise TypeError(f"Invalid IS-DESPOT C++ prcess interface: " 
                            f"Expected '{DespotBridge}', got '{type(self.despot_connection)}'.")
        
        log_info(f"setting up TCP connection for communication with {Config.Despot.VARIANT}...")
        # general control connection
        self.despot_connection.establish_connection(Connection.DESPOT)


    def close_connections(self):
        if self.despot_connection is None:
            raise TypeError("IS-DESPOT C++ process interface has not been initialized.")
        if not isinstance(self.despot_connection, DespotBridge):
            raise TypeError(f"Invalid IS-DESPOT C++ prcess interface: " 
                            f"Expected '{DespotBridge}', got {type(self.despot_connection)}.")

        self.despot_connection.close_connections()
        self.despot_connection = None


    # ================================================================================================= #
    #                                       IS-DESPOT CORE METHODS                                      #
    # ================================================================================================= #
    def get_speed_action(self, step_counter: int):
        if not isinstance(self.episode_rewards, list):
            raise TypeError(
                f"Invalid episode rewards buffer type: Expected 'list', got '{type(self.episode_rewards)}'."
            )
        # at this point we do not yet have calculated the reward of the current observation
        # because we first need IS-DESPOT's action to do so
        if len(self.episode_rewards) != step_counter:
            raise ValueError(
                f"Invalid number of episode rewards: Expected {step_counter}, got {len(self.episode_rewards)}."
            )
        if not isinstance(self.episode_rewards[-1], float):
            raise TypeError(
                f"Invalid previous reward type: Expected '{float}', got '{type(self.episode_rewards[-1])}'."
            )
        
        if not isinstance(self.episode_observations, list):
            raise TypeError(
                f"Invalid episode observations buffer type: Expected 'list', got '{type(self.episode_observations)}'."
            )
        if len(self.episode_observations) != step_counter:
            raise ValueError(
                f"Invalid number of episode observations: Expected {step_counter}, got {len(self.episode_observations)}."
            )
        if not isinstance(self.episode_observations[-1], tuple):
            raise TypeError(
                f"Invalid previous observation type: Expected '{tuple}', got '{type(self.episode_observations[-1])}'."
            )
            
        # reward of previous observation set by super(ISDespotP, self).get_reward()
        # or dummy reward in case of first step of episode
        previous_reward = self.episode_rewards[-1]
        
        # current observation set by super(ISDespotP, self).get_observation()
        current_observation = self.episode_observations[-1]

        self.despot_connection.send_observation(terminal=self.is_terminal_state,
                                                # IS-DESPOT needs the previous reward because it is required by HyLEAP's NN
                                                reward=previous_reward,
                                                car_position=[current_observation[0], current_observation[1]],
                                                car_speed=current_observation[2],
                                                angle=current_observation[3], 
                                                car_path=self.step_agent_vehicle_future_trajectory,
                                                pedestrian_visibility=self.is_pedestrian_observable,
                                                pedestrian_position=[current_observation[4], current_observation[5]])
        
        # query IS-DESPOT for action
        return self.despot_connection.receive_despot_simulation_result()


    def run_step(self, step_counter: int):
        self.vehicle = self.world.player
        transform = self.vehicle.get_transform()
        start = (self.vehicle.get_location().x, self.vehicle.get_location().y, transform.rotation.yaw)
        end = self.scenario[2]

        obstacles = super(ISDespotP, self).get_obstacles(start)
        (self.step_agent_vehicle_future_trajectory, risk) = super(ISDespotP, self).get_path_simple(start, end, obstacles)

        agent_vehicle_control = carla.VehicleControl(
            throttle=0.0, steer=0.0, brake=0.0, hand_brake=False, reverse=False, manual_gear_shift=False, gear=0
        )

        # buggy path planner returning empty path means we skip this step and just maintain current velocity
        if not len(self.step_agent_vehicle_future_trajectory):
            step_summary = {"skipped_step": True, "control": agent_vehicle_control}
            return step_summary

        super(ISDespotP, self).get_current_observation(step_counter)

        agent_vehicle_control.steer = (self.step_agent_vehicle_future_trajectory[2][2] - start[2]) / 70.0
        # best speed action for the given path
        despot_action, despot_value, despot_policy = self.get_speed_action(step_counter)

        # translate received action into CARLA car control commands
        if despot_action is Action.DECELERATE:
            agent_vehicle_control.brake = 0.6
        elif despot_action is Action.ACCELERATE:
            agent_vehicle_control.throttle = 0.6

        # remember episode
        self.episode_actions.append(despot_action)
        self.episode_controls.append(agent_vehicle_control)
        self.episode_despot_policies.append(despot_policy)

        step_summary = super(ISDespotP, self).get_reward_akash(step_counter)
        step_summary["skipped_step"] = False
        step_summary["control"] = agent_vehicle_control
        step_summary["action"] = despot_action

        return step_summary 


    def finalize_episode(self, episode_counter: int, terminated_early: bool):
        # this function is only intended for clean-up at the moment, 
        # i.e. if we have to synchronize Python & C++ after forcefully terminating an episode early
        if not terminated_early: return
    
        previous_observation = self.episode_observations[-1]
        # terminate episode in C++ by sending the previous oibservation again but with terminal flag
        self.despot_connection.send_observation(terminal=True,
                                                reward=self.episode_rewards[-1],
                                                car_position=[previous_observation[0], previous_observation[1]],
                                                car_speed=previous_observation[2],
                                                angle=previous_observation[3], 
                                                car_path=self.step_agent_vehicle_future_trajectory,
                                                pedestrian_visibility=self.is_pedestrian_observable,
                                                pedestrian_position=[previous_observation[4], previous_observation[5]])
            
        # receive & throw away IS-DESPOT's simulation results
        _, _, _ = self.despot_connection.receive_despot_simulation_result()

        # IS-DESPOT is now waiting for new episode to start
        return super().finalize_episode(episode_counter, terminated_early)