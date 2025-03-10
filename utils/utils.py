"""
Author: Dikshant Gupta
Time: 23.03.21 14:29
"""
import re
import carla
import math
import torch
# native python libraries
import os
import sys
import re
import time
import math
import socket
from pathlib import Path
from subprocess import Popen, CalledProcessError, PIPE
from multiprocessing import Queue

# third-party python libraries
import carla
import numpy as np
import torch
import psutil

# CARLA-CTS01 benchmark python libraries
from utils.config import Config, DespotVariant, Mode, Agent


def save_model(model_state_dict: dict,
               model_dir: str = Config.MODEL_DIR,
               checkpoint: str = "latest",
               name: str = None,
               number: int = 3):
    from utils.logger import log_info, log_debug

    if model_state_dict is None:
        raise ValueError("Model to save must not be empty.")
    if not isinstance(model_state_dict, dict):
        raise TypeError(f"Invalid model type: Expected 'dict', got '{type(model_state_dict)}'.")
    if model_dir is None:
        raise ValueError("Model directory to save into must not be empty.")
    if not isinstance(model_dir, str):
        raise TypeError(f"Invalid model directory type: Expected 'str', got '{type(model_dir)}'.")
    if checkpoint is None:
        raise ValueError("Model checkpoint directory to save into must not be empty.")
    if not isinstance(checkpoint, str):
        raise TypeError(f"Invalid model checkpoint directory type: Expected 'str', got '{type(checkpoint)}'.")
    if name is None: name = Config.AGENT.value
    if not isinstance(name, str):
        raise TypeError(f"Invalid name type: Expected 'str', got '{type(name)}'.")
    if ".pt" in name:
        raise ValueError(f"Invalid model name: Expected name without file type specification.")

    model_checkpoint_dir = os.path.join(model_dir, checkpoint)
    os.makedirs(model_checkpoint_dir, exist_ok=True)
    log_info(f"saving model '{name}' at checkpoint '{model_checkpoint_dir}'...")

    full_model_save_path = os.path.join(model_checkpoint_dir, name) + ".pth"
    torch.save(model_state_dict, full_model_save_path)

    # trim models to specified number
    all_models = sorted(Path(model_checkpoint_dir).iterdir(), key=os.path.getmtime)
    while len(all_models) > number:
        oldest_model = all_models.pop(0)
        log_debug(f"removing: {str(oldest_model)}")
        os.remove(oldest_model)


def load_model(model, model_dir: str = Config.MODEL_DIR, checkpoint: str = Config.MODEL_CHECKPOINT, key: str = None):
    from utils.logger import log_info

    if model is None:
        raise ValueError("Model to load into must not be empty.")
    if not isinstance(model, torch.nn.Module):
        raise TypeError("Can only load PyTorch models.")
    if model_dir is None:
        raise ValueError(f"Model directory to search must not be empty.")
    if not isinstance(model_dir, str):
        raise TypeError(f"Invalid model directory type: Expected 'str', got '{type(checkpoint)}'.")
    if checkpoint is None:
        raise ValueError("Model checkpoint directory to search must not be empty.")
    if not isinstance(checkpoint, str):
        raise TypeError(f"Invalid model checkpoint directory type: Expected 'str', got '{type(checkpoint)}'.")
    if key is None: key = Config.AGENT.value
    if not isinstance(key, str):
        raise TypeError(f"Invalid key type: Expected 'str', got '{type(key)}'.")

    model_checkpoint_dir = os.path.join(model_dir, checkpoint)
    if not os.path.exists(model_checkpoint_dir):
        raise ValueError(f"Model checkpoint directory '{model_checkpoint_dir}' doesn't exist.")
    log_info(f"loading model from {model_checkpoint_dir}...")

    # paths
    all_models = sorted(Path(model_checkpoint_dir).iterdir(), key=os.path.getmtime)
    if len(all_models) == 0: raise ValueError(f"Model checkpoint directory '{model_checkpoint_dir}' is empty.")

    filtered_models = [model for model in all_models if key in str(model)]
    if len(filtered_models) == 0:
        raise ValueError(f"No models matching '{key}' found in model checkpoint directory '{model_checkpoint_dir}'.")

    index = -1
    latest_model_loaded = False
    # latest model might be corrupt (interrupted during save operation)
    while not latest_model_loaded:
        latest_model_path = filtered_models[index]
        try:
            model.load_state_dict(torch.load(latest_model_path))
            latest_model_loaded = True
        except Exception as e:
            log_info(e)
            index -= 1
            if abs(index) > len(filtered_models): raise ValueError(f"All {abs(index)} models appear to be corrupted.")

    log_info(f"loaded latest model '{latest_model_path}' matching key '{key}' "
             f"created at {time.ctime(os.path.getctime(latest_model_path))} "
             f"and last modified at {time.ctime(os.path.getmtime(latest_model_path))}")
    return model


def scenarios_to_string():
    if Config.MODE is Mode.TRAIN:
        if len(Config.TRAIN_SCENARIOS) == 10:
            scenarios = "S-ALL"
        else:
            scenarios = ('S-' + ','.join([str(scenario) for scenario in Config.TRAIN_SCENARIOS])).strip()
    elif Config.MODE is Mode.VAL:
        if len(Config.VAL_SCENARIOS) == 12:
            scenarios = "S-ALL"
        else:
            scenarios = ('S-' + ','.join([str(scenario) for scenario in Config.VAL_SCENARIOS])).strip()
    elif Config.MODE is Mode.TEST:
        if len(Config.TEST_SCENARIOS) == 12:
            scenarios = "S-ALL"
        else:
            scenarios = ('S-' + ','.join([str(scenario) for scenario in Config.TEST_SCENARIOS])).strip()
    else:
        return "S-ALL"
    return scenarios


def run_despot() -> None:
    try:
        from utils.logger import log_info, log_debug, log_exception
        error_message = None

        # construct command depending on despot variant
        if Config.Despot.VARIANT is DespotVariant.IS_DESPOT:
            cd_cmd = "cd agents/planner/isdespot/build/carla_car"
            filename = "is_despot_carla_car"
            exec_cmd = f"./{filename} " + \
                       f"--port {Config.Despot.PORT} " + \
                       f"--scenario {scenarios_to_string()} " + \
                       f"--start_timestamp {Config.GLOBAL_START_TIME} " + \
                       f"--output_directory {Config.METRICS_DIR} " + \
                       f"--track_planning_effort {'true' if Config.Despot.TRACKING else 'false'} " + \
                       f"--agent {Config.AGENT if Config.AGENT not in [Agent.HyLEAR_C, Agent.HyLEAR_AUTOBOTS, Agent.HyLEAR_AUTOBOTS_C] else Agent.HyLEAR} " + \
                       f"--decouple {'true' if Config.HyLEAP.DECOUPLE else 'false'} " + \
                       f"--hacky {'true' if Config.HyLEAP.HACKY else 'false'} " + \
                       f"--no_vertical_pruning {'true' if Config.AGENT is Agent.HyPLAN and Config.MODE is Mode.TRAIN else 'false'} " + \
                       f"--aggressive_belief_updates {'true' if Config.Despot.AGGRESSIVE_BELIEF_UPDATES else 'false'} " + \
                       f"--minimal_noise {'true' if Config.Despot.MINIMAL_NOISE else 'false'} " + \
                       f"--no_importance_sampling {'true' if Config.Despot.NO_IMPORTANCE_SAMPLING else 'false'} " + \
                       f"--no_normalization {'true' if Config.Despot.NO_NORMALIZATION else 'false'} " + \
                       f"--timeout {Config.Despot.TIMEOUT} " + \
                       f"--time_per_planning_step {Config.Despot.TIME_PER_PLANNING_STEP} " + \
                       f"--max_episode_steps {Config.MAX_EPISODE_STEPS} " + \
                       f"--noise {Config.Despot.NOISE} " + \
                       f"--seed {Config.SEED} " + \
                       f"--max_search_depth {Config.Despot.MAX_SEARCH_DEPTH} " + \
                       f"--discount_factor {Config.Despot.DISCOUNT} " + \
                       f"--particle_number {Config.Despot.PARTICLE_NUMBER} " + \
                       f"--gap_reduction_rate {Config.Despot.GAP} " + \
                       f"--max_policy_simulation_length {Config.Despot.MAX_POLICY_SIM_LEN} " + \
                       f"--pruning_constant {Config.Despot.PRUNING_CONSTANT} "


        elif Config.Despot.VARIANT is DespotVariant.HYP_DESPOT:
            raise NotImplementedError(
                '''
                Only the Python/C++ interface of the plain HyP-DESPOT agent is missing. But no Hybrid agent will 
                run with it out of the box and the implementation of HyLEAP + HyPDESPOT is non-trivial. 
                It will require the Python code to run concurrently as well (e.g. belief state evaluation).
                '''
            )

        chmod_cmd = f"chmod +x {filename}"
        concat_cmd = " && "
        cmd = cd_cmd + concat_cmd + chmod_cmd + concat_cmd + exec_cmd
        log_info(f"executing {cmd}")
        # execute commands and log output of spawned child-process
        with Popen([cmd], shell=True, bufsize=1, stdout=PIPE, stderr=PIPE, text=True) as p:
            for line in p.stdout:
                # pretty printing
                line = line.strip("\n").replace("\b", "\n")
                if line: log_info(line)
            error_message = p.stderr.read()

        if p.returncode != 0:
            raise CalledProcessError(p.returncode, p.args)

    except:
        log_exception(*sys.exc_info(), kill_parent=True, error_message=error_message)


def find_free_port() -> int:
    port = None
    # find free port
    while True:
        port = np.random.randint(1024, 65535 + 1)
        # create a new tcp socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # try to bind to the specified port
        try:
            sock.bind(("127.0.0.1", port))
            # don't block socket by checking
            sock.close()
            break
        except socket.error:
            continue
    return port


def run_carla_server(carla_server_queue: Queue):
    from utils.logger import log_info

    # find free port
    port = find_free_port()

    # construct command for starting CARLA server
    cd_cmd = "cd /home/carla/ && "
    misc_cmd = "unset SDL_VIDEODRIVER && "
    carla_cmd = f"./CarlaUE4.sh -vulkan -RenderOffScreen -nosound -carla-rpc-port={port}"
    exec_cmd = cd_cmd + misc_cmd + carla_cmd

    log_info(f"executing {exec_cmd}")

    # running until crash
    carla_server_queue.put(port)
    # actually start CARLA server
    carla_server_process = Popen([exec_cmd], shell=True)
    # wait until termination, i.e. when we go beyond this line the server crashed
    carla_server_process.wait()
    # publish failure
    if carla_server_process.returncode != 0:
        log_info("CARLA server crashed - rebooting")
        carla_server_queue.put(port)


# an agents has stopped if it has been standing still for the past 50 steps
def has_agent_stopped(car_speeds: list, past_steps: int):
    from utils.logger import log_info, log_debug

    if not isinstance(car_speeds, list):
        raise TypeError(f"Invalid past car speeds type: Expected 'list', got '{type(car_speeds)}'.")
    if len(car_speeds) >= Config.MAX_EPISODE_STEPS:
        # log_debug("Episode reached maximum number of simulation steps. Terminating forcefully.")
        return True
    if len(car_speeds) < past_steps:
        # log_debug("Not enough simulation steps: Skipping agent-stop check.")
        return False

    stopped = True
    for car_speed in car_speeds[-past_steps:]:
        if car_speed > 0.1:
            stopped = False
            break
    return stopped


def kill_all_processes(kill_parent: bool):
    # kill all children processes
    this_process = psutil.Process(os.getpid())
    for child_process in this_process.children(recursive=True):
        print(f"killing child-process with pid {child_process.pid}")
        child_process.kill()

    # kill all parent processes
    if kill_parent:
        parent_process = psutil.Process(this_process.ppid())
        print(f"killing parent process with pid {parent_process.pid}")
        parent_process.kill()

    # kill current process
    print(f"killing current-process with pid {this_process.pid}")
    this_process.kill()


def l2_distance(pos1, pos2):
    '''
    :param pos1: (x,y) coordinates of position 1
    :param pos2: (x,y) coordinates of position 2
    :return: Euclidean distance between two points
    '''
    direction = pos1 - pos2
    direction_norm = math.sqrt(direction.x ** 2 + direction.y ** 2)
    return direction_norm


# taken from official CARLA tutorial at https://carla.readthedocs.io/en/latest/tuto_G_bounding_boxes/
def build_projection_matrix(w, h, fov, is_behind_camera=False):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)

    if is_behind_camera:
        K[0, 0] = K[1, 1] = -focal
    else:
        K[0, 0] = K[1, 1] = focal

    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


# taken from official CARLA tutorial at https://carla.readthedocs.io/en/latest/tuto_G_bounding_boxes/
def get_image_point(loc, K, w2c):
    # Calculate 2D projection of 3D coordinate

    # Format the input coordinate (loc is a carla.Position object)
    point = np.array([loc.x, loc.y, loc.z, 1])
    # transform to camera coordinates
    point_camera = np.dot(w2c, point)

    # New we must change from UE4's coordinate system to an "standard"
    # (x, y ,z) -> (y, -z, x)
    # and we remove the fourth component also
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    # now project 3D->2D using the camera matrix
    point_img = np.dot(K, point_camera)
    # normalize
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img[0:2]


# taken from official CARLA tutorial at https://carla.readthedocs.io/en/latest/tuto_G_bounding_boxes/
def point_in_canvas(pos, img_h, img_w):
    """Return true if point is in canvas"""
    if (pos[0] >= 0) and (pos[0] < img_w) and (pos[1] >= 0) and (pos[1] < img_h):
        return True
    return False


# !/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" Module with auxiliary functions. """


def draw_waypoints(world, waypoints, z=0.5):
    """
    Draw a list of waypoints at a certain height given in z.

        :param world: carla.world object
        :param waypoints: list or iterable container with the waypoints to draw
        :param z: height in meters
    """
    for wpt in waypoints:
        wpt_t = wpt.transform
        begin = wpt_t.location + carla.Location(z=z)
        angle = math.radians(wpt_t.rotation.yaw)
        end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
        world.debug.draw_arrow(begin, end, arrow_size=0.3, life_time=1.0)


def get_speed(vehicle):
    """
    Compute speed of a vehicle in Km/h.

        :param vehicle: the vehicle for which speed is calculated
        :return: speed as a float in Km/h
    """
    vel = vehicle.get_velocity()

    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)


def is_within_distance_ahead(target_transform, current_transform, max_distance):
    """
    Check if a target object is within a certain distance in front of a reference object.

    :param target_transform: location of the target object
    :param current_transform: location of the reference object
    :param orientation: orientation of the reference object
    :param max_distance: maximum allowed distance
    :return: True if target object is within max_distance ahead of the reference object
    """
    target_vector = np.array([target_transform.location.x - current_transform.location.x,
                              target_transform.location.y - current_transform.location.y])
    norm_target = np.linalg.norm(target_vector)

    # If the vector is too short, we can simply stop here
    if norm_target < 0.001:
        return True

    if norm_target > max_distance:
        return False

    fwd = current_transform.get_forward_vector()
    forward_vector = np.array([fwd.x, fwd.y])
    d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    return d_angle < 90.0


def is_within_distance(target_location, current_location, orientation, max_distance, d_angle_th_up, d_angle_th_low=0):
    """
    Check if a target object is within a certain distance from a reference object.
    A vehicle in front would be something around 0 deg, while one behind around 180 deg.

        :param target_location: location of the target object
        :param current_location: location of the reference object
        :param orientation: orientation of the reference object
        :param max_distance: maximum allowed distance
        :param d_angle_th_up: upper thereshold for angle
        :param d_angle_th_low: low thereshold for angle (optional, default is 0)
        :return: True if target object is within max_distance ahead of the reference object
    """
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = np.linalg.norm(target_vector)

    # If the vector is too short, we can simply stop here
    if norm_target < 0.001:
        return True

    if norm_target > max_distance:
        return False

    forward_vector = np.array(
        [math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
    d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    return d_angle_th_low < d_angle < d_angle_th_up


def compute_magnitude_angle(target_location, current_location, orientation):
    """
    Compute relative angle and distance between a target_location and a current_location

        :param target_location: location of the target object
        :param current_location: location of the reference object
        :param orientation: orientation of the reference object
        :return: a tuple composed by the distance to the object and the angle between both objects
    """
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = np.linalg.norm(target_vector)

    forward_vector = np.array([math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
    d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    return (norm_target, d_angle)


def distance_vehicle(waypoint, vehicle_transform):
    """
    Returns the 2D distance from a waypoint to a vehicle

        :param waypoint: actual waypoint
        :param vehicle_transform: transform of the target vehicle
    """
    loc = vehicle_transform.location
    x = waypoint.transform.location.x - loc.x
    y = waypoint.transform.location.y - loc.y

    return math.sqrt(x * x + y * y)


def vector(location_1, location_2):
    """
    Returns the unit vector from location_1 to location_2

        :param location_1, location_2: carla.Location objects
    """
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps

    return [x / norm, y / norm, z / norm]


def compute_distance(location_1, location_2):
    """
    Euclidean distance between 3D points

        :param location_1, location_2: 3D points
    """
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps
    return norm


def positive(num):
    """
    Return the given number if positive, else 0

        :param num: value to check
    """
    return num if num > 0.0 else 0.0


def degrees_to_radians(degrees):
    if -360.0 > degrees > 360.0:
        raise ValueError(f"Invalid angle (degrees) received: Expected value in [-360, 360], but got {degrees:.4f}.")

    radians = degrees * np.pi / 180

    if -2.0 > radians > 2.0:
        raise ValueError(f"Invalid angle (radians) received: Expected value in [-2, 2], but got {radians:.4f}.")

    return radians


# theta is given in degrees
# checked for correctness: ground-truth is the car intention image (birdview) generation module
# because they show the corners of the agent vehicle that are actually fed to the NN
# default values for margins gets collision corners of agent vehicle
def get_corners(center_x, center_y, degrees, point_x, point_y, front_margin=0.0, side_margin=0.0, back_margin=0.0):
    from utils.logger import log_info

    radians = degrees_to_radians(degrees)
    corners = []

    # TOP LEFT VERTEX:
    top_left_x = center_x + ((Config.Carla.EGO_VEHICLE_WIDTH / 2) * np.sin(radians)) + \
                 ((Config.Carla.EGO_VEHICLE_LENGTH / 2) * np.cos(radians))
    top_left_y = center_y - ((Config.Carla.EGO_VEHICLE_WIDTH / 2) * np.cos(radians)) + \
                 ((Config.Carla.EGO_VEHICLE_LENGTH / 2) * np.sin(radians))
    corners.append((top_left_x, top_left_y))

    # TOP RIGHT VERTEX:
    top_right_x = center_x - ((Config.Carla.EGO_VEHICLE_WIDTH / 2) * np.sin(radians)) + \
                  ((Config.Carla.EGO_VEHICLE_LENGTH / 2) * np.cos(radians))
    top_right_y = center_y + ((Config.Carla.EGO_VEHICLE_WIDTH / 2) * np.cos(radians)) + \
                  ((Config.Carla.EGO_VEHICLE_LENGTH / 2) * np.sin(radians))
    corners.append((top_right_x, top_right_y))

    # BOTTOM LEFT VERTEX:
    bot_left_x = center_x + ((Config.Carla.EGO_VEHICLE_WIDTH / 2) * np.sin(radians)) - \
                 ((Config.Carla.EGO_VEHICLE_LENGTH / 2) * np.cos(radians))
    bot_left_y = center_y - ((Config.Carla.EGO_VEHICLE_WIDTH / 2) * np.cos(radians)) - \
                 ((Config.Carla.EGO_VEHICLE_LENGTH / 2) * np.sin(radians))
    corners.append((bot_left_x, bot_left_y))

    # BOTTOM RIGHT VERTEX:
    bot_right_x = center_x - ((Config.Carla.EGO_VEHICLE_WIDTH / 2) * np.sin(radians)) - \
                  ((Config.Carla.EGO_VEHICLE_LENGTH / 2) * np.cos(radians))
    bot_right_y = center_y + ((Config.Carla.EGO_VEHICLE_WIDTH / 2) * np.cos(radians)) - \
                  ((Config.Carla.EGO_VEHICLE_LENGTH / 2) * np.sin(radians))
    corners.append((bot_right_x, bot_right_y))

    log_info(f"agent vehicle corner positions:\n"
             f"\t- center: ({center_x:.2f},{center_y:.2f})\n"
             f"\t- theta: {degrees:.2f}\n"
             f"\t- top-left & top-right: ({top_left_x:.2f},{top_left_y:.2f}) & ({top_right_x:.2f},{top_right_y:.2f})\n"
             f"\t- bottom-left & bottom-riht: ({bot_left_x:.2f},{bot_left_y:.2f}) & ({bot_right_x:.2f},{bot_right_y:.2f})\n")

    return corners


def normalized_columns_initializer(size, std=1.0):
    """
    Normalizing over a matrix.
    :param weights: given matrix
    :param std: standard deviation
    :return: normalized matrix
    """
    out = torch.randn(size)
    out *= std / torch.sqrt(out.pow(2).unsqueeze(0).sum(1).expand_as(out))
    return out


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p


def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def soft_update(target, source, tau):
    with torch.no_grad():
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * tau + param.data * (1.0 - tau))


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p


def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def soft_update(target, source, tau):
    with torch.no_grad():
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * tau + param.data * (1.0 - tau))


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
