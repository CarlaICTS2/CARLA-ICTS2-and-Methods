import os
import sys
import json
import logging
from pathlib import Path
import traceback

from utils.utils import kill_all_processes, scenarios_to_string
from utils.config import Config, Mode, Agent, HYBRID_AGENTS

console_logger: logging.Logger = None
learning_metrics_log_path: str = None
performance_metrics_log_path: str = None
pedestrian_path_data_log_path: str = None
is_logging_initialized: bool = False

agent_to_config = {
    Agent.IS_DESPOT: Config.Despot,
    Agent.HYP_DESPOT: Config.Despot,
    Agent.NavA2C: Config.NavA2C,
    Agent.HyLEAR: Config.HyLEAR,
    Agent.HyLEAP: Config.HyLEAP,
    Agent.HyPLAN: Config.HyPLAN,
    Agent.LEADER: Config.Leader,
    Agent.HyLEAR_C: Config.HyLEAR,
    Agent.HyLEAR_AUTOBOTS: Config.HyLEAR,
    Agent.HyLEAR_AUTOBOTS_C: Config.HyLEAR
}


def initialize_logging():
    # reference variables defined outside of function scope
    global is_logging_initialized
    global console_logger
    global learning_metrics_log_path
    global performance_metrics_log_path
    global pedestrian_path_data_log_path

    # prevent adding multiple handlers
    if is_logging_initialized:
        log_debug("loggers already initialized")
        return

    # ============================================================ #
    #           CONSTRUCT OUTPUT DIRECTORY STRUCTURE               #
    # ============================================================ #
    all_output_directory = os.path.join(os.getcwd(), "output")

    # use command-line output directory if provided
    if Config.OUTPUT_DIR is not None:
        run_output_directory = os.path.join(all_output_directory, Config.OUTPUT_DIR)
    # otherwise construct informative directory name
    else:
        directory_name = f"{Config.AGENT}_" + \
                         f"{'hacky_' if Config.HyLEAP.HACKY else ''}" + \
                         f"{'decoupled_' if Config.HyLEAP.DECOUPLE else ''}" + \
                         f"{'num_forward_' + str(Config.HyPLAN.NUM_FORWARD_PASSES) + '_' if Config.AGENT is Agent.HyPLAN else ''}" + \
                         f"{'integrated_' if Config.HyPLAN.TRAIN_INTEGRATION else ''}" + \
                         f"{'improved_' if Config.HyPLAN.IMPROVE_REWARD else ''}" + \
                         f"{'NavA2C_' if Config.HyPLAN.NavA2C else ''}" + \
                         f"{str(Config.Despot.PARTICLE_NUMBER) + 'P' if Config.AGENT is not Agent.NavA2C else ''}_" + \
                         f"{'APP' if Config.RISK_AWARE_PATH else 'BPP'}_" + \
                         f"{'PPP' if Config.PREDICT_PEDESTRIAN_PATH else 'NPPP'}"

        run_output_directory = os.path.join(all_output_directory, directory_name)

    Config.OUTPUT_DIR = run_output_directory

    # create directories
    Config.METRICS_DIR = os.path.join(Config.OUTPUT_DIR, "metrics", f"{Config.MODE}")
    os.makedirs(Config.METRICS_DIR, exist_ok=True)

    Config.MODEL_DIR = os.path.join(Config.OUTPUT_DIR, "models")
    os.makedirs(Config.MODEL_DIR, exist_ok=True)

    Config.DEBUG_DIR = os.path.join(Config.OUTPUT_DIR, "debug")
    os.makedirs(Config.DEBUG_DIR, exist_ok=True)

    Config.DATA_DIR = os.path.join(Config.OUTPUT_DIR, "data")
    os.makedirs(Config.DATA_DIR, exist_ok=True)

    # learning metrics only during training
    if Config.MODE is Mode.TRAIN:
        learning_metrics_log_path = os.path.join(Config.METRICS_DIR,
                                                 f"learning_metrics_{Config.GLOBAL_START_TIME}.json")
        # log pedestrian paths during training
        if Config.RECORD_PEDESTRIAN_DATA:
            pedestrian_path_data_log_path = os.path.join(Config.DATA_DIR,
                                                         f"pedestrian_path_data_{Config.GLOBAL_START_TIME}.json")

    # always record performance metrics
    # differentiate between scenarios to enable parallel testing
    performance_metrics_log_path = os.path.join(Config.METRICS_DIR,
                                                "performance_metrics_" + \
                                                f"{scenarios_to_string()}_" + \
                                                f"{Config.GLOBAL_START_TIME}.json")

    # ============================================================ #
    #                   INITIALIZE CONSOLE LOGGER                  #
    # ============================================================ #
    # determine logging level for console logger
    log_level = logging.DEBUG if Config.VERBOSE else logging.INFO

    console_logger = logging.getLogger("console_logger")
    console_logger.setLevel(log_level)

    # disable propagation to root logger (prevents duplicate messages)
    console_logger.propagate = False

    # create console handler and set level to debug
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(log_level)

    # construct logfile path
    log_filename = f"console_log_{Config.GLOBAL_START_TIME}.txt"
    full_log_file_path = os.path.join(Config.DEBUG_DIR, log_filename)

    # create file handler and set level to debug
    file_handler = logging.FileHandler(filename=Path(full_log_file_path), mode="a")
    file_handler.setLevel(log_level)

    # output time, function and actual message
    console_Logger_formatter = logging.Formatter(
        fmt='[%(asctime)s, %(filename)s->%(funcName)s():%(lineno)s] %(message)s',
        # date format
        datefmt='%d.%m.%Y %I:%M:%S %p'
        )
    console_handler.setFormatter(console_Logger_formatter)
    file_handler.setFormatter(console_Logger_formatter)

    # log to file and console simultaneously,
    console_logger.addHandler(console_handler)
    console_logger.addHandler(file_handler)

    # log error messages to the same file
    sys.excepthook = log_exception
    log_info(f"logging console output at '{full_log_file_path}'")
    log_info(f"logging performance metrics at '{performance_metrics_log_path}'")
    if Config.MODE is Mode.TRAIN: log_info(f"logging learning metrics at '{learning_metrics_log_path}'")
    # done
    is_logging_initialized = True
    # ============================================================ #
    #                   DUMP TRAINING CONFIGURATION                #
    # ============================================================ #
    if Config.MODE is Mode.TRAIN:
        general_config = {key: value for key, value in vars(Config).items() if
                          not callable(value) and not key.startswith("__")}

        carla_config = {key: value for key, value in vars(Config.Carla).items() if
                        not callable(value) and not key.startswith("__")}

        agent_config = {key: value for key, value in vars(agent_to_config[Config.AGENT]).items() if
                        not callable(value) and not key.startswith("__")}

        despot_config = None
        # for all agents that use is-despot
        if Config.AGENT in HYBRID_AGENTS:
            despot_config = {key: value for key, value in vars(Config.Despot).items() if
                             not callable(value) and not key.startswith("__")}

        config_dict = {
            "config": general_config,
            "carla": carla_config,
            "agent": agent_config
        }

        if despot_config is not None: config_dict["despot"] = despot_config

        config_log_path = os.path.join(Config.OUTPUT_DIR, f"configuration_{Config.GLOBAL_START_TIME}.json")
        log_info(f"dumping training configuration at '{config_log_path}'")
        with open(config_log_path, "a") as file:
            file.write(json.dumps(config_dict, default=str, indent=2))


def log_exception(exc_type, exc_value, exc_traceback, kill_parent=False, error_message=None):
    message = "\n" + "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    if error_message: message + "\n" + error_message
    console_logger.critical(message)
    kill_all_processes(kill_parent)


def log_debug(message: str) -> None:
    console_logger.debug(message)


def log_info(message: str) -> None:
    console_logger.info(message)


def log_performance_metrics(message: dict) -> None:
    if performance_metrics_log_path is None:
        raise ValueError("Invalid performance metrics log path 'None'.")
    if not isinstance(message, dict):
        raise TypeError(f"Invalid message type: Expected 'dict', got '{type(message)}'.")

    with open(performance_metrics_log_path, "a") as file:
        file.write(json.dumps(message))
        file.write("\n")


def log_learning_metrics(message: dict) -> None:
    if Config.MODE is not Mode.TRAIN:
        raise ValueError(f"Can't log learning metrics during '{Config.MODE}'.")
    if learning_metrics_log_path is None:
        raise ValueError("Invalid learning metrics log path 'None'.")
    if not isinstance(message, dict):
        raise TypeError(f"Invalid message type: Expected 'dict', got '{type(message)}'.")

    with open(learning_metrics_log_path, "a") as file:
        file.write(json.dumps(message))
        file.write("\n")


def log_data(message: dict) -> None:
    if not Config.RECORD_PEDESTRIAN_DATA:
        return
    if Config.MODE is not Mode.TRAIN:
        raise ValueError(f"Can't log pedestrian data during '{Config.MODE}' mode.")
    if pedestrian_path_data_log_path is None:
        raise ValueError("Invalid pedestrian path data log path: 'None'.")
    if not isinstance(message, dict):
        raise TypeError(f"Invalid message type: Expected 'dict', got '{type(message)}'.")

    with open(pedestrian_path_data_log_path, "a") as file:
        file.write(json.dumps(message))
        file.write("\n")