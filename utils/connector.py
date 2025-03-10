import os
import sys
import socket
import struct

import numpy as np

from utils.config import Address, Config, StringEnum, Action
from utils.logger import log_debug, log_info, initialize_logging
from config import Config as old_Config


# changing these also requires changing the corresponding C++ code
class Connection(StringEnum):
    DESPOT: str = "/tmp/is-despot_connection_"
    LEADER: str = "/tmp/leader_connection_"
    HyLEAP_EVALUATION: str = "/tmp/hyleap_evaluation_connection_"
    HyPLAN_EVALUATION: str = "/tmp/hyplan_evaluation_connection_"


# tcp socket capable of receiving and sending messages
class DespotBridge:
    def __init__(self):
        self.inet_server: socket.socket = None
        self.unix_servers: dict = {}
        self.unix_connections: dict = {}

        # initialize_logging()

        if Config.Despot.PORT is None:
            raise TypeError("Invalid IS-DESPOT TCP port 'None'.")
        # there is always a general despot control connection if a DespotBridge is created
        self.despot_bind_path: str = f"{Connection.DESPOT}{Config.Despot.PORT}"

        self.boot_server()

    # starts a TCP sever to make it known to other processes that this port is occupied
    def boot_server(self) -> None:
        self.inet_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.inet_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_address = (Address.REMOTE.value, Config.Despot.PORT)
        # server connection is just a socket listening on a given ip address and port
        self.inet_server.bind(server_address)
        self.inet_server.listen()
        print(f"claimed address {server_address[0]}:{server_address[1]}")

    def close_server(self) -> None:
        if self.inet_server is None:
            raise TypeError("No AF_INET server to close.")

        print(f"killing TCP server listening at {Address.REMOTE}:{Config.Despot.PORTt}...")
        self.inet_server.close()
        self.inet_server = None

    # blocking function call
    def establish_connection(self, bind_path: Connection = None) -> None:
        # default bind path
        if bind_path is None: bind_path = Connection.DESPOT
        if not isinstance(bind_path, Connection): raise TypeError(
            f"Invalid bind path type: Expected 'Connection', got '{type(bind_path)}'.")
        bind_path = f"{bind_path}{Config.Despot.PORT}"
        if len(self.unix_connections) != 0 and self.unix_connections.get(bind_path) is not None:
            raise ValueError(f"Invalid bind path '{bind_path}': Already in use.")

        if os.path.exists(bind_path): os.remove(bind_path)
        unix_server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

        unix_server.bind(bind_path)
        unix_server.listen()
        print(f"TCP server listening at '{bind_path}'.")

        # wait for incoming connection request
        unix_connection, _ = unix_server.accept()
        print(f"TCP server successfully connected at '{bind_path}'.")

        # update state dictionaries
        self.unix_servers.update({bind_path: unix_server})
        self.unix_connections.update({bind_path: unix_connection})

    def close_connections(self) -> None:
        if self.unix_servers is None:
            raise TypeError("AF_UNIX servers have not been initialized.")
        if self.unix_connections is None:
            raise TypeError("AF_UNIX connections have not been initialized.")
        if len(self.unix_servers) == 0:
            raise ValueError("No AF_UNIX server to close.")
        if len(self.unix_connections) == 0:
            raise ValueError("No AF_UNIX connections to close.")

        print(f"closing connections between {Config.AGENT} Python process & IS-DESPOT C++ process...")
        for (connection_bind_path, unix_connection, unix_server) in zip(list(self.unix_connections.items()),
                                                                        list(self.unix_servers.values())):
            print(f"closing connection at {connection_bind_path}...")
            unix_connection.close()
            unix_server.close()

        self.unix_connections = None
        self.unix_servers = None

    # blocking call until end of message has been received
    def receive_exact_message(self, bind_path: Connection = None) -> str:
        if self.unix_connections is None:
            raise TypeError("AF_UNIX connections have not been initialized.")
        if len(self.unix_connections) == 0:
            raise ValueError("No open AF_UNIX connections.")
        # default bind path
        if bind_path is None: bind_path = Connection.DESPOT
        if not isinstance(bind_path, Connection):
            raise TypeError(f"Invalid bind path type: Expected 'Connection', got '{type(bind_path)}'.")
        bind_path = f"{bind_path}{Config.Despot.PORT}"
        if self.unix_connections.get(bind_path) is None:
            raise ValueError(f"Invalid bind path '{bind_path}': No connection.")

        unix_connection: socket.socket = self.unix_connections[bind_path]

        # receive message length
        total_received_bytes: bytes = b""
        while (len(total_received_bytes) < struct.calcsize("@i")):
            received_bytes = unix_connection.recv(struct.calcsize("@i") - len(total_received_bytes))
            if received_bytes is None:
                raise socket.error("Error while receiving message: Client disconnected.")
            total_received_bytes += received_bytes

        # sanity check: all bytes for inferring message length received?
        if len(total_received_bytes) != struct.calcsize("@i"):
            raise ValueError(
                f"Invalid number of bytes received: Expected {struct.calcsize('@i')}, got {len(total_received_bytes)}.")

        # infer message length
        announced_bytes = struct.unpack("@i", total_received_bytes)[0]
        if not isinstance(announced_bytes, int):
            raise TypeError(f"Invalid message length type: Expected 'int', got '{type(announced_bytes)}'.")
        log_debug(f"{announced_bytes} bytes have been announced")

        # receive actual message
        total_received_bytes: bytes = b""
        while (len(total_received_bytes) < announced_bytes):
            received_bytes = unix_connection.recv(announced_bytes - len(total_received_bytes))
            if received_bytes is None:
                raise socket.error("Error while receiving message: Client disconnected.")
            total_received_bytes += received_bytes

        # sanity check: all bytes for inferring message received?
        if len(total_received_bytes) != announced_bytes:
            raise ValueError(
                f"Invalid number of bytes received: Expected {announced_bytes}, got {len(total_received_bytes)}.")
        log_debug(f"received bytes: {total_received_bytes}")

        # sanity check: can all bytes be converted into a valid sequence of doubles without leftovers?
        num_doubles: float = len(total_received_bytes) / struct.calcsize("@d")
        log_debug(f"number of double values {num_doubles}")
        if not num_doubles.is_integer():
            raise TypeError(f"Invalid byte to double conversion ratio: Got {len(total_received_bytes)} bytes, "
                            f"resulting in {num_doubles:.4f} double values "
                            f"(with each occupying {struct.calcsize('@d')} bytes.)")
        # actualy infer message
        return list(struct.unpack(f"@{int(num_doubles)}d", total_received_bytes))

    def receive_despot_simulation_result(self):
        message = self.receive_exact_message()
        if not isinstance(message, list):
            raise TypeError(f"Invalid message type for despot policy: Expected 'list', got '{type(message)}'.")
        if (len(np.array(message).shape) > 1) or np.array(message).shape[0] != 5:
            raise ValueError(f"Invalid message dimensions for IS-DESPOT simulation result:"
                             f"Expected (5,), got {np.array(message).shape}.")
        despot_policy = message[:3]
        if abs(sum(despot_policy) - 1) > 1e-6:
            raise ValueError(
                f"Invalid IS-DESPOT policy received: Expected sum()==1, got sum()=={sum(despot_policy):.6f}.")
        despot_action = message[3]
        if not message[3].is_integer():
            raise ValueError(f"Invalid IS-DESPOT action received: Expected one of (0.0, 1.0, 2.0), got {message[3]}.")
        despot_action = Action(despot_action)
        despot_value = message[4]

        return despot_action, despot_value, despot_policy

    # message format is as follows:
    # 1. Config.HyLEAP.LSTM_STATE_SIZE
    # 2. number of nodes
    # 3. for each node:
    #   3.1 car state (x, y, theta)
    #   3.2 pedestrian state (x, y)
    #   3.3 car velocity
    # 4. previous action
    # 5. previous reward
    # 6. past ego vehicle trajectory (simulated during planning for correctly drawing intention images)
    def receive_expanded_nodes(self, bind_path: Connection):
        message = self.receive_exact_message(bind_path)
        if not isinstance(message, list):
            raise TypeError(f"Invalid message type for expanded nodes: Expected 'list', got '{type(message)}'.")
        if (len(np.array(message).shape) > 1):
            raise ValueError(
                f"Invalid message dimensions for expanded nodes: Expected (n,), got {np.array(message).shape}.")
        num_nodes: float = message[Config.HyLEAP.LSTM_STATE_SIZE]
        if not num_nodes.is_integer():
            raise ValueError(f"Invalid number of nodes: Expected an integral number, got {num_nodes:.4f}.")
        num_nodes = int(num_nodes)

        # split message into its components
        lstm_hidden_state = np.array(message[:Config.HyLEAP.LSTM_STATE_SIZE // 2], np.float64)
        lstm_hidden_state = np.tile(lstm_hidden_state, num_nodes) \
            .reshape(num_nodes, Config.HyLEAP.LSTM_STATE_SIZE // 2)

        lstm_cell_state = np.array(message[Config.HyLEAP.LSTM_STATE_SIZE // 2:Config.HyLEAP.LSTM_STATE_SIZE],
                                   np.float64)
        lstm_cell_state = np.tile(lstm_cell_state, num_nodes) \
            .reshape(num_nodes, Config.HyLEAP.LSTM_STATE_SIZE // 2)

        # past trajectories of agent vehicle of each received node
        agent_vehicle_past_simulated_trajectories = []
        # equivalent to observations
        nodes = []
        # initial offset
        offset = Config.HyLEAP.LSTM_STATE_SIZE + 1
        # extract information associated with each received osbervation
        for node_index in range(num_nodes):
            # extract new observation
            node = message[offset:offset + Config.Carla.OBSERVATION_SIZE]
            nodes.append(node)
            # number of already traversed waypoints
            num_trajectory_entries = int(message[offset + Config.Carla.OBSERVATION_SIZE])
            # past trajectory of this node
            trajectory = []
            # traverse trajectory waypoints
            for index in range(int(offset + Config.Carla.OBSERVATION_SIZE + 1), \
                               int(offset + Config.Carla.OBSERVATION_SIZE + 1 + num_trajectory_entries), 2):
                trajectory.append((message[index], message[index + 1]))
            agent_vehicle_past_simulated_trajectories.append(trajectory)
            # update offset for next node
            offset = offset + Config.Carla.OBSERVATION_SIZE + 1 + num_trajectory_entries

            '''            
            # debug info
            log_debug(
                f"node #{node_index}:\n"
                f"\t- car (x, y, theta, vel): ({node[0]:.2f}, {node[1]:.2f}, {node[2]:.2f}, {node[5]:.2f})\n"
                f"\t- pedestrian (x, y): ({node[3]:.2f}, {node[4]:.2f})\n"
                f"\t- previous action: {node[6]:.2f}\n"
                f"\t- previous reward: {node[7]:.2f}\n"
                f"\t- past trajectory ({num_trajectory_entries}): {trajectory}"
            )
            '''

        # shape observations
        nodes = np.array(nodes, np.float64) \
            .reshape(num_nodes, Config.Carla.OBSERVATION_SIZE)

        # whether the expanded node is the root node
        if message[-1] == 1.0:
            is_root_node = True
            if num_nodes != 1:
                raise ValueError(f"Invalid message format for receiving expanded nodes: "
                                 f"Messages querying for the root node, can only contain the root node itself.")
        elif message[-1] == -1.0:
            is_root_node = False
        else:
            raise ValueError(f"Invalid message format for receiving expanded nodes: Expected last value to either be "
                             f"1.0 or -1.0, but got {message[-1]} instead.")

        return lstm_hidden_state, lstm_cell_state, nodes, agent_vehicle_past_simulated_trajectories, is_root_node

    def receive_belief_and_observation(self, bind_path: Connection):
        message = self.receive_exact_message(bind_path)
        if not isinstance(message, list):
            raise TypeError(f"Invalid message type for LEADER communication: Expected 'list', got '{type(message)}'.")
        if (len(np.array(message).shape) > 1):
            raise ValueError(
                f"Invalid message dimensions for LEADER communication: Expected (n,), got {np.array(message).shape}.")
        if (len(message) != Config.Leader.ATTENTION_SIZE + Config.Leader.OBSERVATION_SIZE):
            raise ValueError(
                f"Invalid number of message entries for LEADER communication: "
                f"Expected {Config.Leader.ATTENTION_SIZE + Config.Leader.OBSERVATION_SIZE}, got {len(message)}."
            )

        belief = message[:Config.Leader.ATTENTION_SIZE]
        observation = message[Config.Leader.ATTENTION_SIZE:]
        return belief, observation

    def send_exact_message(self, message: list, unix_connection: socket.socket) -> None:
        if not isinstance(unix_connection, socket.socket):
            raise TypeError(
                f"Invalid connection for sending message: Expected socket.socket, got '{type(unix_connection)}'.")
        if not isinstance(message, list):
            raise TypeError(f"Invalid message type: Expected 'list', got '{type(message)}'.")
        if len(np.array(message).shape) > 1:
            raise ValueError(
                f"Invalid number of dimensions for sending message: Expected (n,), got {np.array(message).shape}.")

        # convert message to bytes
        message_bytes: bytes = struct.pack(f"@{len(message)}d", *message)
        # announce number of bytes that constitute the message
        unix_connection.sendall(len(message_bytes).to_bytes(struct.calcsize("@i"), sys.byteorder))
        # send the actual message
        unix_connection.sendall(message_bytes)

    # method is only called by general is-despot control connection
    def send_observation(self,
                         terminal: bool,
                         reward: float,
                         car_position: list,
                         car_speed: float,
                         angle: float,
                         car_path: list,
                         pedestrian_visibility: bool,
                         pedestrian_position: list,
                         pedestrian_path: list = None) -> None:
        if self.unix_connections is None:
            raise TypeError("AF_UNIX connections have not been initialized.")
        if len(self.unix_connections) == 0:
            raise ValueError("No open AF_UNIX connections.")
        if self.unix_connections.get(self.despot_bind_path) is None:
            raise ValueError(f"Invalid bind path '{self.despot_bind_path}': No open connection.")
        if not isinstance(terminal, bool) and not isinstance(terminal, np.bool_):
            raise TypeError(f"Invalid type of parameter 'terminal': Expected 'bool', got '{type(terminal)}'.")
        if not isinstance(reward, float):
            raise TypeError(f"Invalid type of parameter 'reward': Expected 'float', got '{type(reward)}'.")
        if not isinstance(car_position, list):
            raise TypeError(f"Invalid type of parameter 'car_position': Expected 'list', got '{type(car_position)}'.")
        if len(np.shape(car_position)) > 1 or len(car_position) != 2:
            raise ValueError(
                f"Invalid dimensions of parameter 'car_position': Expected (2,), got {np.shape(car_position)}.")
        if not isinstance(angle, float):
            raise TypeError(f"Invalid type of parameter 'angle': Expected 'float', got '{type(angle)}'.")
        if not isinstance(car_path, list):
            raise TypeError(f"Invalid type of parameter 'car_path': Expected 'list', got '{type(car_path)}'.")
            # planned car path can be 0 for the last step of en episode that reached the goal
        # if len(np.shape(car_path)) != 2 or np.shape(car_path)[1] != 3:
        #    raise ValueError(f"Invalid dimensions of parameter 'car_path': Expected (n, 3), got {np.shape(car_path)}.")
        if not isinstance(pedestrian_visibility, bool) and not isinstance(pedestrian_visibility, np.bool_):
            raise TypeError(
                f"Invalid type of parameter 'pedestrian_visibility': Expected 'bool', got '{type(pedestrian_visibility)}'.")
        if not isinstance(pedestrian_position, list):
            raise TypeError(
                f"Invalid type of parameter 'pedestrian_position': Expected 'list', got '{type(pedestrian_position)}'.")
        if len(np.shape(pedestrian_position)) > 1 or len(pedestrian_position) != 2:
            raise ValueError(
                f"Invalid dimensions of parameter 'pedestrian_position': Expected (1, 2), got {np.shape(pedestrian_position)}.")
        if pedestrian_path is not None and not isinstance(pedestrian_path, list):
            raise TypeError(
                f"Invalid type of parameter 'pedestrian_path': Expected 'list', got '{type(pedestrian_path)}'.")
        if pedestrian_path is not None and (len(np.shape(pedestrian_path)) != 2 or np.shape(pedestrian_path)[1] != 2):
            raise ValueError(
                f"Invalid dimensions of parameter 'pedestrian_path': Expected (n, 2), got {np.shape(pedestrian_path)}.")
        if Config.Carla.NUM_PEDESTRIANS != 1:
            raise NotImplementedError("Invalid number of pedestrians: Missing C++/IS-DESPOT implementation.")

        unix_connection: socket.socket = self.unix_connections[self.despot_bind_path]

        message: list = []
        # meta information
        message.append(1.0) if terminal else message.append(-1.0)
        message.append(reward)

        # car information
        [message.append(car_coordinate) for car_coordinate in car_position]
        message.append(car_speed)
        message.append(angle)
        message.append(len(np.array(car_path).flatten()))
        # car path waypoint like (x, y, theta)
        [message.append(element) for waypoint in car_path for element in waypoint]

        # pedestrian information
        # pedestrian position like (x, y)
        if pedestrian_visibility:
            message.append(1.0)
            [message.append(pedestrian_coordinate) for pedestrian_coordinate in pedestrian_position]
        else:
            message.append(-1.0)
            [message.append(pedestrian_coordinate) for pedestrian_coordinate in [0.0, 0.0]]

        # predicted pedestrian path (only if enabled)
        if pedestrian_path is not None:
            message.append(1.0)
            [message.append(pedestrian_coordinate) for pedestrian_position in pedestrian_path for pedestrian_coordinate
             in pedestrian_position]
        else:
            message.append(-1.0)

        self.send_exact_message(message, unix_connection)

    # method used by hyleap and hyplan
    def send_drla_prediction(self, message: str, bind_path: Connection) -> None:
        if self.unix_connections is None:
            raise TypeError("AF_UNIX connections have not been initialized.")
        if len(self.unix_connections) == 0:
            raise ValueError("No open AF_UNIX connections.")
        if not isinstance(bind_path, Connection):
            raise TypeError(f"Invalid bind path type: Expected 'Connection', got '{type(bind_path)}'.")
        bind_path = f"{bind_path}{Config.Despot.PORT}"
        if self.unix_connections.get(bind_path) is None:
            raise ValueError(f"Invalid bind path '{bind_path}': No open connection.")

        unix_connection: socket.socket = self.unix_connections[bind_path]
        self.send_exact_message(message, unix_connection)

    # method used by leader
    # send is_weights [p(g_1), ..., p(g_n)] as a string "p(g_1),...,p(g_n)\n "
    def send_is_weights(self, message: list, bind_path: Connection):
        if self.unix_connections is None:
            raise TypeError("AF_UNIX connections have not been initialized.")
        if len(self.unix_connections) == 0:
            raise ValueError("No open AF_UNIX connections.")
        if not isinstance(bind_path, Connection):
            raise TypeError(f"Invalid bind path type: Expected 'Connection', got '{type(bind_path)}'.")
        bind_path = f"{bind_path}{Config.Despot.PORT}"
        if self.unix_connections.get(bind_path) is None:
            raise ValueError(f"Invalid bind path '{bind_path}': No open connection.")

        unix_connection: socket.socket = self.unix_connections[bind_path]
        self.send_exact_message(message, unix_connection)
