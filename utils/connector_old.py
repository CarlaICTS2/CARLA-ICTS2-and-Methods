import socket
import struct

from utils.config import StringEnum


class Connection(StringEnum):
    DESPOT: str = "/tmp/is-despot_connection_"
    LEADER: str = "/tmp/leader_connection_"
    HyLEAP_EVALUATION: str = "/tmp/hyleap_evaluation_connection_"
    HyPLAN_EVALUATION: str = "/tmp/hyplan_evaluation_connection_"



class Connector:
    def __init__(self, port):
        self.port = port
        self.connection = None
        self.inet_server: socket.socket = None
        self.unix_servers: dict = {}
        self.unix_connections: dict = {}


    def establish_connection(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = ('localhost', self.port)
        sock.bind(server_address)
        sock.listen()
        print("Connect to port {}".format(self.port))
        self.connection, client_address = sock.accept()
        print("Connection to port {} established!".format(self.port))

    def receive_exact_message(self, bind_path: Connection = None) -> str:
        if self.unix_connections is None:
            raise TypeError("AF_UNIX connections have not been initialized.")
        if len(self.unix_connections) == 0:
            raise ValueError("No open AF_UNIX connections.")
        # default bind path
        if bind_path is None: bind_path = Connection.DESPOT
        if not isinstance(bind_path, Connection):
            raise TypeError(f"Invalid bind path type: Expected 'Connection', got '{type(bind_path)}'.")
        bind_path = f"{bind_path}{1245}"
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
        # log_debug(f"{announced_bytes} bytes have been announced")

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
        # log_debug(f"received bytes: {total_received_bytes}")

        # sanity check: can all bytes be converted into a valid sequence of doubles without leftovers?
        num_doubles: float = len(total_received_bytes) / struct.calcsize("@d")
        # log_debug(f"number of double values {num_doubles}")
        if not num_doubles.is_integer():
            raise TypeError(f"Invalid byte to double conversion ratio: Got {len(total_received_bytes)} bytes, "
                            f"resulting in {num_doubles:.4f} double values "
                            f"(with each occupying {struct.calcsize('@d')} bytes.)")
        # actualy infer message
        return list(struct.unpack(f"@{int(num_doubles)}d", total_received_bytes))

    def receive_message(self):
        message = ""
        while True:
            m = self.connection.recv(1024).decode('utf-8')
            # print("Received {}!".format(m))
            message += m
            if len(m) == 0:
                print(m)
            elif m[-1] == "\n":
                break
        return message

    def send_message(self, terminal, reward, angle, car_pos, car_speed,
                     pedestrian_positions, path, pedestrian_path=None):
        message = ""
        if terminal:
            temp = "true"
        else:
            temp = "false"
        message += temp + ";" + str(reward) + ";" + str(angle) + ";"
        message += str(car_pos[0]) + ";" + str(car_pos[1]) + ";" + str(car_speed) + ";"
        for pos in pedestrian_positions:
            if len(pos) == 0:
                message += ";" + ";"
            else:
                message += str(pos[0]) + ";" + str(pos[1]) + ";"  # Pedestrian position: (x, y)
        for wp in path:
            message += str(wp[0]) + "," + str(wp[1]) + "," + str(wp[2]) + ","  # Waypoint: (x, y, theta)

        message = message[:-1] + ";"
        if pedestrian_path is not None:
            for pos in pedestrian_path:
                message += str(pos[0]) + "," + str(pos[1]) + ","
        else:
            message += "null,"
        message = message[:-1] + "\n"
        message = message.encode('utf-8')
        self.connection.sendall(message)


if __name__ == '__main__':
    conn = Connector(1245)
    conn.establish_connection()
    conn.receive_message()
    for _ in range(10):
        conn.send_message(False, 0.0, 0.0, [100.0, 100.0], 8.5, [[89.0, 92.0]], [[101.0, 101.0, 0.0],
                                                                                 [102.0, 102.0, 0.0],
                                                                                 [103.0, 103.0, 0.0]])
        msg = conn.receive_message()
        print(msg[0] == '0', msg[0])
