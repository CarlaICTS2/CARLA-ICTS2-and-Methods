#ifndef CONNECTOR_DESPOT
#define CONNECTOR_DESPOT

#include <sys/un.h>
#include <netinet/in.h>
#include <thread>
#include "float.h"

#include "despot/solver/despot.h"
#include "despot/interface/default_policy.h"
#include "despot/util/logging.h"

#include "state.h"
#include "path.h"
#include "param.h"

using namespace std;

class Message {
    public:
        bool terminal;
        double reward;
        std::pair<double, double> car_position;
        double car_angle;
        double car_speed;
        Path car_path;
        bool pedestrian_in_scene;
        std::pair<double, double> pedestrian_position;
        bool pedestrian_path_prediction;
        std::vector<std::pair<double, double>> pedestrian_path;
        std::tuple<double, double, double> obstacle;

        void print() {
            logc << "Terminal: " << terminal << endl;
            logc << "Reward: " << reward << endl;
            logc << "Car (x, y, speed, angle): (" 
                           << car_position.first << ", " 
                           << car_position.second << ", " 
                           << car_speed << ", "
                           << car_angle << ")" << endl;
            logc << "Car path [(x, y, angle), ...]: [";
            for (int i = 0; i < car_path.size(); i++) {
                logc << "(" << car_path[i].x
                     << ", " << car_path[i].y
                     << ", " << car_path[i].theta
                     << ")";
                if (i != car_path.size()-1) logc << " ,";
            }
            logc << "]" << endl;
            if (pedestrian_in_scene) {
                logc << "Pedestrian (x, y): (" << pedestrian_position.first 
                     << ", " << pedestrian_position.second << ")" << endl;
            } else {
                logc << "No pedestrian in scene." << endl;
            }
            if (!pedestrian_path.empty()) {
                logc << "Pedestrian path [(x, y), ...]: [";
                for (int i = 0; i < pedestrian_path.size(); i++) {
                    logc << "(" << pedestrian_path[i].first 
                         << ", " << pedestrian_path[i].second 
                         << ")";
                    if (i != pedestrian_path.size()-1) logc << " ,";
                }    
                logc << "]" << endl;            
            }
            else {
                logc << "No pedestrian path prediction enabled." << endl;
            }
        }
};


class TCPConnector 
{
    private:
        inline static Path car_path;
        int unix_connection;

    public:
        int establish_connection(std::string bind_path, int port) 
        {
            bind_path += std::to_string(port);

            if ((unix_connection = socket(AF_UNIX, SOCK_STREAM, 0)) < 0) {
                printf("IS-DESPOT::[CONNECTOR] Socket creation error %s.\n", strerror(errno));
                return -1;
            }

            struct sockaddr_un saddr;
            memset(&saddr, 0, sizeof(struct sockaddr_un));
            saddr.sun_family = AF_UNIX;
            strcpy(saddr.sun_path, bind_path.c_str());

            int tries = 0, res = -1;
            while ((res = connect(unix_connection, (struct sockaddr *) &saddr, sizeof(struct sockaddr_un))) < 0) {
                std::this_thread::sleep_for (std::chrono::seconds(1));

                if(++tries == 5){
                    printf("IS-DESPOT::[CONNECTOR] Timeout error connecting to '%s': %s.\n", 
                           bind_path.c_str(), strerror(errno));
                    exit(-1);
                }
            }
            logc << "IS-DESPOT::[CONNECTOR] Successfully bound to TCP server file " << bind_path << "." << endl;
            return unix_connection;
        }


        // only read as many bytes from the socket as have been actually announced
        std::vector<double> receive_exact_message()
        {
            logc << __FUNCTION__ << endl;
            // bytes used for representing uint32_t is platform agnostic
            uint32_t announced_bytes = 0, total_received_bytes = 0, received_bytes = 0;
            logc << "\t- expect " << static_cast<int>(sizeof(announced_bytes)) << " bytes to be received" << endl;
            // until we have read all bytes required to infer the length of the incoming message
            while (total_received_bytes < sizeof(announced_bytes)) {
                // shift bytes so we do not overwrite and reduce bytes we want to read by the number of
                // bytes that have alreday been received in successive calls
                received_bytes = recv(unix_connection, 
                                      reinterpret_cast<char*>(&announced_bytes) + total_received_bytes, 
                                      sizeof(announced_bytes) - total_received_bytes, 0);
                // error handling
                if (received_bytes <= 0) {
                    printf("IS-DESPOT::[CONNECTOR] Error while reading from socket: %s.\n", strerror(errno));
                    exit(-1);
                }
                total_received_bytes += received_bytes;
                logc << "\t- message length: received bytes " << received_bytes 
                     << " (total " << total_received_bytes << ")" << endl;
            }

            // sanity check: all bytes received?
            if (total_received_bytes != sizeof(announced_bytes)) {
                printf("IS-DESPOT::[CONNECTOR] Invalid number of bytes received for inferring message length.\n");
                exit(-1);
            }

            logc << "\t- " << announced_bytes << " bytes have been announced" << endl;

            // sanity check: can all bytes be converted to doubles?
            double number_of_values = announced_bytes / (sizeof(double)*1.0);
            if (floor(number_of_values) != number_of_values) {
                printf("IS-DESPOT::[CONNECTOR] Invalid byte to double conversion ratio: "
                       "Expected an integral number of doubles, but got %.4f.\n", number_of_values);
                exit(-1);
            }

            // create buffer which exactly matches the number of announced bytes
            std::vector<double> buffer(static_cast<int>(number_of_values));
            total_received_bytes = 0, received_bytes = 0;
            // until we have read all bytes required to construct the incoming message
            while (total_received_bytes < announced_bytes) {
                // actually read from the socket at most the number of announced bytes
                received_bytes = recv(unix_connection, 
                                      reinterpret_cast<char*>(buffer.data()) + total_received_bytes, 
                                      announced_bytes - total_received_bytes, 0);
                // error handling
                if (received_bytes <= 0) {
                    printf("IS-DESPOT::[CONNECTOR] Error while reading from socket: %s.\n", strerror(errno));
                    exit(-1);
                }
                total_received_bytes += received_bytes;
                logc << "\t- message content: received bytes " << received_bytes 
                     << " (total " << total_received_bytes << ")" << endl;
            }

            // sanity check: all bytes received?
            if (total_received_bytes != announced_bytes) {
                printf("IS-DESPOT::[CONNECTOR] Invalid number of bytes received for inferring message contents.\n");
                exit(-1);
            }
            logc << "\t- received message with " << buffer.size() << " double entries" << endl;

            return buffer;
        }


        // general python-despot comunication
        Message receive_observation() 
        {
            std::vector<double> buffer = receive_exact_message();

            logc << __FUNCTION__ << endl
                 << "\t- received observation: ";
            for (int i = 0; i < buffer.size(); i++) {
                if (i % 8 == 0) logc << "\n\t  ";
                logc << buffer[i];
                if (i != buffer.size()-1) logc << ", ";
                else logc << "\n\t- with sizeof(observation) in bytes: " << buffer.size()*sizeof(double) << endl;   
            }

            // construct message object
            Message message = Message();

            if (buffer[0] == -1.0) message.terminal = false;
            else if (buffer[0] == 1.0) message.terminal = true;
            else {
                printf("IS-DESPOT::[CONNECTOR] Invalid value of message field 'terminal':" 
                       "Expected either -1.0 or 1.0, got %.4f.\n", buffer[0]);
                exit(-1);
            }
            message.reward = buffer[1];

            // car information
            message.car_position.first = buffer[2];
            message.car_position.second = buffer[3];
            message.car_speed = buffer[4];
            message.car_angle = buffer[5];

            // car path
            double car_path_length = -1, fractional_part = -1;
            fractional_part = modf(buffer[6], &car_path_length);
            if (fractional_part != 0.0) {
                printf("IS-DESPOT::[CONNECTOR] Invalid car path length provided: "
                       "Expected integral number, got %.4f.\n", fractional_part);
                exit(-1);
            }
            if (car_path_length == 0) {
                printf("IS-DESPOT::[CONNECTOR] Received future agent vehicle trajectory of length zero. "
                       "Skipping planning for this step. Finalizing episode.\n");
            }
            // car path length in individual elements
            for (int i = 7; i < static_cast<int>(car_path_length)+7; i+=3) {
                message.car_path.push_back(COORD(buffer[i], buffer[i+1], buffer[i+2]));
            }
            TCPConnector::car_path = message.car_path;

            // pedestrian
            // is a pedestrian in the scene?
            if (buffer[7+car_path_length] == -1.0) {
                message.pedestrian_in_scene = false;
                message.pedestrian_position.first = DBL_MIN;
                message.pedestrian_position.second = DBL_MIN;             
            }
            else if (buffer[7+car_path_length] == 1.0) {
                message.pedestrian_in_scene = true;
                // extract pedestrian position
                message.pedestrian_position.first = buffer[7+car_path_length+1];
                message.pedestrian_position.second = buffer[7+car_path_length+2];
            }
            else {
                printf("IS-DESPOT::[CONNECTOR] Invalid value of message field 'pedestrian_in_scene': " 
                       "Expected either -1.0 or 1.0, got %.4f.\n", buffer[7+car_path_length]);
                exit(-1);
            }

            // is pedestrian path prediction enabled?
            if (buffer[7+car_path_length+3] == -1.0) {
                message.pedestrian_path_prediction = false;
            }
            else if (buffer[7+car_path_length+3] == 1.0) {
                message.pedestrian_path_prediction = true;
                // extract predicted pedestrian path
                for (int i = 7+car_path_length+4; i < buffer.size(); i+=2) {
                    message.pedestrian_path.push_back(std::pair<double, double>(buffer[i], buffer[i+1]));
                }
            }
            else {
                printf("IS-DESPOT::[CONNECTOR] Invalid value of message field 'pedestrian_path_prediction': " 
                       "Expected either -1.0 or 1.0, got %.4f.\n", buffer[7+car_path_length+3]);
                exit(-1);
            }

            // save pedestrian as obstacle
            message.obstacle = tuple<double, double, double>(message.pedestrian_position.first, 
                                                             message.pedestrian_position.first, 
                                                             0.0);
            message.print();
            return message;
        }


        // hyleap: state evaluation connection
        void send_expanded_nodes(std::vector<despot::VNode*> expanded_nodes, bool root = false)
        {
            logc << __FUNCTION__ << endl
                 << "\t- sending " << expanded_nodes.size() << " node(s)"  << endl;

            // sanity check: is there a valid car path?
            if (TCPConnector::car_path.size() == 0) {
                printf("IS-DESPOT::[CONNECTOR] Invalid car path: Empty.\n");
                exit(-1);
            }
            // buffer starts with lstm state
            std::vector<double> buffer(ModelParams::LSTM_STATE_SIZE, 0);
            // send lstm state of root
            if (root) {
                buffer = expanded_nodes.back()->lstm_state;
                logc << "\t- previous root node lstm state: ";
            // any other node in the tree (that is guaranteed to have a parent)
            } else {
                // double parent call, because only VNodes hold lstm states
                buffer = expanded_nodes.back()->parent()->parent()->lstm_state;
                logc << "\t- parent lstm state: ";
            }
            for (int i = 0; i < buffer.size(); i++) {
                if (i % 8 == 0) logc << "\n\t  ";
                logc << buffer[i];
                if (i != buffer.size()-1) logc << ", ";
                else logc << endl;
            }	
            buffer.push_back(static_cast<double>(expanded_nodes.size())); 
            // for each expanded node
            for(int i = 0; i < expanded_nodes.size(); i++){
                // get the first particle
                PomdpState* state = static_cast<PomdpState*>(expanded_nodes[i]->particles()[0]);
                // and the car's position
                COORD car_position = TCPConnector::car_path.at(state->car.pos);
                // car
                buffer.push_back(car_position.x);
                buffer.push_back(car_position.y);
                buffer.push_back(car_position.theta);
                // pedestrian
                buffer.push_back(state->peds[0].pos.x);
                buffer.push_back(state->peds[0].pos.y);

                buffer.push_back(state->car.vel);
                if (!root) {
                    // add the action that lead to this node (only qnodes have action edges, vnodes have observation edges)
                    // any vnode that is not the root node has a qnode and vnode parent
                    buffer.push_back(expanded_nodes[i]->parent()->edge());
                    // add step_reward of the parent qnode (as all other values are heuristic estimates)
                    buffer.push_back(expanded_nodes[i]->parent()->step_reward);
                } else {
                    // action that has actually been executed in the previous step (defaults to 1/maintain for step==0)
                    buffer.push_back(static_cast<double>(despot::DESPOT::previous_action.action));
                    // reward that has actually been received in the previous step (defaults to 0 for step==0)
                    buffer.push_back(despot::DESPOT::previous_action.value);
                }
                // announce number of double values in past_trajectory
                buffer.push_back(state->past_trajectory.size()*2);
                // communicate already traversed path by agent vehicle
                for (const COORD& coordinates: state->past_trajectory) {
                    buffer.push_back(coordinates.x);
                    buffer.push_back(coordinates.y);
                }
            }
            // make it known wether the node is the root node
            buffer.push_back(root ? 1.0 : -1.0);
            // actually send data
            send_message(buffer);
        }


        // hyleap: state evaluation connection
        void receive_drla_predictions(std::vector<despot::VNode*>& expanded_nodes)
        {
            std::vector<double> buffer = receive_exact_message();
            logc << __FUNCTION__ << endl;
            // LSTM hidden + cell state + belief state value estimate + predicted action + estimated uncertainty
            // HyLEAP's uncertainty value is set to -1 as default
            // i.e. the buffer space occupied by one node
            int individual_node_length = ModelParams::LSTM_STATE_SIZE + 3;
            for (std::size_t i = 0; i < expanded_nodes.size(); i ++) {
                // extract new lstm state
                expanded_nodes[i]->lstm_state = std::vector<double>(buffer.begin() + i*individual_node_length,
                                                                    buffer.begin() + i*individual_node_length
                                                                                   + ModelParams::LSTM_STATE_SIZE);
                // extract and save drla's prediction
                despot::ACT_TYPE drla_action = static_cast<despot::ACT_TYPE>(buffer[i*individual_node_length + ModelParams::LSTM_STATE_SIZE]);
                double drla_value = buffer[i*individual_node_length + ModelParams::LSTM_STATE_SIZE + 1];
                // this will contain a dummy value for HyLEAP
                double drla_uncertainty = buffer[i*individual_node_length + ModelParams::LSTM_STATE_SIZE + 2];
                expanded_nodes[i]->drla_prediction = despot::ValuedAction(drla_action, drla_value, drla_uncertainty);

                logc << "\t- predicted action: " << drla_action << endl
                     << "\t- estimated value: " << drla_value << endl
                     << "\t- estimated uncertainty: " << drla_uncertainty << endl
                     << "\t- current lstm state: ";
                     for (int j = 0; j < expanded_nodes[i]->lstm_state.size(); j++) {
                        if (j % 8 == 0) logc << "\n\t  ";
                        logc << expanded_nodes[i]->lstm_state[j];
                        if (j != expanded_nodes[i]->lstm_state.size()-1) logc << ", ";
                        else logc << endl;
                     }	            
            }
        }


        // leader control connection
        std::vector<double> receive_is_weights() 
        {
            return receive_exact_message();
        }


        // equivalent to python's sendall method
        void send_message(const double message) 
        {
            std::vector<double> buffer(1, message);
            send_message(buffer);
        }


        // equivalent to python's sendall method
        void send_message(const std::vector<double> message) 
        {
            logc << __FUNCTION__ << endl;

            // infer message length
            const uint32_t message_length = static_cast<const uint32_t>(message.size() * sizeof(double));

            logc << "\t- sending message: ";
            for (int i = 0; i < message.size(); i++) {
                if (i != 0 && i % 8 == 0) logc << "\n\t  ";
                logc << message[i];
                if (i != message.size()-1) logc << ", ";
                else logc << "\n\t- with sizeof(message) in bytes: " << message_length << endl;   
            }
  
            // send message length
            int total_sent_bytes = 0, sent_bytes = 0;
            while (total_sent_bytes < sizeof(message_length)) {
                sent_bytes = send(unix_connection, 
                                  reinterpret_cast<const char*>(&message_length) + total_sent_bytes, 
                                  sizeof(message_length) - total_sent_bytes, 0);
                if (sent_bytes <= 0) {
                    printf("IS-DESPOT::[CONNECTOR] Error while sending message byte length: %s.\n", strerror(errno));
                    exit(-1);
                }
                total_sent_bytes += sent_bytes;
                logc << "\t- message length: sent bytes " << sent_bytes 
                     << " (total " << total_sent_bytes << ")" << endl;
            }
            if (total_sent_bytes != sizeof(message_length)) {
                printf("IS-DESPOT::[CONNECTOR] Invalid number of sent bytes for announcing message length.\n");
                exit(-1);
            }

            // send actual message
            total_sent_bytes = 0, sent_bytes = 0;
            while (total_sent_bytes < message_length) {
                sent_bytes = send(unix_connection, reinterpret_cast<const char*>(message.data()) + total_sent_bytes, 
                                  message_length - total_sent_bytes, 0);
                if (sent_bytes <= 0) {
                    printf("IS-DESPOT::[CONNECTOR] Error while sending message: %s.\n", strerror(errno));
                    exit(-1);
                }
                total_sent_bytes += sent_bytes;
                logc << "\t- message content: sent bytes " << sent_bytes 
                     << " (total " << total_sent_bytes << ")" << endl;
            }
            if (total_sent_bytes != message_length) {
                printf("IS-DESPOT::[CONNECTOR] Invalid number of sent bytes for publishing message.\n");
                exit(-1);
            }       
        }
};
#endif