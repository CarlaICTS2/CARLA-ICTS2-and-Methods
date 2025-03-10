#include <random>
#include <ctime>
#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <cstdio>
#include <limits.h>
#include <algorithm>
#include <iomanip>

#include "despot/solver/despot.h"
#include "despot/core/particle_belief.h"
#include "despot/util/util.h"

#include "connector.h"
#include "ped_pomdp.h"
#include "math_utils.h"
#include "optionparser.h"


// IS-DESPOT
static TCPConnector control_conn;
static int port = -1;

// HyLEAP & HyPLAN
static TCPConnector eval_conn;
static int dud_episodes = 0;

// LEADER
static TCPConnector leader_conn;


class Simulator {
public:
    typedef pair<float, Pedestrian> PedDistPair;

    inline static WorldModel worldModel = WorldModel();
    inline static PedPomdp* pomdp = new PedPomdp(worldModel);

    // action always maintain, value from mininum number of steps until colliision
    inline static despot::ScenarioLowerBound *lower_bound = pomdp->CreateScenarioLowerBound("SMART");
    inline static despot::ScenarioUpperBound *upper_bound = pomdp->CreateScenarioUpperBound("SMART");

    inline static despot::DESPOT *solver;

    inline static void init_solver() {
        // activate node evaluation connection if we use either "vanilla" hyleap or hyplan
        if (despot::Globals::config.AGENT == HyLEAP || despot::Globals::config.AGENT == HyPLAN) {
            solver = new despot::DESPOT(pomdp, lower_bound, upper_bound, &eval_conn);
        }
        else solver = new despot::DESPOT(pomdp, lower_bound, upper_bound); 
    }

    inline static void run(int episode_counter) {
        WorldStateTracker stateTracker(worldModel);
        WorldBeliefTracker beliefTracker(worldModel, stateTracker);

        // for pomdp planning and print world info
        PomdpState s;
        // for tracking world state
        PomdpStateWorld world_state;

        // reset HyLEAP variables for each new episode
        // reset root node lstm state 
        despot::DESPOT::previous_root_lstm_state = std::vector<double>(ModelParams::LSTM_STATE_SIZE, 0);
        // reset previous action (default: maintain) and reward (default: 0.0) for each episode
        despot::DESPOT::previous_action.action = 1;
        despot::DESPOT::previous_action.value = 0;

        int step_counter = 1;
        for(;; step_counter++) {
            logs << __FUNCTION__ << endl
                 << "======================= STEP=" << step_counter << "=======================" << endl;
			// receive current observation from server
            Message message = control_conn.receive_observation();

            // this can happen when episodes are terminated early due to surpasing the maximum allowed number of steps
            // or the agent glitching out; when this happens episodes are not considered for final performance evaluation
            // nor for training any learning agent; send dummy values and finalize episode
            if (message.car_path.size() == 0) {
                std::vector<double> result = {0.0, 0.0, 1.0, 1.0, 0.0};
                control_conn.send_message(result);
                break;
            }
            // make pedestrian information known
            pomdp->num = message.pedestrian_in_scene == true ? 1 : 0;
            worldModel.n_peds = message.pedestrian_in_scene == true ? 1 : 0;
            world_state.num = message.pedestrian_in_scene == true ? 1 : 0;

            // reward for executing the action of the previous step
            // Python sends 0 as default for the first step of every episode
            despot::DESPOT::previous_action.value = message.reward;
            
            // print debug information
            logs << "\t- previous action: " << despot::DESPOT::previous_action.action << endl
                 << "\t- previous reward: " << despot::DESPOT::previous_action.value << endl;
            /*            << "\t- previous root node lstm state: ";
            for (int i = 0; i < despot::DESPOT::previous_root_lstm_state.size(); i++) {
                if (i != 0 && i % 8 == 0) log_simulator << "\n\t  ";
                log_simulator << despot::DESPOT::previous_root_lstm_state[i];
                if (i != despot::DESPOT::previous_root_lstm_state.size()-1) log_simulator << ", ";
                else log_simulator << endl;
            }
            */
            worldModel.path = message.car_path;

			// set current position to be 0 (will probably always stay that way)
			world_state.car.pos = 0;
			world_state.car.vel = message.car_speed; // in m/s

            for(int i = 0; i < world_state.num; i++) {
                world_state.peds[i].goal = -1;
                world_state.peds[i].pos = COORD(message.pedestrian_position.first, message.pedestrian_position.second);
                world_state.peds[i].id = i;
            }

            COORD car_path_position = worldModel.path[world_state.car.pos];
            logs << "\t- agent vehicle position (x, y, theta): ("
                 << car_path_position.x << ", " << car_path_position.y << ", " << car_path_position.theta << ")" << endl;
                 //<< "\t- inCollision(): " << worldModel.inCollision(world_state) << endl;

			stateTracker.updateCar(worldModel.path[world_state.car.pos]);
			stateTracker.updateVel(world_state.car.vel);

			// TODO ... update the peds in stateTracker and the pedestrians
			for (int i = 0; i < world_state.num; i++) {
				Pedestrian p(world_state.peds[i].pos.x, world_state.peds[i].pos.y, world_state.peds[i].id);
				stateTracker.updatePed(p);
                /*
                logs << "\t- pedestrian position (x, y): (" 
                     << stateTracker.ped_list.front().x
                     << ", "
                     << stateTracker.ped_list.front().y
                     << ")" << endl;
                */

				PedStruct p_mod(COORD(world_state.peds[i].pos.x, world_state.peds[i].pos.y+10), -1, world_state.peds[i].id);
                logs << "\t- modified pedestrian position (x, y+10): (" << p_mod.pos.x << ", " << p_mod.pos.y << ")" << endl
                     // << "\t- inCollision(): " << worldModel.inCollision(world_state) << endl
                     << "\t- inFrontNew(): " << worldModel.inFrontNew(p_mod.pos, world_state.car.pos) << endl
                     << "\t- isMovingAwayNew(): " << worldModel.isMovingAwayNew(world_state, 0) << endl;
			}

            if (world_state.num == 0) {
                stateTracker.ped_list.clear();
            }

            std::vector<PedDistPair> sorted_peds = stateTracker.getSortedPeds();

            // 1. update all pedestrian beliefs for each new scene simulation step
            //    new pedestrian beliefs define a uniform probability distibution over all possible goal directions
            // 2. existing pedestrian beliefs are updated according to the direction and distance that an associated
            //    pedestrian has moved since the last scene simulation step
            // 3. this only takes information from the state tracker into account
            beliefTracker.update(step_counter);

            // send updated belief to leader
            if (despot::Globals::config.AGENT == LEADER && stateTracker.ped_list.size() == 1) {
                std::vector<double> ped_belief = beliefTracker.belief(worldModel.n_peds-1);
                ped_belief.push_back(car_path_position.x);
                ped_belief.push_back(car_path_position.y);
                ped_belief.push_back(car_path_position.theta);
                ped_belief.push_back(world_state.car.vel);
                ped_belief.push_back(stateTracker.ped_list.front().x);
                ped_belief.push_back(stateTracker.ped_list.front().y);

                // send belief over first pedestrian and current observation
                leader_conn.send_message(ped_belief);
                // wait for leader's response
                std::vector<double> is_weights = leader_conn.receive_is_weights();

                // sanity check: number of weights = number of pedestrian goal directions?
                int num_goal_directions = beliefTracker.sorted_beliefs[0].prob_goals.size();
                if (is_weights.size() != num_goal_directions) {
                    printf("IS-DESPOT::[SIMULATOR] Invalid number of importance weights: Expected %d, got %lu.\n",
                            num_goal_directions, is_weights.size()); 
                    exit(-1);
                }
                logb << __FUNCTION__ << endl
                     << "\t- LEADER importance weights:" << endl;
                worldModel.print_probability_distribution(is_weights, 0.0);

	            // prepare particle importance weights vector
                beliefTracker.sorted_beliefs[0].particle_importance_weights.resize(num_goal_directions);

                // calculate particle importance weights as prob_goals / leader_attention (unnormalized)
                std::transform(
                    beliefTracker.sorted_beliefs[0].prob_goals.begin(), 
                    beliefTracker.sorted_beliefs[0].prob_goals.end(), 
                    is_weights.begin(),
                    beliefTracker.sorted_beliefs[0].particle_importance_weights.begin(),
                    std::divides<double>()
                );
                logb << __FUNCTION__ << endl
                     << "\t- LEADER-attention adjusted importance weights:" << endl;
                worldModel.print_probability_distribution(beliefTracker.sorted_beliefs[0].particle_importance_weights, 0.01);

                // normalize leader weights for sampling pedestrian goal directiuons if enabled
                // (which is always done according to official LEADER's code base: 
                // https://github.com/modanesh/LEADER/blob/master/car_hyp_despot/src/planner/crowd_belief.cpp 
                // lines 148-186)
                double leader_sum = std::accumulate(is_weights.begin(), is_weights.end(), 0.0);
                if (fabs(leader_sum - 1.0) > 1e-6) {
                    for (double& weight: is_weights) { weight /= leader_sum; }
                    double leader_sum_norm = std::accumulate(is_weights.begin(), is_weights.end(), 0.0);
                    if (fabs(leader_sum_norm - 1.0) > 1e-6) {
                        printf("IS-DESPOT::[SIMULATOR] Invalid probability distribution: " 
                               "Expected total probability mass of 1, got %.4f.\n", leader_sum_norm);
				        exit(-1);
                    }
                }
                // assign intention importance weights
                beliefTracker.sorted_beliefs[0].leader_attention = is_weights;
                logb <<  "\t- LEADER's attention distribution for sampling pedestrian goal directions:" << endl;
                worldModel.print_probability_distribution(is_weights, 0.01);
            }

            // 1. particles describe a specific state of the world - there is no uncertainty!
            //    i.e. pedestrians have a specific goal direction which is sampled based on the 
            //    current belief (probability distribution over all goal directions)
            std::vector<PomdpState> samples = beliefTracker.sample(despot::Globals::config.PARTICLE_NUMBER);

            // 1. particle = sample + actually allocated memory
            // 2. all particles have uniform weight of 1.0/#particles
            std::vector<despot::State*> particles = pomdp->ConstructParticles(samples);

            // adjust importance weight of particles according to intention of the pedestrian
            // as defined by the recently updated distribution incorporating LEADER's attention
            if (despot::Globals::config.AGENT == LEADER && stateTracker.ped_list.size() == 1) {
                double total_weight = 0;                
                for (auto& particle: particles) {
                    PomdpState* state = static_cast<PomdpState*>(particle);
                    // get importance weight of pedestrian goal direction
                    state->weight *= beliefTracker.sorted_beliefs[0].particle_importance_weights[state->peds[0].goal];
                    total_weight += state->weight;
                }
                // traditionally normalizing importance distribution has beneficial effects on performance
                // (at least for "standard" IS-DESPOT: https://adacomp.comp.nus.edu.sg/wp-content/uploads/2020/08/ijrr19isdespot.pdf)
                if (!despot::Globals::config.NO_NORMALIZATION) {
                    double total_weight_norm = 0;
                    for (auto& particle: particles) {
                        PomdpState* state = static_cast<PomdpState*>(particle);
                        state->weight /= total_weight;
                        total_weight_norm += state->weight;
                    }
                    if (fabs(total_weight_norm - 1.0) > 1e-6) {
                        printf("IS-DESPOT::[SIMULATOR] Invalid probability distribution: " 
                               "Expected total probability mass of 1, got %.4f.\n", total_weight_norm);
                        exit(-1);                        
                    }
                }
            }

            // randomly shuffles particles
            // checks that weight of all particles equals 1
            // can artificially inflate number of particles by duplication
            despot::ParticleBelief* pb = new despot::ParticleBelief(particles, pomdp, NULL, false);

            // set root belief
            // beliefs shift across scene simulation steps
            // beliefs depends on past & current pedestreian movement
            solver->belief(pb);

            // execute IS-DESPOT, i.e. obtain action
            // random streams influence the probability with which the ego vehicle's velocity updates are actually applied
            // and how noisy pedestrians' goal directions are, i.e. how accurately they are walking
            // the weight of particles is changed during each planning step
            despot::ValuedAction optimal_action = solver->Search();

            // save current action as the previous action of the next step
            despot::DESPOT::previous_action.action = optimal_action.action;

            logs << __FUNCTION__ << endl
                 << "\t- 'improved' despot policy being sent to Python:" 
                 << "\n\t\t-- DEC: " << solver->action_probabilities[0] 
                 << "\n\t\t-- MAIN: " << solver->action_probabilities[1] 
                 << "\n\t\t-- ACC: " << solver->action_probabilities[2]
                 << endl; 

            // this is needed for HyLEAP
            std::vector<double> result = solver->action_probabilities;
            // this is always needed
            result.push_back(optimal_action.action);
            result.push_back(optimal_action.value);
            // send final message of step
            control_conn.send_message(result);

            // we need to send this seperately in order to preserve synchronicity
            // only do so when there is at least one pedestrian in the scene simulation step
            // otherwise there is no point in calling LEADER; since it generated attention distributions for pedestrians
            if (despot::Globals::config.AGENT == LEADER && stateTracker.ped_list.size() == 1) {
                leader_conn.send_message(optimal_action.value);
            }

            delete pb;

            if (message.terminal) {
                break;
            }

        } // episode

        // log metrics
        despot::logging::track_episode(step_counter, episode_counter);
        // debug statistics
        despot::logging::print_resource_consumption();
    }
};


int main(int argc, char** argv) {
    // auto-flush
    setbuf(stdout, NULL); 

    OptionParser::parse_arguments(argc, argv);

    control_conn.establish_connection("/tmp/is-despot_connection_", despot::Globals::config.PORT);
    switch (despot::Globals::config.AGENT) {
        case IS_DESPOT:
            printf("IS-DESPOT::[SETUP] Executing vanilla IS-DESPOT code.\n");
            break;
        case HyLEAP:
            printf("IS-DESPOT::[SETUP] Executing HyLEAP code snippets.\n");
            eval_conn.establish_connection("/tmp/hyleap_evaluation_connection_", despot::Globals::config.PORT);
            break;                
        case HyLEAR:
            printf("IS-DESPOT::[SETUP] Executing HyLEAR code snippets.\n");
            break;    
        case HyPLAN:
            printf("IS-DESPOT::[SETUP] Executing HyPLAN code snippets.\n");
            eval_conn.establish_connection("/tmp/hyplan_evaluation_connection_", despot::Globals::config.PORT);
            break;    
        case LEADER:
            printf("IS-DESPOT::[SETUP] Executing LEADER code snippets.\n");
            leader_conn.establish_connection("/tmp/leader_connection_", despot::Globals::config.PORT);
            break;   
        default:
            printf("IS-DESPOT::[SETUP] Invalid agent specified.\n");
            exit(-1);
    }

    despot::logging::print_config();
    despot::logging::print_pomdp_model_parameters();

    despot::logging::init_cpu_currently_used();
    despot::logging::init_cpu_currently_used_by_current_process();
    
    Simulator::init_solver();

    int episode_counter = 1;
    while (true) {
        logs << "++++++++++++++++++++++ EPISODE= " << episode_counter << " ++++++++++++++++++++++" << endl;
        Simulator::run(episode_counter);
        episode_counter++;
    }
}
