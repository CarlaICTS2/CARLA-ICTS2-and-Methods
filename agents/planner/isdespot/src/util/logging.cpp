#include <despot/util/logging.h>

using namespace std;

namespace despot {

streambuf* log_ostream::log_buf::setbuf(char* s, streamsize n) 
{
	return this;
}

std::string logging::bool_to_string(bool b) 
{
	if (b) return "true";
	else return "false";
}

void logging::set_scope(Scope scope, bool setting) 
{
	logging::scope_to_bool[scope] = setting;
}

bool logging::get_scope(Scope scope) 
{
	if (logging::scope_to_bool.empty()) {
        return false;
	}
	if (logging::scope_to_bool.find(scope) == logging::scope_to_bool.end()) {
        return false;
	}
	if (logging::scope_to_bool[scope]) {
		return true;
	} else {
		return false;
	}
}

log_ostream& logging::stream(Scope scope) 
{
	return *(streams_[scope]);
}

void logging::stream(Scope scope, ostream& out) 
{
	streams_[scope] = new log_ostream(out, markers_[scope]);
}

vector<log_ostream*> logging::initialize_log_streams() 
{
	vector<log_ostream*> streams(6);

	streams[CONNECTOR] = new log_ostream(std::cout, markers_[CONNECTOR]);
	streams[SIMULATOR] = new log_ostream(std::cout, markers_[SIMULATOR]);
	streams[DESPOT] = new log_ostream(std::cout, markers_[DESPOT]);
	streams[IMPORTANCE_SAMPLING] = new log_ostream(std::cout, markers_[IMPORTANCE_SAMPLING]);
	streams[POMDP] = new log_ostream(std::cout, markers_[POMDP]);
	streams[WORLD] = new log_ostream(std::cout, markers_[WORLD]);
	streams[BELIEF] = new log_ostream(std::cout, markers_[BELIEF]);

	return streams;
}

void logging::initialize_settings(bool default_setting)
{
	for (int i = CONNECTOR; i != LAST_DUMMY; i++) {
		logging::set_scope(static_cast<logging::Scope>(i), default_setting);
	}
}

void logging::print_settings() 
{
	for (int scope = CONNECTOR; scope != LAST_DUMMY; scope++) {
		printf("IS-DESPOT::[LOGGING] Scope %s: %s.\n", 
		markers_[scope].c_str(), (get_scope(Scope(scope)) ? "enabled" : "disabled"));
	}
}

void logging::print_config() 
{
    printf("\bIS-DESPOT::[SETUP] IS-DESPOT CONFIGURATION"
		   "\b\t--aggressive_belief_updates = %s"
		   "\b\t--minimal_noise = %s"
		   "\b\t--no_importance_sampling = %s"
		   "\b\t--no_normalization = %s"
		   "\b\t--timeout = %.4f"
		   "\b\t--time_per_step = %.4f"
		   "\b\t--maximum_episode_steps = %d"
           "\b\t--noise = %.4f"
		   "\b\t--seed = %d"
           "\b\t--max_search_depth = %d"
           "\b\t--discount_factor = %.4f"
           "\b\t--particle_number = %d"
           "\b\t--gap_reduction_rate = %.4f"
           "\b\t--max_policy_simulation_length = %d"
           "\b\t- pruning_constant = %.4f \n"
           , bool_to_string(Globals::config.AGGRESSIVE_BELIEF_UPDATES).c_str()
           , bool_to_string(Globals::config.MINIMAL_NOISE).c_str()
           , bool_to_string(Globals::config.NO_IMPORTANCE_SAMPLING).c_str()
           , bool_to_string(Globals::config.NO_NORMALIZATION).c_str()
           , Globals::config.TIMEOUT
           , Globals::config.TIME_PER_PLANNING_STEP
           , Globals::config.MAX_EPISODE_STEPS
           , Globals::config.NOISE
           , Globals::config.SEED
           , Globals::config.MAX_SEARCH_DEPTH
           , Globals::config.DISCOUNT
           , Globals::config.PARTICLE_NUMBER
           , Globals::config.GAP_REDUCTION_RATE
           , Globals::config.MAX_POLICY_SIM_LEN
           , Globals::config.PRUNING_CONSTANT);
}
 
void logging::print_pomdp_model_parameters()
{
    printf("\bIS-DESPOT::[SETUP] WORLD SIMULATION PARAMETERS"
        "\b\t- BELIEF_ANGLE_DISCRETIZATION = %d"
        "\b\t- CRASH_PENALTY = %0.4f"
        "\b\t- REWARD_FACTOR_VEL = %0.4f"
        "\b\t- VEL_MAX = %0.4f"
        "\b\t- NOISE_GOAL_ANGLE = %0.4f"
        "\b\t- REWARD_BASE_CRASH_VEL = %0.4f"
        "\b\t- BELIEF_SMOOTHING = %0.4f"
        "\b\t- NOISE_ROBVEL = %0.4f"
        "\b\t- COLLISION_DISTANCE = %0.4f"
        "\b\t- COLLISION_SIDE_DISTANCE = %0.4f"
        "\b\t- IN_FRONT_ANGLE_DEG = %0.4f"
        "\b\t- CAR_WIDTH = %0.4f"
        "\b\t- CAR_LENGTH = %0.4f"
        "\b\t- MAX_EPISODE_LENGTH = %d"
        "\b\t- N_PED_IN = %d"
        "\b\t- N_PED_WORLD = %d"
        "\b\t- LASER_RANGE = %0.4f"
        "\b\t- pos_rln = %0.4f "
        "\b\t- vel_rln = %0.4f"
        "\b\t- PATH_STEP = %0.4f"
        "\b\t- GOAL_TOLERANCE = %0.4f"
        "\b\t- PED_SPEED = %0.4f"
        "\b\t- map_height = %d"
        "\b\t- map_width = %d"
        "\b\t- control_freq = %0.4f"
        "\b\t- AccSpeed = %0.4f"
        "\b\t- GOAL_REWARD = %0.4f"
        "\b\t- NUM_PEDESTRIANS = %d"
        "\b\t- LSTM_STATE_SIZE = %d"
        "\b\t- OBSERVATION_SIZE = %d \n"
        , ModelParams::BELIEF_ANGLE_DISCRETIZATION
        , ModelParams::CRASH_PENALTY
        , ModelParams::REWARD_FACTOR_VEL
        , ModelParams::VEL_MAX
        , ModelParams::NOISE_GOAL_ANGLE
        , ModelParams::REWARD_BASE_CRASH_VEL
        , ModelParams::BELIEF_SMOOTHING
        , ModelParams::NOISE_ROBVEL
        , ModelParams::COLLISION_DISTANCE
        , ModelParams::COLLISION_SIDE_DISTANCE
        , ModelParams::IN_FRONT_ANGLE_DEG
        , ModelParams::CAR_WIDTH
        , ModelParams::CAR_LENGTH
        , ModelParams::MAX_EPISODE_LENGTH
        , ModelParams::N_PED_IN
        , ModelParams::N_PED_WORLD
        , ModelParams::LASER_RANGE
        , ModelParams::pos_rln
        , ModelParams::vel_rln
        , ModelParams::PATH_STEP
        , ModelParams::GOAL_TOLERANCE
        , ModelParams::PED_SPEED
        , ModelParams::map_height
        , ModelParams::map_width
        , ModelParams::control_freq
        , ModelParams::AccSpeed
        , ModelParams::GOAL_REWARD
        , ModelParams::NUM_PEDESTRIANS
        , ModelParams::LSTM_STATE_SIZE
        , ModelParams::OBSERVATION_SIZE);
}

const std::vector<std::string> logging::get_markers()
{
	return markers_;
}

void logging::initialize_tracking() 
{
    std::string stats_filename = "despot_tracking_" + despot::Globals::config.SCENARIO + 
                                 "_" + despot::Globals::config.START_TIMESTAMP + ".csv";
 	stats_tracking_file_path = despot::Globals::config.OUTPUT_DIRECTORY + "/" + stats_filename;
    printf("IS-DESPOT::[SETUP] Tracking planning effort in %s.\n", stats_tracking_file_path.c_str());

    // remove file if already exists
    std::remove(stats_tracking_file_path.c_str());

    // init tracking file
    ofstream despot_tracking;

    // dump is-despot hyper-parameters
    despot_tracking.open(stats_tracking_file_path, ios::out | ios::app);
    despot_tracking << "{'aggressive_belief_updates': " << std::noboolalpha << despot::Globals::config.AGGRESSIVE_BELIEF_UPDATES << ", "
                    << "'minimal_noise': " << std::noboolalpha << despot::Globals::config.MINIMAL_NOISE << ", "
                    << "'no_importance_sampling': " << std::noboolalpha << despot::Globals::config.NO_IMPORTANCE_SAMPLING << ", "
                    << "'no_normalization': " << std::noboolalpha << despot::Globals::config.MAX_SEARCH_DEPTH << ", "
                    << "'timeout': " << std::to_string(despot::Globals::config.TIMEOUT) << ", "
                    << "'time_per_step': " << std::to_string(despot::Globals::config.TIME_PER_PLANNING_STEP) << ", "
                    << "'maximum_episode_steps': " << std::to_string(despot::Globals::config.MAX_EPISODE_STEPS) << ", "
                    << "'noise': " << std::to_string(despot::Globals::config.NOISE) << ", "
                    << "'seed': " << std::to_string(despot::Globals::config.SEED) << ", "
                    << "'max_search_depth': " << std::to_string(despot::Globals::config.MAX_SEARCH_DEPTH) << ", "
                    << "'discount_factor': " << std::to_string(despot::Globals::config.DISCOUNT) << ", "
                    << "'num_scenarios': " << std::to_string(despot::Globals::config.PARTICLE_NUMBER) << ", "
                    << "'gap_reduction_rate': " << std::to_string(despot::Globals::config.GAP_REDUCTION_RATE) << ", "
                    << "'max_policy_sim_len': " << std::to_string(despot::Globals::config.MAX_POLICY_SIM_LEN) << ", "
                    << "'pruning_constant': " << std::to_string(despot::Globals::config.PRUNING_CONSTANT) << "}" 
                    << endl;

    // dump csv header line
    despot_tracking << // planning effort
                       "episode," <<
                       "step," <<

                       "tree_nodes," <<
                       "backups," <<
                       "terminal_states," <<

                       "initial_root_lower_bound," <<
                       "initial_root_upper_bound," <<

                       "final_root_lower_bound," <<
                       "final_root_upper_bound," <<

                       "average_root_lower_bound," <<
                       "median_root_lower_bound," <<
                       "std_root_lower_bound," <<
                       "min_root_lower_bound," <<
                       "max_root_lower_bound," <<

                       "average_root_upper_bound," <<
                       "median_root_upper_bound," <<
                       "std_root_upper_bound," <<    
                       "min_root_upper_bound," <<
                       "max_root_upper_bound," <<

                       "average_initial_lower_bound," <<
                       "median_initial_lower_bound," <<
                       "std_initial_lower_bound," <<
                       "min_initial_lower_bound," <<
                       "max_initial_lower_bound," <<

                       "average_initial_upper_bound," <<
                       "median_initial_upper_bound," <<
                       "std_initial_upper_bound," <<
                       "min_initial_upper_bound," <<
                       "max_initial_upper_bound," <<

                       "num_trials," <<
                       "average_trial_depth," <<
                       "median_trial_depth," <<
                       "std_trial_depth," <<
                       "min_trial_depth," <<
                       "max_trial_depth," <<

                       "average_WEU," <<
                       "median_WEU," <<
                       "std_WEU," <<
                       "min_WEU," <<
                       "max_WEU," <<

                       "average_observations," <<
                       "median_observations," <<
                       "std_observations," <<
                       "min_observations," <<
                       "max_observations,";


    if (despot::Globals::config.AGENT == HyLEAP || despot::Globals::config.AGENT == HyPLAN) {
    despot_tracking << "python_communication_time,"
                    << "python_interactions,";
    }

    if (despot::Globals::config.AGENT == HyPLAN && !despot::Globals::config.NO_VERTICAL_PRUNING) {
    despot_tracking << "average_uncertainty," <<
                       "median_uncertainty," <<
                       "std_uncertainty," <<
                       "min_uncertainty," <<
                       "max_uncertainty,";
    } 

    // execution times
    despot_tracking << "search_execution_time," <<
                       "construct_tree_execution_time," <<
                       "trial_execution_time," <<
                       "expand_execution_time," <<
                       "qcreation_execution_time," <<
                       "copy_execution_time," <<
                       "step_execution_time," <<
                       "free_execution_time," <<
                       "norm_execution_time," <<
                       "vcreation_execution_time," <<
                       "init_bounds_execution_time," <<
                       "backup_execution_time" << endl;

    despot_tracking.close();     
}

void logging::track_episode(int num_steps, int episode) {
    const auto enter = Clock::now();

    using namespace despot;
    if (!Globals::config.TRACKING) {
        return;
    }

    if (logging::tree_nodes.size() != num_steps) {
        printf("IS-DESPOT::[ERROR] Invalid step number for logging tree nodes: Expected %d, got %lu.\n", 
               num_steps, tree_nodes.size()); exit(-1);
    }
    if (logging::backups.size() != num_steps) {
        printf("IS-DESPOT::[ERROR] Invalid step number for logging backups: Expected %d, got %lu.\n", 
               num_steps, backups.size()); exit(-1);
    }
    if (logging::terminal_states.size() != num_steps) {
        printf("IS-DESPOT::[ERROR] Invalid step number for logging terminal states: Expected %d, got %lu.\n", 
               num_steps, terminal_states.size()); exit(-1);        
    }
    if (logging::root_lower_bounds.size() != num_steps) {
        printf("IS-DESPOT::[ERROR] Invalid step number for logging root lower bounds: Expected %d, got %lu.\n", 
               num_steps, root_lower_bounds.size()); exit(-1);         
    }
    if (logging::root_upper_bounds.size() != num_steps) {
        printf("IS-DESPOT::[ERROR] Invalid step number for logging root upper bounds: Expected %d, got %lu.\n", 
               num_steps, root_upper_bounds.size()); exit(-1);         
    }
    if (logging::initial_lower_bounds.size() != num_steps) {
        printf("IS-DESPOT::[ERROR] Invalid step number for logging initial lower bounds: Expected %d, got %lu.\n", 
               num_steps, initial_lower_bounds.size()); exit(-1);         
    }
    if (logging::initial_upper_bounds.size() != num_steps) {
        printf("IS-DESPOT::[ERROR] Invalid step number for logging initial upper bounds: Expected %d, got %lu.\n", 
               num_steps, initial_upper_bounds.size()); exit(-1);          
    }
    if (logging::trial_depths.size() != num_steps) {
        printf("IS-DESPOT::[ERROR] Invalid step number for logging trial depths: Expected %d, got %lu.\n", 
               num_steps, trial_depths.size()); exit(-1);          
    }
    if (logging::excess_uncertainties.size() != num_steps) {
        printf("IS-DESPOT::[ERROR] Invalid step number for logging excess uncertainties: Expected %d, got %lu.\n", 
               num_steps, excess_uncertainties.size()); exit(-1);        
    }
    if (logging::observations.size() != num_steps) {
        printf("IS-DESPOT::[ERROR] Invalid step number for logging observations: Expected %d, got %lu.\n", 
               num_steps, observations.size()); exit(-1);         
    }
    // HyLEAP 
    if (despot::Globals::config.AGENT == HyLEAP || despot::Globals::config.AGENT == HyPLAN) {
        if (logging::python_communication_times.size() != num_steps) {
            printf("IS-DESPOT::[ERROR] Invalid step number for logging python communication times: Expected %d, got %lu.\n", 
                   num_steps, python_communication_times.size()); exit(-1);              
        }
        if (logging::python_interactions.size() != num_steps) {
            printf("IS-DESPOT::[ERROR] Invalid step number for logging python interactions: Expected %d, got %lu.\n", 
                   num_steps, python_interactions.size()); exit(-1);               
        }
    } 
    // HyPLAN
    if (despot::Globals::config.AGENT == HyPLAN && !despot::Globals::config.NO_VERTICAL_PRUNING) {
        if (logging::uncertainty_values.size() != num_steps) {
            printf("IS-DESPOT::[ERROR] Invalid step number for logging HyPLAN uncertainty estimates: Expected %d, got %lu.\n", 
                   num_steps, uncertainty_values.size()); exit(-1);               
        }
    }
    // execution times
    if (logging::search_execution_times.size() != num_steps) {
        printf("IS-DESPOT::[ERROR] Invalid step number for logging search times: Expected %d, got %lu.\n", 
               num_steps, search_execution_times.size()); exit(-1);  
    }
    if (logging::construct_tree_execution_times.size() != num_steps) {
        printf("IS-DESPOT::[ERROR] Invalid step number for logging tree construction times: Expected %d, got %lu.\n", 
               num_steps, construct_tree_execution_times.size()); exit(-1);  
    }
    if (logging::trial_execution_times.size() != num_steps) {
        printf("IS-DESPOT::[ERROR] Invalid step number for logging trial times: Expected %d, got %lu.\n", 
               num_steps, trial_execution_times.size()); exit(-1);  
    }
    if (logging::expand_execution_times.size() != num_steps) {
        printf("IS-DESPOT::[ERROR] Invalid step number for logging expand times: Expected %d, got %lu.\n", 
               num_steps, expand_execution_times.size()); exit(-1);  
    }
    if (logging::qcreation_execution_times.size() != num_steps) {
        printf("IS-DESPOT::[ERROR] Invalid step number for logging qcreation times: Expected %d, got %lu.\n", 
               num_steps, qcreation_execution_times.size()); exit(-1); 
    }
    if (logging::copy_execution_times.size() != num_steps) {
        printf("IS-DESPOT::[ERROR] Invalid step number for logging copy times: Expected %d, got %lu.\n", 
               num_steps, copy_execution_times.size()); exit(-1); 
    }
    if (logging::step_execution_times.size() != num_steps) {
        printf("IS-DESPOT::[ERROR] Invalid step number for logging step times: Expected %d, got %lu.\n", 
               num_steps, step_execution_times.size()); exit(-1); 
    }
    if (logging::free_execution_times.size() != num_steps) {
        printf("IS-DESPOT::[ERROR] Invalid step number for logging free times: Expected %d, got %lu.\n", 
               num_steps, free_execution_times.size()); exit(-1); 
    }
    if (logging::norm_execution_times.size() != num_steps) {
        printf("IS-DESPOT::[ERROR] Invalid step number for logging norm times: Expected %d, got %lu.\n", 
               num_steps, norm_execution_times.size()); exit(-1); 
    }
    if (logging::vcreation_execution_times.size() != num_steps) {
        printf("IS-DESPOT::[ERROR] Invalid step number for logging vcreation times: Expected %d, got %lu.\n", 
               num_steps, vcreation_execution_times.size()); exit(-1); 
    }
    if (logging::init_bounds_execution_times.size() != num_steps) {
        printf("IS-DESPOT::[ERROR] Invalid step number for logging bound initialization times: Expected %d, got %lu.\n", 
               num_steps, init_bounds_execution_times.size()); exit(-1); 
    }
    if (logging::backup_execution_times.size() != num_steps) {
        printf("IS-DESPOT::[ERROR] Invalid step number for logging backup times: Expected %d, got %lu.\n", 
               num_steps, backup_execution_times.size()); exit(-1); 
    }

    // open tracking file and append
    std::ofstream despot_tracking;
    despot_tracking.open(stats_tracking_file_path, ios::out | ios::app);
               
    // dump entire episode
    for(int step = 0; step < num_steps; step++) {

        std::pair<double, double> avg_and_std = MathUtils::get_average_and_stdev(logging::root_lower_bounds[step]);
        std::pair<double, double> min_and_max = MathUtils::get_min_and_max(logging::root_lower_bounds[step]);

                        // meta
        despot_tracking << std::to_string(episode) << ","
                        << std::to_string(step+1) << ","

                        // planning effort
                        << std::to_string(logging::tree_nodes[step]) << ","
                        << std::to_string(logging::backups[step]) << ","
                        << std::to_string(logging::terminal_states[step]) << ","

                        << std::to_string(logging::root_lower_bounds[step].front()) << ","
                        << std::to_string(logging::root_upper_bounds[step].front()) << ","

                        << std::to_string(logging::root_lower_bounds[step].back()) << ","
                        << std::to_string(logging::root_upper_bounds[step].back()) << ","
 
                        // root lower bound
                        << std::to_string(avg_and_std.first) << ","
                        << std::to_string(MathUtils::get_median(logging::root_lower_bounds[step])) << ","
                        << std::to_string(avg_and_std.second) << ","
                        << std::to_string(min_and_max.first) << ","
                        << std::to_string(min_and_max.second) << ",";

        // root upper bound
        avg_and_std = MathUtils::get_average_and_stdev(logging::root_upper_bounds[step]);
        min_and_max = MathUtils::get_min_and_max(logging::root_upper_bounds[step]);  

        despot_tracking << std::to_string(avg_and_std.first) << ","
                        << std::to_string(MathUtils::get_median(logging::root_upper_bounds[step])) << ","
                        << std::to_string(avg_and_std.second) << ","
                        << std::to_string(min_and_max.first) << ","
                        << std::to_string(min_and_max.second) << ",";

        // initial lower bound
        avg_and_std = MathUtils::get_average_and_stdev(logging::initial_lower_bounds[step]);
        min_and_max = MathUtils::get_min_and_max(logging::initial_lower_bounds[step]); 
        despot_tracking << std::to_string(avg_and_std.first) << ","
                        << std::to_string(MathUtils::get_median(logging::initial_lower_bounds[step])) << ","
                        << std::to_string(avg_and_std.second) << ","
                        << std::to_string(min_and_max.first) << ","
                        << std::to_string(min_and_max.second) << ",";

        // initial upper bound
        avg_and_std = MathUtils::get_average_and_stdev(logging::initial_upper_bounds[step]);
        min_and_max = MathUtils::get_min_and_max(logging::initial_upper_bounds[step]); 
        despot_tracking << std::to_string(avg_and_std.first) << ","
                        << std::to_string(MathUtils::get_median(logging::initial_upper_bounds[step])) << ","
                        << std::to_string(avg_and_std.second) << ","
                        << std::to_string(min_and_max.first) << ","
                        << std::to_string(min_and_max.second) << ",";

        // trial depth
        avg_and_std = MathUtils::get_average_and_stdev(logging::trial_depths[step]);
        min_and_max = MathUtils::get_min_and_max(logging::trial_depths[step]); 
        despot_tracking << std::to_string(logging::trial_depths[step].size()) << ","
                        << std::to_string(avg_and_std.first) << ","
                        << std::to_string(MathUtils::get_median(logging::trial_depths[step])) << ","
                        << std::to_string(avg_and_std.second) << ","
                        << std::to_string(min_and_max.first) << ","
                        << std::to_string(min_and_max.second) << ",";

        // weighted excess uncertainty
        avg_and_std = MathUtils::get_average_and_stdev(logging::excess_uncertainties[step]);
        min_and_max = MathUtils::get_min_and_max(logging::excess_uncertainties[step]); 
        despot_tracking << std::to_string(avg_and_std.first) << ","
                        << std::to_string(MathUtils::get_median(logging::excess_uncertainties[step])) << ","
                        << std::to_string(avg_and_std.second) << ","
                        << std::to_string(min_and_max.first) << ","
                        << std::to_string(min_and_max.second) << ",";

        // observations
        avg_and_std = MathUtils::get_average_and_stdev(logging::observations[step]);
        min_and_max = MathUtils::get_min_and_max(logging::observations[step]); 
        despot_tracking << std::to_string(avg_and_std.first) << ","
                        << std::to_string(MathUtils::get_median(logging::observations[step])) << ","
                        << std::to_string(avg_and_std.second) << ","
                        << std::to_string(min_and_max.first) << ","
                        << std::to_string(min_and_max.second) << ",";

         // HyLEAP
        if (despot::Globals::config.AGENT == HyLEAP || despot::Globals::config.AGENT == HyPLAN) {
        despot_tracking << std::to_string(logging::python_communication_times[step]) << ","
                        << std::to_string(logging::python_interactions[step]) << ",";
        }

        // HyPLAN
        if (despot::Globals::config.AGENT == HyPLAN && !despot::Globals::config.NO_VERTICAL_PRUNING) {
        avg_and_std = MathUtils::get_average_and_stdev(logging::uncertainty_values[step]);
        min_and_max = MathUtils::get_min_and_max(logging::uncertainty_values[step]); 
        despot_tracking << std::to_string(avg_and_std.first) << ","
                        << std::to_string(MathUtils::get_median(logging::uncertainty_values[step])) << ","
                        << std::to_string(avg_and_std.second) << ","
                        << std::to_string(min_and_max.first) << ","
                        << std::to_string(min_and_max.second) << ",";
        }

        // execution time tracking
        despot_tracking << std::to_string(logging::search_execution_times[step]) << ","
                        << std::to_string(logging::construct_tree_execution_times[step]) << ","
                        << std::to_string(logging::trial_execution_times[step]) << ","
                        << std::to_string(logging::expand_execution_times[step]) << ","
                        << std::to_string(logging::qcreation_execution_times[step]) << ","
                        << std::to_string(logging::copy_execution_times[step]) << ","
                        << std::to_string(logging::step_execution_times[step]) << ","
                        << std::to_string(logging::free_execution_times[step]) << ","
                        << std::to_string(logging::norm_execution_times[step]) << ","
                        << std::to_string(logging::vcreation_execution_times[step]) << ","
                        << std::to_string(logging::init_bounds_execution_times[step]) << ","
                        << std::to_string(logging::backup_execution_times[step]) << endl;
    }
    despot_tracking.close();

    // clear logs for next scene
    logging::tree_nodes.clear();
    logging::backups.clear();
    logging::terminal_states.clear();

    logging::root_lower_bounds.clear();
    logging::root_upper_bounds.clear(); 

    logging::initial_lower_bounds.clear();
    logging::initial_upper_bounds.clear();

    logging::trial_depths.clear();
    logging::excess_uncertainties.clear(); 
    logging::observations.clear();

    logging::python_communication_times.clear();
    logging::python_interactions.clear();
    
    logging::uncertainty_values.clear();
    
    logging::search_execution_times.clear(); 
    logging::construct_tree_execution_times.clear();
    logging::trial_execution_times.clear();
    logging::expand_execution_times.clear();
    logging::qcreation_execution_times.clear();
    logging::copy_execution_times.clear(); 
    logging::step_execution_times.clear(); 
    logging::free_execution_times.clear();
    logging::norm_execution_times.clear();
    logging::vcreation_execution_times.clear(); 
    logging::init_bounds_execution_times.clear();

    logging::backup_execution_times.clear();

    double duration = std::chrono::duration_cast<dsec>(Clock::now() - enter).count()*1000;
    printf("IS-DESPOT::[TRACKING] Time taken for tracking data %0.4fms.\n", duration);
}


// this essentially makes memory leaks visible
void logging::print_resource_consumption() {
    // print usage statistics of:
    // CPU
    printf("IS-DESPOT::[DEBUG] Current CPU usage: %.0f%%.\n", get_cpu_currently_used_by_current_process());
    
    // virutal memory
    double virt_mem = -1.0;
    if (initial_ep) {
        initial_virt_mem = static_cast<float>(get_virtual_memory_currently_used_by_current_process())/1000;
        virt_mem = initial_virt_mem;
    }
    else {
        virt_mem = static_cast<float>(get_virtual_memory_currently_used_by_current_process())/1000;
    }

    printf("IS-DESPOT::[DEBUG] Currently used virtual memory: %.0f MB. "
           "Increase in virtual memory usage since initial execution: %.2f%%.\n",
            virt_mem, (virt_mem/initial_virt_mem));

    // physical memory
    double phys_mem = -1.0;
    if (initial_ep) {
        initial_phys_mem = static_cast<float>(get_physical_memory_currently_used_by_current_process())/1000;
        phys_mem = initial_phys_mem;
    }
    else {
        phys_mem = static_cast<float>(get_physical_memory_currently_used_by_current_process())/1000;
    }
    double value = (phys_mem/initial_phys_mem);
    printf("IS-DESPOT::[DEBUG] Currently used physical memory: %.0f MB. "
           "Increase in physical memory usage since initial execution: %.2f%%.\n", phys_mem, value);
    initial_ep = false;
}

void logging::init_cpu_currently_used() {
    FILE* file = fopen("/proc/stat", "r");
    fscanf(file, "cpu %llu %llu %llu %llu", &lastTotalUser, &lastTotalUserLow, &lastTotalSys, &lastTotalIdle);
    fclose(file);
}

double logging::get_cpu_currently_used() {
    double percent;
    FILE* file;
    unsigned long long totalUser, totalUserLow, totalSys, totalIdle, total;

    file = fopen("/proc/stat", "r");
    fscanf(file, "cpu %llu %llu %llu %llu", &totalUser, &totalUserLow, &totalSys, &totalIdle);
    fclose(file);

    if (totalUser < lastTotalUser || 
		totalUserLow < lastTotalUserLow || 
		totalSys < lastTotalSys || 
		totalIdle < lastTotalIdle) {
        // Overflow detection. Just skip this value.
        percent = -1.0;
    }
    else{
        total = (totalUser - lastTotalUser) + (totalUserLow - lastTotalUserLow) + (totalSys - lastTotalSys);
        percent = total;
        total += (totalIdle - lastTotalIdle);
        percent /= total;
        percent *= 100;
    }

    lastTotalUser = totalUser;
    lastTotalUserLow = totalUserLow;
    lastTotalSys = totalSys;
    lastTotalIdle = totalIdle;

    return percent;
}

void logging::init_cpu_currently_used_by_current_process() {
    FILE* file;
    struct tms timeSample;
    char line[128];

    lastCPU = times(&timeSample);
    lastSysCPU = timeSample.tms_stime;
    lastUserCPU = timeSample.tms_utime;

    file = fopen("/proc/cpuinfo", "r");
    numProcessors = 0;
    while(fgets(line, 128, file) != NULL){
        if (strncmp(line, "processor", 9) == 0) numProcessors++;
    }
    fclose(file);
}

double logging::get_cpu_currently_used_by_current_process() {
    struct tms timeSample;
    clock_t now;
    double percent;

    now = times(&timeSample);
    if (now <= lastCPU || timeSample.tms_stime < lastSysCPU || timeSample.tms_utime < lastUserCPU){
        // Overflow detection. Just skip this value.
        percent = -1.0;
    }
    else{
        percent = (timeSample.tms_stime - lastSysCPU) +
            (timeSample.tms_utime - lastUserCPU);
        percent /= (now - lastCPU);
        percent /= numProcessors;
        percent *= 100;
    }
    lastCPU = now;
    lastSysCPU = timeSample.tms_stime;
    lastUserCPU = timeSample.tms_utime;

    return percent;
}

long long logging::get_total_virtual_memory() {
    struct sysinfo memInfo;

    sysinfo (&memInfo);
    long long totalVirtualMem = memInfo.totalram;
    // Add other values in next statement to avoid int overflow on right hand side...
    totalVirtualMem += memInfo.totalswap;
    totalVirtualMem *= memInfo.mem_unit;
    return totalVirtualMem;
}

long long logging::get_virtual_memory_currently_used() {
    struct sysinfo memInfo;

    sysinfo (&memInfo);
    long long virtualMemUsed = memInfo.totalram - memInfo.freeram;
    // Add other values in next statement to avoid int overflow on right hand side...
    virtualMemUsed += memInfo.totalswap - memInfo.freeswap;
    virtualMemUsed *= memInfo.mem_unit;
    return virtualMemUsed;
}

int logging::parseLine(char* line){
    // This assumes that a digit will be found and the line ends in " Kb".
    int i = strlen(line);
    const char* p = line;
    while (*p <'0' || *p > '9') p++;
    line[i-3] = '\0';
    i = atoi(p);
    return i;
}

int logging::get_virtual_memory_currently_used_by_current_process() { //Note: this value is in KB!
    FILE* file = fopen("/proc/self/status", "r");
    int result = -1;
    char line[128];

    while (fgets(line, 128, file) != NULL){
        if (strncmp(line, "VmSize:", 7) == 0){
            result = parseLine(line);
            break;
        }
    }
    fclose(file);
    return result;
}

long long logging::get_total_physical_memory() {
    struct sysinfo memInfo;

    sysinfo (&memInfo);
    long long totalPhysMem = memInfo.totalram;
    // Multiply in next statement to avoid int overflow on right hand side...
    totalPhysMem *= memInfo.mem_unit;
    return totalPhysMem;
}

long long logging::get_physical_memory_currently_used() {
    struct sysinfo memInfo;

    sysinfo (&memInfo);
    long long physMemUsed = memInfo.totalram - memInfo.freeram;
    // Multiply in next statement to avoid int overflow on right hand side...
    physMemUsed *= memInfo.mem_unit;
    return physMemUsed;
}

int logging::get_physical_memory_currently_used_by_current_process() { //Note: this value is in KB!
    FILE* file = fopen("/proc/self/status", "r");
    int result = -1;
    char line[128];

    while (fgets(line, 128, file) != NULL){
        if (strncmp(line, "VmRSS:", 6) == 0){
            result = parseLine(line);
            break;
        }
    }
    fclose(file);
    return result;
}

} // namespace despot
