#ifndef LOGGING_H
#define LOGGING_H

#include <iostream>
#include <sstream>
#include <vector>
#include <map>
#include <chrono>
#include "sys/types.h"
#include "sys/sysinfo.h"
#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include "sys/times.h"

#include <despot/core/globals.h>
#include "../../../carla_car/src/param.h"
#include "../../../carla_car/src/math_utils.h"

// typedef for convenience
typedef std::chrono::steady_clock Clock;
typedef std::chrono::duration<double> dsec;

namespace despot {

class log_ostream: public std::ostream {
	private:
		class log_buf: public std::stringbuf {
			private:
				std::ostream& out_;
				std::string marker_;

			public:
				log_buf(std::ostream& out, std::string marker = "") : out_(out), marker_(marker) {;}
				~log_buf() {;}
				virtual std::streambuf* setbuf(char* s, std::streamsize n);
		};
		log_buf buffer_;

	public:
		log_ostream(std::ostream& out, std::string marker = "") : std::ostream(&buffer_), buffer_(out, marker) {;}

};

class logging {
	private:
		// debug resource consumption
		inline static double initial_virt_mem = -1.0;
		inline static double initial_phys_mem = -1.0;
		inline static bool initial_ep = true;
		inline static unsigned long long lastTotalUser, lastTotalUserLow, lastTotalSys, lastTotalIdle;
		inline static clock_t lastCPU, lastSysCPU, lastUserCPU;
		inline static int numProcessors;

		inline static const std::vector<std::string> markers_ = { 
			"CONNECTOR", "SIMULATOR", "DESPOT", "POMDP", "WORLD", "BELIEF", "IMPORTANCE_SAMPLING" 
		};
		inline static std::map<int, bool> scope_to_bool;
		inline static std::string stats_tracking_file_path;

		static std::vector<log_ostream*> initialize_log_streams();
		inline static std::vector<log_ostream*> streams_ = initialize_log_streams();

		static double get_cpu_currently_used();
		static double get_cpu_currently_used_by_current_process();
		static long long get_total_virtual_memory();
		static long long get_virtual_memory_currently_used();
		static int parseLine(char* line);
		static int get_virtual_memory_currently_used_by_current_process();
		static long long get_total_physical_memory();
		static long long get_physical_memory_currently_used();
		static int get_physical_memory_currently_used_by_current_process();

	public:
		// tracking of all scene simulation steps of one scene
		inline static std::vector<int> tree_nodes;
		inline static std::vector<int> backups;
		inline static std::vector<int> terminal_states;

		inline static std::vector<std::vector<double>> root_lower_bounds;
		inline static std::vector<std::vector<double>> root_upper_bounds;

		inline static std::vector<std::vector<double>> initial_lower_bounds;
		inline static std::vector<std::vector<double>> initial_upper_bounds;
 
		inline static std::vector<std::vector<double>> trial_depths;
		inline static std::vector<std::vector<double>> excess_uncertainties; 
		inline static std::vector<std::vector<double>> observations;
 
		// fined-grained execution time tracking
		inline static std::vector<double> search_execution_times;
		inline static std::vector<double> construct_tree_execution_times;
 
		inline static std::vector<double> trial_execution_times;
		inline static std::vector<double> expand_execution_times;
		inline static std::vector<double> qcreation_execution_times;
		inline static std::vector<double> copy_execution_times;
		inline static std::vector<double> step_execution_times;
		inline static std::vector<double> free_execution_times;
		inline static std::vector<double> norm_execution_times;
		inline static std::vector<double> vcreation_execution_times;
		inline static std::vector<double> init_bounds_execution_times;
 
		inline static std::vector<double> backup_execution_times;
 
		inline static std::vector<double> python_communication_times;
		inline static std::vector<int> python_interactions;

		inline static std::vector<std::vector<double>>  uncertainty_values;

		enum Scope {CONNECTOR, SIMULATOR, DESPOT, POMDP, WORLD, BELIEF, IMPORTANCE_SAMPLING, LAST_DUMMY};
		inline static std::map<std::string, Scope> marker_to_scope = {
			{"CONNECTOR", CONNECTOR},
			{"SIMULATOR", SIMULATOR},
			{"DESPOT", DESPOT},
			{"POMDP", POMDP},
			{"WORLD", WORLD},
			{"BELIEF", BELIEF},
			{"IMPORTANCE_SAMPLING", IMPORTANCE_SAMPLING}
		};

		static std::string bool_to_string(bool b);
		
		// debug information
		static void init_cpu_currently_used_by_current_process();
		static void init_cpu_currently_used();
		static void initialize_settings(bool default_setting);
		static void print_settings();
		static void print_config();
		static void print_pomdp_model_parameters();
		static void set_scope(Scope scope, bool setting);
		static bool get_scope(Scope scope);
		static log_ostream& stream(Scope scope);
		static void stream(Scope scope, std::ostream& out);
		static const std::vector<std::string> get_markers();
		static void initialize_tracking();
		static void track_episode(int num_steps, int episode);
		static void print_resource_consumption();
};

} // namespace despot

#define LOG(scope) if (!despot::logging::get_scope(scope)) ; else std::cout

#define logc LOG(despot::logging::CONNECTOR)
#define logs LOG(despot::logging::SIMULATOR)
#define logd LOG(despot::logging::DESPOT)
#define logis LOG(despot::logging::IMPORTANCE_SAMPLING)
#define logp LOG(despot::logging::POMDP)
#define logw LOG(despot::logging::WORLD)
#define logb LOG(despot::logging::BELIEF)

#endif
