#ifndef STATISTICS_LOGGER_H
#define STATISTICS_LOGGER_H

#include <despot/interface/belief.h>
#include <despot/core/globals.h>
#include <despot/interface/pomdp.h>
#include <despot/interface/world.h>
#include <despot/util/util.h>

#include <despot/core/pomdp_world.h>

#include "numeric"
#include <chrono>


// typedef for convenience
typedef std::chrono::steady_clock Clock;
typedef std::chrono::duration<double> dsec;

namespace despot {

/* =============================================================================
 * EvalLog class
 * =============================================================================*/

class EvalLog {
private:
	std::vector<std::string> runned_instances;
	std::vector<int> num_of_completed_runs;
	std::string log_file_;

public:
	static time_t start_time;

	static double curr_inst_start_time;
	static double curr_inst_target_time; // Targetted amount of time used for each step
	static double curr_inst_budget; // Total time in seconds given for current instance
	static double curr_inst_remaining_budget; // Remaining time in seconds for current instance
	static int curr_inst_steps;
	static int curr_inst_remaining_steps;
	static double allocated_time;
	static double plan_time_ratio;

	// tracking of all scene simulation steps of one scene
	static std::vector<int> tree_nodes;
	static std::vector<int> backups;
	static std::vector<int> terminal_states;

	static std::vector<std::vector<double>> root_lower_bounds;
	static std::vector<std::vector<double>> root_upper_bounds;

	static std::vector<std::vector<double>> initial_lower_bounds;
	static std::vector<std::vector<double>> initial_upper_bounds;

	static std::vector<std::vector<double>> trial_depths;
	static std::vector<std::vector<double>> excess_uncertainties; 
	static std::vector<std::vector<double>> observations;
 
	// fined-grained execution time tracking
	static std::vector<double> search_execution_times;
	static std::vector<double> construct_tree_execution_times;

	static std::vector<double> trial_execution_times;
	static std::vector<double> expand_execution_times;
	static std::vector<double> qcreation_execution_times;
	static std::vector<double> copy_execution_times;
	static std::vector<double> step_execution_times;
	static std::vector<double> free_execution_times;
	static std::vector<double> norm_execution_times;
	static std::vector<double> vcreation_execution_times;
	static std::vector<double> init_bounds_execution_times;

	static std::vector<double> backup_execution_times;

	static std::vector<double> python_communication_times;
	static std::vector<int> python_interactions;

	static std::vector<std::vector<double>>  uncertainty_values;


	EvalLog(std::string log_file);

	void Save();
	void IncNumOfCompletedRuns(std::string problem);
	int GetNumCompletedRuns() const;
	int GetNumRemainingRuns() const;
	int GetNumCompletedRuns(std::string instance) const;
	int GetNumRemainingRuns(std::string instance) const;
	double GetUsedTimeInSeconds() const;
	double GetRemainingTimeInSeconds() const;

	// Pre-condition: curr_inst_start_time is initialized
	void SetInitialBudget(std::string instance);
	double GetRemainingBudget(std::string instance) const;
};

/* =============================================================================
 * Logger class
 * =============================================================================*/

/** Interface for evaluating a solver's performance by simulating how it runs
 * in a real.
 */
class Logger {
protected:
	DSPOMDP* model_;
	World* world_;

	State* state_;
	Belief* belief_;
	//std::string belief_type_;
	//Solver* solver_;
	clock_t start_clockt_;

	std::string world_type_;

	int step_;
	double target_finish_time_;
	std::ostream* out_;

	std::vector<double> discounted_round_rewards_;
	std::vector<double> undiscounted_round_rewards_;
	double reward_;
	double total_discounted_reward_;
	double total_undiscounted_reward_;

public:
	Logger(DSPOMDP* model, Belief* belief, Solver* solver, World* world,
			std::string world_type, clock_t start_clockt, std::ostream* out,
			double target_finish_time = -1, int num_steps = -1);
	virtual ~Logger();

	inline void out(std::ostream* o) {
		out_ = o;
	}

	inline void rewards(std::vector<double> rewards) {
		undiscounted_round_rewards_ = rewards;
	}

	inline std::vector<double> rewards() {
		return undiscounted_round_rewards_;
	}

	inline int step() {
		return step_;
	}
	inline double target_finish_time() {
		return target_finish_time_;
	}
	inline void target_finish_time(double t) {
		target_finish_time_ = t;
	}
	inline Belief* belief() {
		return belief_;
	}
	inline void belief(Belief* b) {
		belief_ = b;
	}
	inline DSPOMDP* model() {
		return model_;
	}
	inline void model(DSPOMDP* m) {
		model_ = m;
	}

	virtual void InitRound(State* state);
	virtual double EndRound(); // Return total undiscounted reward for this round.
	virtual bool SummarizeStep(int step, int round, bool terminal, ACT_TYPE action,
			OBS_TYPE obs, double step_start_t);
	virtual void PrintStatistics(int num_runs);

	double AverageUndiscountedRoundReward() const;
	double StderrUndiscountedRoundReward() const;
	double AverageDiscountedRoundReward() const;
	double StderrDiscountedRoundReward() const;

	void CheckTargetTime() const;
};

} // namespace despot

#endif
