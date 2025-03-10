#ifndef CONFIG_H
#define CONFIG_H

#include <string>

#include <despot/util/util.h>

enum Agent {
	IS_DESPOT,
	HyLEAP,
	HyLEAR,
	HyPLAN,
	LEADER
};

inline static std::map<std::string, Agent> string_to_agent = {
	{"IS-DESPOT", IS_DESPOT},
	{"HyLEAP", HyLEAP},
	{"HyLEAR", HyLEAR},
	{"HyPLAN", HyPLAN},
	{"LEADER", LEADER}
};

namespace despot {
	
struct Config {
	// misc
	int PORT; // TCP port for connection to python
	std::string SCENARIO;
	std::string START_TIMESTAMP;
	// logging
	std::string OUTPUT_DIRECTORY;
	bool TRACKING;
	// agent-specific
	Agent AGENT;
	bool DECOUPLE; // decouple either hyleap or hyplan training 
	bool HACKY; // use Florian's hacky code for hyleap or hyplan
	bool NO_VERTICAL_PRUNING; // prevent uncertainty-based lower bound weighing of hyplan
	bool ATTENTION_SAMPLING;
	// meta
	bool AGGRESSIVE_BELIEF_UPDATES;
	bool MINIMAL_NOISE;
	bool NO_IMPORTANCE_SAMPLING;
	bool NO_NORMALIZATION;
	double TIMEOUT;
	double TIME_PER_PLANNING_STEP;
	int MAX_EPISODE_STEPS; // number of steps to run the simulation for any given episode
	double NOISE;
	unsigned SEED;
	// planning
	double TARGET_GAP;
	int MAX_SEARCH_DEPTH;
	double DISCOUNT;
	int PARTICLE_NUMBER;
	double GAP_REDUCTION_RATE; // xi * gap(root) is the target uncertainty at the root
	int MAX_POLICY_SIM_LEN; // maximum number of steps for simulating the default policy
	double PRUNING_CONSTANT;

	Config() :
		// misc
		PORT(-1),
		SCENARIO("ALL"),
		START_TIMESTAMP(""),
		OUTPUT_DIRECTORY(get_cwd()),
		TRACKING(false),
		// agent-specific
		AGENT(IS_DESPOT),
		DECOUPLE(false),
		HACKY(false),
		NO_VERTICAL_PRUNING(false),
		ATTENTION_SAMPLING(false),
		// meta
		AGGRESSIVE_BELIEF_UPDATES(false),
		MINIMAL_NOISE(false),
		NO_IMPORTANCE_SAMPLING(false),
		NO_NORMALIZATION(false),
		TIMEOUT(0.25),
		TIME_PER_PLANNING_STEP(0.25),
		MAX_EPISODE_STEPS(500),
		NOISE(0.1),
		SEED(42),
		// planning
		TARGET_GAP(1e-6),
		MAX_SEARCH_DEPTH(20),
		DISCOUNT(0.99),
		PARTICLE_NUMBER(500),
		GAP_REDUCTION_RATE(0.95),
		MAX_POLICY_SIM_LEN(90),
		PRUNING_CONSTANT(0)
		{}
};

} // namespace despot

#endif
