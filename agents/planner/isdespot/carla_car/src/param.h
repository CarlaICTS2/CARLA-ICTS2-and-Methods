#ifndef MODELPARAMS_H
#define MODELPARAMS_H

#include <string>
#include "sin_cos_lookup_table.cpp"
#include "map"

namespace ModelParams {
    extern int BELIEF_ANGLE_DISCRETIZATION;

	extern double CRASH_PENALTY;
	extern double REWARD_FACTOR_VEL;
	extern double VEL_MAX;

	extern double NOISE_GOAL_ANGLE;
	extern double REWARD_BASE_CRASH_VEL;
	extern double BELIEF_SMOOTHING;
	extern double NOISE_ROBVEL;

	extern double IN_FRONT_ANGLE_DEG;

    extern double CAR_WIDTH;
    extern double CAR_LENGTH;
	extern double COLLISION_DISTANCE;
	extern double COLLISION_SIDE_DISTANCE;

	extern double front_nearmiss_margin;
	extern double side_nearmiss_margin;
	extern double back_nearmiss_margin;

    extern double goalX;
    extern double goalZ;

    extern int MAX_EPISODE_LENGTH;

	const int N_PED_IN = 1;
	const int N_PED_WORLD = 1;
	const int NUM_PEDESTRIANS = 1;

    extern double LASER_RANGE;

	extern double pos_rln; // position resolution
	extern double vel_rln; // velocity resolution

	extern double PATH_STEP;
	extern double GOAL_TOLERANCE;
	extern double PED_SPEED;

	extern int map_height;
	extern int map_width;

	extern bool debug;

	extern double control_freq;
	extern double AccSpeed;

	extern double GOAL_REWARD;



	// HyLEAP's LSTM state is composed of hidden and cell state
	const int LSTM_STATE_SIZE = 2 * 128;
	const int OBSERVATION_SIZE = 4 + 2 * NUM_PEDESTRIANS;

    extern SinCosLookupTable* lookupTable;
	
	enum {
		ACT_DEC = 0,
		ACT_CUR = 1,
		ACT_ACC = 2
	};

	extern std::map<int, std::string> action_idx_to_string;

};

#endif

