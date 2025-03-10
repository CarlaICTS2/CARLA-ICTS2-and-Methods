#include "param.h"

namespace ModelParams 
{
    int BELIEF_ANGLE_DISCRETIZATION = 2;

    double CRASH_PENALTY = -1.0;
    double REWARD_FACTOR_VEL = 0.5;
	// in m/s
	double VEL_MAX = 50 * 0.27778;

	double NOISE_GOAL_ANGLE = M_PI * 0.06; // 0 for debugging
    double BELIEF_SMOOTHING = 0.05;
    double NOISE_ROBVEL = 0.05;

    double REWARD_BASE_CRASH_VEL = 0.5;
    double IN_FRONT_ANGLE_DEG = 180;

    double COLLISION_DISTANCE = 5.0;
    double COLLISION_SIDE_DISTANCE = 1.8;

	double front_nearmiss_margin = ModelParams::COLLISION_DISTANCE;
	double side_nearmiss_margin = ModelParams::CAR_WIDTH / 2.0 + ModelParams::COLLISION_SIDE_DISTANCE;
	double back_nearmiss_margin = ModelParams::CAR_LENGTH + 0.1;

    double CAR_WIDTH = 1.994;
    double CAR_LENGTH = 4.182;

	int MAX_EPISODE_LENGTH = 500;

	double LASER_RANGE = 50.0;

	double pos_rln = 1.0; //0.25; // position resolution
	double vel_rln = 5 * 0.2778; // velocity resolution m/s

	double PATH_STEP = 0.2;
	double GOAL_TOLERANCE = 3;
	double PED_SPEED = 1.2;

    int map_height = 305;
    int map_width = 387;

	bool debug = false;

	double control_freq = 4;

	double AccSpeed = 1.3888; // 5 km/h 

	double GOAL_REWARD = 1.0;

	SinCosLookupTable* lookupTable = new SinCosLookupTable();

	std::map<int, std::string> action_idx_to_string = {{0, "DEC"}, {1, "MAIN"}, {2, "ACC"}};
}

