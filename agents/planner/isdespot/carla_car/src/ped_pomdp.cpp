#include "despot/core/builtin_policy.h"
#include "despot/core/globals.h"
#include "state.h"
#include "ped_pomdp.h"
#include <algorithm>
#include <limits>

using namespace std;

PedPomdp::PedPomdp(WorldModel& _world_model) : world(_world_model), random_(despot::Random(despot::Seeds::Next())) { ; }


class PedPomdpParticleLowerBound : public despot::ParticleLowerBound 
{
	private:
		const PedPomdp* ped_pomdp_;

	public:
		PedPomdpParticleLowerBound(const despot::DSPOMDP* model) : 
		despot::ParticleLowerBound(model), ped_pomdp_(static_cast<const PedPomdp*>(model)) { ; }

		virtual despot::ValuedAction Value(const vector<despot::State*>& particles) const 
		{
			PomdpState* state = static_cast<PomdpState*>(particles[0]);

			int min_step = numeric_limits<int>::max();
			auto& carpos = ped_pomdp_->world.path[state->car.pos];
			double carvel = state->car.vel + 0.3;

			double min_dist = 100000;

			// Find mininum num of steps for car-pedestrian collision
			for (int i = 0; i < state->num; i++) {
				auto& p = state->peds[i];
				// is pedestrian NOT to the left, right or in front of the car?
				if (!ped_pomdp_->world.inFrontNew(p.pos, state->car.pos) ||
					// is agent vehicle moving away from pedestrian?
					ped_pomdp_->world.isMovingAwayNew(*state, i) ||
					// did the pedestrian stop moving (only happens when on sidewalk)
					ped_pomdp_->world.goals[state->peds[i].goal] == -1) { 
					// if not do not consider pedestrian
					break;
				}

				// 3.25 is maximum distance to collision boundary from front laser (see collsion.cpp)
				double dist = max(COORD::EuclideanDistance(carpos, p.pos) - 3.25, 0.0);
				min_dist = min(dist, min_dist);
				int step = int(ceil(ModelParams::control_freq * dist / ((p.vel + carvel))));
				min_step = min(step, min_step);
			}

			// the faster we go get better, i.e. higher the lower bound
			double move_penalty = ped_pomdp_->MovementPenalty(*state);

			// Case 1, no pedestrian: Constant car speed
			double value = move_penalty / (1 - despot::Globals::Discount());

			// Case 2, with pedestrians: Constant car speed, head-on collision with nearest neighbor
			if (min_step != numeric_limits<int>::max()) {
				double crash_penalty = ped_pomdp_->CrashPenalty(*state);
				value = (move_penalty) * (1 - despot::Globals::Discount(min_step)) / (1 - despot::Globals::Discount())
					+ crash_penalty * despot::Globals::Discount(min_step);
			}
			return despot::ValuedAction(ModelParams::ACT_CUR, despot::State::Weight(particles) * value);
		}
};


class PedPomdpScenarioLowerBound : public despot::DefaultPolicy
{
	protected:
		const PedPomdp* ped_pomdp_;

	public:
		PedPomdpScenarioLowerBound(const despot::DSPOMDP* model, despot::ParticleLowerBound* bound) :
			despot::DefaultPolicy(model, bound), ped_pomdp_(static_cast<const PedPomdp*>(model)) { ; }

		int Action(
			const std::vector<despot::State*>& particles, 
			despot::RandomStreams& streams, 
			despot::History& history
		) const 
		{
			return ped_pomdp_->world.defaultPolicy(particles);
		}
};


despot::ScenarioLowerBound* PedPomdp::CreateScenarioLowerBound(std::string particle_bound_name) const 
{
	if (particle_bound_name == "SMART") {
		return new PedPomdpScenarioLowerBound(this, new PedPomdpParticleLowerBound(this));
	} else {
		cerr << "Unsupported scenario lower bound: " << particle_bound_name << endl;
		exit(0);
	}
}


class PedPomdpParticleUpperBound : public despot::ParticleUpperBound {
	protected:
		const PedPomdp* ped_pomdp_;

	public:
		PedPomdpParticleUpperBound(const despot::DSPOMDP* model) : ped_pomdp_(static_cast<const PedPomdp*>(model)) {;}

		double Value(const despot::State& s) const 
		{
			const PomdpState& state = static_cast<const PomdpState&>(s);
			// unless we are in a collision state
			if (ped_pomdp_->world.inCollision(state)) { return ped_pomdp_->CrashPenalty(state); }
			// we return the goal reward diooscounted by the number of steps until the agent vehilce would reach it,
			// if it were to continue driving with constant velocity and other exo agents' positions kept fixed
			// in other words: there is no risk of collision
			return ModelParams::GOAL_REWARD * despot::Globals::Discount(ped_pomdp_->world.minStepToGoal(state)) + 0.2;
		}
};


despot::ScenarioUpperBound* PedPomdp::CreateScenarioUpperBound(std::string particle_bound_name) const
{
	if (particle_bound_name == "SMART") {
		return new PedPomdpParticleUpperBound(this);
	} else {
		cerr << "Unsupported scenario upper bound: " << particle_bound_name << endl;
		exit(0);
	}
}


uint64_t PedPomdp::Observe(const despot::State& state) const 
{
    const PomdpState &state_ = static_cast<const PomdpState&>(state);

    static vector<int> obs_vec;
    obs_vec.resize(state_.num * 2 + 2);

    obs_vec[0] = state_.car.pos;
    obs_vec[1] = int(state_.car.vel / ModelParams::vel_rln);

    int i = 2;
    for(int j = 0; j < state_.num; j ++) {
    	obs_vec[i++] = int(state_.peds[j].pos.x / ModelParams::pos_rln);
    	obs_vec[i++] = int(state_.peds[j].pos.y / ModelParams::pos_rln);
    }

	hash<vector<int>> myhash;
	return myhash(obs_vec);
}


double PedPomdp::GetMaxReward() const 
{
	cerr << __PRETTY_FUNCTION__ << " is called, but hasn't been implemented." << endl;
	exit(-1);
}


despot::ValuedAction PedPomdp::GetBestAction() const 
{
	cerr << __PRETTY_FUNCTION__ << " is called, but hasn't been implemented." << endl;
	exit(-1);
	return despot::ValuedAction(
		0, ModelParams::CRASH_PENALTY * (ModelParams::VEL_MAX*ModelParams::VEL_MAX + ModelParams::REWARD_BASE_CRASH_VEL)
	);
}


int PedPomdp::NumActiveParticles() const 
{
	return memory_pool_.num_allocated();
}


int PedPomdp::NumObservations() const 
{
	cerr << __PRETTY_FUNCTION__ << " is called, but hasn't been implemented." << endl;
	exit(-1);
    return std::numeric_limits<int>::max();
}


int PedPomdp::ParallelismInStep() const 
{
    return ModelParams::N_PED_IN;
}


// Very high cost for collision
double PedPomdp::CrashPenalty(const PomdpState& state) const 
{ 
    return -0.2 + ModelParams::CRASH_PENALTY * pow(0.5 + state.car.vel / ModelParams::VEL_MAX, 1.4);
}


// Very high cost for collision
double PedPomdp::CrashPenalty(const PomdpStateWorld& state) const 
{ 
    return -0.2 + ModelParams::CRASH_PENALTY * pow(0.5 + state.car.vel / ModelParams::VEL_MAX, 1.4);
}


// Avoid frequent dec or acc
double PedPomdp::ActionPenalty(int action) const 
{
    if (action == ModelParams::ACT_DEC || ModelParams::ACT_ACC) { return -0.01; }
    else { return 0; }
}


// Less penalty for longer distance travelled
double PedPomdp::MovementPenalty(const PomdpState& state) const 
{
    return (ModelParams::REWARD_FACTOR_VEL * (state.car.vel - ModelParams::VEL_MAX) / ModelParams::VEL_MAX) / 100.0;
}


// Less penalty for longer distance travelled
double PedPomdp::MovementPenalty(const PomdpStateWorld& state) const 
{
    return (ModelParams::REWARD_FACTOR_VEL * (state.car.vel - ModelParams::VEL_MAX) / ModelParams::VEL_MAX) / 100.0;
}


bool PedPomdp::Step(despot::State& state_, double rNum, int action, double& reward, uint64_t& obs) const {
	PomdpState& state = static_cast<PomdpState&>(state_);
	reward = 0.0;

	// CHECK: relative weights of each reward component
	// Terminate upon reaching goal

	if (world.isGlobalGoal(state.car) || state.car.dist_travelled >= 25) {
        reward = ModelParams::GOAL_REWARD;
		return true;
	}

 	// Safety control: collision; Terminate upon collision
    if(state.car.vel > 0.01 && world.inCollision(state) ) {
		reward = CrashPenalty(state); //, closest_ped, closest_dist);
		//cout << "Sample crash state: " << reward << "\n";
		return true;
	}

	// Forbidden actions
    double carvel = state.car.vel;
	/*if (action == ACT_CUR && 0.1 < carvel && carvel < 0.6) {
		reward = CrashPenalty(state);
		return true;
	}*/
    if (action == ModelParams::ACT_ACC && carvel >= ModelParams::VEL_MAX) {
		reward = CrashPenalty(state);
		return true;
    }
    if (action == ModelParams::ACT_DEC && carvel <= 0.01) {
		reward = CrashPenalty(state);
		return true;
    }

	// Smoothness control
	reward += ActionPenalty(action);

	// Speed control: Encourage higher speed
	reward += MovementPenalty(state);

	// State transition
	despot::Random random(rNum);
	double acc;
	if (action == ModelParams::ACT_DEC) {
		acc = -ModelParams::AccSpeed;
	} else if (action == ModelParams::ACT_ACC) {
		acc = ModelParams::AccSpeed / ModelParams::control_freq;
	} else {
		acc = 0;
	}

	world.RobStep(state.car, random);
	world.RobVelStep(state.car, acc, random);
	for (int i = 0; i < state.num; i++) {
		if (despot::Globals::config.MINIMAL_NOISE) { world.PedStepDeterministic(state.peds[i], 1); }
		else { world.PedStep(state.peds[i], random); }
	}
	// Observation
	obs = Observe(state);
	return false;
}


bool PedPomdp::ImportanceSamplingStep(
	despot::State& state_, 
	double rNum, 
	int action, 
	double& reward, 
	uint64_t& obs
) const 
{
	PomdpState& state = static_cast<PomdpState&>(state_);
	reward = 0.0;

	if (world.isGlobalGoal(state.car) || state.car.dist_travelled >= 25) {
        reward = ModelParams::GOAL_REWARD;
		return true;
	}

 	// Safety control: collision; Terminate upon collision
	//if (closest_front_dist < ModelParams::COLLISION_DISTANCE) {
    if(state.car.vel > 0.01 && world.inCollision(state) ) { /// collision occurs only when car is moving
		reward = CrashPenalty(state); //, closest_ped, closest_dist);
		//cout << "Sample crash state: " << reward << "\n";
		return true;
	}

	// Forbidden actions

    double carvel = state.car.vel;
	/*if (action == ACT_CUR && 0.1 < carvel && carvel < 0.6) {
		reward = CrashPenalty(state);
		return true;
	}*/
    if (action == ModelParams::ACT_ACC && carvel >= ModelParams::VEL_MAX) {
		reward = CrashPenalty(state);
		return true;
    }
    if (action == ModelParams::ACT_DEC && carvel <= 0.01) {
		reward = CrashPenalty(state);
		return true;
    }

	// Smoothness control
	reward += ActionPenalty(action);

	// Speed control: Encourage higher speed
	reward += MovementPenalty(state);

	// State transition
	despot::Random random(rNum);
	double acc;
	// decelerate is quicker than accelerate 
	if (action == ModelParams::ACT_DEC) {
		acc = -ModelParams::AccSpeed;
	} else if (action == ModelParams::ACT_ACC) {
		acc = ModelParams::AccSpeed / ModelParams::control_freq;
	} else {
		acc = 0;
	}

	// push back old ego vehicle coordinates and angle before they are overwritten
	state.past_trajectory.push_back(state.car.coordinates);
	world.RobStep(state.car, random);
	MyVector car_vec(state.car.coordinates.x, state.car.coordinates.y);
	//logb << "agent vehicle vector angle [DEG]: " << car_vec.GetAngleDeg() << ", [RAD]: " << car_vec.GetAngleRad() << endl;
	MyVector car_move(
		state.car.coordinates.x-state.past_trajectory.back().x, 
		state.car.coordinates.y-state.past_trajectory.back().y
	);
	//logb << "agent vehicle movement angle [DEG]: " << car_move.GetAngleDeg() << ", [RAD]: " << car_move.GetAngleRad() << endl;

	// always apply correct speed action: why wouldn't we? wtf
	state.weight *= world.ISRobVelStep(state.car, acc, random);
	
	for(int i=0;i<state.num;i++)
		// this only changes the weight according to ped-car angle
		state.weight *= world.ISPedStep(state.car, state.peds[i], random, false);

	// Observation
	obs = Observe(state);
	
	return false;
}


bool PedPomdp::ImportanceSamplingStep(
	despot::State& state_, 
	double rNum, 
	int action, 
	double& reward, 
	uint64_t& obs, 
	double& x, 
	double& y
) const 
{
    PomdpState& state = static_cast<PomdpState&>(state_);
    reward = 0.0;

    if (world.isGlobalGoal(state.car) || state.car.dist_travelled >= 25) {
        reward = ModelParams::GOAL_REWARD;
        return true;
    }

    // Safety control: collision; Terminate upon collision
    //if (closest_front_dist < ModelParams::COLLISION_DISTANCE) {
    if(state.car.vel > 0.01 && world.inCollision(state) ) { /// collision occurs only when car is moving
        reward = CrashPenalty(state); //, closest_ped, closest_dist);
        //cout << "Sample crash state: " << reward << "\n";
        return true;
    }

    // Forbidden actions

    double carvel = state.car.vel;
    /*if (action == ACT_CUR && 0.1 < carvel && carvel < 0.6) {
        reward = CrashPenalty(state);
        return true;
    }*/
    if (action == ModelParams::ACT_ACC && carvel >= ModelParams::VEL_MAX) {
        reward = CrashPenalty(state);
        return true;
    }
    if (action == ModelParams::ACT_DEC && carvel <= 0.01) {
        reward = CrashPenalty(state);
        return true;
    }

    // Smoothness control
    reward += ActionPenalty(action);

    // Speed control: Encourage higher speed
    reward += MovementPenalty(state);

    // State transition
    despot::Random random(rNum);

	double acc;
	if (action == ModelParams::ACT_DEC) {
		acc = -ModelParams::AccSpeed;
	} else if (action == ModelParams::ACT_ACC) {
		acc = ModelParams::AccSpeed;
	} else {
		acc = 0;
	}
	
	world.RobStep(state.car, random);
    state.weight *= world.ISRobVelStep(state.car, acc, random);
    //world.RobVelStep(state.car, acc, random);

    for(int i=0;i<state.num;i++)
        //world.PedStep(state.peds[i], random);
        state.weight *= world.ISPedStep(state.car, state.peds[i], random, x, y);

    // Observation
    obs = Observe(state);
    return false;
}


double PedPomdp::ObsProb(uint64_t obs, const despot::State& s, int action) const 
{
	cerr << __PRETTY_FUNCTION__ << " is called, but hasn't been implemented." << endl;
	exit(-1);
	// return obs == Observe(s);
}


vector<vector<double>> PedPomdp::GetBeliefVector(const std::vector<despot::State*> particles) const 
{
	cerr << __PRETTY_FUNCTION__ << " is called, but hasn't been implemented." << endl;
	exit(-1);
}


despot::Belief* PedPomdp::InitialBelief(const despot::State* start, std::string type) const 
{
	cerr << __PRETTY_FUNCTION__ << " is called, but hasn't been implemented." << endl;
	exit(-1);
}


void PedPomdp::PrintState(const despot::State& s, std::ostream& out) const 
{
	cerr << __PRETTY_FUNCTION__ << " is called, but hasn't been implemented." << endl;
	exit(-1);
}


void PedPomdp::PrintObs(const despot::State&state, uint64_t obs, std::ostream& out) const 
{
	cerr << __PRETTY_FUNCTION__ << " is called, but hasn't been implemented." << endl;
	exit(-1);
}


void PedPomdp::PrintAction(int action, std::ostream& out) const 
{
	cerr << __PRETTY_FUNCTION__ << " is called, but hasn't been implemented." << endl;
	exit(-1);
}


// print belief node id, depth, bounds, importance distribution and observation statistics,
// i.e. average, min, max and std values for agent vehicle position, velocity, theta 
// as well as pedestrain position across all particles that approximate the belief
void PedPomdp::PrintBelief(int id, int depth, double lower_bound, double upper_bound, std::vector<despot::State*> particles) const 
{
	// printing debug information about belief states is disabled
	if (!despot::logging::get_scope(despot::logging::DESPOT)) { return; }

	logd << "\t\t- depth(b" << id << "): " << depth << endl
		 << "\t\t- l(b" << id << "): " << lower_bound 
		 << ", u(b" << id << "): " << upper_bound << endl;

	std::vector<double> agent_vehicle_x;
	std::vector<double> agent_vehicle_y;
	// agent vehilce velocity and angle shouldn't really differ between particles so this is more of a sanity check
	std::vector<double> agent_vehicle_velocity; 
	std::vector<double> agent_vehicle_theta;
	std::vector<double> pedestrian_x;
	std::vector<double> pedestrian_y;

	// additional information which might be of interest
	std::vector<double> agent_vehicle_pedestrian_distance;
	std::vector<double> agent_vehicle_distance_travelled;
	std::vector<int> is_in_front;
	std::vector<int> is_moving_away;

	for (const auto& particle: particles) {
		const PomdpState* pomdp_particle = static_cast<PomdpState*>(particle);
		agent_vehicle_x.push_back(pomdp_particle->car.coordinates.x);
		agent_vehicle_y.push_back(pomdp_particle->car.coordinates.y);
		agent_vehicle_velocity.push_back(pomdp_particle->car.vel);
		agent_vehicle_theta.push_back(pomdp_particle->car.coordinates.theta);
		agent_vehicle_distance_travelled.push_back(pomdp_particle->car.dist_travelled);

		if (pomdp_particle->num == 1) {
			pedestrian_x.push_back(pomdp_particle->peds[0].pos.x);
			pedestrian_y.push_back(pomdp_particle->peds[0].pos.y);

			agent_vehicle_pedestrian_distance.push_back(
				COORD::EuclideanDistance(pomdp_particle->car.coordinates, pomdp_particle->peds[0].pos)
			);

			is_in_front.push_back(world.inFrontNew(pomdp_particle->peds[0].pos, pomdp_particle->car.pos));
			is_moving_away.push_back(world.isMovingAwayNew(*pomdp_particle, 0));
		}
	}

	logd << "\t\t- b" << id << " statistics (across all particles): " << endl;
	std::pair<double, double> agent_vehicle_x_avg_and_std = MathUtils::get_average_and_stdev(agent_vehicle_x);
	std::pair<double, double> agent_vehicle_x_min_and_max = MathUtils::get_min_and_max(agent_vehicle_x);
	logd << "\t\t\t- agent vehicle x (avg: " <<  agent_vehicle_x_avg_and_std.first 
		 << ", std: " << agent_vehicle_x_avg_and_std.second
		 << ", min: " << agent_vehicle_x_min_and_max.first 
		 << ", max: " << agent_vehicle_x_min_and_max.second << ")" << endl;

	std::pair<double, double> agent_vehicle_y_avg_and_std = MathUtils::get_average_and_stdev(agent_vehicle_y);
	std::pair<double, double> agent_vehicle_y_min_and_max = MathUtils::get_min_and_max(agent_vehicle_y);
	logd << "\t\t\t- agent vehicle y (avg: " <<  agent_vehicle_y_avg_and_std.first 
		 << ", std: " << agent_vehicle_y_avg_and_std.second
		 << ", min: " << agent_vehicle_y_min_and_max.first 
		 << ", max: " << agent_vehicle_y_min_and_max.second << ")" << endl;

	std::pair<double, double> agent_vehicle_velocity_avg_and_std = MathUtils::get_average_and_stdev(agent_vehicle_velocity);
	std::pair<double, double> agent_vehicle_velocity_min_and_max = MathUtils::get_min_and_max(agent_vehicle_velocity);
	logd << "\t\t\t- agent vehicle velocity (avg: " <<  agent_vehicle_velocity_avg_and_std.first 
		 << ", std: " << agent_vehicle_velocity_avg_and_std.second
		 << ", min: " << agent_vehicle_velocity_min_and_max.first 
		 << ", max: " << agent_vehicle_velocity_min_and_max.second << ")" << endl;

	std::pair<double, double> agent_vehicle_theta_avg_and_std = MathUtils::get_average_and_stdev(agent_vehicle_theta);
	std::pair<double, double> agent_vehicle_theta_min_and_max = MathUtils::get_min_and_max(agent_vehicle_theta);
	logd << "\t\t\t- agent vehicle angle [DEG] (avg: " <<  agent_vehicle_theta_avg_and_std.first 
		 << ", std: " << agent_vehicle_theta_avg_and_std.second
		 << ", min: " << agent_vehicle_theta_min_and_max.first 
		 << ", max: " << agent_vehicle_theta_min_and_max.second << ")" << endl;

	std::pair<double, double> pedestrian_x_avg_and_std = MathUtils::get_average_and_stdev(pedestrian_x);
	std::pair<double, double> pedestrian_x_min_and_max = MathUtils::get_min_and_max(pedestrian_x);
	logd << "\t\t\t- pedestrian x (avg: " <<  pedestrian_x_avg_and_std.first 
		 << ", std: " << pedestrian_x_avg_and_std.second
		 << ", min: " << pedestrian_x_min_and_max.first 
		 << ", max: " << pedestrian_x_min_and_max.second << ")" << endl;

	std::pair<double, double> pedestrian_y_avg_and_std = MathUtils::get_average_and_stdev(pedestrian_y);
	std::pair<double, double> pedestrian_y_min_and_max = MathUtils::get_min_and_max(pedestrian_y);
	logd << "\t\t\t- pedestrian y (avg: " <<  pedestrian_y_avg_and_std.first 
		 << ", std: " << pedestrian_y_avg_and_std.second
		 << ", min: " << pedestrian_y_min_and_max.first 
		 << ", max: " << pedestrian_y_min_and_max.second << ")" << endl;

	std::pair<double, double> distance_between_avg_and_std = MathUtils::get_average_and_stdev(agent_vehicle_pedestrian_distance);
	std::pair<double, double> distance_between_min_and_max = MathUtils::get_min_and_max(agent_vehicle_pedestrian_distance);
	logd << "\t\t\t- agent vehicle-pedestrian distance (avg: " <<  distance_between_avg_and_std.first 
		 << ", std: " << distance_between_avg_and_std.second
		 << ", min: " << distance_between_min_and_max.first 
		 << ", max: " << distance_between_min_and_max.second << ")" << endl;

	std::pair<double, double> distance_travelled_avg_and_std = MathUtils::get_average_and_stdev(agent_vehicle_distance_travelled);
	std::pair<double, double> distance_travelled_min_and_max = MathUtils::get_min_and_max(agent_vehicle_distance_travelled);
	logd << "\t\t\t- agent vehicle distance travelled (avg: " <<  distance_travelled_avg_and_std.first 
		 << ", std: " << distance_travelled_avg_and_std.second
		 << ", min: " << distance_travelled_min_and_max.first 
		 << ", max: " << distance_travelled_min_and_max.second << ")" << endl;

	logd << "\t\t\t- is_in_front(): " << MathUtils::get_average_and_stdev(is_in_front).first*100 << "%" << endl
		 << "\t\t\t- is_moving_away(): " << MathUtils::get_average_and_stdev(is_moving_away).first*100 << "%" << endl;

	PrintParticles(particles);
}


// prints the importance distribution encoded in the provided particles vector 
// in ascending order of pedestrian goal direction weight
void PedPomdp::PrintParticles(const std::vector<despot::State*> particles) const 
{
	// printing debug information about importance distributions is disabled
	if (!despot::logging::get_scope(despot::logging::IMPORTANCE_SAMPLING)) { return; }

	logis << "\t\t- importance distribution [particles: " 
		  << particles.size() << ", weight: " << despot::State::Weight(particles) << "]:" << endl;
	// how are pedestrian goal angles distributed across particles?
	std::map<int, int> ped_goal_count;
	// how much weight does each pedestrian goal angle have associated with it?
	std::map<int, double> ped_goal_weight;

	for(const auto& particle: particles) {
		const PomdpState* pomdp_particle = static_cast<PomdpState*>(particle);
		if (pomdp_particle->num == 0) {
			logis << "\t\t\t- no pedestrian in scene simulation step" << endl;
			return;
		} else if (pomdp_particle->num > 1) {
			printf("IS-DESPOT::[%s] Invalid number of pedestrians in scene simulation step: Expected 1, got %d.\n",
				   __PRETTY_FUNCTION__, pomdp_particle->num);
			exit(-1);
		} else {
			// goal direction in degree
			ped_goal_count[pomdp_particle->peds[0].goal*2]++;
			ped_goal_weight[pomdp_particle->peds[0].goal*2] += pomdp_particle->weight;
		}
	}
	for (const auto& el: despot::SortByValue(ped_goal_weight)) {
		logis << "\t\t\t- pedestrian goal angle [DEG]: " << el.first 
			  << ", weight: " << el.second 
			  << ", #particles: " << ped_goal_count[el.first] << endl;
	}
}


despot::State* PedPomdp::Allocate(int state_id, double weight) const 
{
	PomdpState* particle = memory_pool_.Allocate();
	particle->state_id = state_id;
	particle->weight = weight;
	return particle;
}


std::vector<despot::State*> PedPomdp::ConstructParticles(std::vector<PomdpState>& samples) 
{
	int num_particles=samples.size();
	std::vector<despot::State*> particles;
	for(int i=0;i<samples.size();i++) {
		PomdpState* particle = static_cast<PomdpState*>(Allocate(-1, 1.0/num_particles));
		(*particle) = samples[i];
		particle->SetAllocated();
		particle->weight = 1.0/num_particles;
		particles.push_back(particle);
	}
	return particles;
}


despot::State* PedPomdp::Copy(const despot::State* particle) const 
{
	PomdpState* new_particle = memory_pool_.Allocate();
	*new_particle = *static_cast<const PomdpState*>(particle);
	new_particle->SetAllocated();
	return new_particle;
}


void PedPomdp::Free(despot::State* particle) const 
{
	memory_pool_.Free(static_cast<PomdpState*>(particle));
}