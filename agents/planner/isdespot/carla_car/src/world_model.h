#pragma once

#include "path.h"
#include "param.h"
#include "despot/util/random.h"

#include <limits>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <fstream>
#include <iostream>
#include "math_utils.h"
#include <map>

class PomdpState;
class State;
class PomdpStateWorld;
struct PedStruct;
struct CarStruct;
class Pedestrian;
class COORD;

/* Belief over ONE pedestrian */
struct PedBelief {
	int id;
    COORD pos;
    double vel;
    std::vector<double> prob_goals;

    // this is only relevant for leader
    std::vector<double> particle_importance_weights;
    std::vector<double> leader_attention;

    int sample_goal() const;
    int sample_goal_leader() const;
    std::size_t maxlikely_goal() const;
};

class WorldModel {
public:

    WorldModel();

    void print_probability_distribution(std::vector<double> pd, double threshold);
	bool isMovingAway(const PomdpState& state, int ped);
    bool isMovingAway(const PomdpStateWorld& state, int ped);
    bool isMovingAwayNew(const PomdpStateWorld& state, int ped);
    bool isMovingAwayNew(const PomdpState& state, int ped);

	void getClosestPed(
        const PomdpState& state, 
        int& closest_front_ped, 
        double& closest_front_dist, 
        int& closest_side_ped, 
        double& closest_side_dist
    );

	double getMinCarPedDist(const PomdpState& state);
	double getMinCarPedDistAllDirs(const PomdpState& state);
	int defaultPolicy(const std::vector<despot::State*>& particles);

    bool isGlobalGoal(const CarStruct& car);
	bool inFront(COORD ped_pos, int car) const;
    bool inFrontNew(COORD ped_pos, int car) const;

    bool inCollision(const PomdpState& state);
    bool inCollision(const PomdpStateWorld& state);
    
    bool inCollision(const PomdpState& state, int &id);
    bool inCollision(const PomdpStateWorld& state, int &id);
    
    int minStepToGoal(const PomdpState& state);

	void PedStep(PedStruct &ped, despot::Random& random);
    double ISPedStep(CarStruct &car, PedStruct &ped, despot::Random& random, bool debug);//importance sampling PedStep
    double ISPedStep(CarStruct &car, PedStruct &ped, despot::Random& random, double& x, double& y);//importance sampling PedStep
    void PedStepDeterministic(PedStruct& ped, int step);
	void RobStep(CarStruct &car, despot::Random& random);
    void RobVelStep(CarStruct &car, double acc, despot::Random& random);
    double ISRobVelStep(CarStruct &car, double acc, despot::Random& random);//importance sampling RobvelStep

    double pedMoveProb(COORD p0, COORD p1, int goal_id);
    void setPath(Path path);
    void updatePedBelief(PedBelief& b, const PedStruct& curr_ped, int step_counter);
    PedBelief initPedBelief(const PedStruct& ped);


	Path path;
    std::vector<double> goals;
    double freq;
    double in_front_angle_cos;
    int n_peds;
};

class WorldStateTracker {
public:
    typedef std::pair<float, Pedestrian> PedDistPair;
    COORD carpos;
    double carvel;
    std::vector<Pedestrian> ped_list;
    WorldModel& model;

    WorldStateTracker(WorldModel& _model): model(_model) {}

    void updatePed(const Pedestrian& ped);
    void updateCar(const COORD& car);
    void updateVel(double vel);
    void cleanPed();
    std::vector<PedDistPair> getSortedPeds();
    PomdpState getPomdpState();
};

class WorldBeliefTracker {
public:
    WorldModel& model;
    WorldStateTracker& stateTracker;
    CarStruct car;
    std::map<int, PedBelief> peds;
	std::vector<PedBelief> sorted_beliefs;

    WorldBeliefTracker(WorldModel& _model, WorldStateTracker& _stateTracker): model(_model), stateTracker(_stateTracker) {;}

    // return the probability distribution over all possible pedestrian goal directions given the pedestrian's ID
    std::vector<double> belief(int id);

    void update(int step_counter);
    PomdpState sample();
    std::vector<PomdpState> sample(int num);

    void PrintState(const despot::State& s, std::ostream& out = std::cout) const;
    void printBelief() const;
};

