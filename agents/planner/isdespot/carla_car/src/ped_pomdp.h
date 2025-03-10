#ifndef PED_POMDP_H
#define PED_POMDP_H

#include "despot/interface/pomdp.h"
#include "world_model.h"

#include <cmath>
#include <utility>
#include <string>

class PedPomdp : public despot::DSPOMDP {
private:
	mutable despot::MemoryPool<PomdpState> memory_pool_;
	mutable despot::Random random_;

public:
	WorldModel& world;
	int num;

	PedPomdp(WorldModel& _world_model);
/* =============================================================================
 * 								 STEP FUNCTIONS	
 * =============================================================================*/	
	bool Step(despot::State& state_, double rNum, int action, double& reward, uint64_t& obs) const;
	bool ImportanceSamplingStep(despot::State& state_, double rNum, int action, double& reward, uint64_t& obs) const;
    bool ImportanceSamplingStep(despot::State& state_, double rNum, int action, double& reward, uint64_t& obs, double& x, double& y) const;
/* =============================================================================
 * 								PENALTY FUNCTIONS	
 * =============================================================================*/	
    double CrashPenalty(const PomdpState& state) const; 
	double CrashPenalty(const PomdpStateWorld& state) const; 
    double ActionPenalty(int action) const;
    double MovementPenalty(const PomdpState& state) const;
    double MovementPenalty(const PomdpStateWorld& state) const;
/* =============================================================================
 * 						STATE, OBSERVATION & BELIEF FUNCTIONS	
 * =============================================================================*/	
	uint64_t Observe(const despot::State&) const;
	double ObsProb(uint64_t z, const despot::State& s, int action) const;
    despot::State* CreateStartState(std::string type = "DEFAULT") const { return 0; }
	std::vector<std::vector<double>> GetBeliefVector(const std::vector<despot::State*> particles) const;
	despot::Belief* InitialBelief(const despot::State* start, std::string type) const;
/* =============================================================================
 * 								BOUND FUNCTIONS	
 * =============================================================================*/	
	despot::ScenarioUpperBound* CreateScenarioUpperBound(std::string particle_bound_name) const;
	despot::ScenarioLowerBound* CreateScenarioLowerBound(std::string particle_bound_name) const;
/* =============================================================================
 * 								HELPER FUNCTIONS	
 * =============================================================================*/		
	inline int NumActions() const { return 3; }
	despot::ValuedAction GetBestAction() const;
	double GetMaxReward() const;
	int NumActiveParticles() const;
	int ParallelismInStep() const;
    int NumObservations() const;
/* =============================================================================
 * 								 	DEBUG
 * =============================================================================*/
	void PrintState(const despot::State& state, std::ostream& out = std::cout) const;
	void PrintObs(const despot::State& state, uint64_t obs, std::ostream& out = std::cout) const;
	void PrintAction(int action, std::ostream& out = std::cout) const;
	void PrintBelief(int id, int depth, double lower_bound, double upper_bound, std::vector<despot::State*> particles) const;
	void PrintParticles(const std::vector<despot::State*> particles) const;
/* =============================================================================
 * 								MEMORY MANAGEMENT	
 * =============================================================================*/
	despot::State* Allocate(int state_id, double weight) const;
	despot::State* Copy(const despot::State* particle) const;
	void Free(despot::State* particle) const;
	std::vector<despot::State*> ConstructParticles(std::vector<PomdpState> & samples);
};
#endif

