#include <despot/interface/default_policy.h>
#include <despot/interface/pomdp.h>
#include <despot/interface/lower_bound.h>
#include <despot/interface/upper_bound.h>

using namespace std;

namespace despot {

/* =============================================================================
 * State class
 * =============================================================================*/

ostream& operator<<(ostream& os, const State& state) 
{
	cerr << __PRETTY_FUNCTION__ << " is called, but hasn't been implemented." << endl;
	exit(-1);
}

State::State() : state_id(-1) { ; }

State::State(int _state_id, double _weight) : state_id(_state_id), weight(_weight) { ; }

State::~State() { ; }

string State::text() const 
{
	cerr << __PRETTY_FUNCTION__ << " is called, but hasn't been implemented." << endl;
	exit(-1);
}

double State::Weight(const vector<State*>& particles) 
{
	double weight = 0;
	for (int i = 0; i < particles.size(); i++)
		weight += particles[i]->weight;
	return weight;
}
/* =============================================================================
 * StateIndexer class
 * =============================================================================*/
StateIndexer::~StateIndexer() { ; }

/* =============================================================================
 * StatePolicy class
 * =============================================================================*/
StatePolicy::~StatePolicy() { ; }

/* =============================================================================
 * DSPOMDP class
 * =============================================================================*/

DSPOMDP::DSPOMDP() { ; }

DSPOMDP::~DSPOMDP() { ; }

double DSPOMDP::Reward(const State& state, ACT_TYPE action) const 
{
	cerr << __PRETTY_FUNCTION__ << " is called, but hasn't been implemented." << endl;
	exit(-1);
}

bool DSPOMDP::ImportanceSamplingStep(State& state, double random_num, ACT_TYPE action, double& reward, OBS_TYPE& obs) const
{
	return Step(state, random_num, action, reward, obs);
}

vector<double> DSPOMDP::ImportanceWeight(vector<State*> particles) const
{
	vector <double> importance_weight;
	for(int i=0; i<particles.size();i++){
		importance_weight.push_back(particles[i]->weight);
	}
	return importance_weight;
}

vector<double> DSPOMDP::Feature(const State& state) const
{
	cerr << __PRETTY_FUNCTION__ << " is called, but hasn't been implemented." << endl;
	exit(-1);
}

State* DSPOMDP::CreateStartState(std::string type) const 
{
	cerr << __PRETTY_FUNCTION__ << " is called, but hasn't been implemented." << endl;
	exit(-1);
}

ParticleUpperBound* DSPOMDP::CreateParticleUpperBound(string name) const 
{
	cerr << __PRETTY_FUNCTION__ << " is called, but hasn't been implemented." << endl;
	exit(-1);
}

ScenarioUpperBound* DSPOMDP::CreateScenarioUpperBound(string name, string particle_bound_name) const 
{
	cerr << __PRETTY_FUNCTION__ << " is called, but hasn't been implemented." << endl;
	exit(-1);
}

ParticleLowerBound* DSPOMDP::CreateParticleLowerBound(string name) const 
{
	cerr << __PRETTY_FUNCTION__ << " is called, but hasn't been implemented." << endl;
	exit(-1);
}

ScenarioLowerBound* DSPOMDP::CreateScenarioLowerBound(string name, string particle_bound_name) const 
{
	cerr << __PRETTY_FUNCTION__ << " is called, but hasn't been implemented." << endl;
	exit(-1);
}

vector<State*> DSPOMDP::Copy(const vector<State*>& particles) const 
{
	vector<State*> copy;
	for (int i = 0; i < particles.size(); i++)
		copy.push_back(Copy(particles[i]));
	return copy;
}

} // namespace despot
