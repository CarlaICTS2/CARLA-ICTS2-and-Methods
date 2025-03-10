#include <despot/interface/default_policy.h>
#include <despot/interface/pomdp.h>
#include <unistd.h>

using namespace std;

namespace despot {

/* =============================================================================
 * Policy class
 * =============================================================================*/

DefaultPolicy::DefaultPolicy(const DSPOMDP* model, ParticleLowerBound* particle_lower_bound) :
	ScenarioLowerBound(model), particle_lower_bound_(particle_lower_bound) 
{
	if (particle_lower_bound_ == NULL) {
		printf("IS-DESPOT::[%s] Invalid particle lower bound: NULL.\n", __PRETTY_FUNCTION__);
		exit(-1);
	}
}

DefaultPolicy::~DefaultPolicy() { ; }

ValuedAction DefaultPolicy::Value(const vector<State*>& particles, RandomStreams& streams, History& history) const 
{	
	vector<State*> copy;
	for (int i = 0; i < particles.size(); i++) {
		copy.push_back(model_->Copy(particles[i]));
	}
	initial_depth_ = history.Size();
	ValuedAction va = RecursiveValue(copy, streams, history);

	for (int i = 0; i < copy.size(); i++) {
		model_->Free(copy[i]);
	}
	return va;
}

ValuedAction DefaultPolicy::RecursiveValue(const vector<State*>& particles, RandomStreams& streams, History& history) const 
{
	if (streams.Exhausted() || (history.Size() - initial_depth_ >= Globals::config.MAX_POLICY_SIM_LEN)) {
		return particle_lower_bound_->Value(particles);
	} else {
		ACT_TYPE action = Action(particles, streams, history);

		double value = 0;

		map<OBS_TYPE, vector<State*> > partitions;
		OBS_TYPE obs;
		double reward;
		for (int i = 0; i < particles.size(); i++) {
			State* particle = particles[i];
			bool terminal = model_->Step(*particle, streams.Entry(particle->scenario_id), action, reward, obs);
			value += reward * particle->weight;
			if (!terminal) {
				partitions[obs].push_back(particle);
			}
		}

		for (map<OBS_TYPE, vector<State*> >::iterator it = partitions.begin(); it != partitions.end(); it++) {
			OBS_TYPE obs = it->first;
			history.Add(action, obs);
			streams.Advance();
			ValuedAction va = RecursiveValue(it->second, streams, history);
			value += Globals::Discount() * va.value;
			streams.Back();
			history.RemoveLast();
		}
		return ValuedAction(action, value);
	}
}

void DefaultPolicy::Reset() { ; }

ParticleLowerBound* DefaultPolicy::particle_lower_bound() const 
{
	return particle_lower_bound_;
}

} // namespace despot
