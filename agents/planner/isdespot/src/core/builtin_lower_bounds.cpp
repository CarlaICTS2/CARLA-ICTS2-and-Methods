#include <despot/core/builtin_lower_bounds.h>
#include <despot/interface/pomdp.h>
#include <despot/core/node.h>

using namespace std;

namespace despot {
/* =============================================================================
 * TrivialParticleLowerBound class
 * =============================================================================*/

TrivialParticleLowerBound::TrivialParticleLowerBound(const DSPOMDP* model) :
	ParticleLowerBound(model) {
}

ValuedAction TrivialParticleLowerBound::Value(
	const vector<State*>& particles) const {
	ValuedAction va = model_->GetBestAction();
	va.value *= State::Weight(particles) / (1 - Globals::Discount());
	return va;
}


} // namespace despot
