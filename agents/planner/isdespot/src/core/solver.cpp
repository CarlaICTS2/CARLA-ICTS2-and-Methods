#include <despot/core/solver.h>
#include <despot/util/logging.h>
#include <despot/interface/pomdp.h>
#include <despot/interface/belief.h>

using namespace std;

namespace despot {
/* =============================================================================
 * Solver class
 * =============================================================================*/

Solver::Solver(const DSPOMDP* model, Belief* belief) :
	model_(model),
	belief_(belief),
	history_(History()) {
}

Solver::~Solver() {
}

void Solver::BeliefUpdate(ACT_TYPE action, OBS_TYPE obs) {
	double start = get_time_second();

	belief_->Update(action, obs);
	history_.Add(action, obs);
}

void Solver::belief(Belief* b) {
	belief_ = b;
	history_.Truncate(0);
}

Belief* Solver::belief() {
	return belief_;
}

} // namespace despot
