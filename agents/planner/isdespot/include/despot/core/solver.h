#ifndef SOLVER_H
#define SOLVER_H

#include <despot/core/globals.h>
#include <despot/core/history.h>

namespace despot {

class DSPOMDP;
class Belief;
struct ValuedAction;

/* =============================================================================
 * Solver class
 * =============================================================================*/

class Solver {
protected:
	const DSPOMDP* model_;
	Belief* belief_;
	History history_;

public:
	Solver(const DSPOMDP* model, Belief* belief);
	virtual ~Solver();

	/**
	 * Find the optimal action for current belief, and optionally return the
	 * found value for the action. Return the value Globals::NEG_INFTY if the
	 * value is not to be used.
	 */
	virtual ValuedAction Search() = 0;

	/**
	 * Update current belief, history, and any other internal states that is
	 * needed for Search() to function correctly.
	 */
	virtual void BeliefUpdate(ACT_TYPE action, OBS_TYPE obs);

	/**
	 * Set initial belief for planning. Make sure internal states associated with
	 * initial belief are reset. In particular, history need to be cleaned, and
	 * allocated memory from previous searches need to be cleaned if not.
	 */
	virtual void belief(Belief* b);
	Belief* belief();
};

} // namespace despot

#endif
