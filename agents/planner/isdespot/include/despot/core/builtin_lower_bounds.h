#ifndef BUILTIN_LOWER_BOUND_H
#define BUILTIN_LOWER_BOUND_H

#include <despot/interface/lower_bound.h>

namespace despot {
/* =============================================================================
 * TrivialParticleLowerBound class
 * =============================================================================*/

class TrivialParticleLowerBound: public ParticleLowerBound {
public:
	TrivialParticleLowerBound(const DSPOMDP* model);

public:
	virtual ValuedAction Value(const std::vector<State*>& particles) const;
};

} // namespace despot

#endif
