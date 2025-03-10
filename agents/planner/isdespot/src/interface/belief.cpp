#include <despot/interface/belief.h>
#include <despot/interface/pomdp.h>

using namespace std;

namespace despot {

/* =============================================================================
 * ParticleBelief class
 * =============================================================================*/

Belief::Belief(const DSPOMDP* model) :
	model_(model) {
}

Belief::~Belief() {
}

string Belief::print() const {
	return "AbstractBelief";
}

ostream& operator<<(ostream& os, const Belief& belief) {
	os << (&belief)->print();
	return os;
}

} // namespace despot
