#ifndef GLOBALS_H
#define GLOBALS_H

#include <typeinfo>
#include <memory>
#include <set>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <list>
#include <algorithm>
#include <ctime>
#include <vector>
#include <queue>
#include <cmath>
#include <limits>
#include <sstream>
#include <map>
#include <inttypes.h>
#include <despot/config.h>

#include <despot/util/logging.h>

namespace despot {

typedef uint64_t OBS_TYPE;
typedef int ACT_TYPE;

namespace Globals {
extern const double NEG_INFTY;
extern const double POS_INFTY;
extern const double INF;
extern const double TINY;

extern Config config;

inline bool Fequals(double a, double b) {
	return std::fabs(a - b) < TINY;
}

inline double Discount() {
	return config.DISCOUNT;
}

inline double Discount(int d) {
	return std::pow(config.DISCOUNT, d);
}

} // namespace

} // namespace despot

#endif
