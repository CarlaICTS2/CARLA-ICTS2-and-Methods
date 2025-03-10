#ifndef PED_STATE_H
#define PED_STATE_H

#include "coord.h"
#include "param.h"
#include "despot/interface/pomdp.h"

#include <vector>
#include <utility>
#include "string.h"

struct PedStruct {
	COORD pos;
	int goal;
	int id;
	double vel;

	// aren't there scenarios with different pedestrians speeds?
	// that would mean that ped speed has to be set when executing despot script
	PedStruct(){
        vel = ModelParams::PED_SPEED;
    }
	PedStruct(COORD pos_, int goal_, int id_) {
		pos = pos_;
		goal = goal_;
		id = id_;
        vel = ModelParams::PED_SPEED;
	}
};

class Pedestrian
{
public:
	double x, y;
	int id = -1;
	double last_update;

	Pedestrian() {}
	Pedestrian(double x_,double y_,int id_) {
		x = x_;
		y = y_;
		id = id_;
    }
	Pedestrian(double x_, double y_) {
		x = x_;
		y = y_;
	}
};

// why does the car not have a COORD struct for its position
struct CarStruct {
	int pos;
	COORD coordinates;
	double vel;
	double dist_travelled;
};

class PomdpState : public despot::State {
public:
	CarStruct car;
	std::vector<COORD> past_trajectory;
	int num;
	PedStruct peds[ModelParams::N_PED_IN];
	PomdpState() {}

	std::string text() const {
		return despot::concat(car.vel);
	}
};

class PomdpStateWorld : public despot::State {
public:
	CarStruct car;
	int num;
	PedStruct peds[ModelParams::N_PED_WORLD];
	PomdpStateWorld() {}

	std::string text() const {
		return despot::concat(car.vel);
	}
};

#endif
