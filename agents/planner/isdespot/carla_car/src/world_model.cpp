#include "state.h"
#include "coord.h"
#include "world_model.h"

#ifndef M_PI
#define M_PI    3.14159265358979323846f
#endif

using namespace std;

enum {
    ACT_DEC = 0,
    ACT_CUR = 1,
    ACT_ACC = 2
};

WorldModel::WorldModel(): freq(ModelParams::control_freq),
                          in_front_angle_cos(cos(ModelParams::IN_FRONT_ANGLE_DEG / 180.0 * M_PI)) 
{
    logw << __FUNCTION__ << endl;
                
    // theoretically, pedestrians can turn in any direction (360°)
    // we reduce granurality by discretization, i.e. binning with width 2, 
    // resulting in 180 possible goal positions
    // numerically, goal positions range from 0 ~ 6 (which is 360° in radiant)
    for(int i = 0; i < 360; i+= ModelParams::BELIEF_ANGLE_DISCRETIZATION){
        // 1° = 3.14 / 180 = 0.017 radiant
        // convert degree to radiant in steps of 2
        double angle = ((double) i) * (M_PI / 180.0);
        goals.push_back(angle);
        logw << "\t- goal angle (deg) " << i << ", (rad) " << angle << endl;
    }
    // pedestrian comes to a halt, i.e. stop intention
    goals.push_back(-1);
}


bool WorldModel::isGlobalGoal(const CarStruct& car) {
    double d = COORD::EuclideanDistanceSimple(path[car.pos], path[path.size()-1]);
    return (d<(ModelParams::GOAL_TOLERANCE*ModelParams::GOAL_TOLERANCE));
}

// reactive controller:
// if breaking distance is greater than distance to closest pedestrian with current velocity
//     decellerate
// if breaking distance is less than distance to closest pedestrian with increased velocity
//     acellerate
// else
//     maintain
int WorldModel::defaultPolicy(const std::vector<despot::State*>& particles)  {
	const PomdpState *state=static_cast<const PomdpState*>(particles[0]);
        //printf("reactive controller\n");
        double mindist = std::numeric_limits<double>::infinity();
        auto& carpos = path[state->car.pos];
        double carvel = state->car.vel + 1.5;

        double mindist_all = std::numeric_limits<double>::infinity();

        // Closest pedestrian in front
        for (int i=0; i<state->num; i++) {
    	    auto& p = state->peds[i];

    	    double d = COORD::EuclideanDistance(carpos, p.pos);
            if (d >= 0 && d < mindist_all)
                mindist_all = d;

    	    if(!WorldModel::inFront(p.pos, state->car.pos))
                continue;

            d = COORD::EuclideanDistance(carpos, p.pos);
            if (d >= 0 && d < mindist)
    			mindist = d;
        }

    	double brakingDistance = ((carvel*3.6)*(carvel*3.6)) / (250*0.8);

        if(state->car.vel > 0.1 && (brakingDistance > mindist || mindist_all < 2)){
            return ACT_DEC;
        }

        double nextBrakingDistance = (((carvel+ModelParams::AccSpeed)*3.6)*((carvel+ModelParams::AccSpeed)*3.6)) / (250*0.8);
        if((state->car.vel + 0.1 < ModelParams::VEL_MAX) && (nextBrakingDistance < mindist)){
            return ACT_ACC;
        }

        return ACT_CUR;
}


bool WorldModel::inFrontNew(COORD ped_pos, int car) const {
	const COORD& car_pos = path[car];
    // gives the angle of the pedestrian relative to the agent vehicle
    // agent vehicle's position is equivalent to coordinate system's origin
    // angle increases clockwise, i.e 
    // pedestrian on LHS level with agent vehicle -> angle = 0°
    // pedestrian right in fron of vehicle (perfectly aligned) -> angle = 90°
    // pedestrian on RHS level with agent vehicle -> angle = 180°
    MyVector car_to_ped_vector(car_pos.x - ped_pos.x, car_pos.y - ped_pos.y);
    double car_to_ped_angle = car_to_ped_vector.GetAngleDeg();
    // is pedestrian to the left, right or in front of the car?
    if (car_to_ped_angle < -10 || car_to_ped_angle > 190) { return false; }
    else { return true; }
}


bool WorldModel::inFront(COORD ped_pos, int car) const {
    if(ModelParams::IN_FRONT_ANGLE_DEG > 180.0) {
        // inFront check is disabled
        return true;
    }

	const COORD& car_pos = path[car];
	const COORD& forward_pos = path[path.forward(car, 1.0)];

	double d0 = COORD::EuclideanDistance(car_pos, ped_pos);
	if (d0 <= 0.7) return false;

	double d1 = COORD::EuclideanDistance(car_pos, forward_pos);
	if (d1 <= 0) return false;

	double dot = DotProduct(
        forward_pos.x - car_pos.x, forward_pos.y - car_pos.y, ped_pos.x - car_pos.x, ped_pos.y - car_pos.y
    );
	double cosa = dot / (d0 * d1);

    if(!(cosa <= 1.0 + 1E-8 && cosa >= -1.0 - 1E-8)){
        cout << "\nCosa value: " << cosa << "\n";
        cout << "Dot value: " << dot << "\n";
        cout << "carpos value: " << car_pos.x << ", " << car_pos.y << "\n";
        cout << "pedpos value: " << ped_pos.x << ", " << ped_pos.y << "\n";
        cout << "d0 value: " << d0 << "\n";
        cout << "d1 value: " << d1 << "\n";
        printf("IS-DESPOT::[WORLD] Invalid angle calculation values.\n");
        exit(-1);
    }
    return cosa > in_front_angle_cos;
}

bool WorldModel::isMovingAwayNew(const PomdpStateWorld& state, int ped) {
    const auto& carpos = path[state.car.pos];
	const auto& nextcarpos = path[path.forward(state.car.pos, 1.0)];
	const auto& pedpos = state.peds[ped].pos;
    double current_distance = COORD::EuclideanDistanceSimple(pedpos, carpos);
    double next_distance = COORD::EuclideanDistanceSimple(pedpos, nextcarpos);
    return next_distance > current_distance;
}

bool WorldModel::isMovingAwayNew(const PomdpState& state, int ped) {
    const auto& carpos = path[state.car.pos];
	const auto& nextcarpos = path[path.forward(state.car.pos, 1.0)];
	const auto& pedpos = state.peds[ped].pos;
    double current_distance = COORD::EuclideanDistanceSimple(pedpos, carpos);
    double next_distance = COORD::EuclideanDistanceSimple(pedpos, nextcarpos);
    return next_distance > current_distance;
}


bool WorldModel::isMovingAway(const PomdpStateWorld& state, int ped) {
    const auto& carpos = path[state.car.pos];
	const auto& nextcarpos = path[path.forward(state.car.pos, 1.0)];

	const auto& pedpos = state.peds[ped].pos;
	double goalAngle = goals[state.peds[ped].goal];
	MyVector goal_vec(ModelParams::lookupTable->cos(goalAngle),ModelParams::lookupTable->sin(goalAngle));
    COORD goalpos(pedpos.x + goal_vec.dw, pedpos.y + goal_vec.dh);

	if (goalAngle == -1) { return false; }

	return DotProduct(goalpos.x - pedpos.x, goalpos.y - pedpos.y, nextcarpos.x - carpos.x, nextcarpos.y - carpos.y) > 0;
}

bool WorldModel::isMovingAway(const PomdpState& state, int ped) {
    const auto& carpos = path[state.car.pos];
	const auto& nextcarpos = path[path.forward(state.car.pos, 1.0)];

	const auto& pedpos = state.peds[ped].pos;
	double goalAngle = goals[state.peds[ped].goal];
	MyVector goal_vec(ModelParams::lookupTable->cos(goalAngle),ModelParams::lookupTable->sin(goalAngle));
    COORD goalpos(pedpos.x + goal_vec.dw, pedpos.y + goal_vec.dh);

	if (goalAngle == -1) { return false; }

	return DotProduct(goalpos.x - pedpos.x, goalpos.y - pedpos.y, nextcarpos.x - carpos.x, nextcarpos.y - carpos.y) > 0;
}

/**
 * H: center of the head of the car
 * N: a point right in front of the car
 * M: an arbitrary point
 *
 * Check whether M is in the safety zone
 */
bool inCollision(const double Mx, const double My, const COORD& car_pos, bool debug);

bool WorldModel::inCollision(const PomdpState& state) {
    const int car = state.car.pos;
	const COORD& car_pos = path[car];

    for(int i = 0; i < state.num; i++) {
        const COORD& pedpos = state.peds[i].pos;

        // COORD::EuclideanDistance(pedpos,car_pos) >= 5
        if(COORD::EuclideanDistanceSimple(pedpos,car_pos) >= 50){
            continue;
        } 

        if(::inCollision(pedpos.x, pedpos.y, car_pos, false)) {
            return true;
        }
    }
    return false;
}

/**
 * Checks whether a collision occurs in the current state.
 * @param state the current state
 * @return true, if a collision occurs, else otherwise
 */
bool WorldModel::inCollision(const PomdpStateWorld& state) {
    const int car = state.car.pos;
    const COORD& car_pos = path[car];

    logs << __FUNCTION__ << endl;

    // 

    // 1. pedestrian on front-side collision edge
    COORD front_side_collision(car_pos.x, car_pos.y - ModelParams::CAR_LENGTH / 2);
    logs << "\t- front_side_collision(x: " << front_side_collision.x 
         << ", y: " << front_side_collision.y << "): " 
         << ::inCollision(front_side_collision.x, front_side_collision.y, car_pos, false) << endl;
    
    // 2. pedestrian on right-side collision edge
    COORD right_side_collision(car_pos.x + ModelParams::CAR_WIDTH / 2, car_pos.y);
    logs << "\t- right_side_collision(x: " << right_side_collision.x 
         << ", y: " << right_side_collision.y << "): " 
         << ::inCollision(right_side_collision.x, right_side_collision.y, car_pos, false) << endl;
    
    // 3. pedestrian on back-side collision edge
    COORD back_side_collision(car_pos.x, car_pos.y + ModelParams::CAR_LENGTH / 2);
    logs << "\t- back_side_collision(x: " << back_side_collision.x 
         << ", y: " << back_side_collision.y << "): " 
         << ::inCollision(back_side_collision.x, back_side_collision.y, car_pos, false) << endl;
    
    // 4. pedestrian on left-side collision edge
    COORD left_side_collision(car_pos.x - ModelParams::CAR_WIDTH / 2, car_pos.y);
    logs << "\t- left_side_collision(x: " << left_side_collision.x 
         << ", y: " << left_side_collision.y << "): " 
         << ::inCollision(left_side_collision.x, left_side_collision.y, car_pos, false) << endl;


    // 5. pedestrian on front-side nearmiss edge
    COORD front_side_nearmiss(car_pos.x, car_pos.y - ModelParams::front_nearmiss_margin);
    logs << "\t- front_side_nearmiss(x: " << front_side_nearmiss.x 
         << ", y: " << front_side_nearmiss.y << "): " 
         << ::inCollision(front_side_nearmiss.x, front_side_nearmiss.y, car_pos, false) << endl;

    // 6. pedestrian on right-side nearmiss edge
    COORD right_side_nearmiss(car_pos.x + ModelParams::side_nearmiss_margin - 0.1, car_pos.y);
    logs << "\t- right_side_nearmiss(x: " << right_side_nearmiss.x 
         << " [-0.1 corrected], y: " << right_side_nearmiss.y << "): " 
         << ::inCollision(right_side_nearmiss.x, right_side_nearmiss.y, car_pos, false) << endl;

    // 7. pedestrian on back-side nearmiss edge
    COORD back_side_nearmiss(car_pos.x, car_pos.y + ModelParams::back_nearmiss_margin - 1);
    logs << "\t- back_side_nearmiss(x: " << back_side_nearmiss.x 
         << ", y: " << back_side_nearmiss.y << " [-1 corrected]): " 
         << ::inCollision(back_side_nearmiss.x, back_side_nearmiss.y, car_pos, false) << endl;

    // 8. pedestrian on left-side nearmiss edge
    COORD left_side_nearmiss(car_pos.x - ModelParams::side_nearmiss_margin, car_pos.y);
    logs << "\t- left_side_nearmiss(x: " << left_side_nearmiss.x 
         << ", y: " << left_side_nearmiss.y << "): " 
         << ::inCollision(left_side_nearmiss.x, left_side_nearmiss.y, car_pos, false) << endl;


    // 9. pedestrian outside front-side nearmiss edge
    COORD front_side_outside(car_pos.x, car_pos.y - ModelParams::front_nearmiss_margin - 1);
    logs << "\t- front_side_outside(x: " << front_side_outside.x 
         << ", y: " << front_side_outside.y << " [-1 corrected]): " 
         << ::inCollision(front_side_outside.x, front_side_outside.y, car_pos, false) << endl;

    // 10. pedestrian outside right-side nearmiss edge
    COORD right_side_outside(car_pos.x + ModelParams::side_nearmiss_margin + 0.1, car_pos.y);
    logs << "\t- right_side_outside(x: " << right_side_outside.x 
         << " [+0.1 corrected], y: " << right_side_outside.y << "): " 
         << ::inCollision(right_side_outside.x, right_side_outside.y, car_pos, false) << endl;

    // 11. pedestrian outisde back-side nearmiss edge
    COORD back_side_outisde(car_pos.x, car_pos.y + ModelParams::back_nearmiss_margin -0.9);
    logs << "\t- back_side_outisde(x: " << back_side_outisde.x 
         << ", y: " << back_side_outisde.y << " [-0.9 corrected]): " 
         << ::inCollision(back_side_outisde.x, back_side_outisde.y, car_pos, false) << endl;

    // 12. pedestrian outside left-side nearmiss edge
    COORD left_side_outside(car_pos.x - ModelParams::side_nearmiss_margin - 0.1, car_pos.y);
    logs << "\t- left_side_outside(x: " << left_side_outside.x 
         << ", y: " << left_side_outside.y << "): " 
         << ::inCollision(left_side_outside.x, left_side_outside.y, car_pos, false) << endl;

    exit(-1);

    for(int i=0; i<state.num; i++) {
        const COORD& pedpos = state.peds[i].pos;

        if(COORD::EuclideanDistanceSimple(pedpos, car_pos) >= 50.0){
            continue;
        } 

        if(::inCollision(pedpos.x, pedpos.y, car_pos, true)) {
            return true;
        }
    }
    return false;
}

/**
 * Checks whether a collision occurs in the current LOCAL state. If a collusion occurs,
 * the id of ONE pedestrian is stored in @param id
 * @param state the current state
 * @param id the id of ONE participant that the car collides with, -1 if no collision
 * @return true, if a collision occurs, else otherwise
 */
bool WorldModel::inCollision(const PomdpState& state, int &id) {
	id=-1;
    const int car = state.car.pos;
	const COORD& car_pos = path[car];

    for(int i=0; i<state.num; i++) {
        const COORD& pedpos = state.peds[i].pos;

        if(COORD::EuclideanDistanceSimple(pedpos,car_pos) >= 50){
            continue;
        } 

        if(::inCollision(pedpos.x, pedpos.y, car_pos, false)) {
        	id=state.peds[i].id;
            return true;
        }
    }
    return false;
}

/**
 * Checks whether a collision occurs in the current WORLD state. If a collusion occurs,
 * the id of ONE pedestrian is stored in @param id
 * @param state the current state
 * @param id the id of ONE participant that the car collides with, -1 if no collision
 * @return true, if a collision occurs, else otherwise
 */
bool WorldModel::inCollision(const PomdpStateWorld& state, int &id) {
    id=-1;
    const int car = state.car.pos;
    const COORD& car_pos = path[car];

    for(int i=0; i<state.num; i++) {
        const COORD& pedpos = state.peds[i].pos;

        if(COORD::EuclideanDistanceSimple(pedpos,car_pos) >= 50){
            continue;
        } 

        if(::inCollision(pedpos.x, pedpos.y, car_pos, false)) {
        	id=state.peds[i].id;
            return true;
        }
    }
    return false;
}


void WorldModel::getClosestPed(const PomdpState& state, 
                               int& closest_front_ped,
                               double& closest_front_dist,
                               int& closest_side_ped,
                               double& closest_side_dist)
{
	closest_front_ped = -1;
	closest_front_dist = numeric_limits<double>::infinity();
	closest_side_ped = -1;
	closest_side_dist = numeric_limits<double>::infinity();
    const auto& carpos = path[state.car.pos];

	// Find the closest pedestrian in front
    for(int i=0; i<state.num; i++) {
		const auto& p = state.peds[i];
		bool front = inFront(p.pos, state.car.pos);
        double d = COORD::EuclideanDistance(carpos, p.pos);
        if (front) {
			if (d < closest_front_dist) {
				closest_front_dist = d;
				closest_front_ped = i;
			}
		} else {
			if (d < closest_side_dist) {
				closest_side_dist = d;
				closest_side_ped = i;
			}
		}
    }
}

// get the min distance between car and the peds in its front
double WorldModel::getMinCarPedDist(const PomdpState& state) {
    double mindist = numeric_limits<double>::infinity();
    const auto& carpos = path[state.car.pos];

	// Find the closest pedestrian in front
    for(int i=0; i<state.num; i++) {
		const auto& p = state.peds[i];
		if(!inFront(p.pos, state.car.pos)) continue;
        double d = COORD::EuclideanDistance(carpos, p.pos);
        if (d >= 0 && d < mindist) mindist = d;
    }

	return mindist;
}

///get the min distance between car and the peds
double WorldModel::getMinCarPedDistAllDirs(const PomdpState& state) {
    double mindist = numeric_limits<double>::infinity();
    const auto& carpos = path[state.car.pos];

	// Find the closest pedestrian in front
    for(int i=0; i<state.num; i++) {
		const auto& p = state.peds[i];
        double d = COORD::EuclideanDistance(carpos, p.pos);
        if (d >= 0 && d < mindist) mindist = d;
    }

	return mindist;
}

int WorldModel::minStepToGoal(const PomdpState& state) {
    double d = COORD::EuclideanDistance(path[state.car.pos], path[path.size()-1]);
    if (d < 0) d = 0;
    return int(ceil(d / (ModelParams::VEL_MAX/freq)));
}

void WorldModel::PedStep(PedStruct &ped, despot::Random& random) {
    const double& goal = goals[ped.goal];
	if (goal == -1) {  //stop intention
		return;
	}

    double a = goal;
	double noise = random.NextGaussian() * ModelParams::NOISE_GOAL_ANGLE;
    a += noise;

	//TODO noisy speed
    MyVector move(a, ped.vel/freq, 0);

    if((std::isnan(move.dw)) || (std::isnan(move.dh))){
        cout << "a: " << a << "\n";
        cout << "ped.vel/freq" << (ped.vel/freq) << "\n";
        double dw=(ped.vel/freq)*ModelParams::lookupTable->cos(a);
        double dh=(ped.vel/freq)*ModelParams::lookupTable->sin(a);
        cout << "sin: " << ModelParams::lookupTable->sin(a) << "\n";
        cout << "cos: " << ModelParams::lookupTable->cos(a) << "\n";
        cout << "dw: " << dw << "\n";
        cout << "dh: " << dh << "\n";
        cout << "dw nan: " << dw << "\n";
        cout << "dh nan: " << dh << "\n";
        printf("IS-DESPOT::[WORLD] Invalid pedestrina step calcluation.\n");
        exit(-1);
    }

    ped.pos.x += move.dw;
    ped.pos.y += move.dh;

    return;
}

float normal_pdf(float x, float m, float s)
{
    static const float inv_sqrt_2pi = 0.3989422804014327;
    float a = (x - m) / s;

    return inv_sqrt_2pi / s * std::exp(-0.5f * a * a);
}


double gaussian_prob(double x, double stddev) {
    double a = 1.0 / stddev / sqrt(2 * M_PI);
    double b = - x * x / 2.0 / (stddev * stddev);
    return a * exp(b);
}

double WorldModel::ISPedStep(CarStruct &car, PedStruct &ped, despot::Random& random, bool debug) {
    //logis << __FUNCTION__ << endl;
    // gaussian + distance
    double max_is_angle = 7.0 * M_PI / 64.0;

    //logis << "maximum IS angle [DEG]: " << max_is_angle * (180/M_PI) << endl;

    COORD carpos = path[car.pos];
    if (COORD::EuclideanDistance(ped.pos, carpos) > 3.5) {
        // pedestrian goal angle in radians
        const double& goal = goals[ped.goal];
        // stop intention
        if (goal == -1) { return 1; }

        double a = goal;
        double noise = random.NextGaussian() * ModelParams::NOISE_GOAL_ANGLE;
        a += noise;

        MyVector move(a, ped.vel/freq, 0);

        ped.pos.x += move.dw;
        ped.pos.y += move.dh;
        return 1;
    } else {
        if (debug) logis << __FUNCTION__ << endl;
        double weight = 1.0;
        // pedestrian goal angle in radians
        const double& goal = goals[ped.goal];
        // stop intention
        
        if (goal == -1) { 
            if (debug) logis << "\t- stop intention: unchanged weight" << endl;
            return weight; 
        }

        //if (debug) logs << "\t- max is angle [DEG]: " << max_is_angle*(180/M_PI) << ", [RAD]: " << max_is_angle << endl;

        double goal_angle = goal;
        if (debug) logis << "\t- pedestrian goal angle [DEG]: " << goal_angle*(180/M_PI) << ", [RAD]: " << goal_angle << endl;

        MyVector pedestrian_vec(ped.pos.x, ped.pos.y);
        //if (debug) logs << "\t- pedestrian movement angle [DEG]: " << pedestrian_vec.GetAngleDeg() << ", [RAD]: " << pedestrian_vec.GetAngleRad() << endl;

        MyVector agent_vehicle_vec(path[car.pos].x, path[car.pos].y);
        //if (debug) logs << "\t- agent vehicle angle [DEG]: " << agent_vehicle_vec.GetAngleDeg() << ", [RAD]: " << agent_vehicle_vec.GetAngleRad() << endl;

        // gives the angle of the pedestrian relative to the agent vehicle
        // agent vehicle's position is equivalent to coordinate system's origin
        // angle increases clockwise, i.e 
        // pedestrian on LHS level with agent vehicle -> angle = 0°
        // pedestrian right in fron of vehicle (perfectly aligned) -> angle = 90°
        // pedestrian on RHS level with agent vehicle -> angle = 180°
        MyVector rob_vec(path[car.pos].x - ped.pos.x, path[car.pos].y - ped.pos.y);
        double rob_angle = rob_vec.GetAngleRad();
        if (debug) logis << "\t- rob_vec (x: " << rob_vec.dw << ", y: " << rob_vec.dh << ") with angle [DEG]: " << rob_vec.GetAngleDeg() << ", [RAD]: " << rob_angle << endl;

        // final mean angle
        double final_mean; 

        if (debug) logis << "\t- abs(goal_angle - rob_angle) [DEG]: " << abs(goal_angle - rob_angle)*(180/M_PI) << ", [RAD]: " << abs(goal_angle - rob_angle) << endl; 
        // absolute difference between pedestrian goal angle and difference-angle between pedestrian vs. agent vehicle goal angle
        // difference is small
        // only relevant goal angles are 0° and 180° degrees (or 0 and 1 radians)
        if (abs(goal_angle - rob_angle) <= M_PI) {
            // pedestrian coming from right (goal angle = 180°)
            if(goal_angle > rob_angle) { 
                if (debug) logis << "\t- PED COMING FROM RIGHT" << endl;
                final_mean = goal_angle - min(max_is_angle, goal_angle-rob_angle);
            }
            // pedestrian coming from left (goal angle = 0°)
            else { 
                if (debug) logis << "\t- PED COMING FROM LEFT" << endl;
                final_mean = goal_angle + min(max_is_angle, rob_angle-goal_angle); 
            }
        }
        else {
            if (goal_angle > rob_angle) { final_mean = goal_angle + min(max_is_angle, rob_angle+2*M_PI-goal_angle); }
            else { final_mean = goal_angle - min(max_is_angle, goal_angle+2*M_PI-rob_angle); }
        }

        // final mean angle in [-M_PI, M_PI]
        if (final_mean > M_PI) { final_mean -= M_PI; }
        else if (final_mean < -M_PI) { final_mean += M_PI; }

        // random.NextGaussian() returns a random number sampled from N(0,1)
        double noise = random.NextGaussian() * ModelParams::NOISE_GOAL_ANGLE; // change to the number sampled from N(0, ModelParams::NOISE_GOAL_ANGLE)
        double final_angle = final_mean + noise; //change to the number sampled from N(rob_angle, ModelParams::NOISE_GOAL_ANGLE)

        if (debug) logis << "\t- final angle [DEG]: " << final_angle * (180/M_PI) << ", [RAD]: " << final_angle << endl;
        //TODO noisy speed
        MyVector move(final_angle, ped.vel/freq, 0);

        ped.pos.x += move.dw;
        ped.pos.y += move.dh;

        /*
        increases if gaussian_prob((final_angle - goal_angle) / ModelParams::NOISE_GOAL_ANGLE, 1)
        is bigger than gaussian_prob((final_angle - final_mean) / ModelParams::NOISE_GOAL_ANGLE, 1)

        gaussian_prob((final_angle - goal_angle) / ModelParams::NOISE_GOAL_ANGLE, 1) grows if 
        (final_angle - goal_angle) / ModelParams::NOISE_GOAL_ANGLE is == 0.0

        (final_angle - goal_angle) / ModelParams::NOISE_GOAL_ANGLE approximates 0 if
        (final_angle - goal_angle) << ModelParams::NOISE_GOAL_ANGLE (0.0314),
        which in turn happens if final_angle = goal_angle

        for high weight we want to minimize the denominator N((final_angle-final_mean)/ModelParams::NOISE_GOAL_ANGLE,1),
        which happens when final_angle and final mean are maximally different, because then the term
        (final_angle-final_mean)/ModelParams::NOISE_GOAL_ANGLE will grow. Since the normal distribution has mean == 0,
        anything away from 0 will receive less probability mass
        */
        double nominator_weight = gaussian_prob((final_angle - goal_angle) / ModelParams::NOISE_GOAL_ANGLE, 1);
        double denominator_weight = gaussian_prob((final_angle - final_mean) / ModelParams::NOISE_GOAL_ANGLE, 1);
        weight = nominator_weight / denominator_weight;
        if (debug) logis << "\t- nominator_weight: " << nominator_weight << ", denominator_weight: " << denominator_weight << ", total weight: " << weight << endl;
        return weight;
    }
}

double WorldModel::ISPedStep(CarStruct &car, PedStruct &ped, despot::Random& random, double& x, double& y) {
    // gaussian + distance
    double max_is_angle = 7.0 * M_PI / 64.0;
    COORD carpos = path[car.pos];
    if(COORD::EuclideanDistance(ped.pos, carpos) > 3.5) {
        // pedestrian goal angle in radians
        const double& goal = goals[ped.goal];
        // stop intention
        if (goal == -1) { return 1; }

        // deterministic pedestrian step
        ped.pos.x = x;
        ped.pos.y = y;
        return 1;
    } else {
        double weight = 1.0;
        // pedestrian goal angle in radians
        const double& goal = goals[ped.goal];
        // stop intention
        if (goal == -1) { return weight; }

        double goal_angle = goal;

        // compute angle to robot
        MyVector rob_vec(path[car.pos].x - ped.pos.x, path[car.pos].y - ped.pos.y);
        double rob_angle = rob_vec.GetAngleRad();

        double final_mean; //final mean angle

        if(abs(goal_angle - rob_angle) <= M_PI){
            if(goal_angle > rob_angle) final_mean = goal_angle - min(max_is_angle, goal_angle-rob_angle);
            else  final_mean = goal_angle + min(max_is_angle, rob_angle-goal_angle);
        }
        else{
            if(goal_angle > rob_angle) final_mean = goal_angle + min(max_is_angle, rob_angle+2*M_PI-goal_angle);
            else  final_mean = goal_angle - min(max_is_angle, goal_angle+2*M_PI-rob_angle);
        }

        if(final_mean>M_PI) final_mean -= M_PI;
        else if(final_mean<-M_PI) final_mean += M_PI;

        //random.NextGaussian() returns a random number sampled from N(0,1)
        double noise = random.NextGaussian() * ModelParams::NOISE_GOAL_ANGLE; //change to the number sampled from N(0, ModelParams::NOISE_GOAL_ANGLE)
        double final_angle = final_mean + noise; //change to the number sampled from N(rob_angle, ModelParams::NOISE_GOAL_ANGLE)

        // TODO noisy speed
        ped.pos.x = x;
        ped.pos.y = y;

        weight = gaussian_prob((final_angle - goal_angle) / ModelParams::NOISE_GOAL_ANGLE, 1) /
                 gaussian_prob((final_angle - final_mean) / ModelParams::NOISE_GOAL_ANGLE, 1) ;
        return weight;
    }
}

void WorldModel::PedStepDeterministic(PedStruct& ped, int step) {
    const double& goal = goals[ped.goal];
	if (goal == -1) {  //stop intention
		return;
	}

	MyVector goal_vec(ModelParams::lookupTable->cos(goal),ModelParams::lookupTable->sin(goal));
    goal_vec.AdjustLength(step * ped.vel / freq);

    if((std::isnan(goal_vec.dw)) || (std::isnan(goal_vec.dh))) {
        printf("IS-DESPOT::[WORLD] Invalid goal vector calculation.\n");
        exit(-1);
    }

    ped.pos.x += goal_vec.dw;
    ped.pos.y += goal_vec.dh;
}

double WorldModel::pedMoveProb(COORD prev, COORD curr, int goal_id) {
    //logw << __FUNCTION__ << endl;

	double move_dist = Norm(curr.x-prev.x, curr.y-prev.y);
    //logw << "\t- pedestrian moved distance: " << move_dist << endl;

    // we need a minimal probability of each goal angle when not doing aggressive belief updates,
    // because each angle has to be "recoverable", i.e. it can not be = 0.0
	const double K = (despot::Globals::config.MINIMAL_NOISE) ? 0.0 : 0.001;
    // threshold below which we will assume that pedestrian has stopped moving
	double move_dist_threshold = 1e-6;
    const double& goal_ = goals[goal_id];

    // stop intention
	if (goal_ == -1) {  
    	return (move_dist < move_dist_threshold) ? 1.0 : 0;
    } 
    // pedestrian hasn't moved enough
    else if (move_dist < move_dist_threshold) {
        return 0;
    }

    MyVector goal_vec(ModelParams::lookupTable->cos(goal_),ModelParams::lookupTable->sin(goal_));
    //logw << "\t- goal vector angle " << endl;
    //logw << "\t\t- [DEG]: " << goal_id * 2 << " (given) vs. " << goal_vec.GetAngleDeg() << " (calculated)" << endl;
    //logw << "\t\t- [RAD]: " << goal_ << " (given) vs. " << goal_vec.GetAngleRad() << " (calculated)" << endl;

    //logw << "\t- goal vector before (" << goal_vec.dw << ", " << goal_vec.dh << ") ";
    goal_vec.AdjustLength(1);
    //logw << "and after normalization (" << goal_vec.dw << ", " << goal_vec.dh << ")" << endl;

    MyVector ped_vec(curr.x-prev.x, curr.y-prev.y);
    //logw << "\t- pedestrian vector angle [DEG]: " << ped_vec.GetAngleDeg() << " and [RAD]: " << ped_vec.GetAngleRad() << endl; 
    //logw << "\t- pedestrian vector before (" << ped_vec.dw << ", " << ped_vec.dh << ") ";
    ped_vec.AdjustLength(1);
    //logw << "and after normalization (" << ped_vec.dw << ", " << ped_vec.dh << ")" << endl;

    double cosa = DotProduct(goal_vec, ped_vec);
    double angle = acos(cosa);
    //logw << "\t- ANGLE(goal_vec, ped_vec) [DEG]: " << angle * (180.0/M_PI) << " and [RAD]: " << angle << endl;
    double angle_prob = gaussian_prob(angle, ModelParams::NOISE_GOAL_ANGLE) + K;

    //double angle_new = Angle(DotProduct(goal_vec, ped_vec), Determinant(goal_vec, ped_vec));
    //logw << "\t- ANGLE(goal_vec, ped_vec) [DEG]: " << angle_new * (180.0/M_PI) << " and [RAD]: " << angle_new << endl;
    //double angle_prob = normal_pdf(angle_new, 0, ModelParams::NOISE_GOAL_ANGLE);

    //logw << "\t- N(ANGLE(goal_vec, ped_vec)) [ORIG]: " << angle_prob << " vs. [NEW]: " << angle_prob_new << endl;
    return angle_prob;
}


void WorldModel::RobStep(CarStruct &car, despot::Random& random) {
    double dist = car.vel / freq;

    int nxt = path.forward(car.pos, dist);
    car.pos = nxt;
    car.dist_travelled += dist;
    car.coordinates = path[nxt];
}


void WorldModel::RobVelStep(CarStruct &car, double acc, despot::Random& random) {
    const double N = ModelParams::NOISE_ROBVEL;
    if (N > 0) {
        double prob = random.NextDouble();
        if (prob > N) {
            car.vel += acc;
        }
    } else {
        car.vel += acc;
    }

	car.vel = max(min(car.vel, ModelParams::VEL_MAX), 0.0);

	return;
}

double WorldModel::ISRobVelStep(CarStruct &car, double acc, despot::Random& random) {
    const double N = 4 * ModelParams::NOISE_ROBVEL;
    double weight = 1;
    if (N > 0) {
        double prob = random.NextDouble();
        if (prob > N) {
            car.vel += acc; 
            weight = (1.0 - ModelParams::NOISE_ROBVEL)/(1.0 - N);
        }
        else weight = ModelParams::NOISE_ROBVEL / N;
    } else {
        car.vel += acc;
    }

    car.vel = max(min(car.vel, ModelParams::VEL_MAX), 0.0);

    return weight;
}

void WorldModel::setPath(Path path) {
    this->path = path;
}

void WorldModel::updatePedBelief(PedBelief& b, const PedStruct& curr_ped, int step_counter) {
    logb << __FUNCTION__ << " at STEP=" << step_counter << endl;
	
    double total_weight = 0.0;
    for(int i = 0; i < goals.size(); i++) {
		double prob = pedMoveProb(b.pos, curr_ped.pos, i);

        if (despot::Globals::config.AGGRESSIVE_BELIEF_UPDATES) {
            b.prob_goals[i] = prob;
        }  else {
            b.prob_goals[i] *= prob;
        }

        // prevents numerical instabilities
        b.prob_goals[i] += ModelParams::BELIEF_SMOOTHING / goals.size();
        total_weight += b.prob_goals[i];
	}

    // make valid pd
    if (fabs(total_weight - 1.0) > 1e-6) {
        for(double& w: b.prob_goals) {
            w /= total_weight;
        }
        total_weight = std::accumulate(b.prob_goals.begin(), b.prob_goals.end(), double(0.0));
        if (fabs(total_weight - 1.0) > 1e-6) {
            printf("IS-DESPOT::[BELIEF] Invalid total probability of belief distribution: Epected 1, got %.8f.\n", 
            total_weight);
            exit(-1);
        }
        logb << "\t- normalize with total weight " << total_weight << endl;
    }

    // print pedestrian goal directions that have at least 1% probability
    logb << "\t- probability distribution of pedestrian goal directions based on actually observed pedestrian movement:" << endl;
    print_probability_distribution(b.prob_goals, 0.01);
    double moved_dist = COORD::EuclideanDistance(b.pos, curr_ped.pos);

    // MyVector ped_vec(curr_ped.pos.x, curr_ped.pos.y);
    // logb << "\t- pedestrian vector angle [DEG]: " << ped_vec.GetAngleDeg() << " and [RAD]: " << ped_vec.GetAngleRad() << endl; 

    MyVector ped_move(curr_ped.pos.x-b.pos.x, curr_ped.pos.y-b.pos.y);
    logb << "\t- pedestrian movement angle [DEG]: " << ped_move.GetAngleDeg() << " and [RAD]: " << ped_move.GetAngleRad() << endl; 
    // logb << "\t- pedestrian vector before (" << ped_move.dw << ", " << ped_move.dh << ") ";
    // ped_move.AdjustLength(1);
    // logb << "and after normalization (" << ped_move.dw << ", " << ped_move.dh << ")" << endl;

    // scene simulation steps in CARLA are 50ms apart, *20 for a second
    b.vel = moved_dist * 20;
    logb << "\t- pedestrian velocity " << b.vel << "m/s vs. " << b.vel * 3.6 << "km/h" << endl;

	b.pos = curr_ped.pos;
}

// print probability distribution ordered in ascending order of probability mass until and including mass of threshold 
void WorldModel::print_probability_distribution(std::vector<double> pd, double threshold)
{
    // printing belief-related debug information is disabled
    if (!despot::logging::get_scope(despot::logging::BELIEF)) { return; }

    // sort vector according to highest entry values in ascending order
    // i.e. most likely pedestrian goal directions first
    std::vector<std::size_t> sorted_indeces = MathUtils::sort_indeces(pd);
    for (std::size_t i = 0; i < sorted_indeces.size(); i++) {
        std::size_t angle = sorted_indeces.at(i);
        double probability_mass = pd.at(angle);
        if (probability_mass < threshold) { break; }
        logb << "\t\t- TOP " << i + 1 << ". most likely angle [DEG]: " << angle * 2 //*2 for belief discretization
             << " with p(angle): " << probability_mass << endl;
    }
}


PedBelief WorldModel::initPedBelief(const PedStruct& ped) {
    PedBelief b = {ped.id, 
                   ped.pos, 
                   ModelParams::PED_SPEED, 
                   vector<double>(goals.size(), 1.0/goals.size())};
    return b;
}

double timestamp() {
    //return ((double) clock()) / CLOCKS_PER_SEC;
    static double starttime=despot::get_time_second();
    return despot::get_time_second()-starttime;
}

void WorldStateTracker::cleanPed() {
    vector<Pedestrian> ped_list_new;
    for(int i=0;i<ped_list.size();i++)
    {
        bool insert=true;
        double w1,h1;
        w1=ped_list[i].x;
        h1=ped_list[i].y;
        for(const auto& ped: ped_list_new) {
            double w2,h2;
            w2=ped.x;
            h2=ped.y;
            if (abs(w1-w2)<=0.1&&abs(h1-h2)<=0.1) {
                insert=false;
                break;
            }
        }
        if (timestamp() - ped_list[i].last_update > 0.2) insert=false;
        if (insert)
            ped_list_new.push_back(ped_list[i]);
    }
    ped_list=ped_list_new;
}

void WorldStateTracker::updatePed(const Pedestrian& ped){
    int i=0;
    for(;i<ped_list.size();i++) {
        if (ped_list[i].id==ped.id) {
            ped_list[i].x=ped.x;
            ped_list[i].y=ped.y;
            ped_list[i].last_update = timestamp();
            break;
        }
    }
    if (i==ped_list.size()) {
        ped_list.push_back(ped);
        ped_list.back().last_update = timestamp();
    }
}

void WorldStateTracker::updateCar(const COORD& car) {
    carpos=car;
}

void WorldStateTracker::updateVel(double vel) {
	carvel = vel;
}

vector<WorldStateTracker::PedDistPair> WorldStateTracker::getSortedPeds() {
    vector<PedDistPair> sorted_peds;
    for(const auto& ped: ped_list) {
        COORD cp(ped.x, ped.y);
        float dist = COORD::EuclideanDistance(cp, carpos);
        sorted_peds.push_back(PedDistPair(dist, ped));
    }

    sort(sorted_peds.begin(), sorted_peds.end(),
            [](const PedDistPair& a, const PedDistPair& b) -> bool {
                return a.first < b.first;
            });

    return sorted_peds;
}

PomdpState WorldStateTracker::getPomdpState() {
    auto sorted_peds = getSortedPeds();

    // construct PomdpState
    PomdpState pomdpState;
    pomdpState.car.pos = model.path.nearest(carpos);
    pomdpState.car.vel = carvel;
	pomdpState.car.dist_travelled = 0;
    pomdpState.num = sorted_peds.size();

    for(int i = 0; i < pomdpState.num; i++) {
        const auto& ped = sorted_peds[i].second;
        pomdpState.peds[i].pos.x=ped.x;
        pomdpState.peds[i].pos.y=ped.y;
		pomdpState.peds[i].id = ped.id;
		pomdpState.peds[i].goal = -1;
    }
	return pomdpState;
}

std::vector<double> WorldBeliefTracker::belief(int id) 
{
    if (ModelParams::NUM_PEDESTRIANS != 1) {
        printf("IS-DESPOT::[%s] Invalid number of pedestrians: Expected 1, got %d.\n", 
                __PRETTY_FUNCTION__, ModelParams::NUM_PEDESTRIANS);
        exit(-1);
    }
    if (peds.empty()) {
        printf("IS-DESPOT::[%s] No pedestrian in scene simulation step.\n", __PRETTY_FUNCTION__);
        exit(-1);
    }
    if (peds.find(id) == peds.end()) {
        printf("IS-DESPOT::[%s] Pedestrian %d not in scene simulation step.\n", __PRETTY_FUNCTION__, id);
        exit(-1);        
    }
    return peds[id].prob_goals;
}


void WorldBeliefTracker::update(int step_counter) {
    // update car
    car.pos = model.path.nearest(stateTracker.carpos);
    car.coordinates = model.path[car.pos];
    car.vel = stateTracker.carvel;
	car.dist_travelled = 0;

    auto sorted_peds = stateTracker.getSortedPeds();
    map<int, PedStruct> newpeds;
    for(const auto& dp: sorted_peds) {
        auto& p = dp.second;
        if (p.id >= ModelParams::N_PED_WORLD) {
            printf("IS-DESPOT::[WORLD] Invalid number of pedestrians: Expected at most %d, got %d.\n", ModelParams::N_PED_WORLD-1, p.id);
            exit(-1);
        }
        PedStruct ped(COORD(p.x, p.y), -1, p.id);
        if (ped.id >= ModelParams::N_PED_WORLD) {
            printf("IS-DESPOT::[WORLD] Invalid number of pedestrians: Expected at most %d, got %d.\n", ModelParams::N_PED_WORLD-1, ped.id);
            exit(-1);
        }
        newpeds[p.id] = ped;
    }

    // remove disappeared peds
    vector<int> peds_to_remove;
    for(const auto& p: peds) {
        if (newpeds.find(p.first) == newpeds.end()) {
            peds_to_remove.push_back(p.first);
        }
    }
    if (!peds_to_remove.empty()) {
    }

    for(const auto& i: peds_to_remove) {
        peds.erase(i);
    }

    // update existing peds
    for(auto& kv : peds) {
        PedStruct& curr_ped = newpeds[kv.first];
        model.updatePedBelief(kv.second, curr_ped, step_counter);
    }

    // add new peds
    for(const auto& kv: newpeds) {
		auto& p = kv.second;
        if (p.id >= ModelParams::N_PED_WORLD) {
            printf("IS-DESPOT::[WORLD] Invalid number of pedestrians: Expected at most %d, got %d.\n", ModelParams::N_PED_WORLD-1, p.id);
            exit(-1);
        }
        if (peds.find(p.id) == peds.end()) {
            peds[p.id] = model.initPedBelief(p);
        }
    }

	sorted_beliefs.clear();
	for(const auto& dp: sorted_peds) {
		auto& p = dp.second;
        if (p.id >= ModelParams::N_PED_WORLD) {
            printf("IS-DESPOT::[WORLD] Invalid number of pedestrians: Expected at most %d, got %d.\n", ModelParams::N_PED_WORLD-1, p.id);
            exit(-1);
        }
        if (peds[p.id].id >= ModelParams::N_PED_WORLD) {
            printf("IS-DESPOT::[WORLD] Invalid number of pedestrians: Expected at most %d, got %d.\n", ModelParams::N_PED_WORLD-1, peds[p.id].id);
            exit(-1);
        }
		sorted_beliefs.push_back(peds[p.id]);
	}
    return;
}

int PedBelief::sample_goal_leader() const {
    double r = double(rand()) / RAND_MAX;
    int i = 0;
    r -= leader_attention[i];
    while(r > 0) {
        i++;
        r -= leader_attention[i];
    }
    return i;
}

int PedBelief::sample_goal() const {
    double r = double(rand()) / RAND_MAX;
    int i = 0;
    r -= prob_goals[i];
    while(r > 0) {
        i++;
        r -= prob_goals[i];
    }
    return i;
}

// returns index associated with the most likely pedestrian goal angle
std::size_t PedBelief::maxlikely_goal() const 
{
    return MathUtils::sort_indeces(prob_goals)[0];
}

void WorldBeliefTracker::printBelief() const 
{

}

PomdpState WorldBeliefTracker::sample() {
    PomdpState s;
    s.car = car;
    
	s.num = 0;
    for(int i=0; i < sorted_beliefs.size() && i < ModelParams::N_PED_IN; i++) {
		auto& p = sorted_beliefs[i];
		if (COORD::EuclideanDistance(p.pos, model.path[s.car.pos]) < ModelParams::LASER_RANGE) {
			s.peds[s.num].pos = p.pos;
            // sample pedestrian goal directions according to attention distribution/importance weights
            // generated by LEADER instead of basing it on the actually observed pedestrian movement direction  
            // (which is always done according to official LEADER's code base: 
            // https://github.com/modanesh/LEADER/blob/master/car_hyp_despot/src/planner/crowd_belief.cpp 
            // lines 148-186)
            if (despot::Globals::config.AGENT == LEADER && despot::Globals::config.ATTENTION_SAMPLING) {
                 s.peds[s.num].goal = p.sample_goal_leader(); 
            }
            // for LEADER: this effectively only changes the weight of particles that have been sampled the "standard way"
			// i.e. could be considered a more conservative approach
            else { s.peds[s.num].goal = p.sample_goal(); }
			s.peds[s.num].id = p.id;
            s.peds[s.num].vel = p.vel;
			s.num ++;
            // agent vehicle starting position and angle
            s.car.coordinates = model.path[s.car.pos];
		}
    }

    return s;
}

vector<PomdpState> WorldBeliefTracker::sample(int num) {
    vector<PomdpState> particles;
    for(int i=0; i<num; i++) {
        particles.push_back(sample());
    }
    return particles;
}

void WorldBeliefTracker::PrintState(const despot::State& s, std::ostream& out) const {
	const PomdpState & state=static_cast<const PomdpState&> (s);
    COORD& carpos = model.path[state.car.pos];

	out << "Rob Pos: " << carpos.x<< " " <<carpos.y << endl;
	out << "Rob travelled: " << state.car.dist_travelled << endl;
	for(int i = 0; i < state.num; i ++) {
		out << "Ped Pos: " << state.peds[i].pos.x << " " << state.peds[i].pos.y << endl;
		out << "Goal: " << state.peds[i].goal << endl;
		out << "id: " << state.peds[i].id << endl;
	}
	out << "Vel: " << state.car.vel << endl;
	out<<  "num  " << state.num << endl;
	double min_dist = COORD::EuclideanDistance(carpos, state.peds[0].pos);
	out << "MinDist: " << min_dist << endl;
}

