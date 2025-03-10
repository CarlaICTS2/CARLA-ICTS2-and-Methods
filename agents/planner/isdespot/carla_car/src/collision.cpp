#include "param.h"
#include "coord.h"

#include <iostream>
#include <utility>
#include <cmath>

#include "despot/util/logging.h"

using namespace std;

/**
 *    A-----------N-----------B
 *    |           ^           |
 *    |           |           |
 *    |     |-L<--H-----|     |
 *    |     |           |     |
 *    |     |           |     |
 *    |     |           |     |
 *    |     |           |     |
 *    |     |           |     |
 *    |     |           |     |
 *    |     |           |     |
 *    |     |M          |     |
 *    |     |           |     |
 *    |     |           |     |
 *    |     |-----------|     |
 *    |                       |
 *    |                       |
 *    D-----------------------C
 *
 * H: center of the head of the car
 * N: a point right in front of the car
 * L: a point to the left/right of H
 *
 * A point M is inside the safety zone ABCD iff
 *   ((0 <= HM . HN & (HM . HN)^2 <= (HN . HN) * front_margin^2) || (0 => HM . HN & (HM . HN)^2 < (HN . HN) * back_margin^2))
 *   && (HM . HL)^2 <= (HL . HL) * side_margin^2
 */


bool InRectangle(
	double HNx, 
	double HNy, 
	double HMx, 
	double HMy, 
	double front_margin, 
	double back_margin, 
	double side_margin) 

{
	double HLx = - HNy, // direction after 90 degree anticlockwise rotation
		   HLy = HNx;

	double HM_HN = HMx * HNx + HMy * HNy, // HM . HN
		   HN_HN = HNx * HNx + HNy * HNy; // HN . HN
	if (HM_HN >= 0 && HM_HN * HM_HN > HN_HN * front_margin * front_margin)
		return false;
	if (HM_HN <= 0 && HM_HN * HM_HN > HN_HN * back_margin * back_margin)
		return false;

	double HM_HL = HMx * HLx + HMy * HLy, // HM . HL
		   HL_HL = HLx * HLx + HLy * HLy; // HL . HL
	return HM_HL * HM_HL <= HL_HL * side_margin * side_margin;
}

void rotatePosition(
	double centerX, 
	double centerZ, 
	double theta, 
	double x, 
	double z, 
	double& resX, 
	double& resZ)
{
    double tempX = x - centerX;
    double tempZ = z - centerZ;

    double sin_theta = ModelParams::lookupTable->sin(theta);
    double cos_theta = ModelParams::lookupTable->cos(theta);

    double rotatedX = tempX * cos_theta - tempZ * sin_theta;
    double rotatedY = tempX * sin_theta + tempZ * cos_theta;

    resX = rotatedX + centerX;
    resZ = rotatedY + centerZ;
}

// Mx = ped.pos.x; My = ped.pos.y
bool inCollision(double Mx, double My, const COORD& car_pos, bool debug) {
    const double& carX = car_pos.x;
    const double& carY = car_pos.y;
    const double& carTheta = car_pos.theta;

    double Hx = -1;
    double Hy = -1;

    rotatePosition(carX, carY, carTheta, carX + ModelParams::CAR_LENGTH / 2.0, carY, Hx, Hy);

	// logs << __FUNCTION__ << endl
	//	 << "\t- rotated car position (x: " << Hx << ", y: " << Hy << ")" << endl;

    double Nx = Hx + (Hx - carX); // Move point further in direction of head of the car
    double Ny = Hy + (Hy - carY); // Move point further in direction of head of the car

	double HNx = Nx - Hx;
	double HNy = Ny - Hy;
	double HMx = Mx - Hx;
	double HMy = My - Hy;

	return InRectangle(
		HNx, 
		HNy, 
		HMx, 
		HMy, 
		ModelParams::front_nearmiss_margin, 
		ModelParams::back_nearmiss_margin, 
		ModelParams::side_nearmiss_margin
	);
}