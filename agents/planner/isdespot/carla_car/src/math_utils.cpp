#include "math_utils.h"

#include "numeric"
#include "param.h"
#include "algorithm"
#include <sstream>
#include <iterator>
#include <limits>

MyVector::MyVector(double _dw,double _dh) 
{
    dw=_dw;
    dh=_dh;
}

MyVector::MyVector(double angle,double length,int dummy)
{
	dw=length*ModelParams::lookupTable->cos(angle);
	dh=length*ModelParams::lookupTable->sin(angle);
}

// [-pi, pi] radians
double MyVector::GetAngleRad()  
{
	return atan2(dh,dw);
}

// [-180, 180] degrees
double MyVector::GetAngleDeg()
{
	return GetAngleRad() * (180.0/M_PI);
}

double MyVector::GetLength()
{
	return sqrt(dh*dh+dw*dw);
}

void MyVector::GetPolar(double &angle,double &length)
{
	angle=GetAngleRad();
	length=GetLength();
}

void MyVector::AdjustLength(double length)
{
	if(GetLength() < 0.01) return;   //vector length close to 0
	double rate=length/GetLength();
	dw*=rate;
	dh*=rate;
}

MyVector MyVector::operator + (MyVector  vec)
{
	return MyVector(dw+vec.dw,dh+vec.dh);
}

double DotProduct(MyVector vec1, MyVector vec2)
{
	return vec1.dw*vec2.dw + vec1.dh*vec2.dh;
}

double DotProduct(double x1, double y1, double x2, double y2)
{
	return x1*x2 + y1*y2;
}

double Determinant(MyVector vec1, MyVector vec2)
{
	return vec1.dw*vec2.dh-vec1.dh*vec2.dw;
}

double Angle(double dot_product, double determinant)
{
	return atan2(determinant, dot_product);
}

double Norm(double x,double y)
{
	return sqrt(x*x+y*y);
}

void Uniform(double x,double y,double &ux,double &uy)
{
	double l=Norm(x,y);
	ux=x/l;
	uy=y/l;
}

double MathUtils::get_sum(std::vector<double> input) 
{
	if (input.empty()) return std::numeric_limits<double>::quiet_NaN();
	return std::accumulate(input.begin(), input.end(), 0.0);
}

int MathUtils::get_sum(std::vector<int> input) 
{
	if (input.empty()) return std::numeric_limits<int>::quiet_NaN();
	return std::accumulate(input.begin(), input.end(), 0);
}

int MathUtils::get_median(std::vector<int> input)
{
	if (input.empty()) return std::numeric_limits<int>::quiet_NaN();
    size_t n = input.size() / 2;
    std::nth_element(input.begin(), input.begin()+n, input.end());
    return input[n];
}

double MathUtils::get_median(std::vector<double> input)
{
	if (input.empty()) return std::numeric_limits<double>::quiet_NaN();
	size_t n = input.size();
	return 0.5 * ( input[(n-1)/2] + input[n/2] );
}

std::pair<int, int> MathUtils::get_min_and_max(std::vector<int> input)
{
	if (input.empty()) return std::pair<int, int>(std::numeric_limits<int>::quiet_NaN(), 
								   				  std::numeric_limits<int>::quiet_NaN());

	auto [min, max] = std::minmax_element(begin(input), end(input));
	return std::pair<int, int>(*min, *max);
}

std::pair<double, double> MathUtils::get_min_and_max(std::vector<double> input)
{
	if (input.empty()) return std::pair<double, double>(std::numeric_limits<double>::quiet_NaN(),
										 				std::numeric_limits<double>::quiet_NaN());

	auto [min, max] = std::minmax_element(begin(input), end(input));
	return std::pair<double, double>(*min, *max);
}

std::vector<std::size_t> MathUtils::sort_indeces(const std::vector<double> &v) 
{
	// initialize original index locations
	std::vector<std::size_t> idx(v.size());
	iota(idx.begin(), idx.end(), 0);

	// sort indexes based on comparing values in v
	// using std::stable_sort instead of std::sort
	// to avoid unnecessary index re-orderings
	// when v contains elements of equal values 
	stable_sort(idx.begin(), idx.end(), [&v](std::size_t i1, std::size_t i2) {return v[i1] > v[i2];});

	return idx;
}