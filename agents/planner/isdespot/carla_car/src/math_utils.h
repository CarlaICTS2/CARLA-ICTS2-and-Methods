#ifndef MATHUTILS_H
#define MATHUTILS_H

#include "vector"
#include "string"
#include "numeric"
#include "param.h"
#include "algorithm"
#include <sstream>
#include <iterator>
#include <limits>

class MyVector
{

public:
	MyVector();
	MyVector(double _dw,double _dh);
	MyVector(double angle,double length,int dummy);
	double GetAngleRad();   //[0, 2pi]
	double GetAngleDeg();   //[0, 360]
	double GetLength();
	void GetPolar(double &angle,double &length);
	void AdjustLength(double length);
	//void SetAngle(double angle);
	MyVector  operator + (MyVector  vec);
	double dw,dh;
};

double DotProduct(MyVector vec1, MyVector vec2);
double DotProduct(double x1, double y1, double x2, double y2);

double Determinant(MyVector vec1, MyVector vec2);
double Angle(double dot_product, double determinant);
double Norm(double x,double y);
void Uniform(double x,double y,double &ux,double &uy);
//void AddVector(double in_angle,double in_length,double &out_angle,double &out_length);

class MathUtils {
	public:
		static int get_median(std::vector<int> input);
		static double get_median(std::vector<double> input);
		static std::pair<int, int> get_min_and_max(std::vector<int> input);
		static std::pair<double, double> get_min_and_max(std::vector<double> input);
		static double get_sum(std::vector<double> input);
		static int get_sum(std::vector<int> input);
		static std::vector<std::size_t> sort_indeces(const std::vector<double> &v);

		template<typename T>
		static std::pair<T, T> get_average_and_stdev(std::vector<T> input) 
		{
			if (input.empty()) {
				return std::pair<T, T>(
					std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::quiet_NaN()
				);
			}
			T avg = std::accumulate(input.begin(), input.end(), 0.0) / input.size();
			std::vector<T> diff(input.size());
			std::transform(input.begin(), input.end(), diff.begin(), [avg](T x) { return x - avg; });
			T sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
			T stdev = std::sqrt(sq_sum / input.size());
			return std::pair<T,T>(avg, stdev);
		}

		template<typename T>
		static std::string to_string(std::vector<T> data)
		{
			if (data.empty()) return std::numeric_limits<T>::quiet_NaN();
			std::ostringstream oss;
			std::copy(data.begin(), data.end()-1, std::ostream_iterator<T>(oss, ","));
			oss << data.back();
			return oss.str();
		};
};



#endif
