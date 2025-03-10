#ifndef PATH_H_DESPOT
#define PATH_H_DESPOT

#include <vector>
#include <iostream>

#include "coord.h"
#include "param.h"
#include "math_utils.h"


class Path : public std::vector<COORD> {
public:
    int nearest(COORD pos);
    int forward(int i, double len) const;
};

#endif