#pragma once
#ifndef KNN_HPP
#define KNN_HPP

#include <vector>
#include <queue>
#include <random>
#include <cstdint>
#include <iostream>

#include "metrics.hpp"
#include "neighbor.hpp"


std::vector<Neighbor> kNN(const std::vector<std::vector<float>> &points ,const std::vector<float> &p , int k); 




#endif