#pragma once
#ifndef NEIGHBOR_HPP
#define NEIGHBOR_HPP

#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>

template<typename NumType>
struct Neighbor {
    std::vector<NumType> vec;
    double distance;

    Neighbor(const std::vector<NumType>& v, double d)
        : vec(v), distance(d) {}

    bool operator<(const Neighbor& rhs) const {
        return distance < rhs.distance;
    }
};

template<typename NumType>
std::ostream& operator<<(std::ostream& os, const Neighbor<NumType>& n) {
    os << "Neighbor(distance=" << n.distance << ", vec=[";
    size_t limit = std::min(n.vec.size(), size_t(3));
    for (size_t i = 0; i < limit; ++i) {
        if constexpr (std::is_same_v<NumType, uint8_t> || std::is_same_v<NumType, int8_t>) {
            os << +n.vec[i]; 
        } else {
            os << std::fixed << std::setprecision(4) << n.vec[i];
        }
        if (i < limit - 1) os << ", ";
    }
    if (n.vec.size() > 3) os << ", ...";
    os << "])\n";
    return os;
}


#endif
