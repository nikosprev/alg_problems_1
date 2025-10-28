#pragma once
#ifndef NEIGHBOR_HPP
#define NEIGHBOR_HPP

#include <vector> 


// --- Neighbor struct ---
struct Neighbor {
    std::vector<float> vec;
    double distance;

    Neighbor(std::vector<float> v, double d);
    bool operator<(const Neighbor& rhs) const;
    friend std::ostream& operator<<(std::ostream& os, const Neighbor& n);
};




#endif