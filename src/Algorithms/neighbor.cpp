#include <iostream>
#include <iomanip> 
#include "../../include/Algorithms/neighbor.hpp"

// ----- Neighbor -----
Neighbor::Neighbor(const std::vector<float> v, double d)
    : vec(v), distance(d) {}

bool Neighbor::operator<(const Neighbor& rhs) const {
    return distance < rhs.distance; 
}

std::ostream& operator<<(std::ostream& os, const Neighbor& n) {
    os << "Neighbor(distance=" << n.distance << ", vec=[";
    if (!n.vec.empty()) {
        for(int i = 0 ; i < n.vec.size() && i < 3 ; ++i) {
            os << std::fixed << std::setprecision(4) << n.vec[i] << " ," ;
        }
        if (n.vec.size() > 3){ 
            os << "..." ; 
        }
    }
        
    os << "])\n";
    return os;
}
