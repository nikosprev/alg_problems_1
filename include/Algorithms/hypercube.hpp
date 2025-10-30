#pragma once 
#ifndef HYPERCUBE_HPP
#define HYPERCUBE_HPP 
#include <vector>
#include <queue>
#include <random>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <numeric>
#include <cmath>

#include "neighbor.hpp"
#include "metrics.hpp"
#include "LSH.hpp"



template <typename NumType>
class HyperCube{ 
    std::vector<HashFunction> hf_table;   //HashFunction table to use for finding the slot of a point 
    std::unordered_map<uint64_t ,std::vector<NumType>> bit_table; //A map that stores the points to the appropiate slot  
    float w; 

    size_t k_proj; //How many hashfunctions are needed     

    size_t vec_dim;  //Dimenstion of the point to be saved
    int seed; 

    uint64_t find_slot(std::vector<NumType>& p){ 
        uint64_t mask = 0; 
        for(int i = 0 ; i < hf_table.size() ; ++i ) { 
            if (hf_table[i].calculate(p) > 0){ 
                mask += 1 ; 
            }
            mask <<= 1; 
        }
        return mask; 
    }

    uint64_t next_bit(uint64_t v) {
        uint64_t t = v | (v - 1); // t gets v's least significant 0 bits set to 1
        // Next set to 1 the most significant bit to change, 
        // set to 0 the least significant ones, and add the necessary 1 bits.
        return (t + 1) | (((~t & -~t) - 1) >> (__builtin_ctz(v) + 1));  
    }

    public:

        HyperCube(size_t k_proj_ ,float w_, size_t vec_dim_, int seed)
            : k_proj(k_proj_), HashFunction_size(k_proj_), w(w_), vec_dim(vec_dim_) {
        //Initialize HashFunctions
            hf_table.reserve(k_proj_); 
            for (size_t i = 0; i < k_proj_; ++i)
                hf_table.emplace_back(vec_dim, seed + static_cast<int>(i), w);

        }


        void insertPoint(const std::vector<NumType> &p){ 
            uint64_t slot = find_slot(p); 
            bit_table[slot].push_back(p);
        }


        
    std::vector<Neighbor<NumType>> returnANN(const std::vector<NumType>& p, int M , int k,
                                             bool range_bool = false, float range = 0.0) const {


        
        std::priority_queue<Neighbor<NumType>> topKNeighbors; 
        uint64_t base_slot = find_slot(p); //
        int hammingDist = 0; 
        int exploredCount = 0;                
        uint64_t border = (1ULL << k_proj); 
        bool explorationLimitReached = false; 
        while(hammingDist < k_proj){ //If we reach that point we are cooked  
            if (hammingDist == 0){ //When Hamming Dist is 0 we first explore the values in this slot 
                for(const auto& entry : bit_table[base_slot]){ 
                    double dist = euclidean_distance(p, entry);
                    if (!range_bool || dist < range) {
                        topK.emplace(entry, dist);
                    }
                    if ( (++exploredCount) > M ) {explorationLimitReached = true; break};  
                    
                }
                while (topK.size() > static_cast<size_t>(k) && !range_bool) topK.pop();
                if (explorationLimitReached) break;
                continue; 
            }
            uint64_t currentSlot; 
            uint64_t hammingMask = 0 ; 
            for(int i = 0 ; i < hammingDist ;++i){ 
                hammingMask << 1; 
                hammingMask |= 1; 
            }
            while (hammingMask <=  border){ 
                currentSlot = base_slot ^ hammingMask; 
                for(const auto& entry : bit_table[currentSlot]){ 
                    double dist = euclidean_distance(p, entry);
                    if (!range_bool || dist < range) {
                        topK.emplace(entry, dist);
                    }
                    if ( (++exploredCount) > M ) {explorationLimitReached = true; break}; 
                }
                while (topK.size() > static_cast<size_t>(k) && !range_bool) topK.pop();
                hammingMask = next_bit(hammingMask); 
                if (explorationLimitReached) break;
            }
            if (explorationLimitReached) break;
            hammingMask ++ ;


    }
    std::vector<Neighbor<NumType>> neighbors;
    while (!topK.empty()) {
        neighbors.push_back(topK.top());
        topK.pop();
    }
    std::reverse(neighbors.begin(), neighbors.end());
    return neighbors;
}



    
}; 








#endif


