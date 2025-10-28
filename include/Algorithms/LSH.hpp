#pragma once
#ifndef LSH_HPP
#define LSH_HPP

#include <vector>
#include <queue>
#include <random>
#include <cstdint>
#include <iostream>

#include "metrics.hpp"
#include "neighbor.hpp"

// --- Utility functions ---
std::vector<float> GaussianProjection(size_t size, int* seed);

// --- HashFunction class ---
class HashFunction {
    std::vector<float> w;
    int window;
    float t;
    size_t dim;

public:
    HashFunction(size_t dim, int seed, int window);
    int calculate(const std::vector<float>& p) const;
};

// --- AmplifiedHashFunction class ---
class AmplifiedHashFunction {
    std::vector<HashFunction> hf_table;
    std::vector<int> r_table;

public:
    AmplifiedHashFunction(size_t size, size_t dim, int seed, int window);
    uint64_t calculate_ID(const std::vector<float>& p) const;
};

// --- LSH main class ---
/*
Save all the points to a structure in the lsh once and reference them by pointer to the hashtables 
*/
class LSH {
    std::vector<std::vector<std::vector<std::pair<uint64_t, size_t > >>> HashTables; //save a pair of ID and an idx to the vectort 
    std::vector<std::vector<float>> vectors; //Save the vectors 
    std::vector<AmplifiedHashFunction> IDs; // Definition to the Amplified HashFunctions 

    size_t hashTable_size;
    int num_tables;
    int HashFunction_size;
    int w;
    size_t vec_dim;

public:
    LSH(size_t hashTable_size_, int num_tables_, int HashFunction_size_,
        int w_, size_t vec_dim_, int seed);

    void insert_to_hashTables(const std::vector<float>& p);
    std::vector<Neighbor> returnANN(const std::vector<float>& p, int k ,bool range_bool=false ,float range=0.0) const;
};

#endif // LSH_HPP
