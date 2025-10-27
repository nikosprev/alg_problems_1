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
class LSH {
    std::vector<std::vector<std::vector<std::pair<uint64_t, std::vector<float>>>>> HashTables;
    std::vector<AmplifiedHashFunction> IDs;

    size_t hashTable_size;
    int num_tables;
    int HashFunction_size;
    int w;
    size_t vec_dim;

    int positive_mod(int x, int m) const;

public:
    LSH(size_t hashTable_size_, int num_tables_, int HashFunction_size_,
        int w_, size_t vec_dim_, int seed);

    void insert_to_hashTables(const std::vector<float>& p);
    std::vector<std::vector<float>> return_candidates(const std::vector<float>& p) const;
    std::vector<Neighbor> returnANN(const std::vector<float>& p, int k) const;
};

#endif // LSH_HPP
