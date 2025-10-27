#include "../../include/Algorithms/LSH.hpp"
#include <numeric>
#include <cmath>
#include <cstdlib>
#include <algorithm> 


// ----- GaussianProjection -----
std::vector<float> GaussianProjection(size_t size, int* seed) {
    std::default_random_engine generator(*seed);
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    std::vector<float> r(size);
    for (size_t i = 0; i < size; ++i)
        r[i] = distribution(generator);
    normalize(r);
    return r;
}

// ----- HashFunction -----
HashFunction::HashFunction(size_t dim_, int seed, int window_)
    : window(window_), dim(dim_) {
    w = GaussianProjection(dim, &seed);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(2.0f, static_cast<float>(window));
    t = dist(gen);
}

int HashFunction::calculate(const std::vector<float>& p) const {
    if (p.size() != dim) {
        std::cerr << "Vector size mismatch\n";
        std::exit(EXIT_FAILURE);
    }
    double result = std::inner_product(p.begin(), p.end(), w.begin(), 0.0);
    result = (result + t) / static_cast<double>(window);
    return static_cast<int>(std::floor(result));
}

// ----- AmplifiedHashFunction -----
AmplifiedHashFunction::AmplifiedHashFunction(size_t size, size_t dim, int seed, int window) {
    hf_table.reserve(size);
    for (size_t i = 0; i < size; ++i)
        hf_table.emplace_back(dim, seed + static_cast<int>(i), window);

    r_table.resize(size);
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(INT32_MIN, INT32_MAX);
    for (size_t i = 0; i < size; ++i)
        r_table[i] = dist(gen);
}

uint64_t AmplifiedHashFunction::calculate_ID(const std::vector<float>& p) const {
    const uint64_t M = (1ULL << 16) - 5;
    uint64_t ID = 0;
    for (size_t i = 0; i < hf_table.size(); ++i) {
        ID = (ID + (static_cast<uint64_t>(hf_table[i].calculate(p)) * r_table[i]) % M) % M;
    }
    return ID;
}

// ----- LSH -----
LSH::LSH(size_t hashTable_size_, int num_tables_, int HashFunction_size_,
         int w_, size_t vec_dim_, int seed)
    : hashTable_size(hashTable_size_), num_tables(num_tables_),
      HashFunction_size(HashFunction_size_), w(w_), vec_dim(vec_dim_) {
    HashTables.resize(num_tables);
    IDs.reserve(num_tables);
    for (int i = 0; i < num_tables; ++i) {
        HashTables[i].resize(hashTable_size);
        IDs.emplace_back(HashFunction_size, vec_dim, seed + i * 101, w);
    }
}

int LSH::positive_mod(int x, int m) const {
    int r = x % m;
    return (r < 0) ? r + m : r;
}

void LSH::insert_to_hashTables(const std::vector<float>& p) {
    for (int t = 0; t < num_tables; ++t) {
        uint64_t id = IDs[t].calculate_ID(p);
        int slot = positive_mod(static_cast<int>(id), hashTable_size);
        HashTables[t][slot].emplace_back(id, p);
    }
}

std::vector<std::vector<float>> LSH::return_candidates(const std::vector<float>& p) const {
    std::vector<std::vector<float>> candidates;
    for (int t = 0; t < num_tables; ++t) {
        uint64_t id = IDs[t].calculate_ID(p);
        int slot = positive_mod(static_cast<int>(id), hashTable_size);
        for (const auto& entry : HashTables[t][slot])
            if (entry.first == id)
                candidates.push_back(entry.second);
    }
    return candidates;
}

std::vector<Neighbor> LSH::returnANN(const std::vector<float>& p, int k) const {
    std::priority_queue<Neighbor> topK;
    for (int t = 0; t < num_tables; ++t) {
        uint64_t id = IDs[t].calculate_ID(p);
        int slot = positive_mod(static_cast<int>(id), hashTable_size);
        for (const auto& entry : HashTables[t][slot])
            if (entry.first == id)
                topK.emplace(&entry.second, euclidean_distance(p, entry.second));
        while (topK.size() > static_cast<size_t>(k)) topK.pop();
    }

    std::vector<Neighbor> neighbors;
    while (!topK.empty()) {
        neighbors.push_back(topK.top());
        topK.pop();
    }
    std::reverse(neighbors.begin(), neighbors.end());
    return neighbors;
}
