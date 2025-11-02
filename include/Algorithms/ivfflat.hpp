#pragma once
#ifndef IVF_FLAT_HPP
#define IVF_FLAT_HPP

#include <vector>
#include <queue>
#include <random>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <limits>
#include <numeric>

#include "neighbor.hpp"
#include "metrics.hpp"
#include "kmeans.hpp"


// IVF-Flat (Inverted File Index with Flat lists)


template <typename NumType>
class IVFFlat {
    // Coarse centroids (size = num_clusters). Each centroid is a vector of dimension dim.
    std::vector<std::vector<float>> centroids;

    // Inverted lists: for each centroid, store pairs of (originalIndex, vector).
    std::vector<std::vector<std::pair<size_t, std::vector<NumType>>>> inverted_lists;

    // Original vectors kept to allow optional access; not strictly necessary for search.
    std::vector<std::vector<NumType>> vectors;

    // Configuration
    size_t num_clusters;     // number of coarse clusters (lists)
    size_t vec_dim;          // dimensionality of vectors
    int kmeans_iters;        // iterations of k-means during training

public:
    IVFFlat(size_t num_clusters_, size_t vec_dim_, int kmeans_iters_ = 15)
        : num_clusters(num_clusters_), vec_dim(vec_dim_), kmeans_iters(kmeans_iters_) {
        if (num_clusters == 0 || vec_dim == 0) {
            std::cerr << "IVFFlat: num_clusters and vec_dim must be > 0" << std::endl;
            std::exit(EXIT_FAILURE);
        }
        centroids.resize(num_clusters, std::vector<float>(vec_dim, 0.0f));
        inverted_lists.resize(num_clusters);
    }

    // Insert a single vector prior to training. Training will cluster and assign later.
    void add_vector(const std::vector<NumType>& p) {
        if (p.size() != vec_dim) {
            std::cerr << "IVFFlat::add_vector: vector dim mismatch" << std::endl;
            std::exit(EXIT_FAILURE);
        }
        vectors.push_back(p);
    }

    // Add a batch of vectors.
    void add_vectors(const std::vector<std::vector<NumType>>& pts) {
        for (const auto& p : pts) add_vector(p);
    }

    // Train centroids using k-means 
    void train(int seed = 12345) {
        if (vectors.empty()) {
            std::cerr << "IVFFlat::train: no vectors added" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        // Train via reusable k-means and get final assignment
        std::vector<size_t> final_assignment;
        kmeans::kmeans_train(vectors, num_clusters, vec_dim, kmeans_iters, seed, centroids, final_assignment);

        for (auto& lst : inverted_lists) lst.clear();
        for (size_t i = 0; i < vectors.size(); ++i) {
            size_t c = final_assignment[i];
            inverted_lists[c].emplace_back(i, vectors[i]);
        }
    }

    // Return k approximate nearest neighbors of p by searching only nprobe lists.
    std::vector<Neighbor> query(const std::vector<NumType>& p, int k, int nprobe = 1) const {
        if (p.size() != vec_dim) {
            std::cerr << "IVFFlat::query: vector dim mismatch" << std::endl;
            std::exit(EXIT_FAILURE);
        }
        if (centroids.empty() || inverted_lists.empty()) {
            std::cerr << "IVFFlat::query: index not trained" << std::endl;
            std::exit(EXIT_FAILURE);
        }
        nprobe = std::max(1, std::min(static_cast<int>(num_clusters), nprobe));

        // Find nprobe closest centroids
        std::vector<std::pair<double, size_t>> centroid_dists;
        centroid_dists.reserve(num_clusters);
        for (size_t c = 0; c < num_clusters; ++c) {
            double dist = dist_to_centroid(p, centroids[c]);
            centroid_dists.emplace_back(dist, c);
        }
        std::nth_element(centroid_dists.begin(), centroid_dists.begin() + (nprobe - 1), centroid_dists.end());
        centroid_dists.resize(nprobe);

        //  Scan chosen lists exactly and keep a min-heap of size k (largest on top for easy pop)
        auto cmp = [](const Neighbor& a, const Neighbor& b) { return a.distance < b.distance; };
        std::priority_queue<Neighbor, std::vector<Neighbor>, decltype(cmp)> topK(cmp);

        for (const auto& [_, cid] : centroid_dists) {
            for (const auto& [idx, vec] : inverted_lists[cid]) {
                double dist = euclidean_distance(p, vec);
                topK.emplace(idx, dist);
                if (static_cast<int>(topK.size()) > k) topK.pop();
            }
        }

        // Extract neighbors (ascending distance)
        std::vector<Neighbor> neighbors;
        neighbors.reserve(topK.size());
        while (!topK.empty()) {
            neighbors.push_back(topK.top());
            topK.pop();
        }
        std::reverse(neighbors.begin(), neighbors.end());
        return neighbors;
    }

    // Range query over nprobe lists
    std::vector<Neighbor> range_query(const std::vector<NumType>& p, double range, int nprobe = 1) const {
        if (p.size() != vec_dim) {
            std::cerr << "IVFFlat::range_query: vector dim mismatch" << std::endl;
            std::exit(EXIT_FAILURE);
        }
        nprobe = std::max(1, std::min(static_cast<int>(num_clusters), nprobe));

        std::vector<std::pair<double, size_t>> centroid_dists;
        centroid_dists.reserve(num_clusters);
        for (size_t c = 0; c < num_clusters; ++c) {
            double dist = dist_to_centroid(p, centroids[c]);
            centroid_dists.emplace_back(dist, c);
        }
        std::nth_element(centroid_dists.begin(), centroid_dists.begin() + (nprobe - 1), centroid_dists.end());
        centroid_dists.resize(nprobe);

        std::vector<Neighbor> result;
        for (const auto& [_, cid] : centroid_dists) {
            for (const auto& [idx, vec] : inverted_lists[cid]) {
                double dist = euclidean_distance(p, vec);
                if (dist <= range) result.emplace_back(idx, dist);
            }
        }
        std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) { return a.distance < b.distance; });
        return result;
    }

    // Expose read-only accessors
    const std::vector<std::vector<float>>& get_centroids() const { return centroids; }
    const std::vector<std::vector<std::pair<size_t, std::vector<NumType>>>>& get_inverted_lists() const { return inverted_lists; }

private:
    // Distance to centroid without type conversion of the full vector
    static double dist_to_centroid(const std::vector<NumType>& v, const std::vector<float>& c) {
        double s = 0.0;
        for (size_t i = 0; i < c.size(); ++i) {
            double diff = static_cast<double>(v[i]) - static_cast<double>(c[i]);
            s += diff * diff;
        }
        return std::sqrt(s);
    }
};

#endif


