#pragma once
#ifndef IVF_PQ_HPP
#define IVF_PQ_HPP

#include <vector>
#include <queue>
#include <random>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <limits>
#include <numeric>
#include <cmath>

#include "neighbor.hpp"
#include "metrics.hpp"
#include "kmeans.hpp"


// IVF-PQ (Inverted File Index with Product Quantization)
// Combines a coarse quantizer (IVF) with fine-grained subquantization (PQ)

template <typename NumType>
class IVFPQ {
    // Coarse centroids (IVF part)
    std::vector<std::vector<float>> coarse_centroids;

    // PQ codebooks: M subvectors each with Ks centroids
    std::vector<std::vector<std::vector<float>>> pq_codebooks;

    // encoded inverted lists: each entry is (originalIndex, PQ code)
    std::vector<std::vector<std::pair<size_t, std::vector<uint8_t>>>> inverted_lists;

    // original vectors for training and reconstruction
    std::vector<std::vector<NumType>> vectors;

    // Configuration
    size_t num_coarse_clusters;  // Number of IVF clusters
    size_t vec_dim;              // Dimensionality of input vectors
    size_t M;                    // Number of PQ subvectors
    size_t Ks;                   // Number of PQ centroids per subvector
    int kmeans_iters;            // Iterations for both coarse and PQ kmeans

public:
    IVFPQ(size_t num_coarse_clusters_, size_t vec_dim_, size_t M_, size_t Ks_ = 256, int kmeans_iters_ = 15)
        : num_coarse_clusters(num_coarse_clusters_), vec_dim(vec_dim_), M(M_), Ks(Ks_), kmeans_iters(kmeans_iters_) {
        if (num_coarse_clusters == 0 || vec_dim == 0 || M == 0 || vec_dim % M != 0) {
            std::cerr << "IVFPQ: invalid parameters (check M divides vec_dim)" << std::endl;
            std::exit(EXIT_FAILURE);
        }
        coarse_centroids.resize(num_coarse_clusters, std::vector<float>(vec_dim, 0.0f));
        inverted_lists.resize(num_coarse_clusters);
        pq_codebooks.resize(M);
    }

    void add_vector(const std::vector<NumType>& p) {
        if (p.size() != vec_dim) {
            std::cerr << "IVFPQ::add_vector: vector dim mismatch" << std::endl;
            std::exit(EXIT_FAILURE);
        }
        vectors.push_back(p);
    }

    void add_vectors(const std::vector<std::vector<NumType>>& pts) {
        for (const auto& p : pts) add_vector(p);
    }

    void train(int seed = 12345) {
        if (vectors.empty()) {
            std::cerr << "IVFPQ::train: no vectors added" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        // Train coarse centroids (IVF)
        std::vector<size_t> coarse_assignment;
        kmeans::kmeans_train(vectors, num_coarse_clusters, vec_dim, kmeans_iters, seed, coarse_centroids, coarse_assignment);

        // Assign vectors to coarse centroids
        std::vector<std::vector<std::vector<NumType>>> residuals(num_coarse_clusters);
        for (size_t i = 0; i < vectors.size(); ++i) {
            size_t cid = coarse_assignment[i];
            std::vector<NumType> residual(vec_dim);
            for (size_t j = 0; j < vec_dim; ++j)
                residual[j] = static_cast<float>(vectors[i][j]) - coarse_centroids[cid][j];
            residuals[cid].push_back(std::move(residual));
        }

        // Train PQ codebooks
        size_t subdim = vec_dim / M;
        pq_codebooks.assign(M, std::vector<std::vector<float>>(Ks, std::vector<float>(subdim, 0.0f)));

        // Collect all residuals for PQ training
        std::vector<std::vector<float>> all_residuals;
        for (const auto& cl : residuals)
            for (const auto& v : cl)
                all_residuals.emplace_back(v.begin(), v.end());

        // For each subspace, train kmeans
        for (size_t m = 0; m < M; ++m) {
            std::vector<std::vector<float>> subspace;
            subspace.reserve(all_residuals.size());
            for (const auto& v : all_residuals)
                subspace.emplace_back(v.begin() + m * subdim, v.begin() + (m + 1) * subdim);

            std::vector<size_t> assign_dummy;
            kmeans::kmeans_train(subspace, Ks, subdim, kmeans_iters, seed + m, pq_codebooks[m], assign_dummy);
        }

        // Encode all vectors
        for (auto& lst : inverted_lists) lst.clear();
        for (size_t i = 0; i < vectors.size(); ++i) {
            size_t cid = coarse_assignment[i];
            std::vector<float> residual(vec_dim);
            for (size_t j = 0; j < vec_dim; ++j)
                residual[j] = static_cast<float>(vectors[i][j]) - coarse_centroids[cid][j];
            std::vector<uint8_t> code = encode_vector(residual);
            inverted_lists[cid].emplace_back(i, std::move(code));
        }
    }

    std::vector<Neighbor<NumType>> query(const std::vector<NumType>& q, int k, int nprobe = 1) const {
        if (q.size() != vec_dim) {
            std::cerr << "IVFPQ::query: vector dim mismatch" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        nprobe = std::max(1, std::min(static_cast<int>(num_coarse_clusters), nprobe));

        // Find nprobe closest coarse centroids
        std::vector<std::pair<double, size_t>> coarse_dists;
        coarse_dists.reserve(num_coarse_clusters);
        for (size_t c = 0; c < num_coarse_clusters; ++c) {
            double dist = dist_to_centroid(q, coarse_centroids[c]);
            coarse_dists.emplace_back(dist, c);
        }
        std::nth_element(coarse_dists.begin(), coarse_dists.begin() + (nprobe - 1), coarse_dists.end());
        coarse_dists.resize(nprobe);

        // Precompute PQ distance tables for query residuals
        size_t subdim = vec_dim / M;
        std::vector<std::vector<float>> pq_tables(M, std::vector<float>(Ks, 0.0f));

        for (const auto& [_, cid] : coarse_dists) {
            std::vector<float> q_res(vec_dim);
            for (size_t j = 0; j < vec_dim; ++j)
                q_res[j] = static_cast<float>(q[j]) - coarse_centroids[cid][j];

            for (size_t m = 0; m < M; ++m) {
                for (size_t k_ = 0; k_ < Ks; ++k_) {
                    double d = 0.0;
                    for (size_t d_ = 0; d_ < subdim; ++d_) {
                        double diff = q_res[m * subdim + d_] - pq_codebooks[m][k_][d_];
                        d += diff * diff;
                    }
                    pq_tables[m][k_] = static_cast<float>(d);
                }
            }

            // Search encoded vectors
            auto cmp = [](const Neighbor<NumType>& a, const Neighbor<NumType>& b) {
                return a.distance < b.distance;
            };
            std::priority_queue<Neighbor<NumType>, std::vector<Neighbor<NumType>>, decltype(cmp)> topK(cmp);

            for (const auto& [idx, code] : inverted_lists[cid]) {
                double dist = 0.0;
                for (size_t m = 0; m < M; ++m)
                    dist += pq_tables[m][code[m]];
                topK.emplace(vectors[idx], std::sqrt(dist));
                if (static_cast<int>(topK.size()) > k) topK.pop();
            }

            std::vector<Neighbor<NumType>> neighbors;
            neighbors.reserve(topK.size());
            while (!topK.empty()) {
                neighbors.push_back(topK.top());
                topK.pop();
            }
            std::reverse(neighbors.begin(), neighbors.end());
            return neighbors;
        }

        return {};
    }

    std::vector<Neighbor<NumType>> range_query(const std::vector<NumType>& q, double range, int nprobe = 1) const {
        if (q.size() != vec_dim) {
            std::cerr << "IVFPQ::range_query: vector dim mismatch" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        nprobe = std::max(1, std::min(static_cast<int>(num_coarse_clusters), nprobe));

        // Find nprobe closest coarse centroids
        std::vector<std::pair<double, size_t>> coarse_dists;
        coarse_dists.reserve(num_coarse_clusters);
        for (size_t c = 0; c < num_coarse_clusters; ++c) {
            double dist = dist_to_centroid(q, coarse_centroids[c]);
            coarse_dists.emplace_back(dist, c);
        }
        std::nth_element(coarse_dists.begin(), coarse_dists.begin() + (nprobe - 1), coarse_dists.end());
        coarse_dists.resize(nprobe);

        size_t subdim = vec_dim / M;
        std::vector<Neighbor<NumType>> result;

        for (const auto& [_, cid] : coarse_dists) {
            // Build PQ tables for this centroid's residual
            std::vector<float> q_res(vec_dim);
            for (size_t j = 0; j < vec_dim; ++j)
                q_res[j] = static_cast<float>(q[j]) - coarse_centroids[cid][j];

            std::vector<std::vector<float>> pq_tables(M, std::vector<float>(Ks, 0.0f));
            for (size_t m = 0; m < M; ++m) {
                for (size_t k_ = 0; k_ < Ks; ++k_) {
                    double d = 0.0;
                    for (size_t d_ = 0; d_ < subdim; ++d_) {
                        double diff = q_res[m * subdim + d_] - pq_codebooks[m][k_][d_];
                        d += diff * diff;
                    }
                    pq_tables[m][k_] = static_cast<float>(d);
                }
            }

            for (const auto& [idx, code] : inverted_lists[cid]) {
                double dist = 0.0;
                for (size_t m = 0; m < M; ++m) dist += pq_tables[m][code[m]];
                dist = std::sqrt(dist);
                if (dist <= range) result.emplace_back(vectors[idx], dist);
            }
        }

        std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) { return a.distance < b.distance; });
        return result;
    }

private:
    // Distance between data vector and float centroid
    static double dist_to_centroid(const std::vector<NumType>& v, const std::vector<float>& c) {
        double s = 0.0;
        for (size_t i = 0; i < c.size(); ++i) {
            double diff = static_cast<double>(v[i]) - static_cast<double>(c[i]);
            s += diff * diff;
        }
        return std::sqrt(s);
    }
    // Encode residual into PQ code
    std::vector<uint8_t> encode_vector(const std::vector<float>& residual) const {
        size_t subdim = vec_dim / M;
        std::vector<uint8_t> code(M);
        for (size_t m = 0; m < M; ++m) {
            double best = std::numeric_limits<double>::max();
            uint8_t best_idx = 0;
            for (size_t k = 0; k < Ks; ++k) {
                double d = 0.0;
                for (size_t j = 0; j < subdim; ++j) {
                    double diff = residual[m * subdim + j] - pq_codebooks[m][k][j];
                    d += diff * diff;
                }
                if (d < best) {
                    best = d;
                    best_idx = static_cast<uint8_t>(k);
                }
            }
            code[m] = best_idx;
        }
        return code;
    }
};

#endif
