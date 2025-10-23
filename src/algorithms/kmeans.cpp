#include "../../include/kmeans.hpp"
#include <limits>
#include <unordered_map>
#include <algorithm>



KMeans::KMeans(const Params &p) : params(p), rng(p.seed) {}

std::vector<Vec> KMeans::fit(const Dataset &ds) {
    if (ds.size() == 0) return {};
    size_t n = ds.size();
    size_t dim = ds.dim;
    size_t k = std::min(params.k, n);

    // Random initialization
    std::vector<Vec> centroids;
    centroids.reserve(k);
    std::uniform_int_distribution<size_t> dist(0, n - 1);
    std::unordered_map<size_t, bool> used;
    while (centroids.size() < k) {
        size_t idx = dist(rng);
        if (used.emplace(idx, true).second)
            centroids.push_back(ds.data[idx]);
    }

    std::vector<size_t> assignments(n, 0);
    for (size_t iter = 0; iter < params.max_iters; ++iter) {
        bool changed = false;

        // Assignment step
        for (size_t i = 0; i < n; ++i) {
            Float best = std::numeric_limits<Float>::infinity();
            size_t best_j = 0;
            for (size_t j = 0; j < k; ++j) {
                Float dist = l2_squared(ds.data[i], centroids[j]);
                if (dist < best) { best = dist; best_j = j; }
            }
            if (assignments[i] != best_j) { changed = true; assignments[i] = best_j; }
        }

        // Update step
        std::vector<Vec> sums(k, Vec(dim, 0.0f));
        std::vector<size_t> counts(k, 0);
        for (size_t i = 0; i < n; ++i) {
            size_t a = assignments[i];
            ++counts[a];
            const Vec &v = ds.data[i];
            for (size_t d = 0; d < dim; ++d) sums[a][d] += v[d];
        }
        for (size_t j = 0; j < k; ++j) {
            if (counts[j] == 0) continue;
            for (size_t d = 0; d < dim; ++d) centroids[j][d] = sums[j][d] / counts[j];
        }

        if (!changed) break;
    }
    return centroids;
}