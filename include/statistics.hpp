#ifndef STATISTICS_HPP
#define STATISTICS_HPP

#include <vector>
#include <string>

struct QueryStats {
    double af_sum = 0.0;        // sum of per-query AF (AF_query)
    double recall_sum = 0.0;    // sum of per-query Recall@N

    double tApprox_sum = 0.0;   // sum of approximate query times (ms)
    double tTrue_sum = 0.0;     // sum of true query times (ms)

    int queries = 0;            // number of queries included (Q)

    int total_returned_neighbors = 0; 
};

struct ExperimentSummary {
    std::string dataset_name; // "sift" | "mnist" | fallback
    std::string method_name;  // LSH / Hypercube / IVFFlat / IVFPQ / KNN
    std::string params;       // human readable parameters

    double avgAF = 0.0;
    double avgRecall = 0.0;

    double avgTApprox = 0.0;  // ms
    double avgTTrue = 0.0;    // ms

    double qps = 0.0;         // queries per second (based on approx times)
    double silhouette = 0.0;  // silhouette score
};

// Compute per-query AF using the "paired-by-rank" approach:
// AF_query = average_j( distanceApproximate_j / distanceTrue_j )
// We pass two vectors of distances with the same ordering as returned by
// the ANN and by kNN (true). If the true vector is shorter, we only
// average up to its size.

double computeQueryAF(const std::vector<double>& approxDistances,
                      const std::vector<double>& trueDistances);

// Compute Recall@N for a single query. approxIDs and trueIDs are
// vectors of result indices (ordered). N is the expected top-N.

double computeQueryRecall(const std::vector<int>& approxIDs,
                          const std::vector<int>& trueIDs,
                          int N);

// Update aggregate stats for a single query.
void updateStatsForQuery(QueryStats& stats,
                         double af_query,
                         double recall_query,
                         double tApprox_ms,
                         double tTrue_ms,
                         int returned_neighbors_count);

// Finalize experiment summary from aggregated stats and configuration
ExperimentSummary finalizeSummary(const QueryStats& stats,
                                  const std::string& dataset_name,
                                  const std::string& method_name,
                                  const std::string& params);

// Compute silhouette score for IVFFlat/IVFPQ clustering
// Silhouette score measures how similar an object is to its own cluster compared to other clusters
// Returns average silhouette score: (b - a) / max(a, b)
// where a = mean intra-cluster distance, b = mean nearest-cluster distance
// Template implementation is in statistics.cpp with explicit instantiations
template<typename NumType>
double computeSilhouetteScore(const std::vector<std::vector<float>>& centroids,
                              const std::vector<std::vector<std::pair<size_t, std::vector<NumType>>>>& inverted_lists,
                              const std::vector<std::vector<NumType>>& vectors);

// Append a single-line experiment result to
// <method_name>experiments.txt (e.g. LSHexperiments.txt)
// The line format (single line appended):
// <dataset> <method> <params> AvgAF=<double> Recall=<double> tApprox=<double>ms tTrue=<double>ms QPS=<double> Silhouette=<double>\n
bool appendExperimentLine(const ExperimentSummary& s);

#endif // STATISTICS 

