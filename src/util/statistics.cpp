#include "statistics.hpp"
#include <fstream>
#include <iomanip>
#include <algorithm>

double computeQueryAF(const std::vector<double>& approxDistances,
                      const std::vector<double>& trueDistances) {
    if (approxDistances.empty() || trueDistances.empty()) return 0.0;

    size_t m = std::min(approxDistances.size(), trueDistances.size());
    double sum = 0.0;
    for (size_t i = 0; i < m; ++i) {
        double dA = approxDistances[i];
        double dT = trueDistances[i];
        if (dT <= 0.0) {
            if (dA == 0.0) sum += 1.0;
            else continue; 
        } else {
            sum += (dA / dT);
        }
    }
    if (m == 0) return 10.0; //if m == 0 the lsh found no neighbors -> penalize it 
    return sum / static_cast<double>(m);
}


double computeQueryRecall(const std::vector<int>& approxIDs,
                          const std::vector<int>& trueIDs,
                          int N) {
    if (N <= 0) return 0.0;
    int topN = std::min<int>(N, static_cast<int>(approxIDs.size()));
    if (topN == 0) return 0.0;

    int matches = 0;
    for (int i = 0; i < topN; ++i) {
        int id = approxIDs[i];
        if (std::find(trueIDs.begin(), trueIDs.end(), id) != trueIDs.end()) {
            ++matches;
        }
    }
    return static_cast<double>(matches) / static_cast<double>(N);
}

void updateStatsForQuery(QueryStats& stats,
                         double af_query,
                         double recall_query,
                         double tApprox_ms,
                         double tTrue_ms,
                         int returned_neighbors_count) {
    stats.af_sum += af_query;
    stats.recall_sum += recall_query;
    stats.tApprox_sum += tApprox_ms;
    stats.tTrue_sum += tTrue_ms;
    stats.queries += 1;
    stats.total_returned_neighbors += returned_neighbors_count;
}

ExperimentSummary finalizeSummary(const QueryStats& stats,
                                  const std::string& dataset_name,
                                  const std::string& method_name,
                                  const std::string& params) {
    ExperimentSummary s;
    s.dataset_name = dataset_name;
    s.method_name = method_name;
    s.params = params;
    if (stats.queries > 0) {
        s.avgAF = stats.af_sum / static_cast<double>(stats.queries);
        s.avgRecall = stats.recall_sum / static_cast<double>(stats.queries);
        s.avgTApprox = stats.tApprox_sum / static_cast<double>(stats.queries);
        s.avgTTrue = stats.tTrue_sum / static_cast<double>(stats.queries);
        double totalApproxMs = stats.tApprox_sum;
        if (totalApproxMs > 0.0) {
            s.qps = (static_cast<double>(stats.queries) / (totalApproxMs / 1000.0));
        } else s.qps = 0.0;
    }
    return s;
}

bool appendExperimentLine(const ExperimentSummary& s) {
    std::string fname = s.method_name + "experiments.txt";
    std::ofstream ofs(fname, std::ios::app);
    if (!ofs.is_open()) return false;

    ofs << s.dataset_name << " " << s.method_name << " " << s.params
        << " AvgAF=" << std::fixed << std::setprecision(6) << s.avgAF
        << " Recall=" << std::fixed << std::setprecision(6) << s.avgRecall
        << " tApprox=" << std::fixed << std::setprecision(3) << s.avgTApprox << "ms"
        << " tTrue=" << std::fixed << std::setprecision(3) << s.avgTTrue << "ms"
        << " QPS=" << std::fixed << std::setprecision(2) << s.qps
        << "\n";

    ofs.close();
    return true;
}