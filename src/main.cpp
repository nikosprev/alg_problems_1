#include <iostream>
#include "../include/arg_parser.hpp"
#include "../include/Algorithms/LSH.hpp"
#include "../include/Algorithms/knn.hpp"
//#include "../include/Algorithms/ivfflat.hpp"
#//include "../include/Algorithms/ivfpq.hpp"
#include <fstream>
#include "../include/dataset.hpp"
#include "../include/Algorithms/hypercube.hpp"
#include "../include/statistics.hpp"
#include <chrono>

int main(int argc, char* argv[]) {

        
    // Parse the arguments and print the config parameters
    Config cfg = parseArguments(argc, argv);
    printConfig(cfg);

    std::string query_file = cfg.queryFile; 
    std::string input_file = cfg.inputFile; 
    std::string output_file = cfg.outputFile; 
    Dataset input_data = parseInputFile(input_file ,false);
    Dataset query_data = parseInputFile(query_file); 

    std::ofstream out_file;
    if (!cfg.outputFile.empty()) {
        out_file.open(cfg.outputFile, std::ios::trunc); // clears the file
    }
    

    
    
    QueryStats stats;  // aggregate stats for experiment

    // ------------------ LSH ------------------
    if (cfg.lshFlag) {
        out_file << "METHOD: LSH\n";

        int hashTable_size = 5000;
        int num_tables = cfg.L;
        int hashFunction_size = cfg.k;
        int seed = cfg.seed;
        int N = cfg.N;
        float w = cfg.w;

        if (input_data.type == "vector") {
            int vec_dim = input_data.vectors[0].coordinates.size();
            LSH<float> lsh(hashTable_size, num_tables, hashFunction_size, w, vec_dim, seed);
            for (auto& v : input_data.vectors)
                lsh.insert_to_hashTables(v.coordinates);

            std::vector<std::vector<float>> all_vectors;
            for (const auto& v : input_data.vectors)
                all_vectors.push_back(v.coordinates);

            for (size_t i = 0; i < query_data.vectors.size() ; ++i) {
                const auto& query = query_data.vectors[i].coordinates;

                auto start = std::chrono::high_resolution_clock::now();
                auto approx_neighbors = lsh.returnANN(query, N);
                auto end = std::chrono::high_resolution_clock::now();
                double tApprox = std::chrono::duration<double, std::milli>(end - start).count();

                 auto startTrue = std::chrono::high_resolution_clock::now();
                auto true_neighbors = kNN(all_vectors, query, N);
                auto endTrue = std::chrono::high_resolution_clock::now();
                double tTrue = std::chrono::duration<double, std::milli>(endTrue - startTrue).count();

                std::vector<double> approxDists, trueDists;
                std::vector<int> approxIDs, trueIDs;
                for (auto& n : approx_neighbors) { approxIDs.push_back(n.idx); approxDists.push_back(n.distance); }
                for (auto& n : true_neighbors) { trueIDs.push_back(n.idx); trueDists.push_back(n.distance); }

                double af = computeQueryAF(approxDists, trueDists);
                double recall = computeQueryRecall(approxIDs, trueIDs, N);
                updateStatsForQuery(stats, af, recall, tApprox, tTrue, approx_neighbors.size());

                if (out_file.is_open()) {
                    out_file << "Query: " << i + 1 << "\n";
                    for (size_t j = 0; j < approx_neighbors.size(); ++j) {
                        out_file << "   Nearest neighbor-" << j + 1 << ": " << approx_neighbors[j].idx << "\n";
                        out_file << "   distanceApproximate: " << approx_neighbors[j].distance << "\n";
                        if (j < trueDists.size()) out_file << "   distanceTrue: " << trueDists[j] << "\n";
                    }
                    out_file << "AF=" << af << " Recall@" << N << "=" << recall << "\n";
                    out_file << "QPS: " << tApprox << " ms\n\n";
                }
            }

            ExperimentSummary summary = finalizeSummary(stats, "vector_dataset", "LSH",
                                                        "L=" + std::to_string(cfg.L) +
                                                        ", k=" + std::to_string(cfg.k) +
                                                        ", w=" + std::to_string(cfg.w));
            appendExperimentLine(summary);
        }
        else if (input_data.type == "image") {
            int vec_dim = input_data.images[0].pixels.size();
            LSH<uint8_t> lsh(hashTable_size, num_tables, hashFunction_size, w, vec_dim, seed);
            for (auto& img : input_data.images) lsh.insert_to_hashTables(img.pixels);

            std::vector<std::vector<uint8_t>> all_images;
            for (auto& img : input_data.images) all_images.push_back(img.pixels);

            for (size_t i = 0;   i < query_data.images.size(); ++i) {
                const auto& query = query_data.images[i].pixels;

                auto start = std::chrono::high_resolution_clock::now();
                auto approx_neighbors = lsh.returnANN(query, N);
                auto end = std::chrono::high_resolution_clock::now();
                double tApprox = std::chrono::duration<double, std::milli>(end - start).count();

                auto startTrue = std::chrono::high_resolution_clock::now();
                auto true_neighbors = kNN(all_images, query, N);
                auto endTrue = std::chrono::high_resolution_clock::now();
                double tTrue = std::chrono::duration<double, std::milli>(endTrue - startTrue).count();

                std::vector<double> approxDists, trueDists;
                std::vector<int> approxIDs, trueIDs;
                for (auto& n : approx_neighbors) { approxIDs.push_back(n.idx); approxDists.push_back(n.distance); }
                for (auto& n : true_neighbors) { trueIDs.push_back(n.idx); trueDists.push_back(n.distance); }

                double af = computeQueryAF(approxDists, trueDists);
                double recall = computeQueryRecall(approxIDs, trueIDs, N);
                updateStatsForQuery(stats, af, recall, tApprox, tTrue, approx_neighbors.size());

                if (out_file.is_open()) {
                    out_file << "Query: " << i + 1 << "\n";
                    for (size_t j = 0; j < approx_neighbors.size(); ++j) {
                        out_file << "   Nearest neighbor-" << j + 1 << ": " << approx_neighbors[j].idx << "\n";
                        out_file << "   distanceApproximate: " << approx_neighbors[j].distance << "\n";
                        if (j < trueDists.size()) out_file << "   distanceTrue: " << trueDists[j] << "\n";
                    }
                    out_file << "AF=" << af << " Recall@" << N << "=" << recall << "\n";
                    out_file << "QPS: " << tApprox << " ms\n\n";
                }
            }

            ExperimentSummary summary = finalizeSummary(stats, "image_dataset", "LSH",
                                                        "L=" + std::to_string(cfg.L) +
                                                        ", k=" + std::to_string(cfg.k) +
                                                        ", w=" + std::to_string(cfg.w));
            appendExperimentLine(summary);
        }
    }

    else if (cfg.hypercubeFlag) {
        std::ofstream out_file;
        if (!cfg.outputFile.empty()) out_file.open(cfg.outputFile, std::ios::app);
        out_file << "METHOD: Hypercube\n";

        if (input_data.type == "image") {
            size_t vec_dim = input_data.images[0].pixels.size();
            std::vector<std::vector<uint8_t>> dataset;
            dataset.reserve(input_data.images.size());
            for (auto& img : input_data.images)
                dataset.push_back(img.pixels);

            HyperCube<uint8_t> hypercube(dataset, cfg.kproj, cfg.w, vec_dim, 1);

            for (int i = 0; i < query_data.images.size(); ++i) {
                auto start = std::chrono::high_resolution_clock::now();
                auto closest_neighbors = hypercube.returnANN(
                    query_data.images[i].pixels,
                    cfg.M,
                    cfg.N,
                    cfg.probes
                );
                auto end = std::chrono::high_resolution_clock::now();
                double query_time = std::chrono::duration<double, std::milli>(end - start).count();

                if (out_file.is_open()) {
                    out_file << "Query: " << i+1 << "\n";
                    int nn_count = 1;
                    for (const auto &neigh : closest_neighbors) {
                        out_file << "   Nearest neighbor-" << nn_count++ << ": " << neigh.idx << "\n";
                        out_file << "   distanceApproximate: " << neigh.distance << "\n";
                        out_file << "   distanceTrue: " << neigh.distance << "\n\n";
                    }
                    out_file << "QPS: " << query_time << " ms\n\n";
                }
            }
        }
        else if (input_data.type == "vector") {
            size_t vec_dim = input_data.vectors[0].coordinates.size();
            std::vector<std::vector<float>> dataset;
            dataset.reserve(input_data.vectors.size());
            for (auto& v : input_data.vectors)
                dataset.push_back(v.coordinates);

            HyperCube<float> hypercube(dataset, cfg.kproj, cfg.w, vec_dim, 1);

            for (int i = 0;i < 100  && i < query_data.vectors.size(); ++i) {
                auto start = std::chrono::high_resolution_clock::now();
                auto closest_neighbors = hypercube.returnANN(
                    query_data.vectors[i].coordinates,
                    cfg.M,
                    cfg.N,
                    cfg.probes
                );
                auto end = std::chrono::high_resolution_clock::now();
                double query_time = std::chrono::duration<double, std::milli>(end - start).count();

                if (out_file.is_open()) {
                    out_file << "Query: " << i+1 << "\n";
                    int nn_count = 1;
                    for (const auto &neigh : closest_neighbors) {
                        out_file << "   Nearest neighbor-" << nn_count++ << ": " << neigh.idx << "\n";
                        out_file << "   distanceApproximate: " << neigh.distance << "\n";
                        out_file << "   distanceTrue: " << neigh.distance << "\n\n";
                    }
                    out_file << "QPS: " << query_time << " ms\n\n";
                }
            }
        }

        if (out_file.is_open()) out_file.close();
    }


    else if (cfg.ivfflatFlag){ 
        /*
        int num_clusters = cfg.kclusters, nprobe = cfg.nprobe, N = cfg.N, seed = cfg.seed;
        bool do_range = cfg.rangeFlag; double rangeR = cfg.R;
        std::ofstream out_ivf;
        if (!cfg.outputFile.empty()) out_ivf.open(cfg.outputFile, std::ios::app);
        int iters = 15; // default k-means iterations; could be exposed if needed
        
        if (input_data.type == "image"){ 
            int vec_dim = input_data.images[0].pixels.size();
            
            // Initialize IVFFlat (num_clusters, vec_dim, iters)
            IVFFlat<uint8_t> ivf(num_clusters, vec_dim, iters);
            
            // Insert all data points first
            for (const auto& image : input_data.images) {
                ivf.add_vector(image.pixels);
            }
            
            // Train on the stored vectors
            std::cout << "Training IVFFlat with " << num_clusters << " clusters..." << std::endl;
            ivf.train(seed);
            
            if (do_range) {
                for (int i = 0; i < 10 && i < static_cast<int>(query_data.images.size()); ++i) {
                    std::cout << "Using IVFFlat (range R=" << rangeR << ", nprobe=" << nprobe << ") :\n";
                    auto neighbors = ivf.range_query(query_data.images[i].pixels, rangeR, nprobe);
                    for (const auto &neigh : neighbors) { std::cout << "    " << neigh; if (out_ivf.is_open()) out_ivf << neigh; }
                }
            } else {
                for (int i = 0; i < 10 && i < static_cast<int>(query_data.images.size()); ++i) {
                    std::cout << "Using IVFFlat (top-" << N << ", nprobe=" << nprobe << ") :\n";
                    auto neighbors = ivf.query(query_data.images[i].pixels, N, nprobe);
                    for (const auto &neigh : neighbors) { std::cout << "    " << neigh; if (out_ivf.is_open()) out_ivf << neigh; }
                }
            }
        }
        else if (input_data.type == "vector"){ 
            int vec_dim = input_data.vectors[0].coordinates.size();
            
            // Initialize IVFFlat
            IVFFlat<float> ivf(num_clusters, vec_dim, iters);
            
            // Insert all data points first
            for (const auto& vector : input_data.vectors) {
                ivf.add_vector(vector.coordinates);
            }
            
            // Train on the stored vectors
            std::cout << "Training IVFFlat with " << num_clusters << " clusters..." << std::endl;
            ivf.train(seed);
            
            if (do_range) {
                for (int i = 0; i < 10 && i < static_cast<int>(query_data.vectors.size()); ++i) {
                    std::cout << "Using IVFFlat (range R=" << rangeR << ", nprobe=" << nprobe << ") :\n";
                    auto neighbors = ivf.range_query(query_data.vectors[i].coordinates, rangeR, nprobe);
                    for (const auto &neigh : neighbors) { std::cout << "    " << neigh; if (out_ivf.is_open()) out_ivf << neigh; }
                }
            } else {
                for (int i = 0; i < 10 && i < static_cast<int>(query_data.vectors.size()); ++i) {
                    std::cout << "Using IVFFlat (top-" << N << ", nprobe=" << nprobe << ") :\n";
                    auto neighbors = ivf.query(query_data.vectors[i].coordinates, N, nprobe);
                    for (const auto &neigh : neighbors) { std::cout << "    " << neigh; if (out_ivf.is_open()) out_ivf << neigh; }
                }
            }
            if (out_ivf.is_open()) out_ivf.close();
        }
            */
    }
    else if (cfg.ivfpqFlag){ 
        /*
        int num_clusters = cfg.kclusters, nprobe = cfg.nprobe, N = cfg.N, seed = cfg.seed;
        int M = cfg.M, nbits = cfg.nbits; int iters = 15;
        size_t Ks = (nbits > 0 ? static_cast<size_t>(1u << nbits) : 256u);
        bool do_range = cfg.rangeFlag; double rangeR = cfg.R;
        std::ofstream out;
        if (!cfg.outputFile.empty()) out.open(cfg.outputFile);
        if (input_data.type == "image"){ 
            int vec_dim = input_data.images[0].pixels.size();

            IVFPQ<uint8_t> ivfpq(static_cast<size_t>(num_clusters), static_cast<size_t>(vec_dim), static_cast<size_t>(M), Ks, iters);
            for (const auto& image : input_data.images) ivfpq.add_vector(image.pixels);
            std::cout << "Training IVFPQ (nlist=" << num_clusters << ", M=" << M << ", nbits=" << nbits << ")...\n";
            ivfpq.train(seed);
            for (int i = 0; i < 10 && i < static_cast<int>(query_data.images.size()); ++i) {
                if (do_range) {
                    std::cout << "Using IVFPQ (range R=" << rangeR << ", nprobe=" << nprobe << ") :\n";
                    auto neighbors = ivfpq.range_query(query_data.images[i].pixels, rangeR, nprobe);
                    for (const auto &neigh : neighbors) {
                        std::cout << "    " << neigh;
                        if (out.is_open()) out << neigh;
                    }
                } else {
                    std::cout << "Using IVFPQ (top-" << N << ", nprobe=" << nprobe << ") :\n";
                    auto neighbors = ivfpq.query(query_data.images[i].pixels, N, nprobe);
                    for (const auto &neigh : neighbors) {
                        std::cout << "    " << neigh;
                        if (out.is_open()) out << neigh;
                    }
                }
            }
        }
        else if (input_data.type == "vector"){ 
            int vec_dim = input_data.vectors[0].coordinates.size();

            IVFPQ<float> ivfpq(static_cast<size_t>(num_clusters), static_cast<size_t>(vec_dim), static_cast<size_t>(M), Ks, iters);
            for (const auto& v : input_data.vectors) ivfpq.add_vector(v.coordinates);
            std::cout << "Training IVFPQ (nlist=" << num_clusters << ", M=" << M << ", nbits=" << nbits << ")...\n";
            ivfpq.train(seed);
            for (int i = 0; i < 10 && i < static_cast<int>(query_data.vectors.size()); ++i) {
                if (do_range) {
                    std::cout << "Using IVFPQ (range R=" << rangeR << ", nprobe=" << nprobe << ") :\n";
                    auto neighbors = ivfpq.range_query(query_data.vectors[i].coordinates, rangeR, nprobe);
                    for (const auto &neigh : neighbors) {
                        std::cout << "    " << neigh;
                        if (out.is_open()) out << neigh;
                    }
                } else {
                    std::cout << "Using IVFPQ (top-" << N << ", nprobe=" << nprobe << ") :\n";
                    auto neighbors = ivfpq.query(query_data.vectors[i].coordinates, N, nprobe);
                    for (const auto &neigh : neighbors) {
                        std::cout << "    " << neigh;
                        if (out.is_open()) out << neigh;
                    }
                }
            }
            if (out.is_open()) out.close();
        }
        */
    }
    else { // fallback KNN
    std::cout << "Using KNN\n"; 
    std::ofstream out_file;
    if (!cfg.outputFile.empty()) out_file.open(cfg.outputFile, std::ios::app);
    out_file << "METHOD: KNN\n";
    if (input_data.type == "image"){ 
        std::vector<std::vector<uint8_t>> all_points;
        for (const auto& v : input_data.images)
            all_points.push_back(v.pixels);

        for(int i = 0;i < query_data.images.size(); ++i){ 
            
            auto closest_neighbors = kNN(all_points, query_data.images[i].pixels, cfg.N); 

            if (out_file.is_open()) {
                
                out_file << "Query: " << i +1<< "\n";
                int nn_count = 1;
                for (const auto &neigh : closest_neighbors) {
                    out_file << "   Nearest neighbor-" << nn_count++ << ": " << neigh.idx << "\n";
                    out_file << "   distanceTrue: " << neigh.distance << "\n\n";
                }

            }
        }
    }
    else if (input_data.type == "vector"){ 
        std::vector<std::vector<float>> all_points;
        for (const auto& v : input_data.vectors)
            all_points.push_back(v.coordinates);

        for(int i = 0;  i < query_data.vectors.size(); ++i){ 
            //std::cout << "      Query :" << i+1 << "\n"; 
            auto closest_neighbors = kNN(all_points, query_data.vectors[i].coordinates, cfg.N); 

            if (out_file.is_open()) {
                
                out_file << "Query: " << i << "\n";
                int nn_count = 1;
                for (const auto &neigh : closest_neighbors) {
                    out_file << "   Nearest neighbor-" << nn_count++ << ": " << neigh.idx << "\n";
                    out_file << "   distanceTrue: " << neigh.distance << "\n\n";
                }
        } 
    }
    }

}
    /*
    try {
        Dataset dataset = parseInputFile("input.dat");

        std::cout << "Detected dataset type: " << dataset.type << "\n";

    if (dataset.type == "image") {
        std::cout << "Loaded " << dataset.images.size() << " image(s)\n";
      printRandomImages(dataset,3);
    }

    else if (dataset.type == "vector") {
        std::cout << "Loaded " << dataset.vectors.size() << " vector(s)\n";
        for (size_t i = 0; i < dataset.vectors.size(); ++i) {
            const auto &vec = dataset.vectors[i];
           // std::cout << "Vector " << i + 1 << ": dimension="
          //            << vec.dimension
             //         << ", coords=" << vec.coordinates.size() << "\n";
        }
        
        // Print some random vectors from the dataset
       printRandomVectors(dataset, 1000);
    }
}
catch (std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
}


        return 0;
        */ 







}