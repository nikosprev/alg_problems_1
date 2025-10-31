#include <iostream>
#include "../include/arg_parser.hpp"
#include "../include/Algorithms/LSH.hpp"
#include "../include/Algorithms/knn.hpp"
#include "../include/Algorithms/ivfflat.hpp"
#include "../include/Algorithms/ivfpq.hpp"
#include <fstream>
#include "../include/dataset.hpp"



int main(int argc, char* argv[]) {

        
    // Parse the arguments and print the config parameters
    Config cfg = parseArguments(argc, argv);
    printConfig(cfg);

    std::string query_file = cfg.queryFile; 
    std::string input_file = cfg.inputFile; 
    std::string output_file = cfg.outputFile; 
    Dataset input_data = parseInputFile(input_file);
    Dataset query_data = parseInputFile(query_file); 

    
    

    
    if (cfg.lshFlag){ 
        int hashTable_size = 10000,num_tables = cfg.L  ,hashFunction_size = 5 ,seed = cfg.seed,  N = cfg.N; 
        float w = cfg.w ; 
        //initialize LSH solver 
        if (input_data.type == "image"){ 
            int vec_dim = input_data.images[0].pixels.size() ; 
            LSH<u_int8_t> lsh(hashTable_size, num_tables, hashFunction_size, w, vec_dim, seed); 
            for (auto image : input_data.images){ 
                lsh.insert_to_hashTables(image.pixels);
            }
            std::vector<Neighbor<uint8_t>> closest_neighbors;

            for(int i = 0 ;i< 10 &&  i < query_data.images.size() ; ++i){ 
                std::cout << "Using LSH : " << std::endl ; 
                closest_neighbors = lsh.returnANN(query_data.images[i].pixels ,N ); 
                for (const auto &neigh : closest_neighbors ){ 
                    std::cout << "    " << neigh ; 
                }
            }
        }
        else if (input_data.type == "vector"){ 
            int vec_dim = input_data.vectors[0].coordinates.size() ; 
            LSH<float> lsh(hashTable_size, num_tables, hashFunction_size, w, vec_dim, seed); 
            for (auto vector : input_data.vectors ){ 
                lsh.insert_to_hashTables(vector.coordinates);
            }
            std::vector<Neighbor<float>> closest_neighbors;

            for(int i = 0 ;i< 10 &&  i < query_data.vectors.size() ; ++i){ 
                std::cout << "Using LSH : " << std::endl ; 
                closest_neighbors = lsh.returnANN(query_data.vectors[i].coordinates ,N ); 
                for (const auto &neigh : closest_neighbors ){ 
                    std::cout << "    " << neigh ; 
                }
            }
        }



    }
    else if (cfg.ivfflatFlag){ 
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
    }
    else if (cfg.ivfpqFlag){ 
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
    }
    else { //else use the knn algorithm 

        if (input_data.type == "image"){ 
            std::vector<std::vector<u_int8_t>> all_points;
            for (const auto& v : input_data.images) {
                all_points.push_back(v.pixels);
            }
            for(int i = 0 ;i< 10 &&  i < query_data.images.size() ; ++i){ 
                 
                std::vector<Neighbor<u_int8_t>> closest_neighbors = kNN(all_points ,query_data.images[i].pixels ,2); 
                for (const auto &neigh : closest_neighbors ){ 
                    std::cout << "    " << neigh ; 
                }
            }

        }
        else if (input_data.type == "vector"){ 
            std::vector<std::vector<float>> all_points;
            for (const auto& v : input_data.vectors) {
                all_points.push_back(v.coordinates);
            }
            for(int i = 0 ;i< 10 &&  i < query_data.vectors.size() ; ++i){ 
                
                std::vector<Neighbor<float>> closest_neighbors = kNN(all_points ,query_data.vectors[i].coordinates ,2); 
                for (const auto &neigh : closest_neighbors ){ 
                    std::cout << "    " << neigh ; 
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