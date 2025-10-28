#include <iostream>
#include "../include/arg_parser.hpp"
#include "../include/Algorithms/LSH.hpp"
#include "../include/Algorithms/knn.hpp"
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

    printRandomVectors(input_data , 1);
    

    
    if (cfg.lshFlag){ 
        int hashTable_size = 10000,num_tables = cfg.L  ,hashFunction_size = 5 ,seed = cfg.seed, vec_dim = 128 , N = cfg.N; 
        float w = cfg.w ; 
        //initialize LSH solver 
        LSH lsh(hashTable_size, num_tables, hashFunction_size, w, vec_dim, seed); 
        if (input_data.type == "image"){ 
            for (auto image : input_data.images){ 
                continue;   
            }
        }
        else if (input_data.type == "vector"){ 
            for (auto vector : input_data.vectors ){ 
                lsh.insert_to_hashTables(vector.coordinates);
            }
            std::vector<Neighbor> closest_neighbors;

            for(int i = 0 ;i< 10 &&  i < query_data.vectors.size() ; ++i){ 
                std::cout << "Using LSH : " << std::endl ; 
                closest_neighbors = lsh.returnANN(query_data.vectors[i].coordinates ,N , true ,0.5); 
                for (const auto &neigh : closest_neighbors ){ 
                    std::cout << "    " << neigh ; 
                }
            }
        }



    }
    else if (cfg.ivfflatFlag){ 
        std::cout << "In progress \n"; 
    }
    else if (cfg.ivfpqFlag){ 
        std::cout << "In progress \n"; 
    }
    else { 
        std::cerr << "No algorithm flag \n" ; 
        exit(1); 
    }
     


    /* 
    size_t hashTable_size = 10000;   // slots per table
    int num_tables = 5;          // number of hash tables
    int hashFunction_size = 6;   // number of hash functions in amplified hash
    int w = 20;                   // window size
    size_t vec_dim = 128;        // SIFT vectors are 128-dimensional
    int seed = 42;

    LSH lsh(hashTable_size, num_tables, hashFunction_size, w, vec_dim, seed); 

    // Load SIFT base vectors
    auto base_vectors = read_fvecs("./data/sift/sift_base.fvecs");
    std::cout << "Loaded " << base_vectors.size() << " base vectors." << std::endl;

    // Insert vectors into all hash tables
    for (const auto &vec : base_vectors) {
        lsh.insert_to_hashTables(vec);
    }

    // Load SIFT query vectors
    auto query_vectors = read_fvecs("./data/sift/sift_query.fvecs");
    std::cout << "Loaded " << query_vectors.size() << " query vectors." << std::endl;

    // Run queries and print candidate counts
    std::vector<Neighbor> n; 
    for (size_t i = 0; i < 2 &&  i < query_vectors.size(); ++i) {
        std::cout << "Using LSH : " << std::endl ; 
        n = lsh.returnANN(query_vectors[i] ,1); 
        for (const auto &neigh : n ){ 
            std::cout << "    " << neigh ; 
        }
        std::cout << "Using KNN : " << std::endl ; 
        n = kNN(base_vectors ,query_vectors[i] ,1); 
        for (const auto &neigh : n ){ 
            std::cout << "    " << neigh ; 
        }

    }

    */

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