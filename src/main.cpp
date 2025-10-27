#include <iostream>
#include "../include/arg_parser.hpp"


#include "../include/dataset.hpp"


int main(int argc, char* argv[]) {

    
    // Parse the arguments and print the config parameters
   // Config cfg = parseArguments(argc, argv);
    //printConfig(cfg);


   

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
}