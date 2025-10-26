#include <iostream>
#include "../include/arg_parser.hpp"

#include "../include/prepare_input.hpp"
#include "../include/dataset.hpp"


int main(int argc, char* argv[]) {

    
    // Parse the arguments and print the config parameters
    Config cfg = parseArguments(argc, argv);
    printConfig(cfg);


    //prepare_sift("data/sift.dat", "input.dat");
   // prepare_mnist("data/mnist.dat", "input.dat");


   try {
    Dataset dataset = parseInputFile("input.dat");

    std::cout << "Detected dataset type: " << dataset.type << "\n";

    if (dataset.type == "image") {
        std::cout << "Loaded " << dataset.images.size() << " image(s)\n";
        for (const auto &img : dataset.images) {
            std::cout << "Image: " << img.rows << "x" << img.cols
                      << ", pixels=" << img.pixels.size() << "\n";
        }
    }

    else if (dataset.type == "vector") {
        std::cout << "Loaded " << dataset.vectors.size() << " vector(s)\n";
        for (size_t i = 0; i < dataset.vectors.size(); ++i) {
            const auto &vec = dataset.vectors[i];
           // std::cout << "Vector " << i + 1 << ": dimension="
              //        << vec.dimension
                //      << ", coords=" << vec.coordinates.size() << "\n";
        }
    }
}
catch (std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
}


    return 0;
}