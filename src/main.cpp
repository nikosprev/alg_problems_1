#include <iostream>
#include "../include/arg_parser.hpp"

#include "../include/prepare_input.hpp"


int main(int argc, char* argv[]) {

    
    // Parse the arguments and print the config parameters
    Config cfg = parseArguments(argc, argv);
    printConfig(cfg);


    //prepare_sift("data/sift.dat", "input.dat");
   // prepare_mnist("data/mnist.dat", "input.dat");
    return 0;
}