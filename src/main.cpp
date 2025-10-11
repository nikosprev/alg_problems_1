#include <iostream>
#include "../include/arg_parser.hpp"



int main(int argc, char* argv[]) {

    
    // Parse the arguments and print the config parameters
    Config cfg = parseArguments(argc, argv);
    printConfig(cfg);
    

    

    return 0;
}