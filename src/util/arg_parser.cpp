#include "../../include/arg_parser.hpp"
#include <iostream>

Config parseArguments(int argc, char* argv[]) {
    Config cfg;

    for (int i = 1; i < argc; ++i) {   
        std::string arg = argv[i];

        // Common parameters 
        if (arg == "-d" && i + 1 < argc) cfg.inputFile = argv[++i];
        else if (arg == "-q" && i + 1 < argc) cfg.queryFile = argv[++i];
        else if (arg == "-ο" && i + 1 < argc) cfg.outputFile = argv[++i];
        else if (arg == "-Ν" && i + 1 < argc) cfg.N = std::stoi(argv[++i]);
        else if (arg == "-R" && i + 1 < argc) cfg.R = std::stod(argv[++i]);
        else if (arg == "-type" && i + 1 < argc) cfg.typeFlag = argv[++i];
        else if (arg == "-range" && i + 1 < argc) {
            std::string val = argv[++i];
            cfg.rangeFlag = (val == "true");
        }

        // LSH parameters
        else if (arg == "-k" && i + 1 < argc) cfg.k = std::stoi(argv[++i]);
        else if (arg == "-L" && i + 1 < argc) cfg.L = std::stoi(argv[++i]);
        else if (arg == "-w" && i + 1 < argc) cfg.w = std::stod(argv[++i]);
        else if (arg == "-lsh") cfg.lshFlag = true;

        // Hypercube parameters
        else if (arg == "-kproj" && i + 1 < argc) cfg.kproj = std::stoi(argv[++i]);
        else if (arg == "-M" && i + 1 < argc) cfg.M = std::stoi(argv[++i]);
        else if (arg == "-probes" && i + 1 < argc) cfg.probes = std::stoi(argv[++i]);
        else if (arg == "-hypercube") cfg.hypercubeFlag = true;

        // IVFFlat / IVFPQ parameters
        else if (arg == "-kclusters" && i + 1 < argc) cfg.kclusters = std::stoi(argv[++i]);
        else if (arg == "-nprobe" && i + 1 < argc) cfg.nprobe = std::stoi(argv[++i]);
        else if (arg == "-nbits" && i + 1 < argc) cfg.nbits = std::stoi(argv[++i]);
        else if (arg == "-ivfflat") cfg.ivfflatFlag = true;
        else if (arg == "-ivfpq") cfg.ivfpqFlag = true;
        else if (arg == "-seed" && i + 1 < argc) cfg.seed = std::stoi(argv[++i]);

        else {
            std::cerr << "Unknown or incomplete argument: " << arg << "\n";
        }
    }

    return cfg;
}

void printConfig(const Config& cfg) {
    std::cout << "Input File: " << cfg.inputFile << "\n";
    std::cout << "Query File: " << cfg.queryFile << "\n";
    std::cout << "Output File: " << cfg.outputFile << "\n";
    std::cout << "N: " << cfg.N << ", R: " << cfg.R << ", Type: " << cfg.typeFlag << ", Range: " << cfg.rangeFlag << "\n";

    if (cfg.lshFlag) 
        std::cout << "[LSH] k: " << cfg.k << ", L: " << cfg.L << ", w: " << cfg.w << "\n";
    if (cfg.hypercubeFlag)
        std::cout << "[Hypercube] kproj: " << cfg.kproj << ", M: " << cfg.M << ", probes: " << cfg.probes << "\n";
    if (cfg.ivfflatFlag)
        std::cout << "[IVFFlat] kclusters: " << cfg.kclusters << ", nprobe: " << cfg.nprobe << ", seed: " << cfg.seed << "\n";
    if (cfg.ivfpqFlag)
        std::cout << "[IVFPQ] kclusters: " << cfg.kclusters << ", nprobe: " << cfg.nprobe << ", M: " << cfg.M 
                  << ", nbits: " << cfg.nbits << ", seed: " << cfg.seed << "\n";
}