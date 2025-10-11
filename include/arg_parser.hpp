#include <string>

struct Config {  //struct for the config parameters
    
    string inputFile;
    string queryFile;
    string outputFile;
    int N = 0;
    double R = 0.0;
    string typeFlag;
    bool rangeFlag = false;

    // LSH
    int k = 0, L = 0;
    double w = 0.0;
    bool lshFlag = false;

    // Hypercube
    int kproj = 0, M = 0, probes = 0;
    bool hypercubeFlag = false;

    // IVFFlat / IVFPQ
    int kclusters = 0, nprobe = 0, nbits = 0;
    bool ivfflatFlag = false;
    bool ivfpqFlag = false;
    int seed = 0;
};

void printConfig(const Config&);    // print the config parameters
Config parseArguments(int , char**);    // parse the arguments