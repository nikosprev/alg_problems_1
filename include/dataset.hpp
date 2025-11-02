
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>
#include <regex>

struct Image {
    int rows = 0;
    int cols = 0;
    std::vector<unsigned char> pixels;
};

struct Vector {
    int dimension = 0;
    std::vector<float> coordinates;
};

struct Dataset {
    std::string type;  // "image" or "vector"
    std::vector<Image> images;
    std::vector<Vector> vectors;
};

Dataset parseInputFile(const std::string & ,bool early_exit = true);

std::string detectDatasetType(const std::vector<std::string> &);// Detect dataset type based on keywords

std::string trim(const std::string &);// Helper function to trim whitespace

void printRandomVectors(const Dataset &dataset, int num_vectors = 5);// Print random vectors from the dataset

void printRandomImages(const Dataset &dataset, int num_images = 5);