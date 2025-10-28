#include "../../include/dataset.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include <stdexcept>
#include "../../include/Algorithms/metrics.hpp"
// Endianness Utilities
uint32_t read_uint32_big(std::ifstream &f) {
    uint8_t bytes[4];
    f.read(reinterpret_cast<char *>(bytes), 4);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

uint32_t read_uint32_little(std::ifstream &f) {
    uint8_t bytes[4];
    f.read(reinterpret_cast<char *>(bytes), 4);
    return (bytes[3] << 24) | (bytes[2] << 16) | (bytes[1] << 8) | bytes[0];
}

float read_float32_little(std::ifstream &f) {
    float value;
    f.read(reinterpret_cast<char *>(&value), 4);
    return value;
}

// Dataset Parsing 

Dataset parseInputFile(const std::string &filename) {
    Dataset dataset;
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("Error: Could not open input file " + filename);

    // Try Big Endian (Image dataset)
    f.seekg(0, std::ios::beg);
    uint32_t magic_be = read_uint32_big(f);

    if (magic_be == 2051) {
        dataset.type = "image";
        std::cout << "Detected Big Endian (Image) Dataset\n";

        uint32_t num_images = read_uint32_big(f);
        uint32_t rows = read_uint32_big(f);
        uint32_t cols = read_uint32_big(f);

        dataset.images.reserve(num_images);
        size_t image_size = rows * cols;

        std::cout << "Images: " << num_images << ", Size: " << rows << "x" << cols << "\n";

        for (uint32_t i = 0; i < num_images; ++i) {
            Image img;
            img.rows = rows;
            img.cols = cols;
            img.pixels.resize(image_size);
            f.read(reinterpret_cast<char *>(img.pixels.data()), image_size);
            dataset.images.push_back(std::move(img));
        }

        std::cout << "Loaded " << dataset.images.size() << " images.\n";
        f.close();
        return dataset;
    }

    // Try Little Endian (Vector dataset)
    f.clear();
    f.seekg(0, std::ios::beg);
    dataset.type = "vector";
    std::cout << "Detected Little Endian (Vector) Dataset\n";

    int vectors_loaded = 0;
    const int MAX_VECTORS = 10000;

    while (true) {
        if (f.peek() == EOF) break;
        uint32_t dim;
        f.read(reinterpret_cast<char *>(&dim), 4);
        if (f.eof()) break;

        // assume little-endian integer
        uint8_t *p = reinterpret_cast<uint8_t *>(&dim);
        dim = (p[0]) | (p[1] << 8) | (p[2] << 16) | (p[3] << 24);

        if (dim == 0 || dim > 100000) break; // sanity check

        Vector vec;
        vec.dimension = dim;
        vec.coordinates.resize(dim);

        for (uint32_t i = 0; i < dim; ++i) {
            vec.coordinates[i] = read_float32_little(f);
        }
        //Normalize vec here 
        normalize(vec.coordinates); 
        dataset.vectors.push_back(std::move(vec));
        vectors_loaded++;
        if (vectors_loaded >= MAX_VECTORS) break;
    }

    std::cout << "Total vectors loaded: " << dataset.vectors.size() << "\n";
    f.close();
    return dataset;
}

// Print Random Vectors and images

void printRandomVectors(const Dataset &dataset, int num_vectors) {
    if (dataset.type != "vector") {
        std::cout << "Error: Dataset is not of type 'vector'\n";
        return;
    }

    if (dataset.vectors.empty()) {
        std::cout << "Error: No vectors in the dataset\n";
        return;
    }

    int vectors_to_print = std::min(num_vectors, static_cast<int>(dataset.vectors.size()));
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, dataset.vectors.size() - 1);

    std::cout << "\n=== Random Vectors from Dataset ===\n";
    std::cout << "Total vectors: " << dataset.vectors.size() << "\n";
    std::cout << "Showing " << vectors_to_print << " random vector(s):\n\n";

    for (int i = 0; i < vectors_to_print; ++i) {
        int random_idx = dis(gen);
        const Vector &vec = dataset.vectors[random_idx];
        
        std::cout << "Vector " << random_idx << ":\n";
        std::cout << "  Dimension: " << vec.dimension << "\n";
        std::cout << "  Coordinates: ";
        
        int coords_to_show = std::min(static_cast<int>(vec.coordinates.size()), 10);
        std::cout << std::fixed << std::setprecision(2);
        
        for (int j = 0; j < coords_to_show; ++j) {
            std::cout << vec.coordinates[j];
            if (j < coords_to_show - 1)
                std::cout << ", ";
        }
        
        if (vec.coordinates.size() > 10)
            std::cout << ", ... (" << vec.coordinates.size() << " total coordinates)";
        
        std::cout << "\n\n";
    }
    
    std::cout << "====================================\n";
}

void printRandomImages(const Dataset &dataset, int num_images) {
    if (dataset.type != "image") {
        std::cout << "Error: Dataset is not of type 'image'\n";
        return;
    }

    if (dataset.images.empty()) {
        std::cout << "Error: No images in the dataset\n";
        return;
    }

    int images_to_print = std::min(num_images, static_cast<int>(dataset.images.size()));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, dataset.images.size() - 1);

    std::cout << "\n=== Random Images from Dataset ===\n";
    std::cout << "Total images: " << dataset.images.size() << "\n";
    std::cout << "Showing " << images_to_print << " random image(s):\n\n";

    // ASCII gradient map for visualization
    const std::string shades = " .:-=+*#%@";

    for (int i = 0; i < images_to_print; ++i) {
        int random_idx = dis(gen);
        const Image &img = dataset.images[random_idx];

        std::cout << "Image " << random_idx
                  << " (" << img.rows << "x" << img.cols << ")\n";

        // If image is too big, only show a cropped region
        int max_rows = std::min(img.rows, 28);
        int max_cols = std::min(img.cols, 28);

        for (int r = 0; r < max_rows; ++r) {
            for (int c = 0; c < max_cols; ++c) {
                unsigned char pixel = img.pixels[r * img.cols + c];
                // Map pixel 0–255 to shade index
                int shade_index = (pixel * (shades.size() - 1)) / 255;
                std::cout << shades[shade_index];
            }
            std::cout << "\n";
        }

        if (img.rows > 28 || img.cols > 28)
            std::cout << "... (cropped preview)\n";

        std::cout << "\n";
    }

 
    std::cout << "====================================\n";
}