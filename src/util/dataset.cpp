#include "../../include/dataset.hpp"



// Helper function to trim whitespace
std::string trim(const std::string &s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    size_t end = s.find_last_not_of(" \t\r\n");
    return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
}

// Detect dataset type based on keywords
std::string detectDatasetType(const std::vector<std::string> &lines) {
    for (const auto &line : lines) {
        if (line.find("pixel") != std::string::npos)
            return "image";
        if (line.find("coordinate") != std::string::npos)
            return "vector";
    }
    return "unknown";
}

Dataset parseInputFile(const std::string &filename) {
    Dataset dataset;
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Error opening input.dat");

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(file, line)) {
        if (!trim(line).empty())
            lines.push_back(line);
    }
    file.close();

    dataset.type = detectDatasetType(lines);

    if (dataset.type == "image") {
        int rows = 0, cols = 0, n_images = 0;
        std::vector<unsigned char> pixels;

        for (const auto &ln : lines) {
            std::istringstream iss(ln);
            std::string offset, bits, bitWord, type, value_str;
            if (!(iss >> offset >> bits >> bitWord >> type >> value_str))
                continue;

            std::string desc;
            std::getline(iss, desc);
            desc = trim(desc);

            if (desc.find("number of images") != std::string::npos)
                n_images = std::stoi(value_str);
            else if (desc.find("number of rows") != std::string::npos)
                rows = std::stoi(value_str);
            else if (desc.find("number of columns") != std::string::npos)
                cols = std::stoi(value_str);
            else if (desc.find("pixel") != std::string::npos) {
                if (value_str != "??") {
                    try {
                        unsigned int val = 0;
                        if (value_str.rfind("0x", 0) == 0)
                            val = std::stoul(value_str, nullptr, 16);
                        else
                            val = static_cast<unsigned int>(std::stoi(value_str));
                        pixels.push_back(static_cast<unsigned char>(val));
                    } catch (...) {
                        std::cerr << "Warning: invalid pixel value '" << value_str << "'\n";
                    }
                }
            }
        }

        Image img;
        img.rows = rows;
        img.cols = cols;
        img.pixels = pixels;
        dataset.images.push_back(img);
    }

    else if (dataset.type == "vector") {
        Vector current_vector;

        for (const auto &ln : lines) {
            std::istringstream iss(ln);
            std::string offset, bits, bitWord, type, value_str;
            if (!(iss >> offset >> bits >> bitWord >> type >> value_str))
                continue;

            std::string desc;
            std::getline(iss, desc);
            desc = trim(desc);

            if (desc.find("dimension") != std::string::npos) {
                if (!current_vector.coordinates.empty()) {
                    dataset.vectors.push_back(current_vector);
                    current_vector.coordinates.clear();
                }

                if (std::regex_match(value_str, std::regex("^-?[0-9]+$"))) {
                    current_vector.dimension = std::stoi(value_str);
                } else {
                    std::cerr << "Warning: invalid integer value '"
                              << value_str << "' for dimension, skipping.\n";
                    current_vector.dimension = 0;
                }
            }

            else if (desc.find("coordinate") != std::string::npos) {
                if (value_str != "??") {
                    try {
                        current_vector.coordinates.push_back(std::stof(value_str));
                    } catch (...) {
                        std::cerr << "Warning: invalid coordinate value '"
                                  << value_str << "'\n";
                    }
                }
            }
        }

        if (!current_vector.coordinates.empty())
            dataset.vectors.push_back(current_vector);
    }

    else {
        throw std::runtime_error("Unknown dataset type");
    }

    return dataset;
}

