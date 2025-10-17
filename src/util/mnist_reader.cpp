#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdint>
#include <vector>

using namespace std;

// convert 4 big-endian bytes into an integer
uint32_t readBigEndianInt(ifstream &f) {
    unsigned char bytes[4];
    f.read(reinterpret_cast<char*>(bytes), 4);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}
/*  Format :

0x0000 32 bit integer 0x00000803(2051) magic number
0x0004 32 bit integer 60000 number of images
0x0008 32 bit integer 28 number of rows
0x000C 32 bit integer 28 number of columns
0x0010 unsigned byte 0 pixel no 1
0x0011 unsigned byte 0 pixel no 2
...
0x031F unsigned byte 0 pixel no 784

*/
int prepare_mnist(string inputFile, string outputFile){


    ifstream in(inputFile, ios::binary);
    if (!in.is_open()) {
        cerr << "Error: Cannot open input file.\n";
        return 1;
    }

    ofstream out(outputFile);
    if (!out.is_open()) {
        cerr << "Error: Cannot create output file.\n";
        return 1;
    }

    // Read header
    uint32_t magic = readBigEndianInt(in);
    uint32_t numImages = readBigEndianInt(in);
    uint32_t numRows = readBigEndianInt(in);
    uint32_t numCols = readBigEndianInt(in);

    // Write header info
    out << "0x0000 32 bit integer 0x" << hex << setw(8) << setfill('0') << uppercase << magic
        << "(" << dec << magic << ") magic number\n";
    out << "0x0004 32 bit integer " << numImages << " number of images\n";
    out << "0x0008 32 bit integer " << numRows << " number of rows\n";
    out << "0x000C 32 bit integer " << numCols << " number of columns\n";

    // Compute first image size
    size_t imageSize = numRows * numCols;
    vector<unsigned char> pixels(imageSize);

    in.read(reinterpret_cast<char*>(pixels.data()), imageSize);

    // Write pixel data
    size_t offset = 0x0010;
    for (size_t i = 0; i < imageSize; ++i) {
        out << "0x" << hex << setw(4) << setfill('0') << uppercase << offset + i
            << " unsigned byte " << dec << setw(3) << static_cast<int>(pixels[i])
            << " pixel no " << i + 1 << "\n";
    }

    cout << "Created " << outputFile << " successfully.\n";

    in.close();
    out.close();
    return 0;
}