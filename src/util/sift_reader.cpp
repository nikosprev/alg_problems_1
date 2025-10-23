#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <cstdint>
#include <string>

using namespace std;


/*   Format:
[offset] [type] [value] [description]
0x0000 32 bit integer 128 dimension
0x0004 32 bit float ?? coordinate 001 of 1st vector
0x0008 32 bit float ?? coordinate 002 of 1st vector
........
0x0200 32 bit float ?? coordinate 128 of 1st vector
0x0204 32 bit integer 128 dimension
0x0208 32 bit float ?? coordinate 001 of 2nd vector
........ */
                                     // read 32bit unsigned integer (little endian)
static uint32_t read_u32_le(ifstream &ifs) {
    uint8_t b[4];
    ifs.read(reinterpret_cast<char*>(b), 4);
    if (!ifs) return 0;
    return (uint32_t(b[3])<<24) | (uint32_t(b[2])<<16) | (uint32_t(b[1])<<8) | uint32_t(b[0]);
}


int prepare_sift(string inputFile, string outputFile){
    ifstream in(inputFile, ios::binary);
    if (!in.is_open()) {
    cerr << "Error: cannot open input file.\n";
    return 1;
    }

    ofstream out(outputFile);
    if (!out.is_open()) {
        cerr << "Error: cannot create output file.\n";
        return 1;
    }

    size_t offset = 0;
    size_t vectorIndex = 1;

    while (true) {
        uint32_t dimension;

    // Read dimension 
        in.read(reinterpret_cast<char*>(&dimension), sizeof(dimension));
        if (in.eof()) break; // EOF
        if (in.gcount() < sizeof(dimension)) break;

        out << "0x" << hex << setw(4) << setfill('0') << uppercase << offset
        << " 32 bit integer " << dec << dimension
        << " dimension (vector " << vectorIndex << ")\n";

    offset += 4;

    // Read coordinates
    vector<float> coords(dimension);
    in.read(reinterpret_cast<char*>(coords.data()), dimension * sizeof(float));
    if (in.gcount() < static_cast<std::streamsize>(dimension * sizeof(float))) break;

    for (uint32_t i = 0; i < dimension; ++i) {
        out << "0x" << hex << setw(4) << setfill('0') << uppercase << offset + i * 4
            << " 32 bit float " << dec << fixed << setprecision(6) << coords[i]
            << " coordinate " << setw(3) << setfill('0') << i + 1
            << " of " << vectorIndex << "st vector\n";
    }

    offset += dimension * 4;
    vectorIndex++;
    }

    cout << " Created " << outputFile << " successfully.\n";

    in.close();
    out.close();
    return 0;
}