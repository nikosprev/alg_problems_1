#include <cstdint>
using namespace std;

int prepare_mnist(string inputFile, string outputFile);
int prepare_sift(string inputFile, string outputFile);
uint32_t read_u32_be(ifstream &ifs);
uint32_t read_u32_le(ifstream &ifs);
static float read_float_native(ifstream &ifs);