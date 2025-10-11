# alg_problems_1
Software project for algorithmic problems
```
project1/
├── src/
│ ├── algorithms/
│ │ ├── lsh.c
│ │ ├── hypercube.c
│ └── main.c
├── include/
├── data/
│ ├── input.dat (for MNIST/SIFT dataset)
│ └── query.dat (for query set)
├── Makefile
├── .gitignore
└── readme
```


Step 1: Read data
    MNIST reader:

         Read 32-bit Big Endian integers for header (magic number, number of images, rows, cols).

        Read 784-byte image vectors.

        Store images as std::vector<std::vector<uint8_t>> or std::vector<std::vector<double>>.

    SIFT reader:

        Read Little Endian 32-bit integers for vector dimension (128).

        Read 128 floats per vector, store as std::vector<std::vector<float>>.

        Query file reader: Similar structure, for search vectors.

        Validate inputs: check magic numbers, file sizes, dimensions.