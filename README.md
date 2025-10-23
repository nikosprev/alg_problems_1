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


Step 1: Read data and make input file
    MNIST reader:

         Read 32-bit Big Endian integers for header (magic number, number of images, rows, cols).

        Read 784-byte image vectors.


    SIFT reader:

        Read Little Endian 32-bit integers for vector dimension (128).

        Read 128 floats per vector, store as std::vector<std::vector<float>>.

        Query file reader: Similar structure, for search vectors.

        Validate inputs: check magic numbers, file sizes, dimensions.


Step 2: Algorithms:

IVFFlat employs the following steps:

    Calculate the distance between the query vector and each centroid in the index.
    
    Select the centroid with the smallest distance as the closest centroid to the query 

    Retrieve the vectors associated with the region corresponding to the closest centroid from the inverted index.

    Compute the distances between the query vector and each of the vectors in the retrieved set.
    
    Select the K vectors with the smallest distances as the approximate nearest neighbors to the query.