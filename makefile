# Compiler
CXX = g++

# Directories
SRC_DIR = src
INC_DIR = include
BUILD_DIR = build

# Files
SRC_FILES = $(shell find $(SRC_DIR) -name "*.cpp")  # find all cpp files in subdirectories
OBJ_FILES = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRC_FILES))
TARGET = ex

# Flags
CXXFLAGS = -O2   -I$(INC_DIR)
LDFLAGS = 

# Rules
all: $(TARGET)

$(TARGET): $(OBJ_FILES)
	$(CXX) $(LDFLAGS) $^ -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

# Run commands
run_lsh_sift:
	./ex -lsh -q ./data/sift/sift_query.fvecs -d ./data/sift/sift_base.fvecs -o lsh.txt -N 1 -k 6 -L 5 -w 4 -seed 12345 


run_lsh_sift_default: 
	./ex -lsh -q ./data/sift/sift_query.fvecs -d ./data/sift/sift_base.fvecs -o lsh.txt -N 1 -k 5 -L 5 -w 4 -seed 12345 

run_knn_sift: 
	./ex -q ./data/sift/sift_query.fvecs -d ./data/sift/sift_base.fvecs -o knn.txt -N 10 -seed 1 

run_lsh_mnist: 
	./ex -lsh -q ./data/mnist/train-images.idx3-ubyte -d ./data/mnist/train-images.idx3-ubyte -o out.txt -N 5 -k 4 -L 5 -w 5 -seed 1

run_lsh_mnist_default: 
	./ex -lsh -q ./


run_hypercube_sift: 
	./ex -hypercube -q ./data/sift/sift_query.fvecs -d ./data/sift/sift_base.fvecs -o hypercube.txt -N 5 -k 4 -M 100000 -w 3 -kproj 14 -probes 2 -seed 1
