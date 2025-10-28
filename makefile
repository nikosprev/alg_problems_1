# Compiler
CC = g++

# Directories
SRC_DIR = src
INC_DIR = include
BUILD_DIR = build


# Files
SRC_FILES = $(shell find $(SRC_DIR) -name "*.cpp")            # find all cpp files in sub directories
OBJ_FILES = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRC_FILES))
TARGET = ex

# Rules
all: $(TARGET)

$(TARGET): $(OBJ_FILES)

	$(CC) $(LDFLAGS) $^ -o $@
	
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) ex

run_lsh:
	./ex -lsh -q ./data/sift/sift_query.fvecs -d ./data/sift/sift_base.fvecs -o out.txt -N 5 -k 4 -L 5 -w 100 -seed 1 

