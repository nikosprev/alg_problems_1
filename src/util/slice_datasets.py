#!/usr/bin/env python3
"""
Script to slice/downsample MNIST and SIFT dataset files into smaller subsets

Formats:
- MNIST (.idx3-ubyte): Big-endian format
  - Magic number (2051)
  - Number of images (uint32 big-endian)
  - Rows (uint32 big-endian)
  - Cols (uint32 big-endian)
  - Image data (rows*cols bytes per image)

- SIFT (.fvecs): Little-endian format
  - For each vector:
    - Dimension (uint32 little-endian)
    - Dimension floats (little-endian)
"""

import struct
import os
import sys
from typing import List, Tuple


def read_uint32_be(f):
    """Read big-endian uint32"""
    bytes_data = f.read(4)
    if len(bytes_data) < 4:
        return None
    return struct.unpack('>I', bytes_data)[0]


def write_uint32_be(f, value):
    """Write big-endian uint32"""
    f.write(struct.pack('>I', value))


def read_uint32_le(f):
    """Read little-endian uint32"""
    bytes_data = f.read(4)
    if len(bytes_data) < 4:
        return None
    return struct.unpack('<I', bytes_data)[0]


def write_uint32_le(f, value):
    """Write little-endian uint32"""
    f.write(struct.pack('<I', value))


def read_float_le(f):
    """Read little-endian float"""
    bytes_data = f.read(4)
    if len(bytes_data) < 4:
        return None
    return struct.unpack('<f', bytes_data)[0]


def write_float_le(f, value):
    """Write little-endian float"""
    f.write(struct.pack('<f', value))


def slice_mnist(input_file: str, output_file: str, num_images: int, start_idx: int = 0):
    """
    Slice MNIST IDX file to extract num_images starting from start_idx
    
    Args:
        input_file: Input MNIST .idx3-ubyte file
        output_file: Output file path
        num_images: Number of images to extract
        start_idx: Starting index (default: 0)
    """
    print(f"Slicing MNIST: {input_file} -> {output_file}")
    print(f"  Extracting {num_images} images starting from index {start_idx}")
    
    with open(input_file, 'rb') as inf:
        # Read header
        magic = read_uint32_be(inf)
        if magic != 2051:
            raise ValueError(f"Invalid MNIST magic number: {magic} (expected 2051)")
        
        total_images = read_uint32_be(inf)
        rows = read_uint32_be(inf)
        cols = read_uint32_be(inf)
        image_size = rows * cols
        
        print(f"  Original file: {total_images} images, {rows}x{cols}")
        
        if start_idx + num_images > total_images:
            num_images = total_images - start_idx
            print(f"  Warning: Adjusted to {num_images} images (end of file)")
        
        # Skip to start_idx
        if start_idx > 0:
            inf.seek(16 + start_idx * image_size)  # 16 = 4 bytes magic + 4*3 header fields
        
        # Write output header
        with open(output_file, 'wb') as outf:
            write_uint32_be(outf, magic)  # Magic number
            write_uint32_be(outf, num_images)  # Number of images
            write_uint32_be(outf, rows)
            write_uint32_be(outf, cols)
            
            # Copy image data
            for i in range(num_images):
                image_data = inf.read(image_size)
                if len(image_data) < image_size:
                    print(f"  Warning: Only read {len(image_data)} bytes for image {i}")
                    break
                outf.write(image_data)
                
                if (i + 1) % 1000 == 0:
                    print(f"  Progress: {i + 1}/{num_images} images")
        
        print(f"  Successfully extracted {num_images} images to {output_file}")


def slice_sift(input_file: str, output_file: str, num_vectors: int, start_idx: int = 0):
    """
    Slice SIFT fvecs file to extract num_vectors starting from start_idx
    
    Args:
        input_file: Input SIFT .fvecs file
        output_file: Output file path
        num_vectors: Number of vectors to extract
        start_idx: Starting index (default: 0)
    """
    print(f"Slicing SIFT: {input_file} -> {output_file}")
    print(f"  Extracting {num_vectors} vectors starting from index {start_idx}")
    
    # First pass: find dimensions and count total vectors
    with open(input_file, 'rb') as inf:
        vectors = []
        dim = None
        vector_count = 0
        
        # Skip to start_idx
        if start_idx > 0:
            for i in range(start_idx):
                dim_val = read_uint32_le(inf)
                if dim_val is None:
                    raise ValueError(f"Cannot skip to index {start_idx}, only {i} vectors in file")
                inf.seek(dim_val * 4, 1)  # Skip dimension floats
                vector_count += 1
        
        # Read vectors we want
        for i in range(num_vectors):
            dim_val = read_uint32_le(inf)
            if dim_val is None:
                print(f"  Warning: Only {i} vectors available")
                break
            
            if dim is None:
                dim = dim_val
            elif dim != dim_val:
                raise ValueError(f"Inconsistent dimension: expected {dim}, got {dim_val}")
            
            # Read vector coordinates
            vector = []
            for j in range(dim_val):
                val = read_float_le(inf)
                if val is None:
                    raise ValueError(f"Incomplete vector at index {i}")
                vector.append(val)
            
            vectors.append((dim_val, vector))
            
            if (i + 1) % 1000 == 0:
                print(f"  Progress: {i + 1}/{num_vectors} vectors")
        
        print(f"  Dimension: {dim}, Extracted: {len(vectors)} vectors")
        
        # Write output file
        with open(output_file, 'wb') as outf:
            for dim_val, vector in vectors:
                write_uint32_le(outf, dim_val)
                for val in vector:
                    write_float_le(outf, val)
        
        print(f"  Successfully extracted {len(vectors)} vectors to {output_file}")


def main():
    """Main function"""
    if len(sys.argv) < 4:
        print("Usage: python3 slice_datasets.py <dataset_type> <input_file> <output_file> <num_items> [start_idx]")
        print("  dataset_type: 'mnist' or 'sift'")
        print("  input_file: Input dataset file")
        print("  output_file: Output dataset file")
        print("  num_items: Number of items to extract")
        print("  start_idx: Starting index (optional, default: 0)")
        print("\nExamples:")
        print("  python3 slice_datasets.py mnist train-images.idx3-ubyte train-images-small.idx3-ubyte 1000")
        print("  python3 slice_datasets.py sift sift_base.fvecs sift_base-small.fvecs 5000")
        sys.exit(1)
    
    dataset_type = sys.argv[1].lower()
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    num_items = int(sys.argv[4])
    start_idx = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    
    if dataset_type == 'mnist':
        slice_mnist(input_file, output_file, num_items, start_idx)
    elif dataset_type == 'sift':
        slice_sift(input_file, output_file, num_items, start_idx)
    else:
        print(f"Error: Unknown dataset type '{dataset_type}'. Use 'mnist' or 'sift'")
        sys.exit(1)
    
    print("Done!")


if __name__ == "__main__":
    main()

