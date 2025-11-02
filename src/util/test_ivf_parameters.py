#!/usr/bin/env python3
"""
Script to test IVFFlat and IVFPQ algorithms with different parameters
Tests on MNIST and SIFT datasets

Parameters tested:
- IVFFlat: kclusters, nprobe, N, R
- IVFPQ: kclusters, nprobe, nbits, N, R
"""

import subprocess
import os
from datetime import datetime
from pathlib import Path

# Configuration
RESULTS_FILE = "ivf_parameter_test_results.txt"
SUMMARY_FILE = "ivf_parameter_summary.txt"

# Datasets
DATASETS = {
    "MNIST": {
        "train": "train-images-small.idx3-ubyte",
        "query": "t10k-images-small.idx3-ubyte",
        "type": "mnist"
    },
    "SIFT": {
        "train": "sift_base-small.fvecs",
        "query": "query-small.fvecs",
        "type": "vector"
    }
}

# Common parameters
SEED = 42

# Parameter ranges to test
IVFFLAT_PARAMS = {
    "kclusters": [25, 50, 100, 200],
    "nprobe": [1, 5, 10],
    "N": [1, 5, 10, 20],
    "R": [1000.0, 5000.0, 10000.0]
}

IVFPQ_PARAMS = {
    "kclusters": [50, 100],
    "nprobe": [1, 5],
    "nbits": [4, 8],
    "N": [1, 5, 10, 20],
    "R": [1000.0, 5000.0, 10000.0]
}


def run_command(cmd, output_file):
    """Run a command and capture output"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout per test
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out after 10 minutes"
    except Exception as e:
        return False, "", str(e)


def extract_experiment_result(method_name):
    """Extract the last result from experiments file"""
    exp_file = f"{method_name}experiments.txt"
    if os.path.exists(exp_file):
        with open(exp_file, 'r') as f:
            lines = f.readlines()
            if lines:
                return lines[-1].strip()
    return "No result found"


def test_ivfflat(dataset_name, dataset_config, results_file):
    """Test IVFFlat with different parameters"""
    print(f"\nTesting IVFFlat on {dataset_name}...")
    results_file.write(f"\n{'='*80}\n")
    results_file.write(f"IVFFlat on {dataset_name}\n")
    results_file.write(f"{'='*80}\n\n")
    
    results = []
    total_tests = (len(IVFFLAT_PARAMS["kclusters"]) * len(IVFFLAT_PARAMS["nprobe"]) * 
                   len(IVFFLAT_PARAMS["N"]) * len(IVFFLAT_PARAMS["R"]))
    test_count = 0
    
    for kclusters in IVFFLAT_PARAMS["kclusters"]:
        for nprobe in IVFFLAT_PARAMS["nprobe"]:
            for N in IVFFLAT_PARAMS["N"]:
                for R in IVFFLAT_PARAMS["R"]:
                    test_count += 1
                    output_file = f"out_ivfflat_{dataset_name.lower()}_k{kclusters}_np{nprobe}_N{N}_R{R}.txt"
                    
                    cmd = (
                        f"./ex -d {dataset_config['train']} "
                        f"-q {dataset_config['query']} "
                        f"-kclusters {kclusters} "
                        f"-nprobe {nprobe} "
                        f"-o {output_file} "
                        f"-N {N} "
                        f"-R {R} "
                        f"-type {dataset_config['type']} "
                        f"-range false "
                        f"-ivfflat "
                        f"-seed {SEED}"
                    )
                    
                    print(f"  [{test_count}/{total_tests}] Testing: kclusters={kclusters}, nprobe={nprobe}, N={N}, R={R}")
                    results_file.write(f"Test: kclusters={kclusters}, nprobe={nprobe}, N={N}, R={R}\n")
                    results_file.write(f"Command: {cmd}\n")
                    
                    success, stdout, stderr = run_command(cmd, output_file)
                    
                    if not success:
                        results_file.write(f"ERROR: {stderr}\n")
                    else:
                        result_line = extract_experiment_result("IVFFlat")
                        results_file.write(f"Result: {result_line}\n")
                        results.append({
                            "dataset": dataset_name,
                            "method": "IVFFlat",
                            "kclusters": kclusters,
                            "nprobe": nprobe,
                            "N": N,
                            "R": R,
                            "result": result_line
                        })
                    
                    results_file.write("\n")
                    results_file.flush()
    
    return results


def test_ivfpq(dataset_name, dataset_config, results_file):
    """Test IVFPQ with different parameters"""
    print(f"\nTesting IVFPQ on {dataset_name}...")
    results_file.write(f"\n{'='*80}\n")
    results_file.write(f"IVFPQ on {dataset_name}\n")
    results_file.write(f"{'='*80}\n\n")
    
    results = []
    total_tests = (len(IVFPQ_PARAMS["kclusters"]) * len(IVFPQ_PARAMS["nprobe"]) * 
                   len(IVFPQ_PARAMS["nbits"]) * len(IVFPQ_PARAMS["N"]) * len(IVFPQ_PARAMS["R"]))
    test_count = 0
    
    for kclusters in IVFPQ_PARAMS["kclusters"]:
        for nprobe in IVFPQ_PARAMS["nprobe"]:
            for nbits in IVFPQ_PARAMS["nbits"]:
                for N in IVFPQ_PARAMS["N"]:
                    for R in IVFPQ_PARAMS["R"]:
                        test_count += 1
                        output_file = f"out_ivfpq_{dataset_name.lower()}_k{kclusters}_np{nprobe}_nb{nbits}_N{N}_R{R}.txt"
                        
                        # Calculate M based on dataset dimension (must divide evenly)
                        # For now, use a fixed M that works for both datasets
                        M = 8  # Can be adjusted based on dimension
                        
                        cmd = (
                            f"./ex -d {dataset_config['train']} "
                            f"-q {dataset_config['query']} "
                            f"-kclusters {kclusters} "
                            f"-nprobe {nprobe} "
                            f"-M {M} "
                            f"-nbits {nbits} "
                            f"-o {output_file} "
                            f"-N {N} "
                            f"-R {R} "
                            f"-type {dataset_config['type']} "
                            f"-range false "
                            f"-ivfpq "
                            f"-seed {SEED}"
                        )
                        
                        print(f"  [{test_count}/{total_tests}] Testing: kclusters={kclusters}, nprobe={nprobe}, nbits={nbits}, N={N}, R={R}")
                        results_file.write(f"Test: kclusters={kclusters}, nprobe={nprobe}, nbits={nbits}, N={N}, R={R}, M={M}\n")
                        results_file.write(f"Command: {cmd}\n")
                        
                        success, stdout, stderr = run_command(cmd, output_file)
                        
                        if not success:
                            results_file.write(f"ERROR: {stderr}\n")
                        else:
                            result_line = extract_experiment_result("IVFPQ")
                            results_file.write(f"Result: {result_line}\n")
                            results.append({
                                "dataset": dataset_name,
                                "method": "IVFPQ",
                                "kclusters": kclusters,
                                "nprobe": nprobe,
                                "nbits": nbits,
                                "N": N,
                                "R": R,
                                "M": M,
                                "result": result_line
                            })
                        
                        results_file.write("\n")
                        results_file.flush()
    
    return results


def create_summary(all_results, summary_file):
    """Create a summary table of all results"""
    summary_file.write("IVF Parameter Testing Summary\n")
    summary_file.write("="*100 + "\n\n")
    summary_file.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # Group by dataset and method
    for dataset in ["MNIST", "SIFT"]:
        summary_file.write(f"\n{dataset} Dataset Results\n")
        summary_file.write("-"*100 + "\n\n")
        
        for method in ["IVFFlat", "IVFPQ"]:
            method_results = [r for r in all_results if r["dataset"] == dataset and r["method"] == method]
            if not method_results:
                continue
            
            summary_file.write(f"{method}:\n")
            
            if method == "IVFFlat":
                summary_file.write(f"{'kclusters':<12} {'nprobe':<8} {'N':<6} {'R':<12} {'Result':<60}\n")
                summary_file.write("-"*100 + "\n")
                for r in method_results:
                    summary_file.write(f"{r['kclusters']:<12} {r['nprobe']:<8} {r['N']:<6} {r['R']:<12.1f} {r['result']:<60}\n")
            else:  # IVFPQ
                summary_file.write(f"{'kclusters':<12} {'nprobe':<8} {'nbits':<8} {'N':<6} {'R':<12} {'Result':<50}\n")
                summary_file.write("-"*100 + "\n")
                for r in method_results:
                    summary_file.write(f"{r['kclusters']:<12} {r['nprobe']:<8} {r['nbits']:<8} {r['N']:<6} {r['R']:<12.1f} {r['result']:<50}\n")
            
            summary_file.write("\n")


def main():
    """Main testing function"""
    # Check if executable exists
    if not os.path.exists("./ex"):
        print("Error: ./ex executable not found. Please compile first with 'make'")
        return
    
    # Open results file
    with open(RESULTS_FILE, 'w') as results_file:
        results_file.write("IVF Algorithm Parameter Testing Results\n")
        results_file.write("="*80 + "\n")
        results_file.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        results_file.write(f"\nParameters tested:\n")
        
        ivfflat_total = (len(IVFFLAT_PARAMS['kclusters']) * len(IVFFLAT_PARAMS['nprobe']) * 
                        len(IVFFLAT_PARAMS['N']) * len(IVFFLAT_PARAMS['R']))
        ivfpq_total = (len(IVFPQ_PARAMS['kclusters']) * len(IVFPQ_PARAMS['nprobe']) * 
                      len(IVFPQ_PARAMS['nbits']) * len(IVFPQ_PARAMS['N']) * len(IVFPQ_PARAMS['R']))
        
        results_file.write(f"  IVFFlat: {len(IVFFLAT_PARAMS['kclusters'])} kclusters × {len(IVFFLAT_PARAMS['nprobe'])} nprobe × "
                          f"{len(IVFFLAT_PARAMS['N'])} N × {len(IVFFLAT_PARAMS['R'])} R = {ivfflat_total} combinations\n")
        results_file.write(f"  IVFPQ: {len(IVFPQ_PARAMS['kclusters'])} kclusters × {len(IVFPQ_PARAMS['nprobe'])} nprobe × "
                          f"{len(IVFPQ_PARAMS['nbits'])} nbits × {len(IVFPQ_PARAMS['N'])} N × {len(IVFPQ_PARAMS['R'])} R = {ivfpq_total} combinations\n")
        results_file.write(f"  Datasets: {len(DATASETS)}\n")
        
        total_tests = (ivfflat_total + ivfpq_total) * len(DATASETS)
        results_file.write(f"  Total tests: {total_tests}\n\n")
        
        all_results = []
        
        # Test IVFFlat
        for dataset_name, dataset_config in DATASETS.items():
            results = test_ivfflat(dataset_name, dataset_config, results_file)
            all_results.extend(results)
        
        # Test IVFPQ
        for dataset_name, dataset_config in DATASETS.items():
            results = test_ivfpq(dataset_name, dataset_config, results_file)
            all_results.extend(results)
        
        results_file.write("\n" + "="*80 + "\n")
        results_file.write(f"Testing completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        results_file.write(f"Total tests run: {len(all_results)}\n")
    
    # Create summary
    with open(SUMMARY_FILE, 'w') as summary_file:
        create_summary(all_results, summary_file)
    
    print(f"\n{'='*80}")
    print("Testing completed!")
    print(f"Detailed results: {RESULTS_FILE}")
    print(f"Summary results: {SUMMARY_FILE}")
    print(f"Total tests completed: {len(all_results)}")


if __name__ == "__main__":
    main()
