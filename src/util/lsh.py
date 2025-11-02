import subprocess
import itertools
import os

# Paths
executable = "./../ex"
query_file =  "./../data/mnist/t10k-images.idx3-ubyte"
dataset_file ="./../data/mnist/train-images.idx3-ubyte"
output_dir = "./lsh_experiments-mnist"

# Make sure output directory exists
os.makedirs(output_dir, exist_ok=True)

# --- Default experiment parameters ---
default_params = {
    "N": 1,
    "k": 5,
    "L": 5,
    "w": 4,
    "seed": 1
}

# Run default experiment
default_output_file = os.path.join(output_dir, "lsh_default.txt")
default_cmd = [
    executable,
    "-lsh",
    "-q", query_file,
    "-d", dataset_file,
    "-o", default_output_file,
    "-N", str(default_params["N"]),
    "-k", str(default_params["k"]),
    "-L", str(default_params["L"]),
    "-w", str(default_params["w"]),
    "-seed", str(default_params["seed"])
]
print("Running default experiment...")
subprocess.run(default_cmd ,stdout=subprocess.DEVNULL)
print(f"Default experiment finished. Output: {default_output_file}\n")

# --- Parameter ran
N_values = [1, 5, 10]           # top-N neighbors
k_values = [ 5, 8]           # number of hash functions per table
L_values = [ 5]           # number of hash tables
w_values = [4 ,10,100, 200,300,500,1000] # window size
seed_values = [1]        # seeds for reproducibility

# Create all combinations
param_combinations = list(itertools.product(N_values, k_values, L_values, w_values, seed_values))

# Run all experiments
print("Running all parameter experiments...")
for idx, (N, k, L, w, seed) in enumerate(param_combinations, 1):
    output_file = os.path.join(output_dir, f"lsh_exp_{idx}_N{N}_k{k}_L{L}_w{w}_seed{seed}.txt")
    cmd = [
        executable,
        "-lsh",
        "-q", query_file,
        "-d", dataset_file,
        "-o", output_file,
        "-N", str(N),
        "-k", str(k),
        "-L", str(L),
        "-w", str(w),
        "-seed", str(seed)
    ]
    print(f"[{idx}/{len(param_combinations)}] Running N={N}, k={k}, L={L}, w={w}, seed={seed}")
    subprocess.run(cmd, stdout=subprocess.DEVNULL,)

print("All experiments finished.")
