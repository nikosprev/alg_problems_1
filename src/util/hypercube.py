import subprocess
import itertools
import os

# Paths
executable = "./../ex"
query_file =  "./../data/mnist/t10k-images.idx3-ubyte"
dataset_file ="./../data/mnist/train-images.idx3-ubyte"
output_dir = "./hypercube_experiments-mnist"

# Make sure output directory exists
os.makedirs(output_dir, exist_ok=True)

# --- Default experiment parameters ---
default_params = {
    "N": 1,
    "M": 10,
    "probes": 2,
    "w": 4.0,
    "kproj": 14 ,
    "seed": 1
}

# Run default experiment
default_output_file = os.path.join(output_dir, "hc_default.txt")
default_cmd = [
    executable,
    "-lsh",
    "-q", query_file,
    "-d", dataset_file,
    "-o", default_output_file,
    "-N", str(default_params["N"]),
    "-M", str(default_params["M"]),
    "-kproj" ,str(default_params["kproj"]), 
    "-probes", str(default_params["probes"]),
    "-w", str(default_params["w"]),
    "-seed", str(default_params["seed"])
]
print("Running default experiment...")
subprocess.run(default_cmd ,stdout=subprocess.DEVNULL)
print(f"Default experiment finished. Output: {default_output_file}\n")

# --- Parameter ran
N_values = [ 10]           # top-N neighbors
M_values = [1000 , 10000 , 100000]           # number of hash functions per table
probes_values = [5,10]           # number of hash tables
w_values = [1000 ] # window size
k_proj_values = [5,14,20]
seed_values = [1]        # seeds for reproducibility

# Create all combinations
param_combinations = list(itertools.product(N_values, M_values,k_proj_values , probes_values, w_values, seed_values))

# Run all experiments
print("Running all parameter experiments...")
for idx, (N, M, kproj,probes , w, seed) in enumerate(param_combinations, 1):
    output_file = os.path.join(output_dir, f"hypercube_exp_{idx}_N{N}_M{M}__kproj{kproj}_Probes{probes}_w{w}_seed{seed}.txt")
    cmd = [
        executable,
        "-hypercube",
        "-q", query_file,
        "-d", dataset_file,
        "-o", output_file,
        "-kproj" , str(kproj), 
        "-N", str(N),
        "-probes", str(probes),
        "-M", str(M),
        "-w", str(w),
        "-seed", str(seed)
    ]
    print(f"[{idx}/{len(param_combinations)}] Running kproj={kproj} N={N}, M={M}, probes={probes}, w={w}, seed={seed}")
    subprocess.run(cmd, stdout=subprocess.DEVNULL,)

print("All experiments finished.")
