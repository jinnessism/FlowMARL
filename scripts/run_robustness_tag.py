import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
import sys

# Configurations
python_bin = sys.executable
cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
gpu_id = "2"
project_name = "robustness_tag"

datasets = ["Random", "Medium", "Expert"]
seeds = [0, 1, 2, 42, 77, 99]
configs = [
    {"use_cva": False, "group_prefix": "Baseline"},
    {"use_cva": True, "group_prefix": "CVA"}
]

# Generate all tasks
tasks = []
for dataset in datasets:
    for config in configs:
        group_name = f"{config['group_prefix']}_{dataset}"
        for seed in seeds:
            tasks.append({
                "dataset": dataset,
                "use_cva": config["use_cva"],
                "group_name": group_name,
                "seed": seed
            })

print(f"Total tasks to run: {len(tasks)}")

def run_task(task):
    dataset = task["dataset"]
    use_cva = task["use_cva"]
    group_name = task["group_name"]
    seed = task["seed"]
    
    cmd = [
        python_bin, "continuous_main.py",
        "--env", "mpe",
        "--source", "omar",
        "--scenario", "simple_tag",
        "--dataset", dataset,
        "--data_dir", "./datasets",
        "--offline_steps", "500000",
        "--eval_interval", "50000",
        "--log_interval", "5000",
        "--use_cva", str(use_cva),
        "--seed", str(seed),
        "--project_name", project_name,
        "--run_group", group_name
    ]
    if use_cva:
        cmd += ["--num_heads", "4"]
        
    log_dir = os.path.join(cwd, "exp", "logs_tag")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"{group_name}_seed{seed}.log")
    
    print(f"Starting Task: {group_name} | Seed {seed} -> Logging to {log_file_path}")
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    
    with open(log_file_path, "w") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env
        )
        process.wait()
        
    print(f"Finished Task: {group_name} | Seed {seed} with return code {process.returncode}")
    return process.returncode

# Run tasks with a max workers of 4 (4 concurrent runs on GPU 2)
max_parallel_runs = 4
print(f"Running experiments with max parallel runs = {max_parallel_runs} on GPU {gpu_id}...")
with ThreadPoolExecutor(max_workers=max_parallel_runs) as executor:
    results = list(executor.map(run_task, tasks))

print("All tasks completed!")
print(f"Results summary (return codes): {results}")
