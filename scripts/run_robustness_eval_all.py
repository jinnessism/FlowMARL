# scripts/run_robustness_eval_all.py
import os
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor
import queue

python_bin = sys.executable
cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
gpu_ids = ["0", "1", "2", "3", "4", "5", "6", "7"]
gpu_queue = queue.Queue()

# We evaluate Opponent Robustness on MPE simple_tag Expert
# to verify the generalization advantage of decoupling team representations.
seeds = [0, 1, 2, 42, 77, 99]
configs = [
    {"use_cva": "False", "use_masked_attn": "False", "prefix": "Baseline"},
    {"use_cva": "True", "use_masked_attn": "False", "prefix": "CVA"},
    {"use_cva": "True", "use_masked_attn": "True", "prefix": "MaskedCVA"}
]

def run_task(task):
    config = task["config"]
    seed = task["seed"]
    
    group_name = f"OOD_{config['prefix']}_Expert"
    
    cmd = [
        python_bin, "continuous_main.py",
        "--env", "mpe",
        "--source", "omar",
        "--scenario", "simple_tag",
        "--dataset", "Expert",
        "--data_dir", "./datasets",
        "--offline_steps", "300000",
        "--save_interval", "500000", # do not save checkpoint to save space
        "--eval_interval", "50000",
        "--log_interval", "5000",
        "--use_cva", config["use_cva"],
        "--use_masked_attn", config["use_masked_attn"],
        "--num_heads", "4",
        "--seed", str(seed),
        "--project_name", "robustness_ood",
        "--run_group", group_name
    ]
    
    log_dir = os.path.join(cwd, "exp", "logs_ood")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"{group_name}_seed{seed}.log")
    
    gpu = gpu_queue.get()
    print(f"[OOD EVAL] Starting: {group_name} | Seed {seed} on GPU {gpu} -> Log: {log_file_path}")
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    
    try:
        with open(log_file_path, "w") as log_file:
            process = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env
            )
            process.wait()
    finally:
        gpu_queue.put(gpu)
        
    print(f"[OOD EVAL] Finished: {group_name} | Seed {seed} on GPU {gpu} (Exit Code: {process.returncode})")
    return process.returncode

def main():
    for g in gpu_ids:
        gpu_queue.put(g)
        
    tasks = []
    for config in configs:
        for seed in seeds:
            tasks.append({
                "config": config,
                "seed": seed
            })
            
    print(f"Total OOD evaluation tasks scheduled: {len(tasks)}")
    print(f"Running on GPUs {gpu_ids} with max parallel runs = {len(gpu_ids)}...")
    
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        results = list(executor.map(run_task, tasks))
        
    print("All OOD evaluation tasks completed!")
    print(f"Exit codes: {results}")

if __name__ == "__main__":
    main()
