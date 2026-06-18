import os
import sys
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor
import queue

# Configurations
python_bin = sys.executable
cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
gpu_id = "2"
gpu_queue = queue.Queue()
mem_fraction = "0.20"

scenarios_config = {
    "simple_spread": {
        "env": "mpe",
        "source": "omar",
        "project": "robustness_spread"
    },
    "simple_tag": {
        "env": "mpe",
        "source": "omar",
        "project": "robustness_tag"
    },
    "simple_world": {
        "env": "mpe",
        "source": "omar",
        "project": "robustness_world"
    },
    "2halfcheetah": {
        "env": "gymnasium_mamujoco",
        "source": "og_marl",
        "project": "robustness_halfcheetah"
    },
    "2ant": {
        "env": "gymnasium_mamujoco",
        "source": "og_marl",
        "project": "robustness_ant"
    },
    "6halfcheetah": {
        "env": "gymnasium_mamujoco",
        "source": "og_marl",
        "project": "robustness_6halfcheetah"
    },
    "4ant": {
        "env": "gymnasium_mamujoco",
        "source": "og_marl",
        "project": "robustness_4ant"
    }
}

datasets = ["Random", "Medium", "Expert"]
seeds = [0, 1, 2, 42, 77, 99]
configs = [
    {"use_cva": False, "group_prefix": "Baseline"},
    {"use_cva": True, "group_prefix": "CVA"}
]

def run_task(task):
    scenario = task["scenario"]
    dataset = task["dataset"]
    use_cva = task["use_cva"]
    group_name = task["group_name"]
    seed = task["seed"]
    
    cfg = scenarios_config[scenario]
    
    cmd = [
        python_bin, "continuous_main.py",
        "--env", cfg["env"],
        "--source", cfg["source"],
        "--scenario", scenario,
        "--dataset", dataset,
        "--data_dir", "./datasets",
        "--offline_steps", "300000",
        "--save_interval", "50000",
        "--eval_interval", "50000",
        "--log_interval", "5000",
        "--use_cva", str(use_cva),
        "--seed", str(seed),
        "--project_name", cfg["project"],
        "--run_group", group_name
    ]
    if use_cva:
        cmd += ["--num_heads", "4"]
        
    log_dir = os.path.join(cwd, "exp", "logs", scenario)
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"{group_name}_seed{seed}.log")
    
    # Acquire a GPU from the queue
    gpu = gpu_queue.get()
    
    print(f"[{scenario.upper()}] Starting: {group_name} | Seed {seed} on GPU {gpu} -> Log: {log_file_path}")
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if mem_fraction:
        env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = mem_fraction
    env["XLA_FLAGS"] = "--xla_gpu_enable_command_buffer="
    
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
        
    print(f"[{scenario.upper()}] Finished: {group_name} | Seed {seed} on GPU {gpu} (Exit Code: {process.returncode})")
    return process.returncode

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", choices=list(scenarios_config.keys()), default=["simple_tag"],
                        help="Scenarios to run. e.g. simple_tag 2halfcheetah 2ant")
    parser.add_argument("--max_parallel", type=int, default=4, help="Max parallel jobs on GPU")
    parser.add_argument("--gpu", type=str, default="2", help="CUDA GPU ID to use (can be comma-separated list, e.g. 1,3,5,6,7)")
    parser.add_argument("--mem_fraction", type=str, default="0.20", help="XLA memory fraction constraint")
    args = parser.parse_args()
    
    global gpu_id, mem_fraction
    gpu_id = args.gpu
    mem_fraction = args.mem_fraction
    
    # Initialize the gpu queue
    gpu_ids = [g.strip() for g in args.gpu.split(",") if g.strip()]
    for g in gpu_ids:
        gpu_queue.put(g)
    
    tasks = []
    for scenario in args.scenarios:
        # MuJoCo environments only have Replay quality
        if scenario in ["2halfcheetah", "2ant", "6halfcheetah", "4ant"]:
            scenario_datasets = ["Replay"]
        else:
            scenario_datasets = ["Random", "Medium", "Expert"]
            
        for dataset in scenario_datasets:
            for config in configs:
                group_name = f"{config['group_prefix']}_{dataset}"
                for seed in seeds:
                    tasks.append({
                        "scenario": scenario,
                        "dataset": dataset,
                        "use_cva": config["use_cva"],
                        "group_name": group_name,
                        "seed": seed
                    })
                    
    print(f"Total tasks scheduled: {len(tasks)} across scenarios {args.scenarios}")
    print(f"Running on GPU {gpu_id} with max parallel runs = {args.max_parallel}...")
    
    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        results = list(executor.map(run_task, tasks))
        
    print("All tasks completed!")
    print(f"Exit codes: {results}")

if __name__ == "__main__":
    main()
