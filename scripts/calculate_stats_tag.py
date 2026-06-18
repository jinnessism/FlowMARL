import os
import pandas as pd
import numpy as np
# Define directories
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
base_dir = os.path.join(project_root, "exp", "robustness_tag")
groups = {
    "Random Baseline": "Baseline_Random",
    "Random CVA": "CVA_Random",
    "Medium Baseline": "Baseline_Medium",
    "Medium CVA": "CVA_Medium",
    "Expert Baseline": "Baseline_Expert",
    "Expert CVA": "CVA_Expert"
}

seeds = ["sd000", "sd001", "sd002", "sd042", "sd077", "sd099"]

# We will collect data for each group, step by step
steps = [1, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000]

data = {g: {} for g in groups}
peaks = {g: [] for g in groups} # list of peak values for each seed
completed_seeds_count = {g: 0 for g in groups}

for group_name, folder in groups.items():
    group_path = os.path.join(base_dir, folder)
    if not os.path.exists(group_path):
        continue
    subdirs = os.listdir(group_path)
    
    # Map seed to CSV file
    seed_files = {}
    for s in seeds:
        for sd in subdirs:
            if sd.startswith(s):
                csv_file = os.path.join(group_path, sd, "eval.csv")
                if os.path.exists(csv_file):
                    seed_files[s] = csv_file
                    break
                    
    # Read each seed's data
    seed_dfs = {}
    for s, f in seed_files.items():
        try:
            df = pd.read_csv(f)
            df['step'] = df['step'].astype(int)
            seed_dfs[s] = df
            # Find peak mean episode return
            peak_val = df['evaluation/mean_episode_return'].max()
            peaks[group_name].append(peak_val)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    completed_seeds_count[group_name] = len(seed_dfs)
    
    for step in steps:
        vals = []
        for s, df in seed_dfs.items():
            matches = df[df['step'] == step]['evaluation/mean_episode_return'].values
            if len(matches) > 0:
                vals.append(matches[0])
        if len(vals) > 0:
            mean = np.mean(vals)
            std = np.std(vals) if len(vals) > 1 else 0.0
            data[group_name][step] = (mean, std, len(vals))
        else:
            data[group_name][step] = (None, None, 0)

# Let's print out a summary of active seeds
print("=== Active Seeds Summary ===")
for g in groups:
    print(f"{g}: {completed_seeds_count[g]} seeds loaded")

# Let's print out the LaTeX rows
print("\n=== LaTeX rows for the table ===")
for step in steps:
    row = f"{step:,}"
    for g in groups:
        mean, std, count = data[g][step]
        if mean is not None:
            row += f" & {mean:.2f} \\mypm{{{std:.2f}}}"
        else:
            row += " & [TBD]"
    print(row + " \\\\")

print("\n=== Peak Return row ===")
row = "Peak Return"
for g in groups:
    vals = peaks[g]
    if len(vals) > 0:
        mean = np.mean(vals)
        std = np.std(vals) if len(vals) > 1 else 0.0
        row += f" & {mean:.2f} \\mypm{{{std:.2f}}}"
    else:
        row += " & [TBD]"
print(row + " \\\\")
