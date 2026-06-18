import os
import argparse
import pandas as pd
import numpy as np

scenarios_config = {
    "simple_spread": "test_robustness",
    "simple_tag": "robustness_tag",
    "simple_world": "robustness_world",
    "2halfcheetah": "robustness_halfcheetah",
    "2ant": "robustness_ant",
    "6halfcheetah": "robustness_6halfcheetah",
    "4ant": "robustness_4ant"
}

seeds = ["sd000", "sd001", "sd002", "sd042", "sd077", "sd099"]
steps = [1, 50000, 100000, 150000, 200000, 250000, 300000]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=list(scenarios_config.keys()), required=True,
                        help="Scenario to calculate stats for")
    args = parser.parse_args()
    
    scenario = args.scenario
    project = scenarios_config[scenario]
    
    # Check if this is MuJoCo or MPE
    if scenario in ["2halfcheetah", "2ant", "6halfcheetah", "4ant"]:
        groups = {
            "Replay Baseline": "Baseline_Replay",
            "Replay CVA": "CVA_Replay"
        }
    else:
        groups = {
            "Random Baseline": "Baseline_Random",
            "Random CVA": "CVA_Random",
            "Medium Baseline": "Baseline_Medium",
            "Medium CVA": "CVA_Medium",
            "Expert Baseline": "Baseline_Expert",
            "Expert CVA": "CVA_Expert"
        }
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    base_dir = os.path.join(project_root, "exp", project)
    
    data = {g: {} for g in groups}
    peaks = {g: [] for g in groups}
    completed_seeds_count = {g: 0 for g in groups}
    
    for group_name, folder in groups.items():
        # Pre-initialize data dict to prevent KeyError
        for step in steps:
            data[group_name][step] = (None, None, 0)
            
        group_path = os.path.join(base_dir, folder)
        if not os.path.exists(group_path):
            continue
        subdirs = os.listdir(group_path)
        
        seed_files = {}
        for s in seeds:
            for sd in subdirs:
                if sd.startswith(s):
                    csv_file = os.path.join(group_path, sd, "eval.csv")
                    if os.path.exists(csv_file):
                        seed_files[s] = csv_file
                        break
                        
        seed_dfs = {}
        for s, f in seed_files.items():
            try:
                df = pd.read_csv(f)
                df['step'] = df['step'].astype(int)
                seed_dfs[s] = df
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
                
    print(f"=== Active Seeds Summary for {scenario.upper()} ===")
    for g in groups:
        print(f"{g}: {completed_seeds_count[g]} seeds loaded")
        
    print("\n=== LaTeX rows for the table ===")
    header_line = "Step"
    for g in groups:
        header_line += f" & {g}"
    print(header_line + " \\\\")
    
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

if __name__ == "__main__":
    main()
