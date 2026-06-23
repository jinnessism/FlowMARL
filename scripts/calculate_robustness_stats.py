import os
import pandas as pd
import numpy as np
from scipy import stats

project_root = "/home/pjmin831/FlowMARL/mac-flow"
base_dir = os.path.join(project_root, "exp", "robustness_ood")
seeds = ["sd000", "sd001", "sd002", "sd042", "sd077", "sd099"]

groups = {
    "Baseline": "OOD_Baseline_Expert",
    "GVA (Unmasked)": "OOD_CVA_Expert",
    "Masked GVA": "OOD_MaskedCVA_Expert"
}

metrics = [
    "evaluation/predator_return_default",
    "evaluation/predator_return_random",
    "evaluation/predator_return_heuristic"
]

print("=================== OPPONENT ROBUSTNESS RESULTS ===================")

data = {}
for g_name, folder in groups.items():
    group_path = os.path.join(base_dir, folder)
    if not os.path.exists(group_path):
        print(f"[Folder Missing] {folder}")
        continue
        
    subdirs = os.listdir(group_path)
    seed_files = {}
    for s in seeds:
        matching_dirs = [sd for sd in subdirs if sd.startswith(s)]
        matching_dirs_with_time = []
        for sd in matching_dirs:
            csv_file = os.path.join(group_path, sd, "eval.csv")
            if os.path.exists(csv_file):
                try:
                    mtime = os.path.getmtime(csv_file)
                    matching_dirs_with_time.append((mtime, csv_file))
                except Exception:
                    pass
        if matching_dirs_with_time:
            matching_dirs_with_time.sort(key=lambda x: x[0], reverse=True)
            seed_files[s] = matching_dirs_with_time[0][1]
            
    # Read each seed
    seed_peaks = {m: [] for m in metrics}
    for s, f in seed_files.items():
        try:
            df = pd.read_csv(f)
            for m in metrics:
                if m in df.columns:
                    peak_val = df[m].max()
                    if not np.isnan(peak_val):
                        seed_peaks[m].append(peak_val)
        except Exception as e:
            pass
            
    data[g_name] = seed_peaks

# Print results
for m in metrics:
    print(f"\nMetric: {m}")
    for g_name in groups:
        if g_name in data:
            vals = data[g_name][m]
            if len(vals) > 0:
                mean = np.mean(vals)
                std = np.std(vals) if len(vals) > 1 else 0.0
                print(f"  {g_name:15}: {mean:.2f} \\mypm{{{std:.2f}}} ({len(vals)} seeds)")
            else:
                print(f"  {g_name:15}: [No data]")
                
    # Welch's t-test comparing Masked GVA vs GVA (Unmasked) and Baseline
    if "Masked GVA" in data and "GVA (Unmasked)" in data:
        m_gva = data["Masked GVA"][m]
        gva = data["GVA (Unmasked)"][m]
        if len(m_gva) > 1 and len(gva) > 1:
            t_stat, p_val = stats.ttest_ind(m_gva, gva, equal_var=False)
            print(f"  Welch's t-test (Masked vs Unmasked): t = {t_stat:.4f}, p = {p_val:.4f}")
            
    if "Masked GVA" in data and "Baseline" in data:
        m_gva = data["Masked GVA"][m]
        base = data["Baseline"][m]
        if len(m_gva) > 1 and len(base) > 1:
            t_stat, p_val = stats.ttest_ind(m_gva, base, equal_var=False)
            print(f"  Welch's t-test (Masked vs Baseline): t = {t_stat:.4f}, p = {p_val:.4f}")
