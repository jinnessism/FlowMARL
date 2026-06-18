import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Define directories
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
flowmarl_root = os.path.dirname(project_root)

base_dir = os.path.join(project_root, "exp", "robustness_tag")
groups = {
    "Random Baseline": "Baseline_Random",
    "Random CVA": "CVA_Random",
    "Medium Baseline": "Baseline_Medium",
    "Medium CVA": "CVA_Medium",
    "Expert Baseline": "Baseline_Expert",
    "Expert CVA": "CVA_Expert"
}

# Find all available seeds for each group
def load_group_data(group_folder):
    group_path = os.path.join(base_dir, group_folder)
    if not os.path.exists(group_path):
        return None, None
        
    subdirs = os.listdir(group_path)
    seed_data = []
    
    for sd in subdirs:
        csv_file = os.path.join(group_path, sd, "eval.csv")
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                df['step'] = df['step'].astype(int)
                # Sort by step to ensure correct order
                df = df.sort_values('step')
                seed_data.append(df)
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
                
    if not seed_data:
        return None, None
        
    # Align steps across seeds
    steps = sorted(list(set(seed_data[0]['step'])))
    
    all_returns = []
    for df in seed_data:
        returns = []
        for step in steps:
            matches = df[df['step'] == step]['evaluation/mean_episode_return'].values
            if len(matches) > 0:
                returns.append(matches[0])
            else:
                last_matches = df[df['step'] <= step]['evaluation/mean_episode_return'].values
                if len(last_matches) > 0:
                    returns.append(last_matches[-1])
                else:
                    returns.append(0.0)
        all_returns.append(returns)
        
    all_returns = np.array(all_returns)
    mean = np.mean(all_returns, axis=0)
    std = np.std(all_returns, axis=0)
    
    return np.array(steps), mean, std

# Load data for all conditions
conditions = ["Random", "Medium", "Expert"]
data_dict = {}
for cond in conditions:
    data_dict[cond] = {
        "Baseline": load_group_data(groups[f"{cond} Baseline"]),
        "CVA": load_group_data(groups[f"{cond} CVA"])
    }

# Matplotlib settings
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

kaist_blue = "#1F4899"
baseline_color = "#7F7F7F"

for i, cond in enumerate(conditions):
    ax = axes[i]
    
    # Plot Baseline
    steps_b, mean_b, std_b = data_dict[cond]["Baseline"]
    if steps_b is not None:
        ax.plot(steps_b / 1000, mean_b, label="MAC-Flow (Baseline)", color=baseline_color, linewidth=2, linestyle='--')
        ax.fill_between(steps_b / 1000, mean_b - std_b, mean_b + std_b, color=baseline_color, alpha=0.15)
        
    # Plot CVA
    steps_c, mean_c, std_c = data_dict[cond]["CVA"]
    if steps_c is not None:
        ax.plot(steps_c / 1000, mean_c, label="CVA (Ours)", color=kaist_blue, linewidth=2.5)
        ax.fill_between(steps_c / 1000, mean_c - std_c, mean_c + std_c, color=kaist_blue, alpha=0.15)
        
    ax.set_title(f"{cond} Quality Dataset", fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel("Offline Training Steps (k)", fontsize=11)
    if i == 0:
        ax.set_ylabel("Mean Episode Return", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.grid(True, which='both', linestyle=':', alpha=0.6)
    
    ax.legend(loc="lower right", frameon=True, fontsize=10, facecolor='white', edgecolor='none')

plt.tight_layout()
paper_dir = os.path.join(flowmarl_root, "Paper")
os.makedirs(paper_dir, exist_ok=True)
plot_path = os.path.join(paper_dir, "results_plot_tag.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Plot saved successfully to {plot_path}")
