import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import jax
import jax.numpy as jnp

# Set environment
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from envs.environments import get_environment
from agents.macflow import MACFlowAgent
import ml_collections

# Dynamic path resolution
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
flowmarl_root = os.path.dirname(project_root)
paper_dir = os.path.join(flowmarl_root, "Paper")

def get_latest_run_dir(base_dir, group_folder, seed_prefix):
    group_path = os.path.join(base_dir, group_folder)
    if not os.path.exists(group_path):
        return None
    subdirs = [d for d in os.listdir(group_path) if d.startswith(seed_prefix)]
    if not subdirs:
        return None
    # Sort by directory name (contains timestamp) to get the latest
    subdirs.sort()
    return os.path.join(group_path, subdirs[-1])

# =================================================================================================
# 1. Plot Gating Coefficient (Gamma) Curve
# =================================================================================================
def plot_gamma_curves(scenario):
    base_dir = os.path.join(project_root, "exp", f"robustness_{scenario.split('_')[-1]}")
    if not os.path.exists(base_dir):
        # Fallback for simple_world / 2halfcheetah default directory names
        base_dir = os.path.join(project_root, "exp", f"robustness_{scenario}")
        
    conditions = ["Random", "Medium", "Expert"]
    if scenario in ["2halfcheetah", "2ant"]:
        conditions = ["Replay"]
        
    seeds = ["sd000", "sd001", "sd002", "sd042", "sd077", "sd099"]
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, len(conditions), figsize=(5 * len(conditions), 4.5))
    if len(conditions) == 1:
        axes = [axes]
        
    kaist_blue = "#1F4899"
    kaist_red = "#E61B23"
    
    for idx, cond in enumerate(conditions):
        ax = axes[idx]
        group_folder = f"CVA_{cond}"
        
        all_gamma_bc = []
        all_gamma_onestep = []
        steps = None
        
        for seed in seeds:
            run_dir = get_latest_run_dir(base_dir, group_folder, seed)
            if not run_dir:
                continue
            csv_path = os.path.join(run_dir, "train.csv")
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    if "train/actor/gamma_bc" in df.columns and "train/actor/gamma_onestep" in df.columns:
                        df = df.dropna(subset=["train/actor/gamma_bc", "train/actor/gamma_onestep"])
                        if len(df) > 0:
                            all_gamma_bc.append(df["train/actor/gamma_bc"].values)
                            all_gamma_onestep.append(df["train/actor/gamma_onestep"].values)
                            steps = df["step"].values
                except Exception as e:
                    print(f"Error loading {csv_path}: {e}")
                    
        if steps is not None and len(all_gamma_bc) > 0:
            all_gamma_bc = np.array(all_gamma_bc)
            all_gamma_onestep = np.array(all_gamma_onestep)
            
            mean_bc = np.mean(all_gamma_bc, axis=0)
            std_bc = np.std(all_gamma_bc, axis=0)
            mean_onestep = np.mean(all_gamma_onestep, axis=0)
            std_onestep = np.std(all_gamma_onestep, axis=0)
            
            ax.plot(steps / 1000, mean_bc, label="$\gamma_{bc}$ (BC Flow)", color=kaist_red, linewidth=2)
            ax.fill_between(steps / 1000, mean_bc - std_bc, mean_bc + std_bc, color=kaist_red, alpha=0.15)
            
            ax.plot(steps / 1000, mean_onestep, label="$\gamma_{one}$ (One-step Flow)", color=kaist_blue, linewidth=2)
            ax.fill_between(steps / 1000, mean_onestep - std_onestep, mean_onestep + std_onestep, color=kaist_blue, alpha=0.15)
            
        ax.set_title(f"{cond} Quality Dataset", fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel("Training Steps (k)", fontsize=11)
        if idx == 0:
            ax.set_ylabel("Gating Coefficient ($\gamma$)", fontsize=11)
        ax.tick_params(labelsize=10)
        ax.legend(loc="upper left", frameon=True, fontsize=10, facecolor='white', edgecolor='none')
        ax.grid(True, which='both', linestyle=':', alpha=0.6)
        
    plt.tight_layout()
    os.makedirs(paper_dir, exist_ok=True)
    plot_path = os.path.join(paper_dir, f"gating_curve_{scenario}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Gating curve plot saved to {plot_path}")

# =================================================================================================
# 2. Run Rollout & Record coordinates / attention weights
# =================================================================================================
def run_rollout_recording(scenario, group_folder, seed, ckpt_name="ckpt_300000.npz"):
    # Load Environment
    source = "omar" if "simple" in scenario else "og_marl"
    env_type = "mpe" if "simple" in scenario else "gymnasium_mamujoco"
    
    env = get_environment(source, env_type, scenario, seed=0)
    agent_names = list(env.agents)
    
    # Load Agent configuration
    base_dir = os.path.join(project_root, "exp", f"robustness_{scenario.split('_')[-1]}")
    if not os.path.exists(base_dir):
        base_dir = os.path.join(project_root, "exp", f"robustness_{scenario}")
        
    run_dir = get_latest_run_dir(base_dir, group_folder, seed)
    if not run_dir:
        print(f"Run dir not found for {group_folder} | {seed}")
        return None
        
    ckpt_path = os.path.join(run_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}")
        return None
        
    print(f"Loading checkpoint from {ckpt_path}...")
    
    # Setup agent config
    cfg = ml_collections.ConfigDict(
        dict(
            agent_name='macflow',
            ob_dims=None,
            action_dim=None,
            lr=3e-4,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            actor_layer_norm=False,
            discount=0.995,
            tau=0.005,
            q_agg='mean',
            alpha=1.0,
            flow_steps=10,
            normalize_q_loss=True,
            encoder=None,
            use_cva="CVA" in group_folder,
            num_heads=4,
            use_aw_flow=False,
            aw_temp=1.0,
            aw_warmup_steps=100000,
            use_qg_flow=False,
        )
    )
    
    # Simple dummy batch to initialize agent shapes
    obs_dim = env.obs_dim
    act_dim = env.num_actions
    dummy_obs = jnp.zeros((1, 1, len(agent_names), obs_dim))
    dummy_act = jnp.zeros((1, 1, len(agent_names), act_dim))
    
    agent = MACFlowAgent.create(
        seed=0,
        ex_observations=dummy_obs,
        ex_actions=dummy_act,
        agent_names=agent_names,
        config=cfg,
    )
    
    # Load parameters from checkpoint
    agent.network.load(ckpt_path)
    
    # Rollout loop
    observations, infos = env.reset()
    done = False
    
    traj_coords = [] # list of dicts: agent_name -> position
    attn_weights_history = [] # list of (num_heads, N, N) matrices
    
    # Track positions of landmarks if MPE
    landmark_coords = None
    if hasattr(env.environment, "world") and hasattr(env.environment.world, "landmarks"):
        landmark_coords = [lm.state.p_pos.copy() for lm in env.environment.world.landmarks]
        
    step = 0
    while not done and step < 100:
        # Record current coordinates
        step_coords = {}
        if hasattr(env.environment, "world") and hasattr(env.environment.world, "agents"):
            for i, ag_obj in enumerate(env.environment.world.agents):
                step_coords[agent_names[i]] = ag_obj.state.p_pos.copy()
        traj_coords.append(step_coords)
        
        # Sample actions with attention weights tracking
        # Convert observations to tensor
        from util import concat_agent_id_to_obs
        obs_with_ids = [concat_agent_id_to_obs(observations[agent], i, len(agent_names)) for i, agent in enumerate(agent_names)]
        obs_tensor = jnp.stack(obs_with_ids, axis=0)
        obs_tensor = jnp.where(jnp.isinf(obs_tensor), 0.0, obs_tensor)
        
        noises = jax.random.normal(jax.random.PRNGKey(step), (len(agent_names), cfg['action_dim']))
        
        if cfg.use_cva:
            # Capture intermediates
            out, state = agent.network.apply_fn(
                {'params': agent.network.params},
                obs_tensor, noises,
                name='actor_onestep_flow',
                mutable=['intermediates']
            )
            actions_tensor = jnp.clip(out, -1, 1)
            
            # Extract attention weights
            intermediates = state.get('intermediates', {})
            if 'modules_actor_onestep_flow' in intermediates:
                flow_inter = intermediates['modules_actor_onestep_flow']
                if 'cva' in flow_inter and 'MultiHeadDotProductAttention_0' in flow_inter['cva']:
                    attn_dict = flow_inter['cva']['MultiHeadDotProductAttention_0']
                    if 'attention_weights' in attn_dict:
                        attn_w = attn_dict['attention_weights']
                        if isinstance(attn_w, tuple):
                            attn_w = attn_w[0]
                        if hasattr(attn_w, 'ndim') and attn_w.ndim == 4:
                            attn_w = attn_w[0]
                        attn_weights_history.append(np.array(attn_w))
        else:
            actions_tensor = agent.sample_actions(observations, jax.random.PRNGKey(step))
            
        if isinstance(actions_tensor, dict):
            actions = actions_tensor
        else:
            actions = {agent: np.array(actions_tensor[i]) for i, agent in enumerate(agent_names)}
            
        observations, rewards, terminal, truncation, infos = env.step(actions)
        done = all(terminal.values()) or all(truncation.values())
        step += 1
        
    return {
        "traj_coords": traj_coords,
        "landmark_coords": landmark_coords,
        "attn_weights": attn_weights_history,
        "agent_names": agent_names
    }

# =================================================================================================
# 3. Plot Agent Trajectory Map & Attention Weight Heatmap
# =================================================================================================
def plot_qualitative_results(scenario, cond, seed):
    # 1. Run rollouts
    print(f"Running Baseline rollout for {scenario}...")
    baseline_data = run_rollout_recording(scenario, f"Baseline_{cond}", seed)
    
    print(f"Running CVA rollout for {scenario}...")
    cva_data = run_rollout_recording(scenario, f"CVA_{cond}", seed)
    
    if not baseline_data or not cva_data:
        print("Failed to load rollout data. Make sure checkpoints exist.")
        return
        
    # --- 1) Plot Trajectories ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    
    agent_colors = {
        "agent_0": "#1F4899", # Blue
        "agent_1": "#33A02C", # Green
        "agent_2": "#FF7F00", # Orange
        "agent_3": "#E61B23"  # Red (Prey in simple_tag)
    }
    
    # Plot landmarks
    for ax, data_dict, title in [(ax1, baseline_data, "MAC-Flow (Baseline)"), (ax2, cva_data, "CVA (Ours)")]:
        if data_dict["landmark_coords"] is not None:
            for i, lm in enumerate(data_dict["landmark_coords"]):
                ax.scatter(lm[0], lm[1], c="black", marker="x", s=100, linewidth=2, label="Landmark" if i == 0 else "")
                
        # Plot agent path lines
        agent_names = data_dict["agent_names"]
        traj = data_dict["traj_coords"]
        
        for agent in agent_names:
            xs = [step_pos[agent][0] for step_pos in traj if agent in step_pos]
            ys = [step_pos[agent][1] for step_pos in traj if agent in step_pos]
            if not xs or not ys:
                continue
            
            # Start position
            ax.scatter(xs[0], ys[0], color=agent_colors.get(agent, "purple"), marker="o", s=80, edgecolors='black', zorder=5)
            # Path line
            ax.plot(xs, ys, color=agent_colors.get(agent, "purple"), linewidth=2, alpha=0.8, label=agent.replace("_", " ").title())
            # End position
            ax.scatter(xs[-1], ys[-1], color=agent_colors.get(agent, "purple"), marker="^", s=100, edgecolors='black', zorder=5)
            
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("X coordinate", fontsize=11)
        ax.set_ylabel("Y coordinate", fontsize=11)
        ax.legend(loc="upper right", frameon=True, facecolor='white', edgecolor='none')
        ax.grid(True, linestyle=':', alpha=0.6)
        
    plt.tight_layout()
    os.makedirs(paper_dir, exist_ok=True)
    plot_path = os.path.join(paper_dir, f"trajectory_{scenario}_{cond}_{seed}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Trajectory plot saved to {plot_path}")
    
    # --- 2) Plot Attention Heatmap (Close quarters moment) ---
    if cva_data["attn_weights"]:
        # Find step where agents are closest
        # Compute pairwise distance between agents over time
        traj = cva_data["traj_coords"]
        agent_names = cva_data["agent_names"]
        
        min_dist = float('inf')
        closest_step = 0
        
        for step_idx, step_pos in enumerate(traj):
            # Calculate average pairwise distance between predators
            predators = [ag for ag in agent_names if ag != "agent_3"]
            if len(predators) >= 2:
                dists = []
                if not all(p in step_pos for p in predators):
                    continue
                for i in range(len(predators)):
                    for j in range(i+1, len(predators)):
                        p1 = step_pos[predators[i]]
                        p2 = step_pos[predators[j]]
                        dists.append(np.linalg.norm(p1 - p2))
                avg_dist = np.mean(dists)
                if avg_dist < min_dist:
                    min_dist = avg_dist
                    closest_step = step_idx
                    
        # Extract attention weights at this closest step
        # attn_weights_history shape list of (num_heads, N, N)
        if closest_step < len(cva_data["attn_weights"]):
            attn_matrix = cva_data["attn_weights"][closest_step] # (num_heads, N, N)
            mean_attn = np.mean(attn_matrix, axis=0) # Mean across heads -> (N, N)
            
            plt.figure(figsize=(6, 5))
            labels = [n.replace("_", " ").title() for n in agent_names]
            sns.heatmap(
                mean_attn,
                annot=True,
                fmt=".2f",
                cmap="Blues",
                xticklabels=labels,
                yticklabels=labels,
                cbar_kws={'label': 'Attention Weight ($\\alpha_{ij}$)'}
            )
            plt.title(f"CVA Cross-Attention Weights\n(Close-Quarters Step {closest_step})", fontsize=13, fontweight='bold', pad=10)
            plt.tight_layout()
            
            os.makedirs(paper_dir, exist_ok=True)
            heatmap_path = os.path.join(paper_dir, f"attention_heatmap_{scenario}_{cond}_{seed}.png")
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            print(f"Attention heatmap saved to {heatmap_path}")

# =================================================================================================
# Main block
# =================================================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="simple_tag", help="Scenario name")
    parser.add_argument("--plot_gamma", action="store_true", help="Plot gating curves")
    parser.add_argument("--plot_qualitative", action="store_true", help="Plot trajectories and heatmaps")
    parser.add_argument("--cond", type=str, default="Expert", help="Dataset quality (Expert, Medium, Random)")
    parser.add_argument("--seed", type=str, default="sd002", help="Seed directory prefix (e.g. sd002)")
    args = parser.parse_args()
    
    if args.plot_gamma:
        plot_gamma_curves(args.scenario)
        
    if args.plot_qualitative:
        plot_qualitative_results(args.scenario, args.cond, args.seed)
