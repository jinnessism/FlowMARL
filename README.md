<div align="center">

<div id="user-content-toc" style="margin-bottom: 50px">
  <ul align="center" style="list-style: none;">
    <summary>
      <h1>FlowMARL</h1>
      <h3>Multi-Agent Coordination via Flow Matching</h3>
      <br>
      <h2><a href="https://arxiv.org/abs/2511.05005">Paper</a></h2>
    </summary>
  </ul>
</div>
</div>

## Overview
This codebase provides the official implementation of **FlowMARL**, a framework for offline Multi-Agent Reinforcement Learning (MARL) based on Multi-agent Coordination via Flow Matching. 

## Installation
```bash
# Environment (OG-MARL: https://github.com/instadeepai/og-marl)
conda create -n recipe python=3.8
conda activate recipe
python -m pip install --upgrade pip
pip install -r requirements/datasets.txt
```

Install the SMAC, MPE, MA-MuJoCo environment you plan to use:

```bash
# SMAC v1
bash install_environments/smacv1.sh
pip install -r install_environments/requirements/smacv1.txt

# SMAC v2
bash install_environments/smacv2.sh
pip install -r install_environments/requirements/smacv2.txt

# MPE
pip install -r install_environments/requirements/pettingzoo.txt

# MA-MuJoCo
bash install_environments/mujoco200.sh
pip install -r install_environments/requirements/mamujoco200.txt
```

If you need baseline TensorFlow dependencies:

```bash
pip install -r requirements/baselines.txt
```

## Quick start

```bash
# TD3BC
python smac_main.py --env smac_v1 --source og_marl --scenario 2s3z --dataset Good --seed 0 --agent.alpha 3.0 
```
