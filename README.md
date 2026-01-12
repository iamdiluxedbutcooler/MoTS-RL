# Map-of-the-Soul Reinforcement Learning (MoTS-RL)

## Overview

MoTS-RL extends traditional RL with a four-dimensional affective state vector that influences learning through:
- **Core Affective Dynamics**: Continuous affective state evolution
- **Persona & Shadow Modulators**: Reward scaling based on affective state
- **Ego Modulator**: Dynamic expert policy weighting
- **Self-Regulation**: Entropy-based balance across expert policies

## Installation

```bash
make install
```

Or manually:

```bash
pip install -r requirements.txt
```

## Requirements

- Python ≥3.10
- PyTorch ≥2.1
- gymnasium==0.29.1
- pettingzoo==1.24.1
- stable-baselines3==2.3.0

## Project Structure

```
mots_rl/
├── core.py                # CoreAffectiveDynamics
├── modulators/            # Persona, Shadow, Ego, Self-regulation
├── policy/                # PPO and SAC policy wrappers
├── trainers/              # Training loops
└── utils/                 # Replay buffer, logging

configs/                   # Hydra configuration files
├── stage1/                # Diagnostic experiments
├── stage2/                # Single-agent tasks
├── stage3/                # Multi-agent tasks
├── robustness/            # Robustness tests
└── ablations/             # Ablation studies

scripts/
├── train.py               # Main training script
├── evaluate.py            # Evaluation and plotting
├── eval_boundedness.py    # Boundedness verification
├── eval_cluster.py        # Affective state clustering
└── eval_counterfactual.py # Counterfactual analysis
```

## Quick Start

### Train a Single Configuration

```bash
python scripts/train.py +experiment=stage2/mujoco.yaml
```

### Run All Experiments

```bash
make train-all
```

Or:

```bash
bash scripts/run_experiments.sh
```

### Evaluate Results

```bash
python scripts/evaluate.py
```

Or:

```bash
make eval-all
```

### Run Tests

```bash
make test
```

Or:

```bash
python -m pytest tests/
```

## Experiment Stages

### Stage 1: Diagnostic (4 configs)
- `boundedness.yaml`: Verify affective state boundedness
- `recovery.yaml`: Measure recovery dynamics
- `timescale.yaml`: Analyze temporal characteristics
- `coupled.yaml`: Examine cross-dimensional coupling

### Stage 2: Single-Agent (4 configs)
- `minigrid.yaml`: Discrete navigation (MiniGrid)
- `mujoco.yaml`: Continuous control (MuJoCo)
- `memory.yaml`: Memory-dependent tasks
- `mo_hopper.yaml`: Multi-objective locomotion

### Stage 3: Multi-Agent (4 configs)
- `ipd.yaml`: Iterated Prisoner's Dilemma
- `mpe_nav.yaml`: Multi-agent navigation
- `mpe_pred.yaml`: Predator-prey
- `coin_game.yaml`: Coin collection game

### Robustness (5 configs)
- `r1_reward_swap.yaml`: Mid-training reward swap
- `r2_partner.yaml`: Partner policy swap
- `r3_obs_noise.yaml`: Observation noise injection
- `r4_action_dropout.yaml`: Action dropout
- `r5_zero_shot.yaml`: Zero-shot transfer

### Ablations (9 configs)
- `persona_off.yaml`: Disable persona modulator
- `shadow_off.yaml`: Disable shadow modulator
- `ego_fixed.yaml`: Fixed expert weights
- `self_off.yaml`: Disable self-regulation
- `affect_2d.yaml`: 2D affect space
- `affect_random.yaml`: Random affect updates
- `gating_lstm.yaml`: LSTM-based ego
- `alpha_sweep_fast.yaml`: Fast dynamics (α≈0.99)
- `alpha_sweep_slow.yaml`: Slow dynamics (α≈0.5)

## Core Algorithm

The affective state evolves as:

```
A_{t+1} = α ⊙ A_t + β ⊙ g(r_t, δ_t, φ(s_t))
```

where:
- `A_t ∈ ℝ⁴` is the affective state
- `α, β ∈ (0,1)⁴` are learned decay/update coefficients
- `g` is a two-layer tanh MLP
- `φ` is an observation encoder

Policy mixture:
```
π(a|s,A_t) = Σᵢ ωᵢ(A_t,s_t) πᵢ(a|s,A_t)
```

## Generated Figures

After running experiments and evaluation:
- `results/figures/stage1_core.png`: Diagnostic plots
- `results/figures/stage2_learning.png`: Learning curves
- `results/figures/stage3_social.png`: Multi-agent analysis
- `results/figures/robustness_shift.png`: Robustness results
- `results/figures/ablations_heatmap.png`: Ablation study
- `results/figures/umap_clusters.png`: Affective state clusters

## License

MIT License