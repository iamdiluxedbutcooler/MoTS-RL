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
├── trainers/              # Training loops with robustness mechanisms
├── baselines/             # Baseline implementations (Vanilla, ICM, RND, RL², etc.)
├── envs/                  # Diagnostic environments
└── utils/                 # Replay buffer, logging

configs/                   # Hydra configuration files
├── stage1/                # Diagnostic experiments (4 configs)
├── stage2/                # Single-agent tasks (4 configs)
├── stage3/                # Multi-agent tasks (4 configs)
├── robustness/            # Robustness tests (5 configs)
├── ablations/             # Ablation studies (9 configs)
├── sensitivity/           # Alpha parameter sensitivity (5 configs)
└── baselines/             # Baseline comparisons (6 configs)

scripts/
├── train.py               # Main training script (supports baselines)
├── evaluate.py            # Evaluation and plotting
├── eval_boundedness.py    # Boundedness verification
├── eval_cluster.py        # Affective state clustering
├── eval_counterfactual.py # Counterfactual analysis
└── compare_baselines.py   # Statistical baseline comparison
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

### Sensitivity Analysis (5 configs)
- `alpha_080.yaml`: α = 0.80
- `alpha_085.yaml`: α = 0.85
- `alpha_090.yaml`: α = 0.90 (default)
- `alpha_095.yaml`: α = 0.95
- `alpha_099.yaml`: α = 0.99

### Baselines (6 configs)
- `vanilla.yaml`: Standard PPO/SAC (B1)
- `vanilla_plus.yaml`: Parameter-matched baseline (B2)
- `icm.yaml`: Intrinsic Curiosity Module (B3)
- `rnd.yaml`: Random Network Distillation (B4)
- `dim_affect.yaml`: 2D valence-arousal only (B5)
- `rl2.yaml`: Meta-RL baseline (B6)

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
- `results/figures/stage1_core.png`: Diagnostic plots (boundedness, recovery, timescale, coupling)
- `results/figures/stage2_learning.png`: Single-agent learning curves
- `results/figures/stage3_social.png`: Multi-agent cooperation analysis
- `results/figures/robustness_shift.png`: Robustness under distribution shift
- `results/figures/ablations_heatmap.png`: Ablation study heatmap
- `results/figures/umap_clusters.png`: Affective state clusters (UMAP)
- `results/figures/baseline_comparison.png`: Statistical comparison with all 6 baselines

## Key Features

### Robustness Mechanisms
The trainers now support:
- **Reward Swap** (R1): Mid-training reward sign flip
- **Observation Noise** (R3): Gaussian noise injection
- **Action Dropout** (R4): Random action masking
- **Partner Swap** (R2): Multi-agent partner policy change (planned)
- **Zero-shot Transfer** (R5): Cross-task evaluation (planned)

### Ablation Controls
- `use_persona`: Enable/disable persona modulator
- `use_shadow`: Enable/disable shadow modulator
- `fixed_ego_weights`: Fixed expert arbitration weights
- `random_affect`: Random affect updates (control condition)

### Diagnostic Environments
Custom environments for theoretical validation:
- `diagnostic_boundedness`: Tests boundedness guarantees
- `diagnostic_recovery`: Tests recovery dynamics
- `diagnostic_timescale`: Tests temporal characteristics
- `diagnostic_coupled`: Tests cross-dimensional coupling

### Statistical Analysis
The `compare_baselines.py` script provides:
- Mann-Whitney U tests for all baseline comparisons
- Effect sizes (rank-biserial correlation)
- Significance annotations on plots
- Detailed statistical reports

## License

MIT License