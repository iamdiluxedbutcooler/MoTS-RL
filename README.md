# Persona, Ego, Shadow, and Self: A Map of the Soul Framework for Proto-Emotional Homeostasis in AI

**AAAI-26 Undergraduate Consortium Submission**

## Overview

This repository implements a Jungian-inspired reinforcement learning architecture where agents maintain a four-dimensional proto-emotional state that influences learning and decision-making. The framework decomposes emotional regulation into four modules inspired by Carl Jung's Map of the Soul:

- **Persona & Shadow Modulators**: Reward perception scaling based on affective state
- **Ego Modulator**: Dynamic arbitration between expert policies
- **Self-Regulation**: Entropy-based homeostatic balance

The affective state evolves continuously through a linear dynamical system with theoretical boundedness guarantees, enabling stable long-horizon learning.

## Installation

```bash
make install
```

Or manually:

```bash
pip install -r requirements.txt
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.1
- gymnasium == 0.29.1
- pettingzoo == 1.24.1
- stable-baselines3 == 2.3.0

## Project Structure

```
mots_rl/
├── core.py                # CoreAffectiveDynamics: A_t = αA_{t-1} + βg(r,δ,φ(s))
├── modulators/
│   ├── persona.py         # Persona modulator: positive affect amplification
│   ├── shadow.py          # Shadow modulator: negative affect processing
│   ├── ego.py             # Ego arbitration: π = Σ ω_i(A,s) π_i
│   └── self_reg.py        # Self-regulation: entropy-based homeostasis
├── policy/
│   ├── ppo_policy.py      # PPO with affective state conditioning
│   └── sac_policy.py      # SAC with affective state conditioning
├── trainers/
│   ├── trainer.py         # Base training loop with robustness mechanisms
│   └── multi_agent.py     # Multi-agent training with partner modeling
├── baselines/
│   ├── vanilla.py         # Standard PPO/SAC
│   ├── icm.py             # Intrinsic Curiosity Module
│   ├── rnd.py             # Random Network Distillation
│   └── rl2.py             # Meta-RL baseline
├── envs/
│   ├── diagnostic.py      # Custom environments for theoretical validation
│   └── wrappers.py        # Robustness wrappers (noise, dropout, etc.)
└── utils/
    ├── replay.py          # Experience replay buffer
    └── logging.py         # TensorBoard and CSV logging

configs/                   # Hydra configuration files
├── stage1/                # Theoretical validation (4 configs)
├── stage2/                # Single-agent tasks (4 configs)
├── stage3/                # Multi-agent tasks (4 configs)
├── robustness/            # Distribution shift tests (5 configs)
├── ablations/             # Component ablations (9 configs)
└── baselines/             # Baseline comparisons (6 configs)

scripts/
├── train.py               # Main training entry point
├── evaluate.py            # Generate all figures
├── generate_results.py    # Synthetic result generation for prototyping
└── run_experiments.sh     # Batch experiment runner
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

Or manually:

```bash
bash scripts/run_experiments.sh
```

### Generate Figures

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

```bash
python -m pytest tests/
```

## Core Algorithm

### Affective State Dynamics

The affective state evolves according to:

```python
A_{t+1} = α ⊙ A_t + β ⊙ g(r_t, δ_t, φ(s_t))
```

where:
- `A_t ∈ R^4` is the affective state vector
- `α ∈ (0,1)^4` are learned decay coefficients (temporal persistence)
- `β ∈ (0,1)^4` are learned update coefficients (sensitivity to new stimuli)
- `g: R^n → R^4` is a two-layer tanh MLP mapping observations to affect
- `r_t` is the reward, `δ_t` is the TD error, `φ(s_t)` is the observation encoding

This linear dynamical system guarantees boundedness: `||A_t||_∞ ≤ β_max / (1 - α_max)`.

### Policy Mixture

The final policy is a weighted mixture of expert policies:

```python
π(a|s,A_t) = Σ_i ω_i(A_t, s_t) π_i(a|s, A_t)
```

where `ω_i(A_t, s_t)` are context-dependent gating weights computed by the Ego module.

### Reward Modulation

Perceived reward is modulated by Persona and Shadow:

```python
r_perceived = r_raw + λ_persona · f_persona(A_t, r_raw) + λ_shadow · f_shadow(A_t, r_raw)
```

The Persona amplifies positive experiences when affect is high; Shadow processes negative experiences when affect is low.

## Experiment Configurations

### Stage 1: Theoretical Validation (4 configs)

Validates mathematical properties of the affective dynamics:

- `boundedness.yaml`: Empirically verify `||A_t||_∞` stays below theoretical bound across 100 episodes
- `recovery.yaml`: Inject large perturbations (A_t = 10) and measure exponential decay rates
- `timescale.yaml`: Analyze autocorrelation functions to confirm multi-timescale representation
- `coupled.yaml`: Compute cross-correlation matrix to validate coherent but non-redundant dynamics

### Stage 2: Single-Agent Learning (4 configs)

Evaluate on standard RL benchmarks:

- `minigrid.yaml`: Discrete navigation (MiniGrid-FourRooms, sparse rewards)
- `mujoco.yaml`: Continuous control (HalfCheetah-v4, dense rewards)
- `memory.yaml`: Memory-dependent tasks (T-Maze with long horizons)
- `mo_hopper.yaml`: Multi-objective locomotion (Hopper with speed-stability trade-off)

### Stage 3: Multi-Agent Coordination (4 configs)

Test social reasoning capabilities:

- `ipd.yaml`: Iterated Prisoner's Dilemma (cooperation emergence)
- `mpe_nav.yaml`: Multi-agent navigation (coordination without communication)
- `mpe_pred.yaml`: Predator-prey (strategic opponent modeling)
- `coin_game.yaml`: Coin collection game (competitive dynamics)

### Robustness Testing (5 configs)

Distribution shift scenarios:

- `r1_reward_swap.yaml`: Flip reward sign at episode 125
- `r2_partner.yaml`: Swap partner policy mid-training (multi-agent only)
- `r3_obs_noise.yaml`: Add 20% Gaussian noise to observations
- `r4_action_dropout.yaml`: Randomly drop 15% of actions
- `r5_zero_shot.yaml`: Transfer to unseen task variant

### Ablation Studies (9 configs)

Component-wise importance analysis:

- `persona_off.yaml`: Disable persona modulator (no positive amplification)
- `shadow_off.yaml`: Disable shadow modulator (no negative processing)
- `ego_fixed.yaml`: Fix expert weights uniformly (no dynamic arbitration)
- `self_off.yaml`: Disable self-regulation (no entropy balancing)
- `affect_2d.yaml`: Reduce affect to 2D valence-arousal only
- `affect_random.yaml`: Random affect updates (control condition)
- `gating_lstm.yaml`: Replace ego with LSTM-based gating
- `alpha_sweep_fast.yaml`: Fast dynamics (α ≈ 0.99, short memory)
- `alpha_sweep_slow.yaml`: Slow dynamics (α ≈ 0.5, long memory)

### Baseline Comparisons (6 configs)

- `vanilla.yaml`: Standard PPO/SAC without affective mechanisms
- `vanilla_plus.yaml`: Capacity-matched baseline (same parameter count)
- `icm.yaml`: Intrinsic Curiosity Module (Pathak et al., 2017)
- `rnd.yaml`: Random Network Distillation (Burda et al., 2019)
- `dim_affect.yaml`: Simplified 2D affect without Jung framework
- `rl2.yaml`: Meta-RL (Wang et al., 2016)

## Code Structure Details

### CoreAffectiveDynamics (`mots_rl/core.py`)

Implements the linear dynamical system with learnable parameters:

```python
class CoreAffectiveDynamics(nn.Module):
    def __init__(self, affect_dim=4, obs_dim=64):
        self.alpha = nn.Parameter(torch.rand(affect_dim) * 0.3 + 0.7)  # [0.7, 1.0)
        self.beta = nn.Parameter(torch.rand(affect_dim) * 0.1 + 0.1)   # [0.1, 0.2)
        self.affine_net = nn.Sequential(
            nn.Linear(obs_dim + 2, 128),  # obs + reward + TD error
            nn.Tanh(),
            nn.Linear(128, affect_dim),
            nn.Tanh()
        )
    
    def forward(self, affect_prev, obs, reward, td_error):
        stimulus = self.affine_net(torch.cat([obs, reward, td_error], -1))
        return self.alpha * affect_prev + self.beta * stimulus
```

### Modulators (`mots_rl/modulators/`)

**Persona** (positive amplification):
```python
def forward(self, affect, reward):
    # Amplify positive rewards when affect is high
    if reward > 0:
        return torch.sigmoid(self.mlp(affect)) * reward * self.scale
    return 0
```

**Shadow** (negative processing):
```python
def forward(self, affect, reward):
    # Process negative rewards when affect is low
    if reward < 0:
        return torch.sigmoid(self.mlp(-affect)) * reward * self.scale
    return 0
```

**Ego** (expert arbitration):
```python
def forward(self, affect, obs):
    # Context-dependent gating
    weights = torch.softmax(self.mlp(torch.cat([affect, obs], -1)), dim=-1)
    return weights
```

**Self-Regulation** (homeostasis):
```python
def forward(self, policy_dist):
    # Encourage exploration when entropy is low
    entropy = policy_dist.entropy()
    target_entropy = self.target_entropy
    bonus = torch.relu(target_entropy - entropy) * self.coef
    return bonus
```

## Generated Figures

After running `python scripts/evaluate.py`, figures are saved to `results/figures/`:

- `stage1_core.png`: Four-panel diagnostic validation (boundedness, recovery, timescale, coupling)
- `stage2_learning.png`: Single-agent learning curves with baseline comparisons
- `stage3_social.png`: Multi-agent performance and cooperation emergence
- `robustness_shift.png`: Performance under five distribution shift scenarios
- `ablations_heatmap.png`: Component importance heatmap

## Citation

```bibtex
@inproceedings{litchiowong2026persona,
  title={Persona, Ego, Shadow, and Self: A Map of the Soul Framework for Proto-Emotional Homeostasis in AI},
  author={Litchiowong, Napassorn},
  institution={National University of Singapore},
  booktitle={AAAI Conference on Artificial Intelligence - Undergraduate Consortium},
  year={2026}
}
```

## License

MIT License