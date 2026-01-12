import sys
import torch
import numpy as np
import gymnasium as gym
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mots_rl.core import CoreAffectiveDynamics
from scripts.evaluate import load_results


def main():
    results_dir = Path(__file__).parent.parent / "results"
    
    print("=" * 60)
    print("COUNTERFACTUAL INTERVENTION ANALYSIS")
    print("=" * 60)
    
    results = load_results(results_dir / "stage2" / "mujoco")
    if results is None:
        print("No results found for stage2/mujoco")
        return 1
    
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    
    core = CoreAffectiveDynamics(input_dim=obs_dim + 1, hidden_dim=32, device="cpu")
    
    interventions = [
        torch.tensor([0.5, 0.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.5, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.5, 0.0]),
        torch.tensor([0.0, 0.0, 0.0, 0.5]),
        torch.tensor([-0.5, 0.0, 0.0, 0.0]),
    ]
    
    print("\nRunning counterfactual interventions...")
    
    baseline_returns = []
    intervention_returns = {i: [] for i in range(len(interventions))}
    
    for seed in range(5):
        obs, _ = env.reset(seed=seed)
        core.reset()
        episode_return = 0
        
        for _ in range(200):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            core(next_obs, reward, done)
            episode_return += reward
            
            if done:
                break
            obs = next_obs
        
        baseline_returns.append(episode_return)
    
    for interv_idx, intervention in enumerate(interventions):
        for seed in range(5):
            obs, _ = env.reset(seed=seed)
            core.reset()
            core.set_state(intervention)
            episode_return = 0
            
            for _ in range(200):
                action = env.action_space.sample()
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                core(next_obs, reward, done)
                episode_return += reward
                
                if done:
                    break
                obs = next_obs
            
            intervention_returns[interv_idx].append(episode_return)
    
    baseline_mean = np.mean(baseline_returns)
    print(f"\nBaseline return: {baseline_mean:.2f} ± {np.std(baseline_returns):.2f}")
    
    for interv_idx, intervention in enumerate(interventions):
        interv_mean = np.mean(intervention_returns[interv_idx])
        interv_std = np.std(intervention_returns[interv_idx])
        delta = interv_mean - baseline_mean
        print(f"Intervention {interv_idx}: {interv_mean:.2f} ± {interv_std:.2f} (Δ={delta:+.2f})")
    
    env.close()
    print("\nCounterfactual analysis complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
