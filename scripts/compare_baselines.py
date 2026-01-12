import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.evaluate import load_results

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)


def compute_statistics(results_a, results_b):
    rewards_a = [np.mean(r["episode_rewards"]) for r in results_a if r]
    rewards_b = [np.mean(r["episode_rewards"]) for r in results_b if r]
    
    u_stat, p_value = stats.mannwhitneyu(rewards_a, rewards_b, alternative='two-sided')
    
    n1, n2 = len(rewards_a), len(rewards_b)
    r = 1 - (2*u_stat) / (n1 * n2)
    
    mean_a, std_a = np.mean(rewards_a), np.std(rewards_a)
    mean_b, std_b = np.mean(rewards_b), np.std(rewards_b)
    
    return {
        "mean_a": mean_a,
        "std_a": std_a,
        "mean_b": mean_b,
        "std_b": std_b,
        "p_value": p_value,
        "effect_size": r
    }


def plot_baseline_comparison(results_dir):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    baselines = ["vanilla", "vanilla_plus", "icm", "rnd", "dim_affect", "rl2"]
    mots_results = load_results(results_dir / "stage2" / "mujoco")
    
    comparison_stats = {}
    
    for idx, baseline_name in enumerate(baselines):
        ax = axes[idx // 3, idx % 3]
        
        baseline_results = load_results(results_dir / "baselines" / baseline_name)
        
        if mots_results and baseline_results:
            mots_rewards = [r["episode_rewards"] for r in mots_results]
            baseline_rewards = [r["episode_rewards"] for r in baseline_results]
            
            max_len = max(max(len(r) for r in mots_rewards), max(len(r) for r in baseline_rewards))
            
            mots_matrix = np.full((len(mots_rewards), max_len), np.nan)
            for i, rewards in enumerate(mots_rewards):
                mots_matrix[i, :len(rewards)] = rewards
            
            baseline_matrix = np.full((len(baseline_rewards), max_len), np.nan)
            for i, rewards in enumerate(baseline_rewards):
                baseline_matrix[i, :len(rewards)] = rewards
            
            mots_mean = np.nanmean(mots_matrix, axis=0)
            mots_se = np.nanstd(mots_matrix, axis=0) / np.sqrt(np.sum(~np.isnan(mots_matrix), axis=0))
            
            baseline_mean = np.nanmean(baseline_matrix, axis=0)
            baseline_se = np.nanstd(baseline_matrix, axis=0) / np.sqrt(np.sum(~np.isnan(baseline_matrix), axis=0))
            
            x = np.arange(len(mots_mean))
            
            ax.plot(x, mots_mean, label="MoTS", linewidth=2, color='#1f77b4')
            ax.fill_between(x, mots_mean - mots_se, mots_mean + mots_se, alpha=0.3, color='#1f77b4')
            
            ax.plot(x, baseline_mean, label=baseline_name.replace("_", " ").title(), linewidth=2, color='#ff7f0e')
            ax.fill_between(x, baseline_mean - baseline_se, baseline_mean + baseline_se, alpha=0.3, color='#ff7f0e')
            
            stats_result = compute_statistics(mots_results, baseline_results)
            comparison_stats[baseline_name] = stats_result
            
            sig_marker = "***" if stats_result["p_value"] < 0.001 else "**" if stats_result["p_value"] < 0.01 else "*" if stats_result["p_value"] < 0.05 else "ns"
            
            ax.set_xlabel("Episode")
            ax.set_ylabel("Return")
            ax.set_title(f"MoTS vs {baseline_name.replace('_', ' ').title()} ({sig_marker})")
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = results_dir / "figures" / "baseline_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")
    
    print("\n" + "=" * 80)
    print("BASELINE COMPARISON STATISTICS")
    print("=" * 80)
    for baseline_name, stats_result in comparison_stats.items():
        print(f"\n{baseline_name.upper()}:")
        print(f"  MoTS:     {stats_result['mean_a']:.2f} ± {stats_result['std_a']:.2f}")
        print(f"  Baseline: {stats_result['mean_b']:.2f} ± {stats_result['std_b']:.2f}")
        print(f"  p-value:  {stats_result['p_value']:.4f}")
        print(f"  Effect size (r): {stats_result['effect_size']:.4f}")
        
        if stats_result['p_value'] < 0.05:
            if stats_result['mean_a'] > stats_result['mean_b']:
                print(f"  Result: MoTS significantly BETTER (p < 0.05)")
            else:
                print(f"  Result: MoTS significantly WORSE (p < 0.05)")
        else:
            print(f"  Result: No significant difference (p >= 0.05)")


def main():
    results_dir = Path(__file__).parent.parent / "results"
    
    print("Generating baseline comparison plots and statistics...")
    plot_baseline_comparison(results_dir)
    print("\nBaseline comparison complete.")


if __name__ == "__main__":
    main()
