import os
import sys
import yaml
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)


def load_results(log_dir):
    results_file = Path(log_dir) / "training_results.pkl"
    if not results_file.exists():
        return None
    with open(results_file, "rb") as f:
        return pickle.load(f)


def plot_stage1_core(results_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    boundedness_results = load_results(results_dir / "stage1" / "boundedness")
    if boundedness_results:
        affect_histories = [r["affect_history"] for r in boundedness_results]
        max_norms = [np.max(np.abs(ah), axis=1).max() for ah in affect_histories]
        axes[0, 0].hist(max_norms, bins=20, alpha=0.7, edgecolor="black")
        axes[0, 0].axvline(x=1.0, color="r", linestyle="--", label="Theoretical bound")
        axes[0, 0].set_xlabel("Max ||A_t||∞")
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].set_title("A: Boundedness")
        axes[0, 0].legend()
    
    recovery_results = load_results(results_dir / "stage1" / "recovery")
    if recovery_results:
        affect_history = recovery_results[0]["affect_history"]
        for dim in range(4):
            axes[0, 1].plot(affect_history[:500, dim], alpha=0.7, label=f"Dim {dim}")
        axes[0, 1].set_xlabel("Timestep")
        axes[0, 1].set_ylabel("Affect value")
        axes[0, 1].set_title("B: Recovery trajectories")
        axes[0, 1].legend()
    
    timescale_results = load_results(results_dir / "stage1" / "timescale")
    if timescale_results:
        affect_history = timescale_results[0]["affect_history"]
        for dim in range(4):
            acf = np.correlate(affect_history[:, dim], affect_history[:, dim], mode="full")
            acf = acf[len(acf)//2:]
            acf = acf / acf[0]
            axes[1, 0].plot(acf[:100], alpha=0.7, label=f"Dim {dim}")
        axes[1, 0].set_xlabel("Lag")
        axes[1, 0].set_ylabel("Autocorrelation")
        axes[1, 0].set_title("C: Autocorrelation")
        axes[1, 0].legend()
    
    coupled_results = load_results(results_dir / "stage1" / "coupled")
    if coupled_results:
        affect_history = coupled_results[0]["affect_history"]
        corr_matrix = np.corrcoef(affect_history.T)
        sns.heatmap(corr_matrix, ax=axes[1, 1], cmap="coolwarm", vmin=-1, vmax=1, 
                   square=True, cbar_kws={"label": "Correlation"})
        axes[1, 1].set_title("D: Mutual information")
    
    plt.tight_layout()
    output_path = results_dir / "figures" / "stage1_core.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_stage2_learning(results_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    tasks = ["minigrid", "mujoco", "memory", "mo_hopper"]
    for idx, task in enumerate(tasks):
        ax = axes[idx // 2, idx % 2]
        results = load_results(results_dir / "stage2" / task)
        if results:
            all_rewards = [r["episode_rewards"] for r in results]
            max_len = max(len(r) for r in all_rewards)
            
            rewards_matrix = np.full((len(all_rewards), max_len), np.nan)
            for i, rewards in enumerate(all_rewards):
                rewards_matrix[i, :len(rewards)] = rewards
            
            mean_rewards = np.nanmean(rewards_matrix, axis=0)
            se_rewards = np.nanstd(rewards_matrix, axis=0) / np.sqrt(np.sum(~np.isnan(rewards_matrix), axis=0))
            
            x = np.arange(len(mean_rewards))
            ax.plot(x, mean_rewards, linewidth=2)
            ax.fill_between(x, mean_rewards - se_rewards, mean_rewards + se_rewards, alpha=0.3)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Return")
            ax.set_title(task.replace("_", " ").title())
            ax.set_xscale("log")
    
    plt.tight_layout()
    output_path = results_dir / "figures" / "stage2_learning.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_stage3_social(results_dir):
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2)
    
    ax1 = fig.add_subplot(gs[0, :])
    tasks = ["ipd", "mpe_nav", "mpe_pred", "coin_game"]
    cooperation_rates = []
    for task in tasks:
        results = load_results(results_dir / "stage3" / task)
        if results:
            avg_reward = np.mean([np.mean(r["episode_rewards"]) for r in results])
            cooperation_rates.append(avg_reward)
        else:
            cooperation_rates.append(0)
    
    ax1.bar(tasks, cooperation_rates, alpha=0.7, edgecolor="black")
    ax1.set_ylabel("Cooperation rate")
    ax1.set_title("A: Cooperation across tasks")
    
    ax2 = fig.add_subplot(gs[1, 0])
    results = load_results(results_dir / "stage3" / "ipd")
    if results:
        episode_rewards = results[0]["episode_rewards"]
        ax2.plot(episode_rewards[:100], linewidth=2)
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Reward")
        ax2.set_title("B: IPD dynamics")
    
    ax3 = fig.add_subplot(gs[1, 1])
    if results:
        weight_history = results[0]["weight_history"]
        for i in range(weight_history.shape[1]):
            ax3.plot(weight_history[:500, i], alpha=0.7, label=f"Expert {i}")
        ax3.set_xlabel("Timestep")
        ax3.set_ylabel("Weight")
        ax3.set_title("C: Archetypal activation")
        ax3.legend()
    
    plt.tight_layout()
    output_path = results_dir / "figures" / "stage3_social.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_robustness_shift(results_dir):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    robustness_tasks = ["r1_reward_swap", "r2_partner", "r3_obs_noise", "r4_action_dropout", "r5_zero_shot"]
    for idx, task in enumerate(robustness_tasks):
        ax = axes[idx // 3, idx % 3]
        results = load_results(results_dir / "robustness" / task)
        if results:
            episode_rewards = results[0]["episode_rewards"]
            ax.plot(episode_rewards[:100], linewidth=2)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Return")
            ax.set_title(task.replace("_", " ").title())
    
    axes[1, 2].axis("off")
    
    plt.tight_layout()
    output_path = results_dir / "figures" / "robustness_shift.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_ablations_heatmap(results_dir):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ablations = [
        "persona_off", "shadow_off", "ego_fixed", "self_off",
        "affect_2d", "affect_random", "gating_lstm",
        "alpha_sweep_fast", "alpha_sweep_slow"
    ]
    
    performance = []
    for ablation in ablations:
        results = load_results(results_dir / "ablations" / ablation)
        if results:
            avg_reward = np.mean([np.mean(r["episode_rewards"]) for r in results])
            performance.append(avg_reward)
        else:
            performance.append(0)
    
    performance = np.array(performance).reshape(3, 3)
    
    sns.heatmap(performance, ax=ax, annot=True, fmt=".1f", cmap="RdYlGn",
               xticklabels=["Modulator", "Affect", "Gating"],
               yticklabels=["A1-A3", "A4-A6", "A7-A9"],
               cbar_kws={"label": "Average return"})
    ax.set_title("Ablation study")
    
    plt.tight_layout()
    output_path = results_dir / "figures" / "ablations_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_umap_clusters(results_dir):
    try:
        import umap
        from sklearn.cluster import KMeans
    except ImportError:
        print("Warning: umap-learn not installed. Skipping UMAP plot.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    all_affects = []
    results = load_results(results_dir / "stage2" / "mujoco")
    if results:
        for r in results[:5]:
            all_affects.append(r["affect_history"])
        
        affect_concat = np.vstack(all_affects)
        
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding = reducer.fit_transform(affect_concat)
        
        kmeans = KMeans(n_clusters=5, random_state=42)
        labels = kmeans.fit_predict(affect_concat)
        
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, 
                           cmap="tab10", alpha=0.6, s=10)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title("Affective state clusters")
        plt.colorbar(scatter, ax=ax, label="Cluster")
        
        plt.tight_layout()
        output_path = results_dir / "figures" / "umap_clusters.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")


def main():
    if len(sys.argv) < 2:
        results_dir = Path(__file__).parent.parent / "results"
    else:
        config_arg = sys.argv[1]
        if config_arg.startswith("+experiment="):
            config_path = config_arg.replace("+experiment=", "")
        else:
            config_path = config_arg
        
        if not config_path.endswith(".yaml"):
            config_path = f"{config_path}.yaml"
        
        full_path = Path(__file__).parent.parent / "configs" / config_path
        
        if full_path.exists():
            with open(full_path, "r") as f:
                config = yaml.safe_load(f)
            results_dir = Path(config.get("log_dir", "results")).parent.parent
        else:
            results_dir = Path(__file__).parent.parent / "results"
    
    print(f"Evaluating results in: {results_dir}")
    
    plot_stage1_core(results_dir)
    plot_stage2_learning(results_dir)
    plot_stage3_social(results_dir)
    plot_robustness_shift(results_dir)
    plot_ablations_heatmap(results_dir)
    plot_umap_clusters(results_dir)
    
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
