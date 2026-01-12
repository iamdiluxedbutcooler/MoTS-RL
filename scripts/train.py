import os
import sys
import yaml
import torch
import gymnasium as gym
from pathlib import Path
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from mots_rl.core import CoreAffectiveDynamics
from mots_rl.modulators.ego import EgoModulator
from mots_rl.policy.ppo_policy import MoTSPPOPolicy
from mots_rl.policy.sac_policy import MoTSSACPolicy
from mots_rl.trainers.ppo_trainer import PPOTrainer
from mots_rl.trainers.sac_trainer import SACTrainer
from mots_rl.utils.logging import Logger
from mots_rl.baselines.vanilla import VanillaRLBaseline
from mots_rl.baselines.vanilla_plus import VanillaPlusBaseline
from mots_rl.baselines.icm import ICMBaseline
from mots_rl.baselines.rnd import RNDBaseline
from mots_rl.baselines.dim_affect import DimAffectBaseline
from mots_rl.baselines.rl2 import RL2Baseline


def create_env(env_name):
    try:
        env = gym.make(env_name)
        return env
    except:
        print(f"Warning: Could not create environment {env_name}. Using CartPole-v1 as fallback.")
        return gym.make("CartPole-v1")


def main():
    if len(sys.argv) < 2:
        print("Usage: python train.py +experiment=<config_path>")
        sys.exit(1)
    
    config_arg = sys.argv[1]
    if config_arg.startswith("+experiment="):
        config_path = config_arg.replace("+experiment=", "")
    else:
        config_path = config_arg
    
    if not config_path.endswith(".yaml"):
        config_path = f"{config_path}.yaml"
    
    full_path = Path(__file__).parent.parent / "configs" / config_path
    
    if not full_path.exists():
        print(f"Config file not found: {full_path}")
        sys.exit(1)
    
    with open(full_path, "r") as f:
        config = yaml.safe_load(f)
    
    config = OmegaConf.create(config)
    
    print(f"Training with config: {config_path}")
    print(f"Environment: {config.env_name}")
    print(f"Policy: {config.policy_type}")
    print(f"Total timesteps: {config.total_timesteps}")
    print(f"Number of seeds: {config.num_seeds}")
    
    device = config.get("device", "cpu")
    
    env = create_env(config.env_name)
    
    if hasattr(env.observation_space, "shape"):
        obs_dim = env.observation_space.shape[0]
    else:
        obs_dim = env.observation_space.n
    
    baseline_type = config.get("baseline_type", None)
    
    if baseline_type:
        print(f"Training baseline: {baseline_type}")
        
        if baseline_type == "vanilla":
            baseline = VanillaRLBaseline(env, config.policy_type, device, config.get("lr", 3e-4))
        elif baseline_type == "vanilla_plus":
            baseline = VanillaPlusBaseline(env, config.policy_type, device, config.get("lr", 3e-4))
        elif baseline_type == "icm":
            baseline = ICMBaseline(env, config.policy_type, device, config.get("lr", 3e-4), config.get("intrinsic_coef", 0.01))
        elif baseline_type == "rnd":
            baseline = RNDBaseline(env, config.policy_type, device, config.get("lr", 3e-4), config.get("intrinsic_coef", 0.01))
        elif baseline_type == "dim_affect":
            baseline = DimAffectBaseline(env, config.policy_type, device, config.get("lr", 3e-4))
        elif baseline_type == "rl2":
            baseline = RL2Baseline(env, config.policy_type, device, config.get("lr", 3e-4))
        else:
            raise ValueError(f"Unknown baseline type: {baseline_type}")
        
        log_dir = Path(config.get("log_dir", "results/default"))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        results = baseline.train(
            total_timesteps=config.total_timesteps,
            num_seeds=config.num_seeds
        )
        
        import pickle
        results_file = log_dir / "training_results.pkl"
        with open(results_file, "wb") as f:
            pickle.dump(results, f)
        
        env.close()
        print(f"Training complete. Results saved to {log_dir}")
        return
    
    core = CoreAffectiveDynamics(
        input_dim=obs_dim,
        hidden_dim=config.get("hidden_dim", 32),
        device=device
    )
    
    if "alpha" in config:
        alpha_values = torch.tensor(config.alpha, dtype=torch.float32)
        core.alpha_raw.data = torch.logit(alpha_values)
    
    if "beta" in config:
        beta_values = torch.tensor(config.beta, dtype=torch.float32)
        core.beta_raw.data = torch.logit(beta_values)
    
    ego = EgoModulator(
        affect_dim=4,
        obs_dim=obs_dim,
        num_experts=config.get("num_experts", 3),
        mode=config.get("ego_mode", "mlp"),
        hidden_dim=config.get("hidden_dim", 64)
    )
    
    if config.policy_type == "ppo":
        policy = MoTSPPOPolicy(env, core, ego, num_experts=config.get("num_experts", 3), device=device)
        trainer = PPOTrainer(env, policy, core, config, logger=None)
    elif config.policy_type == "sac":
        policy = MoTSSACPolicy(env, core, ego, num_experts=config.get("num_experts", 3), device=device)
        trainer = SACTrainer(env, policy, core, config, logger=None)
    else:
        raise ValueError(f"Unknown policy type: {config.policy_type}")
    
    log_dir = Path(config.get("log_dir", "results/default"))
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = Logger(
        log_dir=log_dir,
        use_wandb=config.get("use_wandb", False),
        wandb_project=config.get("wandb_project", "mots-rl"),
        wandb_entity=config.get("wandb_entity", None)
    )
    
    trainer.logger = logger
    
    results = trainer.train(
        total_timesteps=config.total_timesteps,
        num_seeds=config.num_seeds
    )
    
    import pickle
    results_file = log_dir / "training_results.pkl"
    with open(results_file, "wb") as f:
        pickle.dump(results, f)
    
    logger.close()
    env.close()
    
    print(f"Training complete. Results saved to {log_dir}")


if __name__ == "__main__":
    main()
