import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO, SAC


class VanillaRLBaseline:

    def __init__(self, env, policy_type="ppo", device="cpu", lr=3e-4):
        self.env = env
        self.policy_type = policy_type
        self.device = device
        
        if policy_type == "ppo":
            self.agent = PPO("MlpPolicy", env, verbose=0, device=device, learning_rate=lr)
        elif policy_type == "sac":
            self.agent = SAC("MlpPolicy", env, verbose=0, device=device, learning_rate=lr)
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")
    
    def predict(self, obs, deterministic=False):
        action, _ = self.agent.predict(obs, deterministic=deterministic)
        return action
    
    def train(self, total_timesteps, num_seeds=1):
        results = []
        
        for seed in range(num_seeds):
            self._set_seed(seed)
            result = self._train_single_run(total_timesteps, seed)
            results.append(result)
        
        return results
    
    def _set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _train_single_run(self, total_timesteps, seed):
        obs, info = self.env.reset(seed=seed)
        
        episode_rewards = []
        episode_reward = 0
        
        for step in range(total_timesteps):
            action = self.predict(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            
            if done:
                episode_rewards.append(episode_reward)
                obs, info = self.env.reset()
                episode_reward = 0
            else:
                obs = next_obs
            
            if step % 2048 == 0 and step > 0:
                self.agent.learn(total_timesteps=2048, reset_num_timesteps=False)
        
        return {
            "episode_rewards": episode_rewards,
            "affect_history": np.zeros((total_timesteps, 4)),
            "weight_history": np.zeros((total_timesteps, 3)),
            "seed": seed
        }
