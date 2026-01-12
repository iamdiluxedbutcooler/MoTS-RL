import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from stable_baselines3 import PPO, SAC


class RNDModule(nn.Module):

    def __init__(self, obs_dim, hidden_dim=256, device="cpu"):
        super().__init__()
        self.device = device
        
        self.target_network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128)
        )
        
        for param in self.target_network.parameters():
            param.requires_grad = False
        
        self.predictor_network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128)
        )
        
        self.optimizer = torch.optim.Adam(self.predictor_network.parameters(), lr=1e-3)
    
    def compute_intrinsic_reward(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        
        with torch.no_grad():
            target_features = self.target_network(obs_t)
        
        predicted_features = self.predictor_network(obs_t)
        
        prediction_error = F.mse_loss(predicted_features, target_features, reduction='none')
        intrinsic_reward = prediction_error.mean(dim=-1).item()
        
        return intrinsic_reward
    
    def update(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        
        with torch.no_grad():
            target_features = self.target_network(obs_t)
        
        predicted_features = self.predictor_network(obs_t)
        
        loss = F.mse_loss(predicted_features, target_features)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class RNDBaseline:

    def __init__(self, env, policy_type="ppo", device="cpu", lr=3e-4, intrinsic_coef=0.01):
        self.env = env
        self.policy_type = policy_type
        self.device = device
        self.intrinsic_coef = intrinsic_coef
        
        obs_dim = env.observation_space.shape[0] if hasattr(env.observation_space, "shape") else env.observation_space.n
        
        self.rnd = RNDModule(obs_dim, device=device)
        
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
        intrinsic_rewards = []
        
        for step in range(total_timesteps):
            intrinsic_reward = self.rnd.compute_intrinsic_reward(obs)
            intrinsic_rewards.append(intrinsic_reward)
            
            action = self.predict(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            self.rnd.update(obs)
            
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
            "intrinsic_rewards": intrinsic_rewards,
            "seed": seed
        }
