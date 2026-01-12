import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from stable_baselines3 import PPO, SAC


class ICMModule(nn.Module):

    def __init__(self, obs_dim, action_dim, hidden_dim=256, device="cpu"):
        super().__init__()
        self.device = device
        
        self.forward_model = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )
        
        self.inverse_model = nn.Sequential(
            nn.Linear(obs_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128)
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def compute_intrinsic_reward(self, obs, action, next_obs):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        action_t = torch.tensor(action, dtype=torch.float32, device=self.device)
        
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
            next_obs_t = next_obs_t.unsqueeze(0)
            action_t = action_t.unsqueeze(0)
        
        encoded_obs = self.encoder(obs_t)
        encoded_next_obs = self.encoder(next_obs_t)
        
        state_action = torch.cat([encoded_obs, action_t], dim=-1)
        predicted_next_state = self.forward_model(state_action)
        
        prediction_error = F.mse_loss(predicted_next_state, encoded_next_obs, reduction='none')
        intrinsic_reward = prediction_error.mean(dim=-1).item()
        
        return intrinsic_reward
    
    def update(self, obs, action, next_obs):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        action_t = torch.tensor(action, dtype=torch.float32, device=self.device)
        
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
            next_obs_t = next_obs_t.unsqueeze(0)
            action_t = action_t.unsqueeze(0)
        
        encoded_obs = self.encoder(obs_t)
        encoded_next_obs = self.encoder(next_obs_t)
        
        state_action = torch.cat([encoded_obs, action_t], dim=-1)
        predicted_next_state = self.forward_model(state_action)
        forward_loss = F.mse_loss(predicted_next_state, encoded_next_obs)
        
        state_pair = torch.cat([encoded_obs, encoded_next_obs], dim=-1)
        predicted_action = self.inverse_model(state_pair)
        inverse_loss = F.mse_loss(predicted_action, action_t)
        
        loss = forward_loss + inverse_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class ICMBaseline:

    def __init__(self, env, policy_type="ppo", device="cpu", lr=3e-4, intrinsic_coef=0.01):
        self.env = env
        self.policy_type = policy_type
        self.device = device
        self.intrinsic_coef = intrinsic_coef
        
        obs_dim = env.observation_space.shape[0] if hasattr(env.observation_space, "shape") else env.observation_space.n
        action_dim = env.action_space.shape[0] if hasattr(env.action_space, "shape") else env.action_space.n
        
        self.icm = ICMModule(obs_dim, action_dim, device=device)
        
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
            action = self.predict(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            intrinsic_reward = self.icm.compute_intrinsic_reward(obs, action, next_obs)
            intrinsic_rewards.append(intrinsic_reward)
            
            augmented_reward = reward + self.intrinsic_coef * intrinsic_reward
            
            self.icm.update(obs, action, next_obs)
            
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
