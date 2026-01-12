import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO, SAC


class RL2Policy(nn.Module):

    def __init__(self, obs_dim, action_dim, hidden_dim=128, device="cpu"):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(
            input_size=obs_dim + action_dim + 1,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        self.hidden = None
    
    def forward(self, obs, prev_action, prev_reward):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        prev_action_t = torch.tensor(prev_action, dtype=torch.float32, device=self.device)
        prev_reward_t = torch.tensor([prev_reward], dtype=torch.float32, device=self.device)
        
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0).unsqueeze(0)
            prev_action_t = prev_action_t.unsqueeze(0).unsqueeze(0)
            prev_reward_t = prev_reward_t.unsqueeze(0).unsqueeze(0)
        
        x = torch.cat([obs_t, prev_action_t, prev_reward_t], dim=-1)
        
        if self.hidden is None:
            lstm_out, self.hidden = self.lstm(x)
        else:
            lstm_out, self.hidden = self.lstm(x, self.hidden)
        
        policy_logits = self.policy_head(lstm_out.squeeze(0).squeeze(0))
        value = self.value_head(lstm_out.squeeze(0).squeeze(0))
        
        return policy_logits, value
    
    def reset(self):
        self.hidden = None


class RL2Baseline:

    def __init__(self, env, policy_type="ppo", device="cpu", lr=3e-4):
        self.env = env
        self.policy_type = policy_type
        self.device = device
        
        obs_dim = env.observation_space.shape[0] if hasattr(env.observation_space, "shape") else env.observation_space.n
        action_dim = env.action_space.shape[0] if hasattr(env.action_space, "shape") else env.action_space.n
        
        self.rl2_policy = RL2Policy(obs_dim, action_dim, device=device)
        
        if policy_type == "ppo":
            self.agent = PPO("MlpPolicy", env, verbose=0, device=device, learning_rate=lr)
        elif policy_type == "sac":
            self.agent = SAC("MlpPolicy", env, verbose=0, device=device, learning_rate=lr)
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")
        
        self.optimizer = torch.optim.Adam(self.rl2_policy.parameters(), lr=lr)
    
    def predict(self, obs, prev_action, prev_reward, deterministic=False):
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
        self.rl2_policy.reset()
        
        episode_rewards = []
        episode_reward = 0
        
        prev_action = np.zeros(self.env.action_space.shape[0] if hasattr(self.env.action_space, "shape") else self.env.action_space.n)
        prev_reward = 0.0
        
        for step in range(total_timesteps):
            action = self.predict(obs, prev_action, prev_reward, deterministic=False)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            
            prev_action = action
            prev_reward = reward
            
            if done:
                episode_rewards.append(episode_reward)
                obs, info = self.env.reset()
                self.rl2_policy.reset()
                episode_reward = 0
                prev_action = np.zeros_like(prev_action)
                prev_reward = 0.0
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
