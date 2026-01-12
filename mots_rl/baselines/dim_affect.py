import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO, SAC


class DimAffectCore(nn.Module):

    def __init__(self, input_dim, hidden_dim=32, device="cpu"):
        super().__init__()
        self.device = device
        self.affect_dim = 2
        
        self.alpha_raw = nn.Parameter(torch.zeros(self.affect_dim))
        self.beta_raw = nn.Parameter(torch.zeros(self.affect_dim))
        
        self.g_net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.affect_dim),
            nn.Tanh()
        )
        
        self.affect_state = torch.zeros(self.affect_dim, device=device)
    
    def get_alpha_beta(self):
        alpha = torch.sigmoid(self.alpha_raw)
        beta = torch.sigmoid(self.beta_raw)
        return alpha, beta
    
    def forward(self, obs, reward, done):
        if done:
            self.affect_state = torch.zeros(self.affect_dim, device=self.device)
            return self.affect_state.clone()
        
        if isinstance(obs, torch.Tensor):
            obs_t = obs.to(self.device)
        else:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        
        reward_t = torch.tensor([reward], dtype=torch.float32, device=self.device)
        
        g_input = torch.cat([reward_t, obs_t.flatten()])
        if g_input.dim() == 1:
            g_input = g_input.unsqueeze(0)
        
        g_output = self.g_net(g_input).squeeze(0)
        
        alpha, beta = self.get_alpha_beta()
        
        self.affect_state = alpha * self.affect_state + beta * g_output
        
        return self.affect_state.clone()
    
    def reset(self):
        self.affect_state = torch.zeros(self.affect_dim, device=self.device)
    
    def get_state(self):
        return self.affect_state.clone()


class DimAffectBaseline:

    def __init__(self, env, policy_type="ppo", device="cpu", lr=3e-4):
        self.env = env
        self.policy_type = policy_type
        self.device = device
        
        obs_dim = env.observation_space.shape[0] if hasattr(env.observation_space, "shape") else env.observation_space.n
        
        self.core = DimAffectCore(obs_dim, device=device)
        
        if policy_type == "ppo":
            self.agent = PPO("MlpPolicy", env, verbose=0, device=device, learning_rate=lr)
        elif policy_type == "sac":
            self.agent = SAC("MlpPolicy", env, verbose=0, device=device, learning_rate=lr)
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")
        
        self.optimizer = torch.optim.Adam(self.core.parameters(), lr=lr)
    
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
        self.core.reset()
        
        episode_rewards = []
        episode_reward = 0
        affect_history = []
        
        for step in range(total_timesteps):
            action = self.predict(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            affect_state = self.core(next_obs, reward, done)
            affect_history.append(affect_state.detach().cpu().numpy())
            
            episode_reward += reward
            
            if done:
                episode_rewards.append(episode_reward)
                obs, info = self.env.reset()
                self.core.reset()
                episode_reward = 0
            else:
                obs = next_obs
            
            if step % 2048 == 0 and step > 0:
                self.agent.learn(total_timesteps=2048, reset_num_timesteps=False)
        
        affect_array = np.array(affect_history)
        affect_padded = np.pad(affect_array, ((0, 0), (0, 2)), mode='constant')
        
        return {
            "episode_rewards": episode_rewards,
            "affect_history": affect_padded,
            "weight_history": np.zeros((total_timesteps, 3)),
            "seed": seed
        }
