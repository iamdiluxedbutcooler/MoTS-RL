import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import SAC


class MoTSSACPolicy(nn.Module):

    def __init__(self, env, core_dynamics, ego_modulator, num_experts=3, device="cpu"):
        super().__init__()
        self.env = env
        self.core = core_dynamics
        self.ego = ego_modulator
        self.num_experts = num_experts
        self.device = device
        
        self.expert_policies = []
        for i in range(num_experts):
            policy = SAC("MlpPolicy", env, verbose=0, device=device)
            self.expert_policies.append(policy)
    
    def predict(self, obs, affect_state, deterministic=False):
        weights = self.ego(affect_state, obs)
        
        expert_actions = []
        for i, expert in enumerate(self.expert_policies):
            action, _ = expert.predict(obs, deterministic=deterministic)
            expert_actions.append(action)
        
        action = np.zeros_like(expert_actions[0])
        for i, a in enumerate(expert_actions):
            action += weights[i].item() * a
        
        return action, weights
    
    def learn(self, total_timesteps, obs, reward, done, affect_state):
        weights = self.ego(affect_state, obs)
        
        for i, expert in enumerate(self.expert_policies):
            weight = weights[i].item()
            if weight > 0.1:
                steps = max(1, int(weight * 1024))
                expert.learn(total_timesteps=steps, reset_num_timesteps=False)
        
        return weights
