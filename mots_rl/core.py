import torch
import torch.nn as nn


class CoreAffectiveDynamics(nn.Module):

    def __init__(self, input_dim, hidden_dim=32, device="cpu"):
        super().__init__()
        self.device = device
        self.affect_dim = 4
        self.obs_dim = input_dim
        
        self.alpha_raw = nn.Parameter(torch.zeros(self.affect_dim))
        self.beta_raw = nn.Parameter(torch.zeros(self.affect_dim))
        
        self.g_net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.affect_dim),
            nn.Tanh()
        )
        
        self.phi_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        self.affect_state = torch.zeros(self.affect_dim, device=device)
        
        self._init_params()
    
    def _init_params(self):
        for m in self.g_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
        for m in self.phi_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
    
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
        
        phi_obs = self.phi_net(obs_t)
        if phi_obs.dim() > 1:
            phi_obs = phi_obs.squeeze(0)
        
        g_input = torch.cat([reward_t, phi_obs])
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
    
    def set_state(self, state):
        self.affect_state = state.clone().to(self.device)
