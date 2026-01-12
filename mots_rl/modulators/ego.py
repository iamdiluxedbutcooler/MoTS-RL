import torch
import torch.nn as nn


class EgoModulator(nn.Module):

    def __init__(self, affect_dim=4, obs_dim=None, num_experts=3, mode="mlp", hidden_dim=64):
        super().__init__()
        self.num_experts = num_experts
        self.mode = mode
        
        if mode == "mlp":
            input_dim = affect_dim + (obs_dim if obs_dim is not None else 0)
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, num_experts)
            )
        elif mode == "lstm":
            input_dim = affect_dim + (obs_dim if obs_dim is not None else 0)
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, num_experts)
            self.hidden = None
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def forward(self, affect_state, obs=None):
        if obs is not None:
            if isinstance(obs, torch.Tensor):
                obs_t = obs
            else:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=affect_state.device)
            if obs_t.dim() == 1:
                x = torch.cat([affect_state, obs_t])
            else:
                x = torch.cat([affect_state, obs_t.flatten()])
        else:
            x = affect_state
        
        if self.mode == "mlp":
            logits = self.net(x)
        else:
            x_in = x.unsqueeze(0).unsqueeze(0)
            if self.hidden is None:
                out, self.hidden = self.lstm(x_in)
            else:
                out, self.hidden = self.lstm(x_in, self.hidden)
            logits = self.fc(out.squeeze(0).squeeze(0))
        
        return torch.softmax(logits, dim=0)
    
    def reset(self):
        if self.mode == "lstm":
            self.hidden = None
