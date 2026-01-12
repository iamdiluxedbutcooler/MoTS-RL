import torch
import torch.nn as nn


class ShadowModulator(nn.Module):

    def __init__(self, affect_dim=4):
        super().__init__()
        self.w_s = nn.Parameter(torch.randn(affect_dim))
        self.b_s = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.w_s, mean=0.0, std=0.1)
    
    def forward(self, affect_state):
        logit = torch.dot(self.w_s, affect_state) + self.b_s
        return torch.exp(logit).squeeze()
