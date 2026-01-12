import torch
import torch.nn as nn


class PersonaModulator(nn.Module):

    def __init__(self, affect_dim=4):
        super().__init__()
        self.w_p = nn.Parameter(torch.randn(affect_dim))
        self.b_p = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.w_p, mean=0.0, std=0.1)
    
    def forward(self, affect_state):
        logit = torch.dot(self.w_p, affect_state) + self.b_p
        return torch.sigmoid(logit).squeeze()
