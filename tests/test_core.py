import pytest
import torch
import numpy as np
from mots_rl.core import CoreAffectiveDynamics


def test_core_initialization():
    core = CoreAffectiveDynamics(input_dim=10, hidden_dim=32, device="cpu")
    assert core.affect_dim == 4
    assert core.affect_state.shape == (4,)


def test_core_forward():
    core = CoreAffectiveDynamics(input_dim=10, hidden_dim=32, device="cpu")
    obs = torch.randn(10)
    reward = 1.0
    done = False
    
    affect_state = core(obs, reward, done)
    assert affect_state.shape == (4,)
    assert torch.all(torch.isfinite(affect_state))


def test_core_reset():
    core = CoreAffectiveDynamics(input_dim=10, hidden_dim=32, device="cpu")
    obs = torch.randn(10)
    core(obs, 1.0, False)
    
    core.reset()
    assert torch.allclose(core.affect_state, torch.zeros(4))


def test_core_boundedness():
    core = CoreAffectiveDynamics(input_dim=10, hidden_dim=32, device="cpu")
    
    alpha = torch.tensor([0.9, 0.8, 0.95, 0.7])
    beta = torch.tensor([0.1, 0.2, 0.05, 0.3])
    core.alpha_raw.data = torch.logit(alpha)
    core.beta_raw.data = torch.logit(beta)
    
    max_norm = 0.0
    for _ in range(1000):
        obs = torch.randn(10)
        reward = np.random.randn()
        affect_state = core(obs, reward, False)
        max_norm = max(max_norm, torch.abs(affect_state).max().item())
    
    assert max_norm <= 1.5


def test_alpha_beta_range():
    core = CoreAffectiveDynamics(input_dim=10, hidden_dim=32, device="cpu")
    alpha, beta = core.get_alpha_beta()
    
    assert torch.all(alpha > 0)
    assert torch.all(alpha < 1)
    assert torch.all(beta > 0)
    assert torch.all(beta < 1)


def test_done_resets_state():
    core = CoreAffectiveDynamics(input_dim=10, hidden_dim=32, device="cpu")
    obs = torch.randn(10)
    
    core(obs, 1.0, False)
    assert not torch.allclose(core.affect_state, torch.zeros(4))
    
    core(obs, 1.0, True)
    assert torch.allclose(core.affect_state, torch.zeros(4))


def test_state_persistence():
    core = CoreAffectiveDynamics(input_dim=10, hidden_dim=32, device="cpu")
    obs = torch.randn(10)
    
    affect1 = core(obs, 1.0, False)
    affect2 = core(obs, 1.0, False)
    
    assert not torch.allclose(affect1, affect2)
