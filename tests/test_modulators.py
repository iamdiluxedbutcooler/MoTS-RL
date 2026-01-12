import pytest
import torch
from mots_rl.modulators.persona import PersonaModulator
from mots_rl.modulators.shadow import ShadowModulator
from mots_rl.modulators.ego import EgoModulator
from mots_rl.modulators.self_reg import SelfRegulator


def test_persona_output_range():
    persona = PersonaModulator(affect_dim=4)
    affect = torch.randn(4)
    output = persona(affect)
    
    assert output.shape == ()
    assert 0 <= output.item() <= 1


def test_shadow_positive():
    shadow = ShadowModulator(affect_dim=4)
    affect = torch.randn(4)
    output = shadow(affect)
    
    assert output.shape == ()
    assert output.item() > 0


def test_ego_mlp():
    ego = EgoModulator(affect_dim=4, obs_dim=10, num_experts=3, mode="mlp")
    affect = torch.randn(4)
    obs = torch.randn(10)
    
    weights = ego(affect, obs)
    
    assert weights.shape == (3,)
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.all(weights >= 0)


def test_ego_lstm():
    ego = EgoModulator(affect_dim=4, obs_dim=10, num_experts=3, mode="lstm", hidden_dim=64)
    affect = torch.randn(4)
    obs = torch.randn(10)
    
    weights = ego(affect, obs)
    
    assert weights.shape == (3,)
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5)


def test_ego_reset():
    ego = EgoModulator(affect_dim=4, obs_dim=10, num_experts=3, mode="lstm")
    affect = torch.randn(4)
    obs = torch.randn(10)
    
    ego(affect, obs)
    ego.reset()
    assert ego.hidden is None


def test_self_regulator_update():
    self_reg = SelfRegulator(lambda_kl=0.01, window_size=100)
    
    for _ in range(50):
        weights = torch.rand(3)
        weights = weights / weights.sum()
        self_reg.update(weights)
    
    assert len(self_reg.weight_history) == 50


def test_self_regulator_loss():
    self_reg = SelfRegulator(lambda_kl=0.01, window_size=100)
    
    for _ in range(50):
        weights = torch.tensor([1.0, 0.0, 0.0])
        self_reg.update(weights)
    
    loss = self_reg.compute_loss()
    assert loss.item() > 0


def test_self_regulator_window():
    self_reg = SelfRegulator(lambda_kl=0.01, window_size=10)
    
    for i in range(20):
        weights = torch.rand(3)
        weights = weights / weights.sum()
        self_reg.update(weights)
    
    assert len(self_reg.weight_history) == 10


def test_self_regulator_reset():
    self_reg = SelfRegulator(lambda_kl=0.01, window_size=100)
    
    for _ in range(10):
        weights = torch.rand(3)
        self_reg.update(weights)
    
    self_reg.reset()
    assert len(self_reg.weight_history) == 0
