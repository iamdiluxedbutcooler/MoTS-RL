import pytest
import gymnasium as gym
from mots_rl.envs.diagnostic import (
    DiagnosticBoundednessEnv,
    DiagnosticRecoveryEnv,
    DiagnosticTimescaleEnv,
    DiagnosticCoupledEnv
)


def test_diagnostic_boundedness_env():
    env = DiagnosticBoundednessEnv()
    obs, info = env.reset()
    
    assert obs.shape == (4,)
    
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs.shape == (4,)
        assert isinstance(reward, float)
        
        if terminated or truncated:
            break


def test_diagnostic_recovery_env():
    env = DiagnosticRecoveryEnv()
    obs, info = env.reset()
    
    assert obs.shape == (4,)
    
    for step in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step == env.perturb_step:
            assert reward == 10.0
        else:
            assert reward == 0.0
        
        if terminated or truncated:
            break


def test_diagnostic_timescale_env():
    env = DiagnosticTimescaleEnv()
    obs, info = env.reset()
    
    assert obs.shape == (4,)
    
    rewards = []
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        
        if terminated or truncated:
            break
    
    assert len(rewards) > 100


def test_diagnostic_coupled_env():
    env = DiagnosticCoupledEnv()
    obs, info = env.reset()
    
    assert obs.shape == (4,)
    
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert -1.0 <= reward <= 1.0
        
        if terminated or truncated:
            break


def test_diagnostic_env_registration():
    env = gym.make("diagnostic_boundedness")
    assert isinstance(env.unwrapped, DiagnosticBoundednessEnv)
    
    env = gym.make("diagnostic_recovery")
    assert isinstance(env.unwrapped, DiagnosticRecoveryEnv)
    
    env = gym.make("diagnostic_timescale")
    assert isinstance(env.unwrapped, DiagnosticTimescaleEnv)
    
    env = gym.make("diagnostic_coupled")
    assert isinstance(env.unwrapped, DiagnosticCoupledEnv)
