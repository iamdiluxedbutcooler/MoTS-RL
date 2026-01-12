import pytest
import torch
import gymnasium as gym
from mots_rl.baselines.vanilla import VanillaRLBaseline
from mots_rl.baselines.vanilla_plus import VanillaPlusBaseline
from mots_rl.baselines.icm import ICMBaseline
from mots_rl.baselines.rnd import RNDBaseline
from mots_rl.baselines.dim_affect import DimAffectBaseline
from mots_rl.baselines.rl2 import RL2Baseline


@pytest.fixture
def env():
    return gym.make("CartPole-v1")


def test_vanilla_baseline(env):
    baseline = VanillaRLBaseline(env, policy_type="ppo", device="cpu")
    action = baseline.predict(env.reset()[0])
    assert action is not None


def test_vanilla_plus_baseline(env):
    baseline = VanillaPlusBaseline(env, policy_type="ppo", device="cpu")
    action = baseline.predict(env.reset()[0])
    assert action is not None


def test_icm_baseline(env):
    baseline = ICMBaseline(env, policy_type="ppo", device="cpu")
    action = baseline.predict(env.reset()[0])
    assert action is not None
    
    obs = env.reset()[0]
    next_obs, _, _, _, _ = env.step(action)
    intrinsic_reward = baseline.icm.compute_intrinsic_reward(obs, action, next_obs)
    assert intrinsic_reward >= 0


def test_rnd_baseline(env):
    baseline = RNDBaseline(env, policy_type="ppo", device="cpu")
    action = baseline.predict(env.reset()[0])
    assert action is not None
    
    obs = env.reset()[0]
    intrinsic_reward = baseline.rnd.compute_intrinsic_reward(obs)
    assert intrinsic_reward >= 0


def test_dim_affect_baseline(env):
    baseline = DimAffectBaseline(env, policy_type="ppo", device="cpu")
    action = baseline.predict(env.reset()[0])
    assert action is not None
    
    obs = env.reset()[0]
    affect_state = baseline.core(obs, 1.0, False)
    assert affect_state.shape == (2,)


def test_rl2_baseline(env):
    baseline = RL2Baseline(env, policy_type="ppo", device="cpu")
    obs = env.reset()[0]
    prev_action = env.action_space.sample()
    prev_reward = 0.0
    action = baseline.predict(obs, prev_action, prev_reward)
    assert action is not None


def test_baseline_training_smoke(env):
    baseline = VanillaRLBaseline(env, policy_type="ppo", device="cpu")
    results = baseline.train(total_timesteps=1000, num_seeds=1)
    
    assert len(results) == 1
    assert "episode_rewards" in results[0]
    assert "affect_history" in results[0]
    assert "weight_history" in results[0]
