import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DiagnosticBoundednessEnv(gym.Env):

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-10, high=10, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        
        self.state = np.zeros(4, dtype=np.float32)
        self.step_count = 0
        self.max_steps = 10000
        self.reward_mode = "adversarial"
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(-1, 1, size=4).astype(np.float32)
        self.step_count = 0
        return self.state, {}
    
    def step(self, action):
        self.step_count += 1
        
        if self.reward_mode == "constant_max":
            reward = 1.0
        elif self.reward_mode == "oscillating":
            reward = np.sin(self.step_count * 0.1)
        elif self.reward_mode == "random":
            reward = self.np_random.uniform(-1, 1)
        elif self.reward_mode == "adversarial":
            reward = 1.0 if self.step_count % 10 < 5 else -1.0
        else:
            reward = 0.0
        
        self.state = self.np_random.uniform(-1, 1, size=4).astype(np.float32)
        
        terminated = self.step_count >= self.max_steps
        truncated = False
        
        return self.state, reward, terminated, truncated, {}


class DiagnosticRecoveryEnv(gym.Env):

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-10, high=10, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        
        self.state = np.zeros(4, dtype=np.float32)
        self.step_count = 0
        self.max_steps = 10000
        self.perturb_step = 100
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(-1, 1, size=4).astype(np.float32)
        self.step_count = 0
        return self.state, {}
    
    def step(self, action):
        self.step_count += 1
        
        if self.step_count == self.perturb_step:
            reward = 10.0
        else:
            reward = 0.0
        
        self.state = self.np_random.uniform(-1, 1, size=4).astype(np.float32)
        
        terminated = self.step_count >= self.max_steps
        truncated = False
        
        return self.state, reward, terminated, truncated, {}


class DiagnosticTimescaleEnv(gym.Env):

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-10, high=10, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        
        self.state = np.zeros(4, dtype=np.float32)
        self.step_count = 0
        self.max_steps = 10000
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(-1, 1, size=4).astype(np.float32)
        self.step_count = 0
        return self.state, {}
    
    def step(self, action):
        self.step_count += 1
        
        reward = (
            0.5 * np.sin(self.step_count * 0.01) +
            0.3 * np.sin(self.step_count * 0.1) +
            0.2 * np.sin(self.step_count * 1.0)
        )
        
        self.state = self.np_random.uniform(-1, 1, size=4).astype(np.float32)
        
        terminated = self.step_count >= self.max_steps
        truncated = False
        
        return self.state, reward, terminated, truncated, {}


class DiagnosticCoupledEnv(gym.Env):

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-10, high=10, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        
        self.state = np.zeros(4, dtype=np.float32)
        self.step_count = 0
        self.max_steps = 10000
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(-1, 1, size=4).astype(np.float32)
        self.step_count = 0
        return self.state, {}
    
    def step(self, action):
        self.step_count += 1
        
        base_reward = self.np_random.uniform(-1, 1)
        
        reward = base_reward
        
        self.state = self.np_random.uniform(-1, 1, size=4).astype(np.float32)
        
        terminated = self.step_count >= self.max_steps
        truncated = False
        
        return self.state, reward, terminated, truncated, {}


gym.register(
    id="diagnostic_boundedness",
    entry_point="mots_rl.envs.diagnostic:DiagnosticBoundednessEnv",
)

gym.register(
    id="diagnostic_recovery",
    entry_point="mots_rl.envs.diagnostic:DiagnosticRecoveryEnv",
)

gym.register(
    id="diagnostic_timescale",
    entry_point="mots_rl.envs.diagnostic:DiagnosticTimescaleEnv",
)

gym.register(
    id="diagnostic_coupled",
    entry_point="mots_rl.envs.diagnostic:DiagnosticCoupledEnv",
)
