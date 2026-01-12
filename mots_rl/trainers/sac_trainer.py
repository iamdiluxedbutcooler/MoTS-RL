import torch
import numpy as np
from mots_rl.modulators.persona import PersonaModulator
from mots_rl.modulators.shadow import ShadowModulator
from mots_rl.modulators.self_reg import SelfRegulator


class SACTrainer:

    def __init__(self, env, policy, core_dynamics, config, logger=None):
        self.env = env
        self.policy = policy
        self.core = core_dynamics
        self.config = config
        self.logger = logger
        
        self.persona = PersonaModulator(affect_dim=4)
        self.shadow = ShadowModulator(affect_dim=4)
        self.self_reg = SelfRegulator(
            lambda_kl=config.get("lambda_kl", 0.01),
            window_size=config.get("window_size", 1000)
        )
        
        self.optimizer = torch.optim.Adam(
            list(self.core.parameters()) +
            list(self.policy.ego.parameters()) +
            list(self.persona.parameters()) +
            list(self.shadow.parameters()),
            lr=config.get("lr", 3e-4)
        )
        
        self.device = config.get("device", "cpu")
    
    def train(self, total_timesteps, num_seeds=1):
        results = []
        
        for seed in range(num_seeds):
            self._set_seed(seed)
            result = self._train_single_run(total_timesteps, seed)
            results.append(result)
        
        return results
    
    def _set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _train_single_run(self, total_timesteps, seed):
        obs, info = self.env.reset(seed=seed)
        self.core.reset()
        self.policy.ego.reset()
        self.self_reg.reset()
        
        episode_rewards = []
        episode_reward = 0
        affect_history = []
        weight_history = []
        
        for step in range(total_timesteps):
            affect_state = self.core.get_state()
            
            action, weights = self.policy.predict(obs, affect_state, deterministic=False)
            
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            persona_mod = self.persona(affect_state)
            shadow_mod = self.shadow(affect_state)
            
            modulated_reward = reward * persona_mod.item() * shadow_mod.item()
            
            self.core(next_obs, modulated_reward, done)
            
            self.self_reg.update(weights)
            
            affect_history.append(affect_state.detach().cpu().numpy())
            weight_history.append(weights.detach().cpu().numpy())
            
            episode_reward += reward
            
            if done:
                episode_rewards.append(episode_reward)
                obs, info = self.env.reset()
                self.core.reset()
                self.policy.ego.reset()
                episode_reward = 0
            else:
                obs = next_obs
            
            if step % 1024 == 0 and step > 0:
                self._update_parameters()
        
        return {
            "episode_rewards": episode_rewards,
            "affect_history": np.array(affect_history),
            "weight_history": np.array(weight_history),
            "seed": seed
        }
    
    def _update_parameters(self):
        self_reg_loss = self.self_reg.compute_loss()
        
        self.optimizer.zero_grad()
        if self_reg_loss.requires_grad:
            self_reg_loss.backward()
        self.optimizer.step()
