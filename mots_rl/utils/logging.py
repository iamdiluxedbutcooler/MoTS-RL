import os
import json
import pickle
import numpy as np
from pathlib import Path


class Logger:

    def __init__(self, log_dir, use_wandb=False, wandb_project=None, wandb_entity=None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_wandb = use_wandb
        self.wandb_run = None
        
        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(project=wandb_project, entity=wandb_entity)
            except ImportError:
                self.use_wandb = False
        
        self.metrics = {}
        self.step = 0
    
    def log_scalar(self, key, value, step=None):
        if step is None:
            step = self.step
        
        if key not in self.metrics:
            self.metrics[key] = []
        
        self.metrics[key].append((step, value))
        
        if self.use_wandb and self.wandb_run is not None:
            self.wandb_run.log({key: value}, step=step)
    
    def log_dict(self, metrics_dict, step=None):
        for key, value in metrics_dict.items():
            self.log_scalar(key, value, step)
    
    def save_metrics(self, filename="metrics.json"):
        filepath = self.log_dir / filename
        with open(filepath, "w") as f:
            json.dump(self.metrics, f, indent=2)
    
    def save_numpy(self, data, filename):
        filepath = self.log_dir / filename
        np.save(filepath, data)
    
    def save_pickle(self, data, filename):
        filepath = self.log_dir / filename
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
    
    def increment_step(self):
        self.step += 1
    
    def close(self):
        self.save_metrics()
        if self.use_wandb and self.wandb_run is not None:
            self.wandb_run.finish()
