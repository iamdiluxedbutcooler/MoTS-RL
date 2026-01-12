import torch


class SelfRegulator:

    def __init__(self, lambda_kl=0.01, window_size=1000):
        self.lambda_kl = lambda_kl
        self.window_size = window_size
        self.weight_history = []
    
    def update(self, weights):
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu()
        self.weight_history.append(weights)
        if len(self.weight_history) > self.window_size:
            self.weight_history.pop(0)
    
    def compute_loss(self):
        if len(self.weight_history) == 0:
            return torch.tensor(0.0)
        
        weights_tensor = torch.stack([w for w in self.weight_history])
        mean_weights = weights_tensor.mean(dim=0)
        
        num_experts = mean_weights.shape[0]
        uniform_target = 1.0 / num_experts
        
        kl_div = (mean_weights * torch.log(num_experts * mean_weights + 1e-8)).sum()
        
        return self.lambda_kl * kl_div
    
    def reset(self):
        self.weight_history = []
