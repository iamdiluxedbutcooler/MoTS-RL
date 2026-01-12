import numpy as np
from collections import deque


class ReplayBuffer:

    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, affect_state, weights):
        self.buffer.append((state, action, reward, next_state, done, affect_state, weights))
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states, actions, rewards, next_states, dones, affect_states, weights = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            np.array(affect_states),
            np.array(weights)
        )
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()
