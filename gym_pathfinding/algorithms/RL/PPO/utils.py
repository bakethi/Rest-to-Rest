import numpy as np

def discount_rewards(rewards, gamma=0.99):
    """Compute discounted future rewards."""
    discounted = np.zeros_like(rewards, dtype=np.float32)
    running_sum = 0
    for t in reversed(range(len(rewards))):
        running_sum = rewards[t] + gamma * running_sum
        discounted[t] = running_sum
    return discounted

def normalize(x):
    """Normalize a NumPy array."""
    return (x - np.mean(x)) / (np.std(x) + 1e-8)
