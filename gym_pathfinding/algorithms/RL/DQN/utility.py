import torch
import numpy as np
import os
import matplotlib.pyplot as plt


# Configuration class for managing hyperparameters
class Config:
    def __init__(
        self,
        batch_size=64,
        gamma=0.99,
        epsilon=0.1,
        learning_rate=1e-3,
        max_episodes=1000,
        target_update_frequency=10,
    ):
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.max_episodes = max_episodes
        self.target_update_frequency = target_update_frequency

    def get(self, param):
        """Get a configuration parameter."""
        return getattr(self, param, None)

    def set(self, param, value):
        """Set a configuration parameter."""
        setattr(self, param, value)


# Plotter class to visualize training progress
class Plotter:
    def __init__(self):
        self.episode_rewards = []
        self.losses = []

    def plot_rewards(self):
        """Plot the total rewards across episodes."""
        plt.plot(self.episode_rewards)
        plt.xlabel('Episodes')
        plt.ylabel('Total Reward')
        plt.title('Episode Rewards')
        plt.show()

    def plot_loss(self):
        """Plot the training loss across episodes."""
        plt.plot(self.losses)
        plt.xlabel('Episodes')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()


# Checkpointer class to save and load models
class Checkpointer:
    def __init__(self, save_path="model_checkpoint.pth"):
        self.save_path = save_path

    def save_model(self, model, optimizer, epoch):
        """Save the model's state dictionary."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, self.save_path)

    def load_model(self, model, optimizer):
        """Load the model's state dictionary."""
        if os.path.exists(self.save_path):
            checkpoint = torch.load(self.save_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            print(f"Model loaded from {self.save_path} (epoch {epoch})")
            return model, optimizer, epoch
        else:
            print(f"No checkpoint found at {self.save_path}")
            return model, optimizer, 0


# Utility function to normalize the state data (example)
def normalize(state, state_min, state_max):
    """Normalize the state to be within a specified range."""
    return (state - state_min) / (state_max - state_min)


# Epsilon-greedy function for action selection
def epsilon_greedy(agent, state, epsilon):
    """Select an action using epsilon-greedy strategy."""
    if np.random.rand() < epsilon:
        return agent.select_random_action()  # Exploration
    else:
        return agent.select_best_action(state)  # Exploitation


# Function to compute various performance metrics
def compute_metrics(total_rewards):
    """Compute average reward and other metrics."""
    avg_reward = np.mean(total_rewards)
    return avg_reward
