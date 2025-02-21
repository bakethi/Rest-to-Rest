import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .ppo_agent import PPOAgent


class PPOTrainer:
    def __init__(self, env, gamma=0.99, clip_epsilon=0.2, update_epochs=10):
        obs_dim = 2 + 1 + env.num_lidar_scans  # Automatically adjust observation size
        self.env = env
        action_dim = 2
        self.agent = PPOAgent(env, action_dim)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.value_function = nn.Linear(obs_dim, 1)  # Simple value function
        self.optimizer_vf = optim.Adam(self.value_function.parameters(), lr=3e-4)

    def compute_advantages(self, rewards, values, dones):
        """Compute advantage estimates using GAE (Generalized Advantage Estimation)"""
        advantages = []
        advantage = 0
        for t in reversed(range(len(rewards))):
            td_error = rewards[t] + (1 - dones[t]) * self.gamma * values[t + 1] - values[t]
            advantage = td_error + self.gamma * 0.95 * (1 - dones[t]) * advantage
            advantages.insert(0, advantage)
        return np.array(advantages)

    def train(self, max_episodes=1000, rollout_size=2048, batch_size=64):
        for episode in range(max_episodes):
            obs, _ = self.env.reset()
            observations, actions, log_probs, rewards, values, dones = [], [], [], [], [], []

            # Collect rollout data
            for _ in range(rollout_size):
                action, log_prob = self.agent.select_action(obs)
                value = self.value_function(torch.tensor(obs, dtype=torch.float32)).item()
                next_obs, reward, done, truncated, _ = self.env.step(action)

                observations.append(obs)
                actions.append(action)
                log_probs.append(float(log_prob))
                rewards.append(reward)
                values.append(value)
                dones.append(bool(done))

                obs = next_obs
                if done:
                    obs = self.env.reset()

            values.append(self.value_function(torch.tensor(obs, dtype=torch.float32)).item())
            advantages = self.compute_advantages(rewards, values, dones)

            # Convert to PyTorch tensors
            observations = torch.tensor(observations, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.float32)
            # Ensure all elements in log_probs are valid floats before conversion
            if isinstance(log_probs, list) and all(isinstance(lp, (float, int)) for lp in log_probs):
                log_probs = torch.tensor(log_probs, dtype=torch.float32)
            else:
                raise ValueError(f"log_probs contains invalid data: {log_probs}")
            advantages = torch.tensor(advantages, dtype=torch.float32)

            # Train using PPO's loss function
            for _ in range(self.update_epochs):
                log_probs_new, entropy = self.agent.evaluate_actions(observations, actions)
                ratio = torch.exp(log_probs_new - log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

                policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
                values = torch.tensor(values, dtype=torch.float32)  # Ensure `values` is a tensor
                value_loss = nn.functional.mse_loss(self.value_function(observations).squeeze(), advantages + values[:-1])


                self.agent.optimizer.zero_grad()
                policy_loss.backward()
                self.agent.optimizer.step()

                self.optimizer_vf.zero_grad()
                value_loss.backward()
                self.optimizer_vf.step()

            print(f"Episode {episode} - Policy Loss: {policy_loss.item()}, Value Loss: {value_loss.item()}")
