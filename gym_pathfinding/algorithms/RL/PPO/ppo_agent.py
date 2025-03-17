import torch
import torch.distributions as dist
from .policy import PPOPolicy


class PPOAgent:
    def __init__(self, env, action_dim, lr=3e-4):
        # Get observation dimension dynamically
        obs_dim = 2 + 1 + 1 + env.num_lidar_scans  # Velocity (2) + Distance (1) + LiDAR readings
        self.policy = PPOPolicy(input_dim=obs_dim, output_dim=action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)


    def select_action(self, obs):
        if isinstance(obs, tuple):  # Ensure we correctly extract the observation
            obs = obs[0]
        obs = torch.tensor(obs, dtype=torch.float32)

        mean, std = self.policy(obs)
        normal_dist = torch.distributions.Normal(mean, std)
        action = normal_dist.sample()  # No tanh() constraint
        log_prob = normal_dist.log_prob(action).sum(dim=-1)

        return action.detach().numpy(), log_prob.detach().numpy()


    def evaluate_actions(self, obs, actions):
        mean, std = self.policy(obs)
        normal_dist = dist.Normal(mean, std)
        log_probs = normal_dist.log_prob(actions).sum(dim=-1)
        entropy = normal_dist.entropy().sum(dim=-1)
        return log_probs, entropy
