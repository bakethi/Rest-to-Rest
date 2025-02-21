import unittest
import torch
import numpy as np
from gym_pathfinding.algorithms.RL.PPO.policy import PPOPolicy
from gym_pathfinding.envs.pathfinding_env import PathfindingEnv
from gym_pathfinding.algorithms.RL.PPO.ppo_agent import PPOAgent
from gym_pathfinding.algorithms.RL.PPO.ppo_trainer import PPOTrainer


class TestPPOTrainer(unittest.TestCase):
    def setUp(self):
        """Initialize environment and PPO trainer before each test."""
        self.env = PathfindingEnv(num_lidar_scans=180)  # Dynamic LiDAR size
        self.trainer = PPOTrainer(env=self.env)

    def test_trainer_initialization(self):
        """Ensure the PPO trainer initializes correctly."""
        self.assertIsInstance(self.trainer, PPOTrainer)
        self.assertEqual(self.trainer.agent.policy.fc1.in_features, 2 + 1 + self.env.num_lidar_scans,
                         "Incorrect input dimension in the policy network.")

    def test_rollout_collection(self):
        """Test if the trainer collects rollouts properly."""
        try:
            obs, _ = self.env.reset()
            observations, actions, log_probs, rewards, values, dones = [], [], [], [], [], []

            for _ in range(10):  # Simulate 10 steps
                action, log_prob = self.trainer.agent.select_action(obs)
                value = self.trainer.value_function(torch.tensor(obs, dtype=torch.float32)).item()
                next_obs, reward, done, truncated, _ = self.env.step(action)

                observations.append(obs)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                values.append(value)
                dones.append(done)

                obs = next_obs
                if done:
                    obs, _ = self.env.reset()

            self.assertEqual(len(observations), 10, "Incorrect number of observations collected.")
            self.assertEqual(len(actions), 10, "Incorrect number of actions collected.")
            self.assertEqual(len(rewards), 10, "Incorrect number of rewards collected.")
        except Exception as e:
            self.fail(f"Rollout collection failed with exception: {e}")

    def test_compute_advantages(self):
        """Ensure advantage computation works correctly."""
        rewards = np.array([1, 1, 1, 1, 1], dtype=np.float32)
        values = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)  # Extra value for bootstrapping
        dones = np.array([0, 0, 0, 0, 1], dtype=np.int32)  # Mark episode end

        try:
            advantages = self.trainer.compute_advantages(rewards, values, dones)
            self.assertEqual(advantages.shape, (5,), "Advantage array shape is incorrect.")
            self.assertTrue(np.all(advantages > 0), "Advantages should be positive.")
        except Exception as e:
            self.fail(f"Advantage computation failed with exception: {e}")

    def test_training_step(self):
        """Ensure that a single training step runs without errors."""
        try:
            self.trainer.train(max_episodes=1, rollout_size=10, batch_size=5)
        except Exception as e:
            self.fail(f"Training step failed with exception: {e}")

    def test_trainer_with_different_lidar_scans(self):
        """Ensure the trainer adapts to different LiDAR scan configurations."""
        for lidar_scans in [90, 180, 360, 720]:
            env = PathfindingEnv(num_lidar_scans=lidar_scans)
            trainer = PPOTrainer(env=env)

            obs_dim = 2 + 1 + lidar_scans
            self.assertEqual(trainer.agent.policy.fc1.in_features, obs_dim,
                             f"Trainer input size mismatch for LiDAR scans {lidar_scans}.")

            try:
                trainer.train(max_episodes=1, rollout_size=10, batch_size=5)
            except Exception as e:
                self.fail(f"Training step failed for LiDAR scans {lidar_scans} with exception: {e}")

class TestPPOAgent(unittest.TestCase):
    def setUp(self):
        """Initialize environment and PPO agent before each test."""
        self.env = PathfindingEnv(num_lidar_scans=180)  # Dynamic LiDAR size
        self.obs_dim = 2 + 1 + self.env.num_lidar_scans  # Velocity (2) + Distance (1) + LiDAR scans
        self.action_dim = 2  # Acceleration (x, y)
        self.agent = PPOAgent(env=self.env, action_dim=self.action_dim)

    def test_agent_initialization(self):
        """Ensure that the PPO agent initializes correctly."""
        self.assertIsInstance(self.agent, PPOAgent)
        self.assertEqual(self.agent.policy.fc1.in_features, self.obs_dim, "Incorrect input dimension in the policy network.")
        self.assertEqual(self.agent.policy.mean_layer.out_features, self.action_dim, "Incorrect output dimension in the policy network.")

    def test_select_action(self):
        """Test if the agent selects an action correctly."""
        obs = np.random.randn(self.obs_dim).astype(np.float32)  # Random environment observation
        try:
            action, log_prob = self.agent.select_action(obs)
            self.assertEqual(len(action), self.action_dim, "Action dimension mismatch.")
            self.assertEqual(log_prob.shape, (), "Log probability should be a scalar.")
        except Exception as e:
            self.fail(f"Action selection failed with exception: {e}")

    def test_action_range(self):
        """Ensure that selected actions remain within the expected range (-1, 1)."""
        obs = np.random.randn(self.obs_dim).astype(np.float32)
        action, _ = self.agent.select_action(obs)
        max_accel = self.env.max_acceleration
        self.assertTrue(np.all(action >= -max_accel) and np.all(action <= max_accel),
                f"Action values are out of expected range (-{max_accel}, {max_accel}).")


    def test_evaluate_actions(self):
        """Test if log probabilities and entropy are computed correctly."""
        obs = torch.randn((5, self.obs_dim))  # Batch of 5 observations
        actions = torch.randn((5, self.action_dim))  # Batch of 5 random actions

        try:
            log_probs, entropy = self.agent.evaluate_actions(obs, actions)
            self.assertEqual(log_probs.shape, (5,), "Log probability output has incorrect shape.")
            self.assertEqual(entropy.shape, (5,), "Entropy output has incorrect shape.")
        except Exception as e:
            self.fail(f"Action evaluation failed with exception: {e}")

    def test_agent_with_different_lidar_scans(self):
        """Ensure the agent can handle different LiDAR scan configurations."""
        for lidar_scans in [90, 180, 360, 720]:
            env = PathfindingEnv(num_lidar_scans=lidar_scans)
            obs_dim = 2 + 1 + lidar_scans
            agent = PPOAgent(env=env, action_dim=self.action_dim)

            self.assertEqual(agent.policy.fc1.in_features, obs_dim, f"Policy input size mismatch for LiDAR scans {lidar_scans}.")

            obs = np.random.randn(obs_dim).astype(np.float32)
            action, _ = agent.select_action(obs)
            self.assertEqual(len(action), self.action_dim, f"Incorrect action size for LiDAR scans {lidar_scans}.")


class TestPPOPolicy(unittest.TestCase):
    def setUp(self):
        """Initialize the policy network dynamically from the environment."""
        self.env = PathfindingEnv(num_lidar_scans=180)  # Change LiDAR scans for testing
        self.obs_dim = 2 + 1 + self.env.num_lidar_scans
        self.action_dim = 2
        self.policy = PPOPolicy(input_dim=self.obs_dim, output_dim=self.action_dim)

    def test_dynamic_input_size(self):
        """Ensure that the policy network adapts to different LiDAR input sizes."""
        obs = torch.randn(self.obs_dim)
        mean, std = self.policy(obs)
        self.assertEqual(mean.shape, (self.action_dim,))
        self.assertEqual(std.shape, (self.action_dim,))

    def test_policy_with_different_lidar_scans(self):
        """Test policy behavior when LiDAR input size changes."""
        for lidar_scans in [90, 180, 360, 720]:
            self.env.num_lidar_scans = lidar_scans
            obs_dim = 2 + 1 + lidar_scans
            policy = PPOPolicy(input_dim=obs_dim, output_dim=self.action_dim)
            obs = torch.randn(obs_dim)
            mean, std = policy(obs)

            self.assertEqual(mean.shape, (self.action_dim,))
            self.assertEqual(std.shape, (self.action_dim,))


    def test_policy_initialization(self):
        """Ensure that the policy network initializes properly."""
        self.assertIsInstance(self.policy, PPOPolicy)

    def test_forward_pass(self):
        """Test if the network processes observations correctly."""
        obs = torch.randn(self.obs_dim)  # Random test input
        try:
            mean, std = self.policy(obs)
            self.assertEqual(mean.shape, (self.action_dim,))
            self.assertEqual(std.shape, (self.action_dim,))
        except Exception as e:
            self.fail(f"Forward pass failed with exception: {e}")

    def test_action_range(self):
        """Ensure that the action means are within the expected range (-1, 1) due to tanh."""
        obs = torch.randn(self.obs_dim)
        mean, _ = self.policy(obs)
        self.assertTrue(torch.all(mean >= -1) and torch.all(mean <= 1), "Mean values are out of range (-1, 1)")

    def test_standard_deviation_positive(self):
        """Ensure that the standard deviation values are always positive."""
        obs = torch.randn(self.obs_dim)
        _, std = self.policy(obs)
        self.assertTrue(torch.all(std > 0), "Standard deviation contains non-positive values")

    def test_action_sampling(self):
        """Test if an action can be correctly sampled from the Gaussian distribution."""
        obs = torch.randn(self.obs_dim)
        mean, std = self.policy(obs)
        normal_dist = torch.distributions.Normal(mean, std)
        try:
            action = normal_dist.sample()
            self.assertEqual(action.shape, (self.action_dim,))
        except Exception as e:
            self.fail(f"Action sampling failed with exception: {e}")

    def test_log_probability_calculation(self):
        """Check if log probabilities of sampled actions can be computed."""
        obs = torch.randn(self.obs_dim)
        mean, std = self.policy(obs)
        normal_dist = torch.distributions.Normal(mean, std)
        action = normal_dist.sample()
        try:
            log_prob = normal_dist.log_prob(action).sum(dim=-1)
            self.assertEqual(log_prob.shape, ())
        except Exception as e:
            self.fail(f"Log probability computation failed with exception: {e}")


if __name__ == "__main__":
    unittest.main()
