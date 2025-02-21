import unittest
import numpy as np
from gym_pathfinding.envs.pathfinding_env import PathfindingEnv
from gym_pathfinding.utils.obstacles import ObstacleManager


class TestPathfindingEnv(unittest.TestCase):
    def setUp(self):
        self.env = PathfindingEnv()
        self.env.reset()

    def test_environment_reset(self):
        self.env.reset()
        self.assertTrue(self.env.agent.position is not None)
        self.assertTrue(self.env.target_position is not None)

    def test_step_function(self):
        action = np.array([1.0, 0.0])  # Example action
        obs, reward, done, _ = self.env.step(action)
        self.assertTrue(obs is not None)
        self.assertTrue(isinstance(reward, float))
        self.assertTrue(isinstance(done, bool))


if __name__ == "__main__":
    unittest.main()
