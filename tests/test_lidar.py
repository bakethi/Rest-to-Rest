import unittest
import numpy as np
import math
from gym_pathfinding.envs.pathfinding_env import PathfindingEnv


class TestLiDAR(unittest.TestCase):
    def test_setup(self):
        self.env = PathfindingEnv(num_lidar_scans=4, lidar_max_range=5)
        self.assertTrue(self.env.num_lidar_scans == 4, "Number of Lidar Scans wrong")
        self.assertTrue(self.env.lidar_max_range == 5, "Lidar Max Range wrong")

    def test_agent_to_target_distance(self):
        self.env = PathfindingEnv(num_lidar_scans=4, lidar_max_range=5)
        self.env.agent.position = [1, 2]
        self.env.target_position = [4, 6]
        self.assertTrue(self.env._getAgentTargetDist() == (5/math.sqrt(100**2 + 100**2)))

    def test_no_obstacles(self):
        self.env = PathfindingEnv(num_lidar_scans=4, lidar_max_range=5)
        self.env.obstacle_manager.obstacles = []  # No obstacles in the environment
        self.bounds = np.array([[0, 0], [100, 100]])  # Example bounds
        self.num_lidar_scans = 4  # Test with 4 rays
        rays = self.env.cast_rays_until_collision()

        # Assert that the rays are cast until the boundary of the environment
        for ray in rays:
            assert self.bounds[0][0] <= ray[0] <= self.bounds[1][0]
            assert self.bounds[0][1] <= ray[1] <= self.bounds[1][1]


if __name__ == "__main__":
    unittest.main()