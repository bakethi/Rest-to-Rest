import unittest
import numpy as np
from gym_pathfinding.envs.pathfinding_env import PathfindingEnv
from gym_pathfinding.utils.pathfinding import Pathfinding
from gym_pathfinding.utils.obstacles import ObstacleManager


class TestPathfinding(unittest.TestCase):
    def setUp(self):
        self.pathfinding = Pathfinding(grid_size=(100, 100), world_bounds=np.array([100.0, 100.0]))
        self.obstacle_manager = ObstacleManager()

    def test_obstacle_map_update(self):
        self.obstacle_manager.add_obstacle(np.array([50.0, 50.0]), size=10.0)
        self.pathfinding.update_obstacle_map(self.obstacle_manager)
        grid_pos = self.pathfinding._world_to_grid(np.array([50.0, 50.0]))
        self.assertEqual(self.pathfinding.obstacle_map[grid_pos], 1)

    def test_pathfinding_algorithm(self):
        self.obstacle_manager.add_obstacle(np.array([50.0, 50.0]), size=10.0)
        self.pathfinding.update_obstacle_map(self.obstacle_manager)
        path = self.pathfinding.find_path(np.array([0.0, 0.0]), np.array([99.0, 99.0]))
        self.assertTrue(len(path) > 0)
        self.assertTrue(np.array_equal(path[-1], np.array([99.0, 99.0])))

    def test_no_path_found(self):
        self.obstacle_manager.add_obstacle(np.array([50.0, 50.0]), size=100.0)
        self.pathfinding.update_obstacle_map(self.obstacle_manager)
        path = self.pathfinding.find_path(np.array([0.0, 0.0]), np.array([99.0, 99.0]))
        self.assertEqual(len(path), 0)

    def test_edge_cases(self):
        self.assertEqual(len(self.pathfinding.find_path(np.array([0.0, 0.0]), np.array([0.0, 0.0]))), 0)
        self.assertRaises(Exception, self.pathfinding.find_path, np.array([-1.0, -1.0]), np.array([99.0, 99.0]))


class TestPathfindingEnv(unittest.TestCase):
    def setUp(self):
        self.env = PathfindingEnv()
        self.env.reset()

    def test_environment_reset(self):
        obs = self.env.reset()
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
