import unittest
import numpy as np
from gym_pathfinding.envs.pathfinding_env import PathfindingEnv
from gym_pathfinding.utils.obstacles import ObstacleManager
from gym_pathfinding.utils.pathfinding import is_target_reachable


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

class TestPathfindingEnvSanityCheck(unittest.TestCase):
    def setUp(self):
        """Initialize the environment before each test."""
        self.env = PathfindingEnv()
        self.env.reset()

    def test_no_obstacle_at_agent_start(self):
        """Ensure that no obstacles are placed at the agent's starting position."""
        agent_pos = self.env.agent.position
        self.assertFalse(
            self.env.obstacle_manager.check_collision_of_point(agent_pos),
            "Obstacle detected at agent's starting position!"
        )

    def test_no_obstacle_at_target(self):
        """Ensure that no obstacles are placed at the target position."""
        target_pos = self.env.target_position
        self.assertFalse(
            self.env.obstacle_manager.check_collision_of_point(target_pos),
            "Obstacle detected at the target position!"
        )

    def test_target_is_reachable(self):
        """Ensure that the target is reachable from the agent's starting position."""
        self.assertTrue(
            is_target_reachable(
                self.env.agent.position,
                self.env.target_position,
                self.env.obstacle_manager,
                self.env.bounds
                ),
            "Target is not reachable from the agent's starting position!"
        )

    def test_target_not_reachable(self):
        """Ensure that the target is not reachable when surrounded by obstacles."""
        # Clear all existing obstacles
        self.env.obstacle_manager.reset()

        # Place obstacles in a way that completely surrounds the target
        target_pos = self.env.target_position
        obstacle_positions = [
            target_pos + np.array([1, 0]),  # Right
            target_pos + np.array([-1, 0]), # Left
            target_pos + np.array([0, 1]),  # Above
            target_pos + np.array([0, -1])  # Below
        ]

        for pos in obstacle_positions:
            self.env.obstacle_manager.add_obstacle(position=pos, size=1.5, shape_type="square")

        # Now check if target is reachable (should be False)
        self.assertFalse(
            is_target_reachable(
                self.env.agent.position,
                self.env.target_position,
                self.env.obstacle_manager,
                self.env.bounds
            ),
            "Target should NOT be reachable, but it is!"
        )



if __name__ == "__main__":
    unittest.main()
