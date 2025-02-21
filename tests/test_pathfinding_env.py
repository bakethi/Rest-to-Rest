import unittest
import numpy as np
from gym_pathfinding.envs.pathfinding_env import PathfindingEnv
from gym_pathfinding.utils.obstacles import ObstacleManager
from gym_pathfinding.utils.pathfinding import is_target_reachable
from gymnasium.utils.env_checker import check_env


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
        obs, reward, terminated, _, _ = self.env.step(action)
        self.assertTrue(obs is not None)
        self.assertTrue(isinstance(reward, float))
        self.assertTrue(isinstance(terminated, bool))

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

class TestGymCompatibility(unittest.TestCase):
    def test_gym_checker(self):
        env = PathfindingEnv()
        check_env(env, warn=True)

    def test_action_space(self):
        env = PathfindingEnv()
        action = env.action_space.sample()
        assert env.action_space.contains(action), "action space does not contain action"

    def test_observation_space(self):
        env = PathfindingEnv()
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)

    def test_reset_output(self):
        """Ensure reset() returns (obs, info) and obs is valid."""
        env = PathfindingEnv()
        result = env.reset()

        # Ensure reset() returns a tuple
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 2, f"Expected 2-tuple, got length {len(result)}"

        obs, info = result
        print("\nReturned observation:", obs)
        print("Observation shape:", obs.shape)
        print("Expected space shape:", env.observation_space.shape)
        print("Does observation fit in space?", env.observation_space.contains(obs))
        # check data type
        assert obs.dtype == np.float32, f"Observation dtype is {obs.dtype}, expected np.float32"

        # Check observation validity
        assert env.observation_space.contains(obs), "Observation not within observation space"

        # Ensure info is a dictionary
        assert isinstance(info, dict), f"Expected dict for info, got {type(info)}"

if __name__ == "__main__":
    unittest.main()
