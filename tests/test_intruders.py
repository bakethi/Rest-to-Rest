import unittest
import numpy as np
import pygame

# Assuming your new env and intruder classes are importable
from gym_pathfinding.envs.intruder_avoidance_env import IntruderAvoidanceEnv
from gym_pathfinding.envs.intruder import Intruder
from gym_pathfinding.envs.visualization import Renderer

class TestIntruderLogic(unittest.TestCase):
    """
    Tests the non-visual, logical behavior of the Intruder class.
    """
    def setUp(self):
        self.bounds = np.array([[0, 0], [100, 100]])
        self.intruder = Intruder(
            initial_position=np.array([50.0, 50.0]),
            bounds=self.bounds,
            change_direction_interval=2.0
        )
        self.agent_pos = np.array([10.0, 10.0])

    def test_intruder_initialization(self):
        """Checks if the intruder and its physics are created correctly."""
        self.assertIsNotNone(self.intruder.physics)
        self.assertEqual(self.intruder.time_since_last_change, 0.0)

    def test_intruder_movement(self):
        """Ensures the intruder moves when its update method is called."""
        initial_pos = self.intruder.physics.position.copy()
        self.intruder.direction = np.array([1, 1])
        # Update for a short time
        self.intruder.update(agent_position=self.agent_pos, dt=0.1)
        self.assertFalse(np.array_equal(initial_pos, self.intruder.physics.position),
                         "Intruder position should change after update.")

    def test_decision_timer(self):
        """Tests that the intruder only changes direction after its interval."""
        # Force an initial direction
        self.intruder.direction = np.array([1.0, 0.0])
        initial_direction = self.intruder.direction.copy()

        # Update for less than the interval, direction should NOT change
        self.intruder.update(agent_position=self.agent_pos, dt=1.0)
        np.testing.assert_array_equal(self.intruder.direction, initial_direction,
                                      "Direction should not change before interval expires.")

        # Update for more than the interval, direction SHOULD change
        self.intruder.update(agent_position=self.agent_pos, dt=1.5)
        self.assertFalse(np.array_equal(self.intruder.direction, initial_direction),
                         "Direction should change after interval expires.")


class TestIntruderVisualization(unittest.TestCase):
    """
    Tests the visual rendering and behavior of intruders in the full environment.
    """
    def setUp(self):
        # Initialize the full environment for visualization
        self.env = IntruderAvoidanceEnv()
        self.renderer = Renderer(self.env) # Your renderer needs to be adapted for the new env

    def test_intruder_rendering(self):
        """Tests if intruders are rendered on the screen without errors."""
        self.env.reset()
        try:
            self.renderer.render(agent= self.env.agent, intruders= self.env.intruders, target_position= self.env.target_position)
            pygame.time.wait(1000) # Keep window open for 1 second
        except Exception as e:
            self.fail(f"Rendering intruders failed with exception: {e}")

    def test_intruder_movement_visualization(self):
        """Runs the simulation for a short period to visually check intruder movement."""
        print("\nRunning visual test for intruder movement. Watch the screen.")
        self.env.reset()
        try:
            # Run for 200 steps
            for _ in range(200):
                # Agent takes a random action, intruders update their logic
                obs, _, _, _, _ = self.env.step(self.env.action_space.sample())
                self.renderer.render(agent= self.env.agent, intruders= self.env.intruders, target_position= self.env.target_position)
                pygame.time.wait(20) # 20ms delay to make movement visible

            pygame.time.wait(2000) # Hold the final frame for 2 seconds
        except Exception as e:
            self.fail(f"Visual movement test failed with exception: {e}")

class TestIntruderLidar(unittest.TestCase):
    """
    Tests that the agent's LiDAR correctly detects intruders.
    """
    def setUp(self):
        self.env = IntruderAvoidanceEnv(number_of_intruders=0) # Start with no intruders
        self.env.reset()

def test_lidar_detects_intruder(self):
    """
    Places a single intruder in front of the agent and checks if the LiDAR ray distance is correct.
    """
    print("\nRunning test for LiDAR detection of a single intruder.")
    # --- 1. Arrange (This part is correct) ---
    self.env.agent.position = np.array([50.0, 50.0])
    self.env.intruders = []
    intruder_pos = np.array([75.0, 50.0])
    intruder_size = 10.0
    intruder = Intruder(initial_position=intruder_pos, bounds=self.env.bounds)
    intruder.physics.size = intruder_size
    self.env.intruders.append(intruder)

    # --- 2. Act (This part is correct) ---
    observation = self.env._get_observation()
    lidar_readings = observation[4:]
    
    # --- 3. Assert (This part needs the fix) ---
    
    # Check the ray pointing right (this part was already correct)
    ray_index_for_right = 0
    expected_distance_to_intruder = intruder_pos[0] - self.env.agent.position[0] - (intruder_size / 2)
    expected_norm_dist_to_intruder = expected_distance_to_intruder / self.env.lidar_max_range
    self.assertAlmostEqual(
        lidar_readings[ray_index_for_right], 
        expected_norm_dist_to_intruder, 
        places=5,
        msg="LiDAR ray pointing at intruder reported an incorrect distance."
    )

    # --- THIS IS THE CORRECTED ASSERTION ---
    # The ray pointing left hits the wall at x=0.
    ray_index_for_left = self.env.num_lidar_scans // 2
    
    # Calculate the expected distance to the left wall.
    expected_distance_to_wall = self.env.agent.position[0] - self.env.bounds[0][0] # 50 - 0 = 50
    expected_normalized_distance_to_wall = expected_distance_to_wall / self.env.lidar_max_range # 50 / 60 = 0.8333...

    self.assertAlmostEqual(
        lidar_readings[ray_index_for_left], 
        expected_normalized_distance_to_wall, # Use the correct expected value
        places=5,
        msg="LiDAR ray pointing away from intruder should report distance to the wall."
    )


if __name__ == "__main__":
    unittest.main()