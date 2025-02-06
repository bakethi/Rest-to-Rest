import unittest
import numpy as np
import pygame
import random
from gym_pathfinding.envs import Renderer
from gym_pathfinding.envs.pathfinding_env import PathfindingEnv
from gym_pathfinding.envs.physics import PhysicsObject
from gym_pathfinding.utils.obstacles import ObstacleManager


class TestVisualization(unittest.TestCase):
    def setUp(self):
        # Initialize the environment and visualization objects
        self.env = PathfindingEnv()
        self.visualization = Renderer(self.env)
        self.agent = PhysicsObject([1, 1])
        self.obstacle_manager = ObstacleManager(world_bounds=np.array([100.0, 100.0]))
        self.target = [10, 10]
        
    def test_initial_rendering(self):
        # Test the initial rendering of the environment
        self.env.reset()  # Reset the environment
        try:
            self.visualization.render(self.agent, self.obstacle_manager, self.target)  # Render the initial state
        except Exception as e:
            self.fail(f"Initial rendering failed with exception: {e}")
        
    def test_update_rendering(self):
        # Test that the environment updates the rendering after each action
        self.env.reset()
        self.env.agent.position= (np.array([1.0, 0.0], dtype=np.float32))  # Move the agent
        try:
            self.visualization.render(self.agent, self.obstacle_manager, self.target)  # Render updated state
            for i in range(100):
                self.env.agent.position= (np.array([(1.0+ i), (1.0+ i)], dtype=np.float32))
                self.visualization.render(self.env.agent, self.obstacle_manager, self.target)
        except Exception as e:
            self.fail(f"Updated rendering failed with exception: {e}")

    def test_path_rendering(self):
        # Test if the path is correctly rendered
        self.env.reset()
        self.env.agent.path_history = np.array([
            [0, 100],
            [75, 54],
            [34, 42],
            [35, 31],
            [49, 82],
            [5, 41],
            [100, 0],
        ], dtype=np.float32)
        try:
            self.visualization.render(self.env.agent, self.obstacle_manager, self.target)  # Render the path
            pygame.time.wait(5000)
        except Exception as e:
            self.fail(f"Path rendering failed with exception: {e}")

    def test_obstacle_rendering(self):
        # Test if obstacles are rendered correctly
        self.env.reset()
        self.env.obstacle_manager.add_obstacle(np.array([50.0, 50.0]), size=10.0)  # Add obstacle
        try:
            self.visualization.render(self.agent, self.obstacle_manager, self.target)  # Render the environment with obstacles
        except Exception as e:
            self.fail(f"Obstacle rendering failed with exception: {e}")

    def test_rendering_performance(self):
        # Test the performance of rendering during continuous updates
        self.env.reset()
        for _ in range(100):  # Perform many steps
            self.env.step(np.array([1.0, 0.0]))  # Move the agent
            try:
                self.visualization.render(self.agent, self.obstacle_manager, self.target)  # Render updated state
            except Exception as e:
                self.fail(f"Rendering failed during performance test with exception: {e}")

    def test_rendering_window(self):
        # Ensure the rendering window appears and stays open
        try:
            self.visualization.render(self.agent, self.obstacle_manager, self.target)  # Render the environment once
            #pygame.time.wait(5000)  # Wait 500ms to see the window
        except Exception as e:
            self.fail(f"Rendering window test failed with exception: {e}")

    def test_rendering_with_target(self):
        # Ensure that the target is rendered and stays visible
        self.env.reset()
        try:
            self.visualization.render(self.agent, self.obstacle_manager, self.target)  # Ensure target is visible after reset
        except Exception as e:
            self.fail(f"Rendering with target failed with exception: {e}")

    def test_path_updates(self):
        # Test that the environment updates the rendering after each action
        self.env.reset()
        self.env.agent.position= (np.array([1.0, 0.0], dtype=np.float32))  # Move the agent
        try:
            self.visualization.render(self.agent, self.obstacle_manager, self.target)  # Render updated state
            for i in range(100):
                self.env.agent.position = (np.array([(1.0+ i), (1.0+ i)], dtype=np.float32))
                self.env.agent.path_history.append(np.array([(1.0+ i), (1.0+ i)], dtype=np.float32))
                self.visualization.render(self.env.agent, self.obstacle_manager, self.target)
        except Exception as e:
            self.fail(f"Updated rendering failed with exception: {e}")

    def test_multiple_obstacles(self):
        # Test if obstacles are rendered correctly
        print("Testing multiple_obstacles")
        self.env.reset()
        self.obstacle_manager.generate_random_obstacles(10)
        try:
            self.visualization.render(self.agent, self.obstacle_manager, self.target)  # Render the environment with obstacles
            pygame.time.wait(5000)
        except Exception as e:
            self.fail(f"Obstacle rendering failed with exception: {e}")


if __name__ == "__main__":
    unittest.main()
