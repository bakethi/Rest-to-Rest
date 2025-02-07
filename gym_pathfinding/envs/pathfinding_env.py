import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .physics import PhysicsObject, PhysicsEngine
from ..utils.obstacles import ObstacleManager


class PathfindingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, number_of_obstacles=1, bounds=np.array([[0, 0], [100, 100]]), bounce_factor=1):
        super(PathfindingEnv, self).__init__()

        # Define action and observation spaces
        # Actions: Acceleration in x and y (range -1 to 1)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observations: [x, y, vx, vy] (position and velocity of agent)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -np.inf, -np.inf], dtype=np.float32),
            high=np.array([100, 100, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32
        )
        self.number_of_obstacles = number_of_obstacles
        self.bounds = bounds
        self.bounce_factor = bounce_factor

        # Initialize environment components
        self.obstacle_manager = ObstacleManager()
        self.generate_random_obstacles()
        self.agent = PhysicsObject(
            position=np.array([0.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            obstacleManager=self.obstacle_manager,
            bounds=self.bounds,
            bounce_factor=self.bounce_factor,
        )
        self.engine = PhysicsEngine([self.agent])


        # Target position
        self.target_position = np.array([90.0, 90.0])

        # Renderer (initialized later when render is called)
        self.renderer = None

    def reset(self):
        """
        Reset the environment to its initial state.
        Returns:
            np.array: Initial observation
        """
        self.agent.reset(position=np.array([0.0, 0.0]), velocity=np.array([0.0, 0.0]))
        self.obstacle_manager.reset()
        return self._get_observation()

    def step(self, action):
        """
        Apply the given action and advance the environment by one step.

        Args:
            action (np.array): Acceleration in x and y.

        Returns:
            tuple: (observation, reward, done, info)
        """
        # Apply action to agent
        self.agent.apply_force(action)
        self.engine.update()

        # Check if the episode is done
        done = self._check_done()

        # Compute reward
        reward = self._compute_reward(done)

        # Return the observation, reward, done flag, and info
        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        """
        Generate the current observation of the environment.

        Returns:
            np.array: [x, y, vx, vy]
        """
        return np.concatenate([self.agent.position, self.agent.velocity])

    def _check_done(self):
        """
        Check if the episode is done.

        Returns:
            bool: True if done, False otherwise
        """
        # Check if agent reaches the target
        if np.linalg.norm(self.agent.position - self.target_position) < 1.0:
            return True

        # Check if agent collides with any obstacles
        if self.obstacle_manager.check_collision(self.agent):
            return True

        return False

    def _compute_reward(self, done):
        """
        Compute the reward for the current step.

        Args:
            done (bool): Whether the episode is done.

        Returns:
            float: The computed reward.
        """
        if done:
            # Positive reward for reaching the target
            if np.linalg.norm(self.agent.position - self.target_position) < 1.0:
                return 100.0
            # Negative reward for collision
            return -100.0

        # Negative reward proportional to the distance to the target
        return -np.linalg.norm(self.agent.position - self.target_position)

    def generate_random_obstacles(self):
        self.obstacle_manager.generate_random_obstacles(self.number_of_obstacles)
        if self.obstacle_manager.obstacles == []:
            raise ValueError("Generating obstacles failed!")
