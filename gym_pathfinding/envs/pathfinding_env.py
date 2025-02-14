import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .physics import PhysicsObject, PhysicsEngine
from ..utils.obstacles import ObstacleManager
import copy


class PathfindingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
            self, 
            number_of_obstacles=1, 
            bounds=np.array([[0, 0], [100, 100]]), 
            bounce_factor=1,
            num_lidar_scans=360,
            lidar_max_range=600,
            lidar_step_size=0.1):
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
        self.num_lidar_scans = num_lidar_scans
        self.lidar_max_range = lidar_max_range
        self.distTarget = None
        self.ray_collisions = None
        self.lidar_step_size = lidar_step_size

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
        self.distTarget = np.array([self._getAgentTargetDist()])
        self.ray_collisions = np.array(self.cast_rays_until_collision())

        return np.concatenate([self.agent.velocity, self.distTarget, self.ray_collisions.flatten()])

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

    def _getAgentTargetDist(self):
        """Returns the normalized distance from p1 to p2 in 2D space."""
        agent_Pos = np.array(self.agent.position)
        target_Pos = np.array(self.target_position)
        # Compute the Euclidean distance between the points
        distance = np.linalg.norm(target_Pos - agent_Pos)

        # Define a maximum possible distance (for normalization)
        # Get the two opposite corners
        corner1 = self.bounds[0]
        corner2 = self.bounds[1]

        # Calculate the Euclidean distance between the corners
        max_distance = np.linalg.norm(corner2 - corner1)

        # Normalize the distance
        normalized_distance = distance / max_distance
        
        return normalized_distance

    def cast_rays_until_collision(self):
        """Casts rays until they hit an obstacle in a 2D grid."""
        rays = []
        angle_step = 2 * np.pi / self.num_lidar_scans  # Divide full circle
        for i in range(self.num_lidar_scans):
            angle = i * angle_step  # Compute angle
            direction = np.array([np.cos(angle), np.sin(angle)])  # Unit vector
            pos = copy.copy(self.agent.position)  # Start position (float for precision)

            # Move along the ray in small steps
            for _ in np.arange(0, self.lidar_max_range, self.lidar_step_size):  # Higher steps for precision
                pos += direction * self.lidar_step_size  # Small step forward

                # Check if the ray goes out of bounds
                if (
                    pos[0] < self.bounds[0][0] or pos[0] >= self.bounds[1][0] or
                    pos[1] < self.bounds[0][1] or pos[1] >= self.bounds[1][1] 
                ):
                    # Apply bounds if provided
                    if self.bounds is not None:
                        pos = np.clip(pos, self.bounds[0], self.bounds[1])
                    rays.append((pos[0], pos[1]))  # Store the hit position
                    break  # Stop if out of bounds

                if self.obstacle_manager is not None:
                    if self.obstacle_manager.check_collision_of_point(pos):
                        rays.append((pos[0], pos[1]))
                        break
        return rays