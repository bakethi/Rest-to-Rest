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
            max_acceleration=5,
            terminate_on_collision=True,
            scaling_factor=0.2,
            random_start_target=False,
            goal_radius=5.0,
            max_collisions = None,
            obstacle_min_size = 1.0,
            obstacle_max_size = 5.0
            ):
        super(PathfindingEnv, self).__init__()

        # Renderer (initialized later when render is called)
        self.obstacle_min_size = obstacle_min_size
        self.obstacle_max_size = obstacle_max_size
        self.number_of_collisions = 0
        self.max_collisions = max_collisions
        self.goal_radius = goal_radius
        self.random_start_target=random_start_target
        self.scaling_factor=scaling_factor
        self.terminate_on_collision = terminate_on_collision
        self.renderer = None
        self.num_lidar_scans = num_lidar_scans
        self.lidar_max_range = lidar_max_range
        self.distTarget = None
        self.ray_collisions = None
        self.max_acceleration = max_acceleration

        # Define action and observation spaces
        # Actions: Acceleration in x and y (range -1 to 1)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observations: (velocity of agent, distance to the target and the lidar sensor readings)
        self.observation_space = spaces.Box(
            low=np.concatenate(([-np.inf, -np.inf], [0], [-1], np.full(self.num_lidar_scans, 0, dtype=np.float32))),  
            high=np.concatenate(([np.inf, np.inf], [1], [1], np.full(self.num_lidar_scans, 1, dtype=np.float32))),  
            dtype=np.float32
        )


        self.number_of_obstacles = number_of_obstacles
        self.bounds = bounds
        self.bounce_factor = bounce_factor

        # Initialize environment components
        self.obstacle_manager = ObstacleManager(self.bounds)


        self.agent = PhysicsObject(
            position=np.array([0.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            obstacleManager=self.obstacle_manager,
            bounds=self.bounds,
            bounce_factor=self.bounce_factor,
        )
        self.engine = PhysicsEngine([self.agent])

        # Target position
        self.generate_target_agent_pos()


        self.generate_random_obstacles()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent.reset(position=np.array([0.0, 0.0]), velocity=np.array([0.0, 0.0]))
        self.number_of_collisions = 0
        self.generate_target_agent_pos()
        self.obstacle_manager.reset()
        self.generate_random_obstacles()
        return self._get_observation(), {}


    def step(self, action):
        """
        Apply the given action and advance the environment by one step.

        Args:
            action (np.array): Acceleration in x and y.

        Returns:
            tuple: (observation, reward, done, info)
        """
        action = np.clip(action, -self.max_acceleration, self.max_acceleration)  # Enforce limits
        # Apply action to agent
        self.agent.apply_force(action)
        self.engine.update()

        # Check if a collision occurred
        collision_occurred = self.obstacle_manager.check_collision(self.agent)

        if collision_occurred:
            self.number_of_collisions += 1

        # Compute reward
        reward = self._compute_reward(collision_occurred)

        #add truncated
        truncated = False


        # Return observation, reward, done flag, and info dictionary
        info = {
            "collision": collision_occurred
        }

        # Check if the episode is done
        terminated = self._check_done()


        # Return the observation, reward, done flag, and info
        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self):
        """
        Generate the current observation of the environment.

        Returns:
            np.array: [x, y, vx, vy]
        """
        self.distTarget = np.array([self._getAgentTargetDist()], dtype=np.float32)
        self.ray_collisions = np.array(self.cast_rays_until_collision(), dtype=np.float32)
        distances = np.linalg.norm(self.ray_collisions - self.agent.position, axis=1)
        normalized_distances = np.clip(distances / self.lidar_max_range, 0, 1).astype(np.float32)
        angle_to_target = np.array([self._get_agent_target_angle()], dtype=np.float32)

        return np.concatenate([self.agent.velocity, self.distTarget, angle_to_target, normalized_distances.flatten()])

    def _check_done(self):
        """
        Check if the episode is done.

        Returns:
            bool: True if done, False otherwise
        """
        # Check if agent reaches the target
        if np.linalg.norm(self.agent.position - self.target_position) < self.goal_radius:
            return True

        # Check if agent collides with any obstacles
        if self.terminate_on_collision and self.obstacle_manager.check_collision(self.agent):
            return True
        
        if self.max_collisions is not None:
            if self.number_of_collisions >= self.max_collisions and self.obstacle_manager.check_collision(self.agent):
                return True

        return False

    def _compute_reward(self, collision_occurred):
        """
        Compute the reward for the current step.

        Args:
            done (bool): Whether the episode is done.

        Returns:
            float: The computed reward.
        """
        reward = -np.linalg.norm(self.agent.position - self.target_position) * 0.1  # Scale-down distance penalty
        
        if collision_occurred:
            reward -= 100

        # 🚀 ADD: Small penalty if the agent doesn't move
        if np.linalg.norm(self.agent.velocity) < 0.01:
            reward -= 1  # Penalize staying still

        # ✅ If the agent reached the target, give it the final reward BEFORE termination
        if np.linalg.norm(self.agent.position - self.target_position) < self.goal_radius:
            reward += 100.0  # Ensure success reward is given


        return reward


    def generate_random_obstacles(self):
        self.obstacle_manager.generate_random_obstacles(
            self.number_of_obstacles,
            agent_position=self.agent.position,
            target_position=self.target_position,
            bounds=self.bounds,
            min_size=self.obstacle_min_size,
            max_size=self.obstacle_max_size
        )


    def _getAgentTargetDist(self):
        """Returns the normalized distance from p1 to p2 in 2D space."""
        agent_Pos = np.array(self.agent.position)
        target_Pos = np.array(self.target_position)
        # Compute the Euclidean distance between the points
        distance = np.linalg.norm(target_Pos - agent_Pos)

        # Define a maximum possible distance (for normalization)
        # Get the two opposite corners
        corner1 = np.array(self.bounds[0], dtype=np.float32)
        corner2 = np.array(self.bounds[1], dtype=np.float32)

        # Calculate the Euclidean distance between the corners
        max_distance = np.linalg.norm(corner2 - corner1)

        # Normalize the distance
        normalized_distance = distance / max_distance
        
        return normalized_distance

    def cast_rays_until_collision(self):
        """Casts rays until they hit an obstacle using optimized intersection calculations."""
        rays = []
        angle_step = 2 * np.pi / self.num_lidar_scans  # Divide full circle

        for i in range(self.num_lidar_scans):
            angle = i * angle_step
            direction = np.array([np.cos(angle), np.sin(angle)])  # Ray direction unit vector
            origin = self.agent.position

            min_dist = self.lidar_max_range  # Max range initially
            hit_point = origin + direction * self.lidar_max_range  # Default end point

            for obstacle in self.obstacle_manager.get_obstacles():
                if obstacle["type"] == "circle":
                    intersection = self.ray_circle_intersection(origin, direction, obstacle)
                else:
                    intersection = self.ray_square_intersection(origin, direction, obstacle)

                if intersection is not None:
                    dist = np.linalg.norm(intersection - origin)
                    if dist < min_dist:
                        min_dist = dist
                        hit_point = intersection

            # **Ensure ray stays within bounds**
            hit_point = np.clip(hit_point, self.bounds[0], self.bounds[1])
            rays.append(hit_point)
        
        return rays


    def ray_circle_intersection(self, origin, direction, obstacle):
        """Finds intersection of a ray with a circle obstacle."""
        center = obstacle["position"]
        radius = obstacle["size"]

        # Solve quadratic equation for intersection
        oc = origin - center
        a = np.dot(direction, direction)
        b = 2 * np.dot(oc, direction)
        c = np.dot(oc, oc) - radius ** 2

        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0:
            return None  # No intersection

        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)

        # Filter valid intersections (only t > 0)
        valid_t = [t for t in (t1, t2) if t > 0]

        if not valid_t:  # If no valid intersections
            return None

        return origin + min(valid_t) * direction  # Use the closest valid intersection


    def ray_square_intersection(self, origin, direction, obstacle):
        """Finds intersection of a ray with an axis-aligned square obstacle using the slab method."""
        half_size = obstacle["size"] / 2
        lower = obstacle["position"] - half_size
        upper = obstacle["position"] + half_size

        t_min = (lower - origin) / direction
        t_max = (upper - origin) / direction

        t1 = np.minimum(t_min, t_max)
        t2 = np.maximum(t_min, t_max)

        t_near = max(t1[0], t1[1])
        t_far = min(t2[0], t2[1])

        if t_near > t_far or t_far < 0:
            return None  # No valid intersection

        return origin + t_near * direction

    def generate_target_agent_pos(self):
        """Randomizes agent and target positions while ensuring a scaled min distance between them."""
        # Target position
        if self.random_start_target:
            env_width = self.bounds[1][0] - self.bounds[0][0]
            env_height = self.bounds[1][1] - self.bounds[0][1]
            # Compute minimum distance as a fraction of the environment diagonal
            min_distance = self.scaling_factor * np.linalg.norm([env_width, env_height])
            while True:
                self.target_position = np.array([
                                                np.random.randint(self.bounds[0][0], self.bounds[1][0]),  # Random x-coordinate
                                                np.random.randint(self.bounds[0][1], self.bounds[1][1])   # Random y-coordinate
                                                ], dtype=np.float64)
                self.agent.position = np.array([
                                                np.random.randint(self.bounds[0][0], self.bounds[1][0]),  # Random x-coordinate
                                                np.random.randint(self.bounds[0][1], self.bounds[1][1])   # Random y-coordinate
                                                ], dtype=np.float64)
                if np.linalg.norm(self.target_position - self.agent.position) > min_distance:
                    break
        else:
            self.agent.position=np.array([0.0, 0.0])
            self.target_position = np.array([90.0, 90.0])

    def _get_agent_target_angle(self):
        """
        Calculate and normalize the angle between the agent and the target.
        Returns a value between -1 and 1.
        """
        delta_pos = self.target_position - self.agent.position
        angle = np.arctan2(delta_pos[1], delta_pos[0])  # Compute angle in radians
        return angle / np.pi  # Normalize to [-1, 1]
