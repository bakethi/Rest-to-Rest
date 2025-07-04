import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .physics import PhysicsObject, PhysicsEngine
from ..utils.obstacles import ObstacleManager
from .intruder import Intruder
import copy


class IntruderAvoidanceEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
            self,
            # --- Existing Parameters ---
            number_of_intruders=1,
            bounds=np.array([[0, 0], [100, 100]]),
            bounce_factor=1,
            num_lidar_scans=24,
            lidar_max_range=50,
            max_acceleration=5,
            terminate_on_collision=False,
            scaling_factor=0.2,
            random_start_target=True,
            goal_radius=5.0,
            max_collisions = None,
            obstacle_min_size = 1.0,
            obstacle_max_size = 5.0,
            dt = 0.1,
            max_intruder_speed = 5,
            change_direction_interval = 3,
            agent_size = 10,
            intruder_size = 3,
            number_of_obstacles = 0,
            terminate_on_target_reached = False,

            # --- PBRS Reward Hyperparameters (from Section 6.2) ---
            gamma: float = 0.99,            # Discount factor of the MDP
            # Safety Objective
            r_collision_reward: float = None, # Collision radius for reward (defaults to agent+intruder size)
            d_safe: float = 20.0,           # Radius of the "safety bubble" (meters)
            k_bubble: float = 100.0,        # Penalty magnitude for entering the safety bubble
            k_decay_safe: float = 0.1,      # Penalty decay rate inside the bubble
            C_collision: float = 1000.0,    # Large terminal penalty for a collision
            # Position-Holding Objective
            k_pos: float = 0.005,           # Scaling constant for position penalty
            # Control Effort Objective
            k_action: float = 0.01,         # Scaling constant for action magnitude penalty
            # Aggregation Weights
            w_safe: float = 0.6,            # Weight for the safety potential
            w_pos: float = 0.4              # Weight for the position-holding potential
            ):
        super(IntruderAvoidanceEnv, self).__init__()
        
        # --- Store all existing parameters ---
        self.number_of_intruders = number_of_intruders
        self.bounds = bounds
        # ... (all your other self.variable = variable assignments)
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
        self.dt = dt
        self.max_intruder_speed = max_intruder_speed
        self.change_direction_interval = change_direction_interval
        self.agent_size = agent_size
        self.intruder_size = intruder_size
        self.terminate_on_target_reached = terminate_on_target_reached
        self.number_of_obstacles = number_of_obstacles
        self.max_collisions = max_collisions
        self.obstacle_min_size = obstacle_min_size
        self.obstacle_max_size = obstacle_max_size

        # --- Store PBRS Reward Hyperparameters ---
        self.gamma = gamma
        # If r_collision_reward is not specified, calculate it from agent/intruder sizes
        self.r_collision_reward = r_collision_reward if r_collision_reward is not None else (self.agent_size / 2 + self.intruder_size / 2)
        self.d_safe = d_safe
        self.k_bubble = k_bubble
        self.k_decay_safe = k_decay_safe
        self.C_collision = C_collision
        self.k_pos = k_pos
        self.k_action = k_action
        self.w_safe = w_safe
        self.w_pos = w_pos

        # --- Initialize other components as before ---
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.concatenate(([-np.inf, -np.inf], [0], [-1], np.full(self.num_lidar_scans, 0, dtype=np.float32))),
            high=np.concatenate(([np.inf, np.inf], [1], [1], np.full(self.num_lidar_scans, 1, dtype=np.float32))),
            dtype=np.float32
        )
        self.obstacle_manager = ObstacleManager(self.bounds)
        self.agent = PhysicsObject(
            position=np.array([0.0, 0.0]), velocity=np.array([0.0, 0.0]),
            obstacleManager=self.obstacle_manager, bounds=self.bounds,
            bounce_factor=bounce_factor, size=self.agent_size
        )
        self.engine = PhysicsEngine([self.agent])
        self.generate_target_agent_pos()
        self.generate_random_obstacles()
        self.intruders = []


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent.reset(position=np.array([0.0, 0.0]), velocity=np.array([0.0, 0.0]))
        self.number_of_collisions = 0
        self.generate_target_agent_pos()
        self.obstacle_manager.reset()
        self.generate_random_obstacles()
        self.intruders = [] # Clear old intruders
        for _ in range(self.number_of_intruders):
            # Find a random starting position for the intruder
            intruder_start_pos = np.random.uniform(low=self.bounds[0], high=self.bounds[1])
            
            # Create a new Intruder instance
            new_intruder = Intruder(
                initial_position=intruder_start_pos,
                bounds=self.bounds,
                max_speed=self.max_intruder_speed,
                change_direction_interval=self.change_direction_interval, 
                size=self.intruder_size
            )
            self.intruders.append(new_intruder)
        
        return self._get_observation(), {}


    def step(self, action):
        """
        Apply the given action and advance the environment by one step.
        """
        # --- Store state `s` for PBRS calculation BEFORE the physics step ---
        state_before_step = {
            'agent_pos': np.copy(self.agent.position),
            'intruder_pos_list': [np.copy(intruder.physics.position) for intruder in self.intruders]
        }

        # Apply action and update physics
        scaled_action = action * self.max_acceleration
        self.agent.apply_force(scaled_action)
        self.engine.update(self.dt)

        # Update all intruders
        for intruder in self.intruders:
            intruder.update(self.agent.position, self.dt)

        # Check for collisions
        collision_occurred = self._check_intruder_collisions()
        if collision_occurred:
            self.number_of_collisions += 1

        # --- Compute reward using PBRS ---
        reward = self._compute_reward(
            state_before_step,
            scaled_action,
            collision_occurred
        )

        # Check for termination and truncation
        terminated = self._check_done()
        truncated = False # You can define your own truncation conditions if needed

        info = {"collision": collision_occurred}

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
        if np.linalg.norm(self.agent.position - self.target_position) < self.goal_radius and self.terminate_on_target_reached:
            return True

        # Check if agent collides with any obstacles
        if self.terminate_on_collision and self.obstacle_manager.check_collision(self.agent):
            return True
        if self.terminate_on_collision and self._check_intruder_collisions():
            return True
        
        if self.max_collisions is not None:
            if self.number_of_collisions >= self.max_collisions and self.obstacle_manager.check_collision(self.agent):
                return True
            if self.number_of_collisions >= self.max_collisions and self._check_intruder_collisions():
                return True

        return False

    def _compute_reward(self, old_state: dict, action: np.ndarray, collision_occurred: bool) -> float:
        """
        Calculates the reward for a given transition using the PBRS framework.
        """
        # 1. Primary Objective: Large penalty for collision (terminal state)
        if collision_occurred:
            return -self.C_collision

        # --- Get potential of the state BEFORE the action (`s`) ---
        phi_current = self._total_potential(
            agent_pos=old_state['agent_pos'],
            intruder_pos_list=old_state['intruder_pos_list']
        )

        # --- Get potential of the state AFTER the action (`s'`) ---
        new_intruder_pos_list = [intruder.physics.position for intruder in self.intruders]
        phi_next = self._total_potential(
            agent_pos=self.agent.position,
            intruder_pos_list=new_intruder_pos_list
        )

        # 2. Potential-Based Shaping Reward (F)
        shaping_reward = self.gamma * phi_next - phi_current

        # 3. Tertiary Objective: Control Effort Penalty (R_effort)
        effort_penalty = -self.k_action * np.sum(action**2)

        # 4. Aggregate the final reward
        total_reward = shaping_reward + effort_penalty
        
        return total_reward



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
        """Casts rays from the agent and finds the closest intersection point with any intruder."""
        rays = []
        angle_step = 2 * np.pi / self.num_lidar_scans

        for i in range(self.num_lidar_scans):
            angle = i * angle_step
            direction = np.array([np.cos(angle), np.sin(angle)])
            origin = self.agent.position

            min_dist = self.lidar_max_range
            hit_point = origin + direction * self.lidar_max_range # Default end point

            # --- THIS IS THE CORRECTED LOGIC ---
            # Loop through the list of Intruder objects
            for intruder in self.intruders:
                # Pass the intruder's physics object to the intersection function
                intersection = self.ray_square_intersection(origin, direction, intruder.physics)

                if intersection is not None:
                    dist = np.linalg.norm(intersection - origin)
                    if dist < min_dist:
                        min_dist = dist
                        hit_point = intersection
            
            # Ensure the final hit point does not exceed the environment bounds
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
        half_size = obstacle.size / 2
        lower = obstacle.position - half_size
        upper = obstacle.position + half_size

        t_min = (lower - origin) / (direction + 1e-8)
        t_max = (upper - origin) / (direction + 1e-8)

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

    def _check_intruder_collisions(self):
        # Helper function to check for collisions
        for intruder in self.intruders:
            distance = np.linalg.norm(self.agent.position - intruder.physics.position)
            # Simple circular collision check
            if distance < (self.agent.size / 2 + intruder.physics.size / 2):
                return True
        return False
    
    def _potential_pos(self, agent_pos: np.ndarray) -> float:
        """Calculates the potential based on negative squared Euclidean distance to the target."""
        dist_sq = np.sum((agent_pos - self.target_position)**2)
        return -self.k_pos * dist_sq

    def _potential_safe(self, agent_pos: np.ndarray, intruder_pos_list: list) -> float:
        """
        Calculates the potential based on the "safety bubble" concept for all intruders.
        The potential is a sum of large negative values for each intruder inside its bubble.
        """
        total_safe_potential = 0.0
        for intruder_pos in intruder_pos_list:
            dist = np.linalg.norm(agent_pos - intruder_pos)
            if dist < self.d_safe:
                # Potential is the negative of the penalty function
                total_safe_potential += -self.k_bubble * np.exp(-self.k_decay_safe * (dist - self.r_collision_reward))
        return total_safe_potential

    def _total_potential(self, agent_pos: np.ndarray, intruder_pos_list: list) -> float:
        """Calculates the total potential of a state as the weighted sum of individual potentials."""
        phi_pos = self._potential_pos(agent_pos)
        phi_safe = self._potential_safe(agent_pos, intruder_pos_list)
        return self.w_pos * phi_pos + self.w_safe * phi_safe