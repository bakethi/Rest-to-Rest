import numpy as np
from .pathfinding import is_target_reachable

class ObstacleManager:
    def __init__(self):
        """
        Initialize the obstacle manager.
        """
        self.obstacles = []

    def add_obstacle(self, position, size, shape_type="square"):
        """
        Add an obstacle to the environment.

        Args:
            position (np.array): The position of the obstacle [x, y].
            size (float): The size of the obstacle (side length or radius).
            shape_type (str): The shape of the obstacle ("square" or "circle").
        """
        self.obstacles.append({
            "position": np.array(position, dtype=np.float32),
            "size": size,
            "type": shape_type,
        })

    def generate_random_obstacles(self, num_obstacles, min_size=1.0, max_size=5.0, agent_position=None, target_position=None, bounds=None):
        """
        Generate random obstacles within the world bounds while ensuring that the agent's start
        position and the target position remain free of obstacles. Also ensures that a valid path
        exists between the agent and the target.

        Args:
            num_obstacles (int): The number of obstacles to generate.
            min_size (float): The minimum size of an obstacle.
            max_size (float): The maximum size of an obstacle.
            agent_position (np.array): The position of the agent.
            target_position (np.array): The position of the target.
            bounds (np.array): The boundaries of the environment.

        Raises:
            ValueError: If a valid environment cannot be generated after multiple attempts.
        """
        max_attempts = 100  # Prevent infinite loops
        attempts = 0

        while attempts < max_attempts:
            self.obstacles = []  # Clear existing obstacles
            for _ in range(num_obstacles):
                valid_position = False
                while not valid_position:
                    position = np.random.uniform(
                        low=[0, 0],
                        high=bounds[1],
                        size=(2,)
                    )
                    size = np.random.uniform(min_size, max_size)
                    shape_type = np.random.choice(["square", "circle"])

                    # Check if the obstacle is placed at the agent or target position
                    if agent_position is not None and np.linalg.norm(position - agent_position) < size:
                        continue
                    if target_position is not None and np.linalg.norm(position - target_position) < size:
                        continue

                    valid_position = True  # Accept the obstacle position

                self.add_obstacle(position, size, shape_type)

            # Ensure the target is reachable
            if is_target_reachable(agent_position, target_position, self, bounds):
                return  # Exit if valid obstacles are generated

            attempts += 1

        raise ValueError("Could not generate a valid obstacle placement within the attempt limit.")



    def check_collision(self, physics_object):
        """
        Check for a collision between a PhysicsObject and any obstacle.

        Args:
            physics_object (PhysicsObject): The object to check for collisions.

        Returns:
            bool: True if a collision occurs, False otherwise.
        """
        for obstacle in self.obstacles:
            if obstacle["type"] == "square":
                half_size = obstacle["size"] / 2
                lower_bound = obstacle["position"] - half_size
                upper_bound = obstacle["position"] + half_size
                if np.all(lower_bound <= physics_object.position) and np.all(physics_object.position <= upper_bound):
                    return True
            elif obstacle["type"] == "circle":
                distance = np.linalg.norm(obstacle["position"] - physics_object.position)
                if distance < obstacle["size"]:
                    return True
        return False

    def get_obstacles(self):
        """
        Get the list of obstacles.

        Returns:
            list[dict]: The list of obstacles.
        """
        return self.obstacles

    def reset(self):
        """
        Reset the obstacle manager, clearing all obstacles.
        """
        self.obstacles = []

    def check_collision_of_point(self, point):
        """
        Check for a collision between a Point and any obstacle.

        Args:
            point (array): The point to check for collisions.

        Returns:
            bool: True if a collision occurs, False otherwise.
        """
        for obstacle in self.obstacles:
            if obstacle["type"] == "square":
                half_size = obstacle["size"] / 2
                lower_bound = obstacle["position"] - half_size
                upper_bound = obstacle["position"] + half_size
                if np.all(lower_bound <= point) and np.all(point <= upper_bound):
                    return True
            elif obstacle["type"] == "circle":
                distance = np.linalg.norm(obstacle["position"] - point)
                if distance < obstacle["size"]:
                    return True
        return False