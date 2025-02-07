import numpy as np


class ObstacleManager:
    def __init__(self, world_bounds=np.array([100.0, 100.0])):
        """
        Initialize the obstacle manager.

        Args:
            world_bounds (np.array): The size of the world in [width, height].
        """
        self.obstacles = []
        self.world_bounds = np.array(world_bounds, dtype=np.float32)

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

    def generate_random_obstacles(self, num_obstacles, min_size=1.0, max_size=5.0):
        """
        Generate random obstacles within the world bounds.

        Args:
            num_obstacles (int): The number of obstacles to generate.
            min_size (float): The minimum size of an obstacle.
            max_size (float): The maximum size of an obstacle.
        """
        for _ in range(num_obstacles):
            position = np.random.uniform(
                low=[0, 0],
                high=self.world_bounds,
                size=(2,)
            )
            size = np.random.uniform(min_size, max_size)
            shape_type = np.random.choice(["square", "circle"])
            self.add_obstacle(position, size, shape_type)

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