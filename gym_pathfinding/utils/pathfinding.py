import numpy as np
import heapq


class Pathfinding:
    def __init__(self, grid_size=(100, 100), world_bounds=np.array([100.0, 100.0])):
        """
        Initialize the pathfinding system.

        Args:
            grid_size (tuple): The grid size (rows, cols).
            world_bounds (np.array): The size of the world in [width, height].
        """
        self.grid_size = grid_size
        self.world_bounds = np.array(world_bounds, dtype=np.float32)
        self.obstacle_map = np.zeros(grid_size, dtype=np.int32)

    def _world_to_grid(self, position):
        """
        Convert world coordinates to grid coordinates.

        Args:
            position (np.array): World coordinates [x, y].

        Returns:
            tuple: Grid coordinates (row, col).
        """
        grid_x = int(position[0] / self.world_bounds[0] * self.grid_size[1])
        grid_y = int(position[1] / self.world_bounds[1] * self.grid_size[0])
        return grid_y, grid_x

    def _grid_to_world(self, grid_position):
        """
        Convert grid coordinates to world coordinates.

        Args:
            grid_position (tuple): Grid coordinates (row, col).

        Returns:
            np.array: World coordinates [x, y].
        """
        world_x = grid_position[1] * self.world_bounds[0] / self.grid_size[1]
        world_y = grid_position[0] * self.world_bounds[1] / self.grid_size[0]
        return np.array([world_x, world_y], dtype=np.float32)

    def update_obstacle_map(self, obstacle_manager):
        """
        Update the obstacle map based on current obstacles.

        Args:
            obstacle_manager (ObstacleManager): The obstacle manager.
        """
        self.obstacle_map.fill(0)
        for obstacle in obstacle_manager.get_obstacles():
            grid_pos = self._world_to_grid(obstacle["position"])
            size_in_cells = int(obstacle["size"] / self.world_bounds[0] * self.grid_size[0])
            for i in range(-size_in_cells, size_in_cells + 1):
                for j in range(-size_in_cells, size_in_cells + 1):
                    row = grid_pos[0] + i
                    col = grid_pos[1] + j
                    if 0 <= row < self.grid_size[0] and 0 <= col < self.grid_size[1]:
                        self.obstacle_map[row, col] = 1

    def find_path(self, start, target):
        """
        Find a path from start to target using A*.

        Args:
            start (np.array): Start position in world coordinates.
            target (np.array): Target position in world coordinates.

        Returns:
            list[np.array]: List of waypoints in world coordinates.
        """
        # Check if start and target positions are within valid bounds (modify as needed)
        if not self._is_valid_position(start):
            raise Exception(f"Invalid start position: {start}")
        if not self._is_valid_position(target):
            raise Exception(f"Invalid target position: {target}")

        start_grid = self._world_to_grid(start)
        target_grid = self._world_to_grid(target)
        grid_path = self._a_star(start_grid, target_grid)
        return [self._grid_to_world(gp) for gp in grid_path]

    def _is_valid_position(self, position):
        """
        Check if the position is within valid bounds.

        Args:
            position (np.array): The position to check.

        Returns:
            bool: True if position is valid, False otherwise.
        """
        # Define bounds for valid positions (example: the grid bounds)
        grid_width, grid_height = self.grid_size  # Assuming you have grid size as an attribute
        x, y = position

        # Check if the position is within the grid boundaries
        return 0 <= x < grid_width and 0 <= y < grid_height

    def _a_star(self, start_grid, target_grid):
        """
        A* algorithm for pathfinding.

        Args:
            start_grid (tuple): Start position in grid coordinates.
            target_grid (tuple): Target position in grid coordinates.

        Returns:
            list[tuple]: List of grid coordinates representing the path.
        """
        open_set = []
        heapq.heappush(open_set, (0, start_grid))
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self._heuristic(start_grid, target_grid)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == target_grid:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            neighbors = [
                (current[0] + dy, current[1] + dx)
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            ]
            for neighbor in neighbors:
                if not (0 <= neighbor[0] < self.grid_size[0] and 0 <= neighbor[1] < self.grid_size[1]):
                    continue
                if self.obstacle_map[neighbor] == 1:
                    continue

                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, target_grid)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []

    def _heuristic(self, node, target):
        """
        Calculate the heuristic (Euclidean distance).

        Args:
            node (tuple): Current grid position.
            target (tuple): Target grid position.

        Returns:
            float: Heuristic distance.
        """
        return np.linalg.norm(np.array(node) - np.array(target))
