import numpy as np
from queue import Queue

def is_target_reachable(agent_position, target_position, obstacle_manager, bounds):
    """
    Use a simple flood-fill algorithm (BFS) to check if there is a valid path
    from the agent's start position to the target.

    Args:
        agent_position (np.array): The agent's starting position.
        target_position (np.array): The target position.
        obstacle_manager (ObstacleManager): The obstacle manager that checks collisions.
        bounds (np.array): The boundaries of the environment.

    Returns:
        bool: True if the target is reachable, False otherwise.
    """
    queue = Queue()
    visited = set()
    
    start = tuple(agent_position)
    target = tuple(target_position)
    
    queue.put(start)
    visited.add(start)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

    while not queue.empty():
        current = queue.get()
        
        if current == target:
            return True  # Path found

        for direction in directions:
            neighbor = (current[0] + direction[0], current[1] + direction[1])

            if bounds[0][0] <= neighbor[0] <= bounds[1][0] and \
               bounds[0][1] <= neighbor[1] <= bounds[1][1]:  # Check bounds

                if not obstacle_manager.check_collision_of_point(neighbor) and neighbor not in visited:
                    visited.add(neighbor)
                    queue.put(neighbor)

    return False  # No path found
