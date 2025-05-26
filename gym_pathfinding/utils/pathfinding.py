import numpy as np
import heapq # For the priority queue

def heuristic(a, b):
    # Manhattan distance for a grid, assuming discrete integer coordinates
    # You might need to cast positions to int/tuple for grid-based A*
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def is_target_reachable(agent_position, target_position, obstacle_manager, bounds):
    # Ensure positions are integers for grid-based A*
    start_node = tuple(map(int, agent_position))
    target_node = tuple(map(int, target_position))
    
    # Priority queue: (f_score, g_score, node)
    open_set = []
    heapq.heappush(open_set, (heuristic(start_node, target_node), 0, start_node))

    # For reconstructing path (not strictly needed for just reachability, but good practice)
    # came_from = {} 

    # g_score: cost from start to current node
    g_score = {start_node: 0}

    # f_score: g_score + heuristic_score
    f_score = {start_node: heuristic(start_node, target_node)}

    # Visited nodes
    closed_set = set()

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # 4-directional movement

    while open_set:
        current_f, current_g, current_node = heapq.heappop(open_set)

        if current_node == target_node:
            return True # Path found

        if current_node in closed_set:
            continue
        closed_set.add(current_node)

        for d_x, d_y in directions:
            neighbor = (current_node[0] + d_x, current_node[1] + d_y)

            # Check bounds
            if not (bounds[0][0] <= neighbor[0] <= bounds[1][0] and \
                    bounds[0][1] <= neighbor[1] <= bounds[1][1]):
                continue

            # Check for collision with obstacles.
            # IMPORTANT: This assumes obstacles are aligned with the grid for point collision.
            # If obstacles are continuous shapes, this check might need to be more sophisticated.
            if obstacle_manager.check_collision_of_point(neighbor): # Reusing existing collision check
                continue

            # Cost to reach neighbor from current node is 1 (for unweighted grid)
            tentative_g_score = current_g + 1

            if tentative_g_score < g_score.get(neighbor, float('inf')):
                # This path to neighbor is better than any previous one. Record it.
                # came_from[neighbor] = current_node # If you need to reconstruct path
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, target_node)
                heapq.heappush(open_set, (f_score[neighbor], g_score[neighbor], neighbor))

    return False # No path found