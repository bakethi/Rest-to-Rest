import unittest
import numpy as np
import math
import pytest
from gym_pathfinding.envs.pathfinding_env import PathfindingEnv


import pytest
import numpy as np
import math
from gym_pathfinding.envs.pathfinding_env import PathfindingEnv


def test_setup():
    env = PathfindingEnv(num_lidar_scans=4, lidar_max_range=5)
    assert env.num_lidar_scans == 4, "Number of Lidar Scans wrong"
    assert env.lidar_max_range == 5, "Lidar Max Range wrong"


def test_agent_to_target_distance():
    env = PathfindingEnv(num_lidar_scans=4, lidar_max_range=5)
    env.agent.position = [1, 2]
    env.target_position = [4, 6]
    expected_distance = 5 / math.sqrt(100**2 + 100**2)
    assert round(env._getAgentTargetDist(), 5) == round(expected_distance, 5)


def test_no_obstacles():
    env = PathfindingEnv(num_lidar_scans=4, lidar_max_range=5)
    env.obstacle_manager.obstacles = []  # No obstacles in the environment
    bounds = np.array([[0, 0], [100, 100]])  # Example bounds
    rays = env.cast_rays_until_collision()

    for ray in rays:
        assert bounds[0][0] <= ray[0] <= bounds[1][0]
        assert bounds[0][1] <= ray[1] <= bounds[1][1]


@pytest.mark.parametrize("num_lidar_scans, lidar_max_range", [
    (12, 50),
    (45, 50),
    (90, 50),
    (180, 50),
    (360, 50),
    (360, 1),
    (360, 10),
    (360, 50),
    (360, 100),
])
def test_lidar_performance(benchmark, num_lidar_scans, lidar_max_range):
    """Test lidar performance with varying configurations."""
    env = PathfindingEnv(
        number_of_obstacles=100,
        bounds=np.array([[0, 0], [100, 100]]),
        bounce_factor=1,
        num_lidar_scans=num_lidar_scans,
        lidar_max_range=lidar_max_range
    )

    # Benchmark the function
    result = benchmark(env.cast_rays_until_collision)


