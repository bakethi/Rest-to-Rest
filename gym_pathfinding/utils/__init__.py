# gym_pathfinding/utils/__init__.py

# Import specific classes
from .obstacles import ObstacleManager

# Define what should be imported when someone imports `gym_pathfinding.envs`
__all__ = ["ObstacleManager"]
