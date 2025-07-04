# Inside gym_pathfinding/__init__.py
from gymnasium.envs.registration import register

register(
     id="gym_pathfinding/IntruderAvoidance-v0",
     entry_point="gym_pathfinding.envs.intruder_avoidance_env:IntruderAvoidanceEnv",
)