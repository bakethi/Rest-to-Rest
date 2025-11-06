import cProfile
import pstats
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from gym_pathfinding.envs.intruder_avoidance_env import IntruderAvoidanceEnv
import os
import random

class RandomizedEnvWrapper(gym.Wrapper):
    """
    This wrapper randomizes the environment's parameters on each reset.
    """
    def __init__(self, env, num_intruders_range, intruder_speed_range):
        super().__init__(env)
        self.num_intruders_range = num_intruders_range
        self.intruder_speed_range = intruder_speed_range

    def reset(self, **kwargs):
        num_intruders = random.randint(self.num_intruders_range[0], self.num_intruders_range[1])
        max_speed = random.uniform(self.intruder_speed_range[0], self.intruder_speed_range[1])
        self.env.unwrapped.number_of_intruders = num_intruders
        self.env.unwrapped.max_intruder_speed = max_speed
        return self.env.reset(**kwargs)

# --- Configuration ---
feature_name = "profiling_run_1Mil_steps"
LOG_DIR = f"./logs/RuntimeStudy/ppo_intruder_{feature_name}"
# For profiling, use a smaller number of timesteps to get results faster
TOTAL_TIMESTEPS = 1000000

os.makedirs(LOG_DIR, exist_ok=True)

# --- TRAINING Environment Setup ---
train_env = IntruderAvoidanceEnv()
train_env = gym.wrappers.TimeLimit(train_env, max_episode_steps=1000)
train_env = gym.wrappers.RecordEpisodeStatistics(train_env)
print("Observation Space:", train_env.observation_space.shape)

# --- Model Creation ---
# Removed logic for loading checkpoints to focus only on training a new model
print("--- Creating new model for profiling ---")
model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log=LOG_DIR)

# Configure logger
new_logger = configure(LOG_DIR, ["stdout", "tensorboard"])
model.set_logger(new_logger)

# --- PROFILING SETUP ---
profiler = cProfile.Profile()
profiler.enable()

print("--- Starting Training for Profiling ---")
# The 'learn' call is simplified, without the callback argument
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    reset_num_timesteps=True, # Set to True as we are not resuming
    tb_log_name=feature_name
)

profiler.disable()

# --- PRINTING PROFILING STATS ---
stats = pstats.Stats(profiler).sort_stats('cumtime')
stats.print_stats(30)  # Print the top 30 functions by cumulative time
stats.dump_stats(f'scripts/RuntimeStudy/{feature_name}.prof')

print(f"✅ Training and profiling complete!")
print(f"Profiling data saved to {feature_name}.prof")