import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from gym_pathfinding.envs.intruder_avoidance_env import IntruderAvoidanceEnv
import datetime
import os

# --- 1. Configuration for Continuous Training ---
# Use a static feature name and paths so the script can find the model on restart.
feature_name = "24_50_no_intruders"
LOG_DIR = f"./logs/ppo_intruder_{feature_name}"
MODEL_PATH = f"models/ppo_intruder_{feature_name}.zip"

# The number of steps to train for in each session of the loop.
TRAINING_STEPS_PER_SESSION = 50_000

# The total number of training sessions to run.
# For an infinite loop, you can change this to a `while True:` loop.
NUM_SESSIONS = 10  # This will result in 40 * 50,000 = 2,000,000 total steps

# Ensure the log and model directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

# --- 2. Environment Setup (Same as before) ---
env = IntruderAvoidanceEnv(
    number_of_intruders=0, 
    bounds=[[0, 0], [100, 100]], 
    # bounce_factor is part of the Intruder class now, not the env
    num_lidar_scans=24, 
    lidar_max_range=50,
    random_start_target=True,
    max_intruder_speed=1,
    intruder_size=3
)

# +++ THIS IS THE ADDED LINE +++
# This wrapper will automatically terminate and reset the episode after 1000 steps.
# This forces the agent to see a new scenario more often. You can tune this value.
env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)

env = gym.wrappers.RecordEpisodeStatistics(env)
print("Observation Space:", env.observation_space.shape)

# --- 3. Load Existing Model or Create New One ---
if os.path.exists(MODEL_PATH):
    print("--- Loading existing model and continuing training ---")
    # Load the previously saved model
    model = PPO.load(MODEL_PATH, env=env)
else:
    print("--- No model found, creating new model ---")
    # If no model exists, create a new one using your parameters
    model = PPO(
        "MlpPolicy",  
        env,  
        policy_kwargs=dict(net_arch=[128, 128]),
        learning_rate=3e-4,  
        n_steps=2048,  
        batch_size=64,  
        n_epochs=10,  
        gamma=0.99,  
        clip_range=0.2,  
        verbose=1,
        tensorboard_log=LOG_DIR,
        device='cpu',
    )

# Configure logger to save to the continuous log directory
new_logger = configure(LOG_DIR, ["stdout", "tensorboard"])
model.set_logger(new_logger)

# --- 4. The Continuous Training Loop ---
for session in range(NUM_SESSIONS):
    print(f"--- Starting Training Session {session + 1}/{NUM_SESSIONS} ---")
    
    # The key to continuous training is `reset_num_timesteps=False`.
    # This ensures the logs and total timesteps continue to increase across each `learn()` call.
    model.learn(
        total_timesteps=TRAINING_STEPS_PER_SESSION,
        reset_num_timesteps=False,  # This is the most important parameter
        tb_log_name="PPO"
    )
    
    # Save the model after each training session
    print(f"--- Session {session + 1} Complete. Saving model to {MODEL_PATH} ---")
    model.save(MODEL_PATH)
    
print(f"✅ All {NUM_SESSIONS} training sessions complete. Final model saved at {MODEL_PATH}")