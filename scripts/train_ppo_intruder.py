import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from gym_pathfinding.envs.intruder_avoidance_env import IntruderAvoidanceEnv
from stable_baselines3.common.monitor import Monitor
import datetime
import os
import random

# +++ 1. DEFINE THE CUSTOM WRAPPER FOR RANDOMIZATION +++
class RandomizedEnvWrapper(gym.Wrapper):
    """
    This wrapper randomizes the environment's parameters on each reset.
    """
    def __init__(self, env, num_intruders_range, intruder_speed_range):
        super().__init__(env)
        self.num_intruders_range = num_intruders_range
        self.intruder_speed_range = intruder_speed_range

    def reset(self, **kwargs):
        # --- Randomize parameters BEFORE the environment is reset ---
        
        # Pick a random integer for the number of intruders
        num_intruders = random.randint(self.num_intruders_range[0], self.num_intruders_range[1])
        
        # Pick a random float for the intruder speed
        max_speed = random.uniform(self.intruder_speed_range[0], self.intruder_speed_range[1])

        # Access the underlying environment via `unwrapped` to set the parameters
        self.env.unwrapped.number_of_intruders = num_intruders
        self.env.unwrapped.max_intruder_speed = max_speed
        
        # You could add other parameters to randomize here, for example:
        # self.env.unwrapped.intruder_size = random.choice([2, 4, 6])
        
        # Now, call the original reset method, which will use the new parameters
        return self.env.reset(**kwargs)

# --- Configuration ---
feature_name = "24_50_PBRS_Training_3" # New name for the randomized run
LOG_DIR = f"./logs/ppo_intruder_{feature_name}"
CHECKPOINT_DIR = f"./models/checkpoints_{feature_name}/"
BEST_MODEL_DIR = f"./models/best_model_{feature_name}/"

CHECKPOINT_FREQ = 500_000
EVAL_FREQ = 25_000 
TOTAL_TIMESTEPS = 5_000_000 

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)

# --- TRAINING Environment Setup ---
base_train_env = IntruderAvoidanceEnv(
    # Initial parameters don't matter as much, as they will be randomized
    change_direction_interval=6,
    number_of_intruders=5, 
    bounds=[[0, 0], [100, 100]], 
    max_intruder_speed=1,
    intruder_size=3,
    # ... other parameters ...
)

# +++ 2. APPLY THE WRAPPER TO THE TRAINING ENVIRONMENT +++
# Define the ranges for your randomization
num_intruders_range = [5, 20]  # e.g., train with 5 to 20 intruders
speed_range = [0.8, 2.0]        # e.g., train with speeds between 0.8 and 2.0

# Wrap the base environment to create the randomized training environment
train_env = RandomizedEnvWrapper(base_train_env, num_intruders_range, speed_range)

# Apply other wrappers as usual
train_env = gym.wrappers.TimeLimit(train_env, max_episode_steps=1000)
train_env = gym.wrappers.RecordEpisodeStatistics(train_env)
print("Observation Space:", train_env.observation_space.shape)


# --- EVALUATION Environment Setup ---
# *** CRUCIAL NOTE: Do NOT randomize the evaluation environment! ***
# The evaluation environment must be consistent to provide a stable benchmark
# for comparing model performance over time.
eval_env = IntruderAvoidanceEnv(
    change_direction_interval=6,
    number_of_intruders=40, # Use a fixed, challenging number for evaluation
    bounds=[[0, 0], [100, 100]], 
    max_intruder_speed=3, # Use a fixed, challenging speed
    intruder_size=3,
    # ... other parameters ...
)
eval_env = gym.wrappers.TimeLimit(eval_env, max_episode_steps=1000)
eval_env = Monitor(eval_env)

# --- Load/Create Model, Callbacks, and Training Loop (remains the same) ---

# Find the latest checkpoint...
latest_checkpoint = None
# ... (your existing code for finding the latest checkpoint) ...

# Load or create a new model
if latest_checkpoint:
    print(f"--- Loading from latest checkpoint: {latest_checkpoint} ---")
    model = PPO.load(latest_checkpoint, env=train_env)
else:
    print("--- No checkpoint found, creating new model ---")
    model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log=LOG_DIR) # Simplified for clarity

# Setup Callbacks
checkpoint_callback = CheckpointCallback(
  save_freq=CHECKPOINT_FREQ,
  save_path=CHECKPOINT_DIR,
  name_prefix=f"ppo_intruder_{feature_name}"
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=BEST_MODEL_DIR,
    log_path=LOG_DIR,
    eval_freq=EVAL_FREQ,
    n_eval_episodes=20 # Increased episodes for a more stable evaluation score
)

callback_list = CallbackList([checkpoint_callback, eval_callback])

# Configure logger and start training
new_logger = configure(LOG_DIR, ["stdout", "tensorboard"])
model.set_logger(new_logger)

print("--- Starting Randomized Training with Auto-Evaluation ---")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    reset_num_timesteps=False,
    tb_log_name="PPO",
    callback=callback_list 
)
    
print(f"✅ Training complete!")
print(f"Checkpoints are saved in {CHECKPOINT_DIR}")
print(f"The best performing model is saved as best_model.zip in {BEST_MODEL_DIR}")