import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
# +++ 1. IMPORT THE CALLBACK +++
from stable_baselines3.common.callbacks import CheckpointCallback
from gym_pathfinding.envs.intruder_avoidance_env import IntruderAvoidanceEnv
import datetime
import os

# --- Configuration ---
feature_name = "24_50_PBRS_Training_2"
LOG_DIR = f"./logs/ppo_intruder_{feature_name}"
# This will be the directory where checkpoints are saved
CHECKPOINT_DIR = f"./models/checkpoints_{feature_name}/" 

# The number of steps between each checkpoint save
CHECKPOINT_FREQ = 500_000 
# Total number of steps to train for
TOTAL_TIMESTEPS = 5_000_000 

# Ensure the log and model directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- Environment Setup (Same as your script) ---
env = IntruderAvoidanceEnv(
    change_direction_interval=6,
    number_of_intruders=5, 
    bounds=[[0, 0], [100, 100]], 
    max_intruder_speed=1,
    intruder_size=3,
    terminate_on_collision=True,
    gamma= 0.99,
    r_collision_reward= None,
    d_safe= 15.0,
    k_bubble= 50.0,
    k_decay_safe= 0.1,
    C_collision= 100.0,
    k_pos= 0.5,
    k_action= 0.001,
    w_safe= 0.3,
    w_pos= 0.7 
)
env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
env = gym.wrappers.RecordEpisodeStatistics(env)
print("Observation Space:", env.observation_space.shape)

# --- Find the latest checkpoint to load from ---
# This part is optional but useful for continuous training. It finds the last saved model.
latest_checkpoint = None
if os.path.exists(CHECKPOINT_DIR) and len(os.listdir(CHECKPOINT_DIR)) > 0:
    # Get all the .zip files in the directory
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.zip')]
    if checkpoints:
        # Sort by the number of steps (extracted from the filename)
        checkpoints.sort(key=lambda x: int(x.split('_')[2]))
        latest_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoints[-1])

# --- Load Existing Model or Create New One ---
if latest_checkpoint:
    print(f"--- Loading from latest checkpoint: {latest_checkpoint} ---")
    model = PPO.load(latest_checkpoint, env=env)
else:
    print("--- No checkpoint found, creating new model ---")
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

# --- 2. SETUP THE CHECKPOINT CALLBACK ---
# This will save a checkpoint every CHECKPOINT_FREQ steps
checkpoint_callback = CheckpointCallback(
  save_freq=CHECKPOINT_FREQ,
  save_path=CHECKPOINT_DIR,
  name_prefix=f"ppo_intruder_{feature_name}"
)

# Configure logger
new_logger = configure(LOG_DIR, ["stdout", "tensorboard"])
model.set_logger(new_logger)

# --- 3. TRAIN WITH THE CALLBACK ---
# The model will now train for the total number of steps, and the
# callback will handle saving checkpoints automatically.
print("--- Starting Training ---")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    reset_num_timesteps=False,
    tb_log_name="PPO",
    # Pass the callback here
    callback=checkpoint_callback 
)
    
print(f"✅ Training complete. Checkpoints are saved in {CHECKPOINT_DIR}")