import gymnasium as gym
from stable_baselines3 import SAC # Changed from PPO to SAC
from stable_baselines3.common.logger import configure
from gym_pathfinding.envs.pathfinding_env import PathfindingEnv
import datetime
import os

# 🔹 Generate a timestamp for logging
feature_name = "24_50" # Changed feature name for SAC
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = f"./logs/sac_pathfinding_{timestamp}_{feature_name}" # Changed log directory to reflect SAC

# Ensure the log directory exists
os.makedirs(log_dir, exist_ok=True)

# 🔹 Create the environment
env = PathfindingEnv(
    number_of_obstacles=10,
    bounds=[[0, 0], [100, 100]],
    bounce_factor=1,
    num_lidar_scans=24,
    lidar_max_range=50,
    random_start_target=True,
    terminate_on_collision=False,
    max_collisions=3
)

# Print environment observation space shape
print("Observation Space:", env.observation_space.shape)

# 🔹 Wrap the environment for logging statistics
env = gym.wrappers.RecordEpisodeStatistics(env)

# 🔹 Instantiate the SAC model with TensorBoard logging enabled
# SAC hyperparameters are different from PPO
model = SAC(
    "MlpPolicy",
    env,
    policy_kwargs=dict(net_arch=[128, 128]),
    learning_rate=3e-4,
    buffer_size=100000, # SAC specific: Size of the replay buffer
    learning_starts=100, # SAC specific: Number of steps before learning starts
    batch_size=256, # SAC specific: Batch size for training
    tau=0.005, # SAC specific: Soft update coefficient (for target networks)
    gamma=0.99,
    train_freq=(1, "episode"), # SAC specific: Train after every episode
    gradient_steps=1, # SAC specific: Number of gradient steps after each rollout
    ent_coef="auto", # SAC specific: Entropy regularization coefficient
    verbose=1,
    tensorboard_log=log_dir,
    device='cpu',
)

# 🔹 Configure logger for more detailed tracking
new_logger = configure(log_dir, ["stdout", "tensorboard"])
model.set_logger(new_logger)

# 🔹 Train the model with TensorBoard logging
model.learn(total_timesteps=2000000, tb_log_name="SAC") # Changed tb_log_name to SAC

# 🔹 Save the trained model
model.save(f"models/sac_pathfinding_{timestamp}_{feature_name}") # Changed model save name to SAC

print(f"✅ Training complete and model saved! Logs available at: {log_dir}")