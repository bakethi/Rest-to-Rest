import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from gym_pathfinding.envs.pathfinding_env import PathfindingEnv
import datetime
import os

# 🔹 Generate a timestamp for logging
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = f"./logs/ppo_pathfinding_{timestamp}"  # TensorBoard log directory

# Ensure the log directory exists
os.makedirs(log_dir, exist_ok=True)

# 🔹 Create the environment
env = PathfindingEnv(
    number_of_obstacles=10, 
    bounds=[[0, 0], [100, 100]], 
    bounce_factor=1, 
    num_lidar_scans=24, 
    lidar_max_range=50,
    random_start_target=True
)

# 🔹 Wrap the environment for logging statistics
env = gym.wrappers.RecordEpisodeStatistics(env)

# 🔹 Instantiate the PPO model with TensorBoard logging enabled
model = PPO(
    "MlpPolicy",  
    env,  
    learning_rate=3e-4,  
    n_steps=2048,  
    batch_size=64,  
    n_epochs=10,  
    gamma=0.99,  
    clip_range=0.2,  
    verbose=1,
    tensorboard_log=log_dir,  # 🔹 Enable TensorBoard logging
    device='cpu'
)

# 🔹 Configure logger for more detailed tracking
new_logger = configure(log_dir, ["stdout", "tensorboard"])
model.set_logger(new_logger)

# 🔹 Train the model with TensorBoard logging
model.learn(total_timesteps=100000, tb_log_name="PPO")

# 🔹 Save the trained model
model.save(f"models/ppo_pathfinding_{timestamp}")

print(f"✅ Training complete and model saved! Logs available at: {log_dir}")

