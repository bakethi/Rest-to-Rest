import gymnasium as gym
from stable_baselines3 import PPO
from gym_pathfinding.envs.pathfinding_env import PathfindingEnv
from gym_pathfinding.algorithms.RL.PPO import PPOTrainer
from gym_pathfinding.envs.visualization import Renderer
import datetime

# Create the environment
env = PathfindingEnv(
    number_of_obstacles=10, 
    bounds=[[0, 0], [100, 100]], 
    bounce_factor=1, 
    num_lidar_scans=24, 
    lidar_max_range=50,
    random_start_target=True
)

# Wrap the environment (Optional but useful for logging)
env = gym.wrappers.RecordEpisodeStatistics(env)

# Instantiate the PPO model
model = PPO(
    "MlpPolicy",  # Uses a multi-layer perceptron (MLP) for function approximation
    env,  
    learning_rate=3e-4,  
    n_steps=2048,  
    batch_size=64,  
    n_epochs=10,  
    gamma=0.99,  
    clip_range=0.2,  
    verbose=1,
    device='cpu'
)

# Train the model
model.learn(total_timesteps=100000)

# Generate timestamp for the filename
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Save the trained model
model.save(f"models/ppo_pathfinding_{timestamp}")

print("Training complete and model saved!")
