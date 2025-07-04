import gymnasium as gym
from stable_baselines3 import PPO, SAC
from gym_pathfinding.envs.intruder_avoidance_env import IntruderAvoidanceEnv
from gym_pathfinding.envs.visualization import Renderer  # If you want to visualize the environment

model_name = "ppo_intruder_24_50_PBRS_Training_2"
# Load the trained model
model = PPO.load(f"/home/bake/Projects/Rest-to-Rest/models/{model_name}.zip")  # Make sure this path matches your saved model

# Create the environment
env = IntruderAvoidanceEnv(
    number_of_intruders=10, 
    bounds=[[0, 0], [100, 100]], 
    bounce_factor=1, 
    num_lidar_scans=24, 
    lidar_max_range=50,
    random_start_target=True,
    terminate_on_collision=False,
    obstacle_max_size=5,
    obstacle_min_size=1, 
    intruder_size=3,
    max_intruder_speed=1,
    change_direction_interval=6,
)


# Initialize the renderer (if visualization is needed)
renderer = Renderer(env, record=True, video_path=f"/home/bake/Projects/Rest-to-Rest/Videos/inference_video_{model_name}.mp4")

# Reset the environment
obs, _ = env.reset()
num_steps = 1000

for _ in range(num_steps):
    # Get action from trained model
    action, _ = model.predict(obs)

    # Take a step in the environment
    obs, reward, done, truncated, info = env.step(action)

    # Render the environment
    renderer.render(env.agent, env.obstacle_manager, env.target_position, env.intruders)

# Save the video and close the renderer
renderer.save_video()
print("Inference completed!")
