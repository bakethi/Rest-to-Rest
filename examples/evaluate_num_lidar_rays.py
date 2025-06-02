import gymnasium as gym
import numpy as np
import pandas as pd
import datetime
import os
from tqdm import tqdm
from stable_baselines3 import PPO
from gym_pathfinding.envs.pathfinding_env import PathfindingEnv

# Define the models to test with their corresponding number of LiDAR scans
# These models were identified from the provided `system_info.txt` files.
models_to_test = [
    {"name": "ppo_pathfinding_2025-05-26_08-00-04_12_50", "num_lidar_scans": 12},
    {"name": "ppo_pathfinding_2025-03-17_13-58-44_24_50", "num_lidar_scans": 24},
    {"name": "ppo_pathfinding_2025-05-26_12-01-19_48_50", "num_lidar_scans": 48},
    {"name": "ppo_pathfinding_2025-05-26_13-30-45_96_50", "num_lidar_scans": 96},
    {"name": "ppo_pathfinding_2025-05-26_15-45-06_182_50", "num_lidar_scans": 182},
    {"name": "ppo_pathfinding_2025-05-27_06-43-07_324_50", "num_lidar_scans": 324},
    {"name": "ppo_pathfinding_2025-05-27_14-23-53_648_50", "num_lidar_scans": 648},
]

# Define the LiDAR ray lengths to test
lidar_ray_lengths_to_test = [25, 50, 100, 200]

# General evaluation parameters
environment_size = 100
obstacle_percentage = 0.1
num_trials = 1000  # Number of trials per setting
step_scale_factor = 3  # Scaling factor for max steps

# Ensure results directory exists
results_dir = "./results/"
os.makedirs(results_dir, exist_ok=True)

all_results = []

# Outer loop for models
for model_info in models_to_test:
    model_name = model_info["name"]
    model_num_lidar_scans = model_info["num_lidar_scans"]

    print(f"\n--- Evaluating Model: {model_name} (LiDAR Scans: {model_num_lidar_scans}) ---")

    # Load the trained model
    try:
        model = PPO.load(f"/home/bake/Projects/Rest-to-Rest/models/{model_name}.zip")
    except Exception as e:
        print(f"Failed to load model {model_name}: {e}")
        continue

    # Loop for different LiDAR ray lengths
    for current_lidar_max_range in lidar_ray_lengths_to_test:
        print(f"  Evaluating with LiDAR Max Range: {current_lidar_max_range}")

        total_steps = []
        total_successes = 0
        total_collisions = 0
        total_truncated = 0 # Initialize counter for truncated runs
        num_obstacles = int((environment_size * environment_size) * (obstacle_percentage / 100))

        # Inner progress bar (Tracking Trials)
        trial_progress = tqdm(
            total=num_trials,
            desc=f"  Model {model_num_lidar_scans} scans, Lidar Range {current_lidar_max_range}",
            position=1,
            leave=False
        )

        for trial in range(num_trials):
            try:
                # Create an environment with the dynamic number of obstacles and current lidar settings
                env = PathfindingEnv(
                    number_of_obstacles=num_obstacles,
                    bounds=[[0, 0], [environment_size, environment_size]],
                    bounce_factor=1,
                    num_lidar_scans=model_num_lidar_scans,  # Use model's trained lidar scans
                    lidar_max_range=current_lidar_max_range, # Use current evaluation lidar range
                    terminate_on_collision=False,
                    random_start_target=True
                )

                agent_position = env.agent.position
                target_position = env.target_position
                euclidean_dist = np.linalg.norm(target_position - agent_position)
                max_steps_per_episode = int(step_scale_factor * euclidean_dist)

                obs, _ = env.reset()
                done = False
                truncated = False
                steps = 0
                episode_collisions = 0

                while not done and not truncated and steps < max_steps_per_episode:
                    action, _ = model.predict(obs)
                    obs, _, done, truncated, info = env.step(action)
                    steps += 1

                    if info["collision"]:
                        episode_collisions += 1

                if done and not truncated:
                    total_successes += 1
                
                if truncated: # Check if the episode was truncated
                    total_truncated += 1

                total_steps.append(steps / environment_size)
                total_collisions += episode_collisions

            except ValueError as e:
                print(f"Env creation not possible for {model_name} (Lidar Range {current_lidar_max_range}): {e}")

            trial_progress.update(1)
        trial_progress.close()

        avg_steps = np.mean(total_steps) if total_steps else 0
        success_rate = (total_successes / num_trials) * 100 if num_trials > 0 else 0
        avg_collisions = total_collisions / num_trials if num_trials > 0 else 0
        truncated_runs_percentage = (total_truncated / num_trials) * 100 if num_trials > 0 else 0 # Percentage of truncated runs

        all_results.append([
            model_name,
            model_num_lidar_scans,
            current_lidar_max_range,
            environment_size,
            obstacle_percentage,
            avg_steps,
            success_rate,
            avg_collisions,
            truncated_runs_percentage # Add truncated runs percentage to results
        ])

# Convert all collected results into a DataFrame
df = pd.DataFrame(all_results, columns=[
    "Model Name",
    "Model LiDAR Scans",
    "Eval LiDAR Range",
    "Environment Size",
    "Obstacle %",
    "Normalized Steps",
    "Success Rate",
    "Avg Collisions",
    "Truncated Runs (%)" # Add new column name
])

# Generate timestamp for the filename
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
csv_filename = f"{results_dir}evaluation_summary_run_{timestamp}.csv"

# Save to CSV
df.to_csv(csv_filename, index=False)
print(f"\n✅ All evaluation results saved to {csv_filename}")