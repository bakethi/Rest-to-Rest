import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os
from tqdm import tqdm  # Progress bar
from stable_baselines3 import SAC # Changed from PPO to SAC
from gym_pathfinding.envs.pathfinding_env import PathfindingEnv

# Load the trained SAC model
# IMPORTANT: Replace "sac_pathfinding_YOUR_TIMESTAMP_YOUR_FEATURE_NAME" with the actual name of your saved SAC model
model_name = "sac_pathfinding_2025-06-10_07-43-11_24_50" # Placeholder for SAC model name
try:
    model = SAC.load(f"/home/bake/Projects/Rest-to-Rest/models/{model_name}.zip") # Changed PPO.load to SAC.load
except Exception as e:
    print(f"Error loading SAC model '{model_name}': {e}")
    print("Please ensure the 'model_name' variable points to your actual trained SAC model .zip file.")
    exit() # Exit if model cannot be loaded

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Define environment sizes and obstacle density percentages
environment_sizes = [25, 50, 100, 150, 200]  # Different world sizes
obstacle_percentages = [.1, .5, 1, 2, 5]  # Percentage of the environment covered in obstacles
step_scale_factor = 3  # Scaling factor for max steps

num_trials = 1000  # Number of trials per setting
results = []

# Ensure results directory exists
results_dir = "./results/"
os.makedirs(results_dir, exist_ok=True)

# Outer progress bar (Tracking Env Sizes & Obstacles)
outer_loop = tqdm(
    total=len(environment_sizes) * len(obstacle_percentages),
    desc="Processing Environments & Obstacle Densities",
    position=0,
    leave=True
)

for size in environment_sizes:
    for obstacle_percent in obstacle_percentages:
        total_steps = []
        total_successes = 0
        total_collisions = 0
        total_truncated = 0 # Counter for truncated runs
        num_obstacles = int((size * size) * (obstacle_percent / 100))  # Scale obstacles to environment size

        # Inner progress bar (Tracking Trials)
        trial_progress = tqdm(
            total=num_trials,
            desc=f"Size {size}, Obstacles {obstacle_percent}%",
            position=1,
            leave=False
        )

        for trial in range(num_trials):
            try:
                # Create an environment with a dynamic number of obstacles
                # The num_lidar_scans and lidar_max_range here should match what your SAC model was trained with
                # For this example, keeping default values from previous contexts,
                # but you might need to adjust these based on your SAC training setup.
                env = PathfindingEnv(
                    number_of_obstacles=num_obstacles,
                    bounds=[[0, 0], [size, size]],
                    bounce_factor=1,
                    num_lidar_scans=24, # Adjust if your SAC model was trained with different scans
                    lidar_max_range=50, # Adjust if your SAC model was trained with different range
                    terminate_on_collision=False,
                    random_start_target=True
                )

                # Compute `max_steps_per_episode` dynamically AFTER environment is initialized
                agent_position = env.agent.position
                target_position = env.target_position
                euclidean_dist = np.linalg.norm(target_position - agent_position)
                max_steps_per_episode = int(step_scale_factor * euclidean_dist)

                obs, _ = env.reset()
                done = False
                truncated_from_env = False # Renamed to avoid confusion with loop condition
                steps = 0
                episode_collisions = 0

                while not done and not truncated_from_env and steps < max_steps_per_episode:
                    action, _ = model.predict(obs, deterministic=True) # SAC prediction can be deterministic for evaluation
                    obs, _, done, truncated_from_env, info = env.step(action)
                    steps += 1

                    # Count collisions
                    if info["collision"]:
                        episode_collisions += 1

                # Correctly determine if the episode was truncated by reaching the step limit
                # An episode is truncated if it reached max_steps_per_episode AND it's not 'done' (i.e., target reached)
                if steps >= max_steps_per_episode and not done:
                    total_truncated += 1

                # Success is counted only if the goal is reached AND it was not truncated due to step limit
                if done and not (steps >= max_steps_per_episode):
                    total_successes += 1

                total_steps.append(steps / size)  # Normalize steps by environment size
                total_collisions += episode_collisions  # Track total collisions

            except ValueError as e:
                print(f"Env creation not possible for size {size}, obstacles {obstacle_percent}%: {e}")
                continue # Continue to next trial if env creation fails

            trial_progress.update(1)  # Update trial progress bar

        trial_progress.close()  # Close trial progress bar

        # Compute statistics
        avg_steps = np.mean(total_steps) if total_steps else 0
        success_rate = (total_successes / num_trials) * 100 if num_trials > 0 else 0
        avg_collisions = total_collisions / num_trials if num_trials > 0 else 0
        truncated_runs_percentage = (total_truncated / num_trials) * 100 if num_trials > 0 else 0

        results.append([
            size,
            obstacle_percent,
            avg_steps,
            success_rate,
            avg_collisions,
            truncated_runs_percentage # Add truncated runs percentage
        ])

        # Convert results into a DataFrame and save after each environment setting
        # This allows intermediate results to be saved
        df = pd.DataFrame(results, columns=[
            "Environment Size",
            "Obstacle %",
            "Normalized Steps",
            "Success Rate",
            "Avg Collisions",
            "Truncated Runs (%)" # Add new column
        ])

        csv_filename = f"{results_dir}run_{timestamp}_model_{model_name}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"\n📂 Intermediate results saved to {csv_filename}")

        # Update the outer progress bar
        outer_loop.update(1)

outer_loop.close()  # Close environment/obstacle progress bar

print(f"\n✅ All evaluation complete! Final results saved to: {csv_filename}")