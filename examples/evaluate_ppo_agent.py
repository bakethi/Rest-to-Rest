import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os
from tqdm import tqdm  # Progress bar
from stable_baselines3 import PPO
from gym_pathfinding.envs.pathfinding_env import PathfindingEnv

# Load the trained model
model_name = "ppo_pathfinding_2025-03-17_13-58-44_penalty_for_standing_still"
model = PPO.load(f"/home/bake/Projects/Rest-to-Rest/models/{model_name}.zip")

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
                env = PathfindingEnv(
                    number_of_obstacles=num_obstacles,
                    bounds=[[0, 0], [size, size]], 
                    bounce_factor=1, 
                    num_lidar_scans=24, 
                    lidar_max_range=50,
                    terminate_on_collision=False,
                    random_start_target=True
                )

                # 🔹 Compute `max_steps_per_episode` dynamically AFTER environment is initialized
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

                    # Count collisions
                    if info["collision"]:  
                        episode_collisions += 1


                # Success is counted only if the goal is reached (not truncated)
                if done and not truncated:
                    total_successes += 1

                total_steps.append(steps / size)  # Normalize steps by environment size
                total_collisions += episode_collisions  # Track total collisions

            except ValueError as e:
                print(f"Env creation not possible: {e}")

            trial_progress.update(1)  # Update trial progress bar

        trial_progress.close()  # Close trial progress bar

        # Compute statistics
        avg_steps = np.mean(total_steps)
        if total_successes != 0:
            success_rate = (total_successes / num_trials) * 100
        else:
            success_rate = 0
            
        avg_collisions = total_collisions / num_trials  # Compute average collisions per trial
        results.append([size, obstacle_percent, avg_steps, success_rate, avg_collisions])

        # Convert results into a DataFrame
        df = pd.DataFrame(results, columns=["Environment Size", "Obstacle %", "Normalized Steps", "Success Rate", "Avg Collisions"])

        # Generate timestamp for the filename

        csv_filename = f"{results_dir}run_{timestamp}_model_{model_name}.csv"

        # Save to CSV
        df.to_csv(csv_filename, index=False)
        print(f"\n📂 Results saved to {csv_filename}")

        # Update the outer progress bar
        outer_loop.update(1)

outer_loop.close()  # Close environment/obstacle progress bar

# Convert results into a DataFrame
df = pd.DataFrame(results, columns=["Environment Size", "Obstacle %", "Normalized Steps", "Success Rate", "Avg Collisions"])

# Generate timestamp for the filename

csv_filename = f"{results_dir}run_{timestamp}_model_{model_name}.csv"

# Save to CSV
df.to_csv(csv_filename, index=False)
print(f"\n📂 Results saved to {csv_filename}")


