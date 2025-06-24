import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os
from tqdm import tqdm  # Progress bar
from stable_baselines3 import PPO
from gym_pathfinding.envs.intruder_avoidance_env import IntruderAvoidanceEnv

# Load the trained model
model_name = "checkpoints_24_50_PBRS_Training_2/ppo_intruder_24_50_PBRS_Training_2_500000_steps"
model = PPO.load(f"/home/bake/Projects/Rest-to-Rest/models/{model_name}.zip")

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Define environment sizes and obstacle density percentages
environment_sizes = [25, 50, 100, 150, 200]  # Different world sizes
intruder_percentages = [.1, .5, 1, 2, 5]  # Percentage of the environment covered in intruders

num_trials = 1000  # Number of trials per setting
# ~~~ MODIFY: Update results list to store new metrics ~~~
results = []

# Ensure results directory exists
results_dir = "./results/"
os.makedirs(results_dir, exist_ok=True)

# Outer progress bar
outer_loop = tqdm(
    total=len(environment_sizes) * len(intruder_percentages),
    desc="Processing Environments & Intruder Densities",
    position=0,
    leave=True
)

for size in environment_sizes:
    for intruder_percentage in intruder_percentages:
        # ~~~ MODIFY: Initialize lists to store metrics for the current setting ~~~
        total_collisions_for_setting = 0
        all_trial_avg_deviations = []
        all_trial_max_deviations = []
        
        num_intruders = int((size * size) * (intruder_percentage / 100))

        # Inner progress bar for trials
        trial_progress = tqdm(
            total=num_trials,
            desc=f"Size {size}, Intruder {intruder_percentage}%",
            position=1,
            leave=False
        )

        for trial in range(num_trials):
            try:
                # ~~~ MODIFY: Removed PathfindingEnv-specific parameters ~~~
                env = IntruderAvoidanceEnv(
                    number_of_intruders=num_intruders,
                    bounds=[[0, 0], [size, size]],
                )

                # ~~~ MODIFY: Use a fixed number of steps for evaluation ~~~
                max_steps_per_episode = 500 # A fixed episode length for consistent evaluation

                obs, _ = env.reset()
                done = False
                
                # +++ ADD: Lists to track deviation for the current episode +++
                episode_collisions = 0
                episode_deviations = []

                for step in range(max_steps_per_episode):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, done, truncated, info = env.step(action)

                    # +++ ADD: Record deviation at each step +++
                    current_deviation = np.linalg.norm(env.agent.position - env.target_position)
                    episode_deviations.append(current_deviation)

                    if info["collision"]:
                        episode_collisions += 1
                    
                    if done:
                        break

                # --- After each episode ---
                total_collisions_for_setting += episode_collisions
                
                # +++ ADD: Calculate and store avg/max deviation for this trial +++
                if episode_deviations: # Ensure the list is not empty
                    all_trial_avg_deviations.append(np.mean(episode_deviations))
                    all_trial_max_deviations.append(np.max(episode_deviations))

            except Exception as e:
                print(f"An error occurred during trial: {e}")

            trial_progress.update(1)

        trial_progress.close()

        # --- After all trials for a setting ---
        # ~~~ MODIFY: Calculate final averages for all new metrics ~~~
        avg_collisions = total_collisions_for_setting / (num_trials * max_steps_per_episode) # Normalize by total steps
        avg_deviation = np.mean(all_trial_avg_deviations) if all_trial_avg_deviations else 0
        avg_max_deviation = np.mean(all_trial_max_deviations) if all_trial_max_deviations else 0
        
        # ~~~ MODIFY: Append all new metrics to the results list ~~~
        results.append([size, intruder_percentage, avg_collisions, avg_deviation, avg_max_deviation])

        # ~~~ MODIFY: Update DataFrame columns and save after each setting ~~~
        df = pd.DataFrame(results, columns=["Environment Size", "Intruder %", "Avg Collisions per Step", "Avg Deviation", "Avg of Max Deviations"])
        csv_filename = f"{results_dir}run_{timestamp}_model_{model_name}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"\n📂 Intermediate results saved to {csv_filename}")

        outer_loop.update(1)

outer_loop.close()

print(f"\n✅ Evaluation complete! Final results saved to {csv_filename}")