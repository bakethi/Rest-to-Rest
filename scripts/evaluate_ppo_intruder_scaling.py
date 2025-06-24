import gymnasium as gym
import numpy as np
import pandas as pd
import datetime
import os
from tqdm import tqdm
from stable_baselines3 import PPO
from gym_pathfinding.envs.intruder_avoidance_env import IntruderAvoidanceEnv

# --- 1. CONFIGURATION ---

# Point to the directory containing all your saved model checkpoints
CHECKPOINT_DIR = "models/best_model_24_50_PBRS_Training_3" 

# +++ ADDED: Define the number of intruders to test against +++
numbers_of_intruders = [5, 10, 20, 40]

# Define the other sets of parameters you want to evaluate against
intruder_sizes = [3, 6, 9]                 # e.g., radius of the intruders
intruder_speeds = [1, 2, 3]          # e.g., max speed of intruders
change_direction_intervals = [6, 9, 12]   # e.g., how often intruders change direction

# --- Evaluation Constants ---
# This parameter will be fixed during the evaluation
EVAL_ENV_SIZE = 100
NUM_TRIALS = 100  # Number of episodes to run for each parameter combination
MAX_STEPS_PER_EPISODE = 500 # Fixed episode length for consistent evaluation

# --- Results File Setup ---
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_dir = "./results/"
os.makedirs(results_dir, exist_ok=True)
csv_filename = f"{results_dir}evaluation_run_{timestamp}.csv"

# --- 2. FIND ALL MODELS TO EVALUATE ---
try:
    model_files = [os.path.join(CHECKPOINT_DIR, f) for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.zip')]
    if not model_files:
        print(f"❌ Error: No model .zip files found in '{CHECKPOINT_DIR}'. Please check the path.")
        exit()
    print(f"✅ Found {len(model_files)} models to evaluate in '{CHECKPOINT_DIR}'.")
except FileNotFoundError:
    print(f"❌ Error: The directory '{CHECKPOINT_DIR}' does not exist.")
    exit()

# This list will hold all results from all models
all_results = []

# --- 3. MAIN EVALUATION LOOP ---

# Outer loop iterates through each found model file
for model_path in tqdm(model_files, desc="Processing Models", position=0, leave=True):
    model_name = os.path.basename(model_path)
    print(f"\n--- Loading model: {model_name} ---")
    model = PPO.load(model_path)

    # +++ ADDED: Loop for the number of intruders +++
    param_loops = tqdm(
        [(num, size, speed, interval) 
         for num in numbers_of_intruders 
         for size in intruder_sizes 
         for speed in intruder_speeds 
         for interval in change_direction_intervals],
        desc=f"  Evaluating Params",
        position=1, 
        leave=False
    )
    for num_intruders, size, speed, interval in param_loops:
        
        # --- Metrics for this specific parameter combination ---
        total_collisions_for_setting = 0
        all_trial_avg_deviations = []
        all_trial_max_deviations = []

        # Run multiple trials for statistical significance
        for _ in range(NUM_TRIALS):
            try:
                # +++ MODIFIED: Instantiate the environment with the current loop parameters +++
                env = IntruderAvoidanceEnv(
                    number_of_intruders=num_intruders,
                    bounds=[[0, 0], [EVAL_ENV_SIZE, EVAL_ENV_SIZE]],
                    intruder_size=size,
                    max_intruder_speed=speed,
                    change_direction_interval=interval
                )

                obs, _ = env.reset()
                done = False
                
                episode_collisions = 0
                episode_deviations = []

                # Run one full episode
                for step in range(MAX_STEPS_PER_EPISODE):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, done, truncated, info = env.step(action)

                    current_deviation = np.linalg.norm(env.agent.position - env.target_position)
                    episode_deviations.append(current_deviation)

                    if info.get("collision", False):
                        episode_collisions += 1
                    
                    if done or truncated:
                        break

                # --- After each episode/trial ---
                total_collisions_for_setting += episode_collisions
                if episode_deviations:
                    all_trial_avg_deviations.append(np.mean(episode_deviations))
                    all_trial_max_deviations.append(np.max(episode_deviations))

            except Exception as e:
                print(f"An error occurred during trial: {e}")

        # --- After all trials for a specific parameter setting ---
        total_steps_in_setting = NUM_TRIALS * MAX_STEPS_PER_EPISODE
        avg_collisions_per_step = total_collisions_for_setting / total_steps_in_setting
        avg_deviation = np.mean(all_trial_avg_deviations) if all_trial_avg_deviations else 0
        avg_max_deviation = np.mean(all_trial_max_deviations) if all_trial_max_deviations else 0
        
        # +++ MODIFIED: Append all relevant data, including number of intruders +++
        all_results.append([
            model_name,
            num_intruders,
            size,
            speed,
            interval,
            avg_collisions_per_step,
            avg_deviation,
            avg_max_deviation
        ])

    # --- After evaluating one model against all parameter sets, save intermediate results ---
    # +++ MODIFIED: New column names for the DataFrame +++
    df = pd.DataFrame(all_results, columns=[
        "Model Checkpoint", "Number of Intruders", "Intruder Size", "Intruder Speed", 
        "Direction Change Interval", "Avg Collisions per Step", "Avg Deviation", "Avg of Max Deviations"
    ])
    df.to_csv(csv_filename, index=False)
    print(f"\n📂 Intermediate results for {model_name} saved to {csv_filename}")


print(f"\n✅ Evaluation complete! Final results for all models saved to {csv_filename}")