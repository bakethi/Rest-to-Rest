import gymnasium as gym
import numpy as np
import pandas as pd
import datetime
import os
import json
import argparse
from tqdm import tqdm
from stable_baselines3 import PPO

# It's good practice to make the environment importable if it's custom
# from gym_pathfinding.envs.intruder_avoidance_env import IntruderAvoidanceEnv

# +++ NEW FOR OPTUNA: Define a placeholder for the environment until it's properly installed
# This allows the script to be read without error, but it must be replaced
# with your actual import.
try:
    from gym_pathfinding.envs.intruder_avoidance_env import IntruderAvoidanceEnv
except ImportError:
    print("⚠️ Warning: Could not import IntruderAvoidanceEnv. Using a dummy class.")
    # If you run this script standalone without the env installed, it will fail at instantiation.
    # This is fine, as it's meant to be called by your main Optuna script where the env is available.
    class IntruderAvoidanceEnv: pass


# --- 1. CONFIGURATION ---

# These parameters define the static evaluation gauntlet that every model must pass through.
# Every model trained by an Optuna trial will be judged against these same conditions.
NUMBERS_OF_INTRUDERS = [5, 10, 20, 40]
INTRUDER_SIZES = [3, 6, 9]
INTRUDER_SPEEDS = [1, 2, 3]
CHANGE_DIRECTION_INTERVALS = [6, 9, 12]

# --- Evaluation Constants ---
EVAL_ENV_SIZE = 100
NUM_TRIALS_PER_CONDITION = 50  # Reduced for faster Optuna trials, adjust as needed
MAX_STEPS_PER_EPISODE = 500

# +++ NEW FOR OPTUNA: Define weights for the final KPI +++
# This is the most important part to customize.
# You need to decide the trade-off between safety (collisions) and efficiency (deviation).
# For example, you might decide that a collision is 100 times worse than a deviation of 1 unit.
W_COLLISION = 100.0  # Weight for collisions
W_DEVIATION = 1.0    # Weight for path deviation

def evaluate_model(model_path: str, log_file: str = None):
    """
    Evaluates a single RL model across a gauntlet of environmental conditions and
    computes a single Key Performance Indicator (KPI).

    Args:
        model_path (str): Path to the saved model .zip file.
        log_file (str, optional): Path to save a detailed CSV log. Defaults to None.
    """
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at '{model_path}'")
        # Return a JSON object with an infinite KPI to signal failure to Optuna
        print(json.dumps({"kpi": float('inf'), "error": "Model not found"}))
        return

    print(f"\n--- Loading model: {os.path.basename(model_path)} ---")
    model = PPO.load(model_path)

    # This list will hold all detailed results before aggregation
    all_results_data = []

    # Create the full list of parameter combinations to test against
    param_combinations = [
        (num, size, speed, interval)
        for num in NUMBERS_OF_INTRUDERS
        for size in INTRUDER_SIZES
        for speed in INTRUDER_SPEEDS
        for interval in CHANGE_DIRECTION_INTERVALS
    ]

    # Loop through the evaluation gauntlet
    param_loops = tqdm(param_combinations, desc="  Evaluating Conditions", position=0, leave=True)
    for num_intruders, size, speed, interval in param_loops:
        
        # --- Metrics for this specific parameter combination ---
        total_collisions_for_setting = 0
        all_trial_avg_deviations = []

        # Run multiple trials for statistical significance
        for _ in range(NUM_TRIALS_PER_CONDITION):
            try:
                env = IntruderAvoidanceEnv(
                    number_of_intruders=num_intruders,
                    bounds=[[0, 0], [EVAL_ENV_SIZE, EVAL_ENV_SIZE]],
                    intruder_size=size,
                    max_intruder_speed=speed,
                    change_direction_interval=interval
                )
                obs, _ = env.reset()
                
                episode_collisions = 0
                episode_deviations = []

                for _ in range(MAX_STEPS_PER_EPISODE):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, done, truncated, info = env.step(action)

                    current_deviation = np.linalg.norm(env.agent.position - env.target_position)
                    episode_deviations.append(current_deviation)

                    if info.get("collision", False):
                        episode_collisions += 1
                    
                    if done or truncated:
                        break
                
                total_collisions_for_setting += episode_collisions
                if episode_deviations:
                    all_trial_avg_deviations.append(np.mean(episode_deviations))

            except Exception as e:
                print(f"An error occurred during trial: {e}")
                # Skip this trial if the environment fails for some reason
                continue

        # Aggregate metrics for this specific setting
        total_steps_in_setting = NUM_TRIALS_PER_CONDITION * MAX_STEPS_PER_EPISODE
        avg_collisions = total_collisions_for_setting / total_steps_in_setting if total_steps_in_setting > 0 else 0
        avg_deviation = np.mean(all_trial_avg_deviations) if all_trial_avg_deviations else 0
        
        all_results_data.append({
            "Number of Intruders": num_intruders,
            "Intruder Size": size,
            "Intruder Speed": speed,
            "Direction Change Interval": interval,
            "Avg Collisions per Step": avg_collisions,
            "Avg Deviation": avg_deviation
        })

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # +++ NEW FOR OPTUNA: AGGREGATE ALL RESULTS INTO A SINGLE KPI +++
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    if not all_results_data:
        print("❌ Error: No evaluation data was collected.")
        print(json.dumps({"kpi": float('inf'), "error": "No data collected"}))
        return

    # Create a DataFrame for easy calculation
    df = pd.DataFrame(all_results_data)

    # Calculate the overall average across all conditions
    # This represents the model's general performance
    overall_avg_collisions = df["Avg Collisions per Step"].mean()
    overall_avg_deviation = df["Avg Deviation"].mean()

    # Combine the metrics into a single KPI score to be MINIMIZED
    # The lower this score, the better the model.
    final_kpi = (W_COLLISION * overall_avg_collisions) + (W_DEVIATION * overall_avg_deviation)

    print(f"\n--- Model Evaluation Summary ---")
    print(f"Overall Avg Collisions: {overall_avg_collisions:.6f}")
    print(f"Overall Avg Deviation:  {overall_avg_deviation:.4f}")
    print(f"Final Weighted KPI:     {final_kpi:.4f}")

    # --- Primary Output for Optuna: Print KPI in JSON format to stdout ---
    output_data = {
        "kpi": final_kpi,
        "details": {
            "overall_avg_collisions": overall_avg_collisions,
            "overall_avg_deviation": overall_avg_deviation
        }
    }
    print("\n---JSON_OUTPUT_START---")
    print(json.dumps(output_data))
    print("---JSON_OUTPUT_END---")


    # --- Secondary Output: Save detailed log file if requested ---
    if log_file:
        df["Model"] = os.path.basename(model_path)
        df.to_csv(log_file, index=False)
        print(f"\n📂 Detailed log saved to {log_file}")

# --- MODIFIED FOR OPTUNA: Main execution block to handle command-line arguments ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a single SB3 model and compute a KPI.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model .zip file to evaluate.")
    parser.add_argument("--log_file", type=str, default=None, help="Optional. Path to save a detailed CSV log file.")
    
    args = parser.parse_args()
    
    evaluate_model(model_path=args.model_path, log_file=args.log_file)