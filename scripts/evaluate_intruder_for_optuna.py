import gymnasium as gym
import numpy as np
import pandas as pd
import json
import argparse
import os
import random
from tqdm import tqdm
from stable_baselines3 import PPO
import multiprocessing

try:
    from gym_pathfinding.envs.intruder_avoidance_env import IntruderAvoidanceEnv
except ImportError:
    class IntruderAvoidanceEnv: pass

# --- CONFIGURATION (remains the same) ---
NUMBERS_OF_INTRUDERS = [5, 10, 20, 40]
INTRUDER_SIZES = [3, 6, 9]
INTRUDER_SPEEDS = [1, 2, 3]
CHANGE_DIRECTION_INTERVALS = [6, 9, 12]
NUM_SAMPLED_CONDITIONS = 24
NUM_TRIALS_PER_CONDITION = 15
MAX_STEPS_PER_EPISODE = 1000
W_COLLISION = 100.0
W_DEVIATION = 1.0

# +++ FIX #1: The worker function now accepts `model_path` instead of `model` +++
def run_single_condition(args):
    model_path, num_intruders, size, speed, interval = args
    
    # +++ Each worker now loads its own copy of the model +++
    model = PPO.load(model_path)
    
    total_collisions_for_setting = 0
    all_trial_avg_deviations = []

    # The rest of this function remains exactly the same
    for _ in range(NUM_TRIALS_PER_CONDITION):
        try:
            env = IntruderAvoidanceEnv(
                number_of_intruders=num_intruders,
                bounds=[[0, 0], [100, 100]],  # Assuming EVAL_ENV_SIZE is 100
                intruder_size=size,
                max_intruder_speed=speed,
                change_direction_interval=interval
                )
            obs, _ = env.reset()
            episode_collisions = 0
            episode_deviations = []

            # Correctly run the episode loop
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
            print(f"Error during single trial evaluation: {e}")
            continue
    
    # ... (rest of the metric calculation and return) ...
    # This part is unchanged
    total_steps_in_setting = NUM_TRIALS_PER_CONDITION * MAX_STEPS_PER_EPISODE
    avg_collisions = total_collisions_for_setting / total_steps_in_setting if total_steps_in_setting > 0 else 0
    avg_deviation = np.mean(all_trial_avg_deviations) if all_trial_avg_deviations else 0
    return {"Number of Intruders": num_intruders, "Intruder Size": size, "Intruder Speed": speed, "Direction Change Interval": interval, "Avg Collisions per Step": avg_collisions, "Avg Deviation": avg_deviation}


def evaluate_model(model_path: str, log_file: str = None):
    if not os.path.exists(model_path):
        print(json.dumps({"kpi": float('inf'), "error": "Model not found"}))
        return

    # +++ FIX #2: We no longer load the model here in the main process +++
    # print(f"\n--- Loading model: {os.path.basename(model_path)} ---")
    # model = PPO.load(model_path) # <-- REMOVE THIS LINE

    all_combinations = [(n, s, sp, i) for n in NUMBERS_OF_INTRUDERS for s in INTRUDER_SIZES for sp in INTRUDER_SPEEDS for i in CHANGE_DIRECTION_INTERVALS]
    param_combinations = random.sample(all_combinations, min(NUM_SAMPLED_CONDITIONS, len(all_combinations)))

    # +++ FIX #3: The pool arguments now contain the `model_path` string +++
    pool_args = [(model_path, *params) for params in param_combinations]
    all_results_data = []

    print(f"--- Evaluating on {len(param_combinations)} conditions in parallel using {os.cpu_count()} cores ---")
    with multiprocessing.Pool(os.cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(run_single_condition, pool_args), total=len(pool_args), desc="  Evaluating Conditions"):
            if result:
                all_results_data.append(result)
    
    # ... (The rest of the script is unchanged) ...
    if not all_results_data:
        print(json.dumps({"kpi": float('inf'), "error": "No data collected"}))
        return
    df = pd.DataFrame(all_results_data)
    overall_avg_collisions = df["Avg Collisions per Step"].mean()
    overall_avg_deviation = df["Avg Deviation"].mean()
    final_kpi = (W_COLLISION * overall_avg_collisions) + (W_DEVIATION * overall_avg_deviation)
    print(f"\n--- Model Evaluation Summary ---")
    print(f"Overall Avg Collisions: {overall_avg_collisions:.6f}")
    print(f"Overall Avg Deviation:  {overall_avg_deviation:.4f}")
    print(f"Final Weighted KPI:     {final_kpi:.4f}")
    output_data = {"kpi": final_kpi, "details": {"overall_avg_collisions": overall_avg_collisions, "overall_avg_deviation": overall_avg_deviation}}
    print("\n---JSON_OUTPUT_START---")
    print(json.dumps(output_data))
    print("---JSON_OUTPUT_END---")
    if log_file:
        df["Model"] = os.path.basename(model_path)
        df.to_csv(log_file, index=False)
        print(f"\n📂 Detailed log saved to {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a single SB3 model and compute a KPI.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--log_file", type=str, default=None)
    args = parser.parse_args()
    evaluate_model(model_path=args.model_path, log_file=args.log_file)