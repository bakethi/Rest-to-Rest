# optuna_study.py
import optuna
import subprocess
import json
import os
import re
import argparse

def objective(trial: optuna.Trial) -> float:
    """
    The main objective function for the Optuna study.
    A single trial consists of:
    1. Suggesting PBRS hyperparameters for the IntruderAvoidanceEnv.
    2. Calling `train.py` to train an agent with these hyperparameters.
    3. Calling `evaluate_for_optuna.py` to get a KPI for the trained agent.
    4. Returning the KPI.
    """
    # Create a unique directory for this trial's artifacts (model, logs)
    # After - A much more robust approach
    trial_dir = f"optuna_trials/{trial.study.study_name}/trial_{trial.number}"
    os.makedirs(trial_dir, exist_ok=True)
    model_path = os.path.abspath(os.path.join(trial_dir, "agent.zip"))
    eval_log_path = os.path.abspath(os.path.join(trial_dir, "evaluation_details.csv"))


    # --- 1. Suggest PBRS hyperparameters for the environment ---
    # The names of these suggestions MUST match the argparse names in train.py
    pbrs_params = {
        "d_safe": trial.suggest_float("d_safe", 10.0, 40.0),
        "k_bubble": trial.suggest_float("k_bubble", 10.0, 200.0),
        "C_collision": trial.suggest_float("C_collision", 500.0, 2000.0),
        "k_pos": trial.suggest_float("k_pos", 0.001, 0.1, log=True),
        "k_action": trial.suggest_float("k_action", 0.001, 0.1, log=True),
        "w_safe": trial.suggest_float("w_safe", 0.1, 0.9),
        "w_pos": trial.suggest_float("w_pos", 0.1, 0.9)
    }
    
    # --- 2. Build and run the training command ---
    train_command = [
        'python', 'scripts/ConvNet/train_optuna.py',
        '--save_path', model_path,
        '--total_timesteps', '2000000',
    ]
    # Dynamically add the hyperparameter arguments
    for key, value in pbrs_params.items():
        train_command.append(f'--{key}')
        train_command.append(str(value))

    print(f"\n--- Starting Trial {trial.number} ---")
    print(f"Parameters: {trial.params}")
    print(f"Training command: {' '.join(train_command)}")
    
    try:
        # Run the training script
        subprocess.run(train_command, check=True)

        # --- 3. Build and run the evaluation command ---
        eval_command = [
            'python', 'scripts/ConvNet/evaluate_intruder_for_optuna.py',
            '--model_path', model_path,
            '--log_file', eval_log_path
        ]
        print(f"Evaluation command: {' '.join(eval_command)}")

        # Run the evaluation script
        result = subprocess.run(eval_command, check=True, capture_output=True, text=True)

        # --- 4. Parse the KPI from the evaluation script's output ---
        json_output_str = re.search(r"---JSON_OUTPUT_START---(.*)---JSON_OUTPUT_END---", result.stdout, re.DOTALL)
        if not json_output_str:
            raise ValueError("Could not find JSON output block in evaluation script stdout.")
            
        kpi_data = json.loads(json_output_str.group(1))
        objective_values = kpi_data['values']

        print(f"✅ Trial {trial.number} finished with values: {objective_values}")
        return tuple(objective_values)

    except subprocess.CalledProcessError as e:
        print(f"❌ Trial {trial.number} failed in a subprocess.")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return float('inf') # Return a large number for failed trials
    except (ValueError, json.JSONDecodeError) as e:
        print(f"❌ Trial {trial.number} failed during KPI parsing: {e}")
        return float('inf')

# --- Main script execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run parallel trials for a single Optuna study.")
    parser.add_argument("--study_name", type=str, required=True, help="Name for the Optuna study.")
    parser.add_argument("--storage_url", type=str, required=True, help="Database URL for Optuna storage.")
    parser.add_argument("--n_trials_per_worker", type=int, default=10, help="Number of trials for this worker to run.")
    args = parser.parse_args()

    # This line connects to the central study database.
    # `load_if_exists=True` is crucial so that all workers join the same study.
    study = optuna.create_study(
        study_name=args.study_name,
        directions=["minimize", "minimize"],
        storage=args.storage_url,
        load_if_exists=True
    )

    # Each worker will independently run a set number of trials.
    study.optimize(objective, n_trials=args.n_trials_per_worker)

    print(f"\n--- Worker finished {args.n_trials_per_worker} trials ---")