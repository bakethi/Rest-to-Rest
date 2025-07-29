# optuna_study.py
import optuna
import subprocess
import json
import os
import re
from optuna.samplers import CmaEsSampler

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
    model_path = os.path.join(trial_dir, "agent.zip")
    eval_log_path = os.path.join(trial_dir, "evaluation_details.csv")

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
        'python', 'scripts/OptunaMultiObjective/train_optuna.py',
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
            'python', 'scripts/OptunaMultiObjective/evaluate_intruder_for_optuna.py',
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
    sampler = CmaEsSampler()
    study = optuna.create_study(
        study_name="IntruderAvoidance-PBRS-MultiObjective-CmaEsSampler", # New name is recommended
        directions=["minimize", "minimize"], # Specify a direction for EACH objective
        storage="sqlite:///pbrs_tuning.db",
        sampler=sampler,
        load_if_exists=True
    )

    study.optimize(objective, n_trials=200)

    print("\n--- Optimization Finished ---")
    # For multi-objective, print the Pareto front (all non-dominated trials)
    print("Best trials (Pareto front):")
    for trial in study.best_trials:
        print(f"  Trial {trial.number}:")
        print(f"    Values: {trial.values}")
        print(f"    Params: {trial.params}")