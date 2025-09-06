import os
import subprocess
import sys

# The name of your evaluation script file
EVALUATION_SCRIPT_NAME = "scripts/OptunaMultiObjective/evaluate_intruder_for_optuna.py"

# --- CONFIGURATION ---
# Define all models to be evaluated here.
# For each model, specify the path to the saved agent (.zip file)
# and the desired output path for the evaluation results (.csv file).
csv_output_path = "/home/bake/Projects/Rest-to-Rest/results/PBRS-Full-Eval"
MODELS_TO_EVALUATE = {
#    "Baseline": {
#        "model_path": "models/baseline_average_params/agent_avg.zip",  # <-- UPDATE THIS PATH
#        "log_path": f"{csv_output_path}/baseline_average_params_evaluation.csv"
#    },
#    "Single-Objective TPE": {
#        "model_path": "models/best_model_24_50_PBRS_SO_TPE/SO-TPE.zip",  # <-- UPDATE THIS PATH
#        "log_path": f"{csv_output_path}/single_objective_TPE_evaluation.csv"
#    },
#    "MO NSGA-II (Safest)": {
#        "model_path": "models/best_model_24_50_PBRS_MO_NSGAII/safest.zip",  # <-- UPDATE THIS PATH
#        "log_path": f"{csv_output_path}/MO_NSGAII/MO_NSGAII_evaluation_safest.csv"
#    },
    "MO NSGA-II (Balanced)": {
        "model_path": "models/best_model_24_50_PBRS_MO_NSGAII/balanced.zip",  # <-- UPDATE THIS PATH
        "log_path": f"{csv_output_path}/MO_NSGAII/evaluation_balanced.csv"
    },
#    "MO NSGA-II (Most Efficient)": {
#        "model_path": "models/best_model_24_50_PBRS_MO_NSGAII/most_efficient.zip",  # <-- UPDATE THIS PATH
#        "log_path": f"{csv_output_path}/MO_NSGAII/evaluation_most_efficient.csv"
#    },
    "MO Random Sampler (Safest)": {
        "model_path": "models/best_model_24_50_PBRS_Random_Sampler/safest.zip", # <-- UPDATE
        "log_path": f"{csv_output_path}/MO_Random_Sampler/evaluation_safest.csv"
    },
#    "MO Random Sampler (Balanced)": {
#        "model_path": "models/best_model_24_50_PBRS_Random_Sampler/balanced.zip", # <-- UPDATE
#        "log_path": f"{csv_output_path}/MO_Random_Sampler/evaluation_balanced.csv"
#    },
    "MO Random Sampler (Most Efficient)": {
        "model_path": "models/best_model_24_50_PBRS_Random_Sampler/most_efficient.zip", # <-- UPDATE
        "log_path": f"{csv_output_path}/MO_Random_Sampler/evaluation_most_efficient.csv"
    },
#    "SO CmaEs": {
#        "model_path": "models/best_model_24_50_PBRS_SO_CmaEs/SO-CmaEs.zip", # <-- UPDATE
#        "log_path": f"{csv_output_path}/SO_CmaEs/evaluation_SO-CmaEs.csv"
#    },
#    "SO GPSampler": {
#        "model_path": "models/best_model_24_50_PBRS_SO_GPSampler/SO-GPSampler.zip", # <-- UPDATE
#        "log_path": f"{csv_output_path}/SO_GPSampler/evaluation_SO-GPSampler.csv"
#    },
#    "MO TPE (Safest)": {
#        "model_path": "models/best_model_24_50_PBRS_MO_TPE/safest.zip",
#        "log_path": f"{csv_output_path}/MO_TPE/evaluation_safest.csv"
#    },
        "MO TPE (Balanced)": {
        "model_path": "models/best_model_24_50_PBRS_MO_TPE/balanced.zip",
        "log_path": f"{csv_output_path}/MO_TPE/evaluation_balanced.csv"
    },
        "MO TPE (Most Efficient)": {
        "model_path": "models/best_model_24_50_PBRS_MO_TPE/most_efficient.zip",
        "log_path": f"{csv_output_path}/MO_TPE/evaluation_most_efficient.csv"
    },
        "MO NSGAIII (Safest)": {
        "model_path": "models/best_model_24_50_PBRS_MO_NSGAIII/safest.zip",
        "log_path": f"{csv_output_path}/MO_NSGAIII/evaluation_safest.csv"
    },
        "MO NSGAIII (Balanced)": {
        "model_path": "models/best_model_24_50_PBRS_MO_NSGAIII/balanced.zip",
        "log_path": f"{csv_output_path}/MO_NSGAIII/evaluation_balanced.csv"
    },
#        "MO NSGAIII (Most Efficient)": {
#        "model_path": "models/best_model_24_50_PBRS_MO_NSGAIII/most_efficient.zip",
#        "log_path": f"{csv_output_path}/MO_NSGAIII/evaluation_most_efficient.csv"
#    },
}


def main():
    """
    Main function to validate paths and then evaluate all specified models.
    """
    print("🚀 Starting master evaluation script...")

    # --- 1. SCRIPT EXISTENCE CHECK ---
    if not os.path.exists(EVALUATION_SCRIPT_NAME):
        print(f"❌ Error: Evaluation script '{EVALUATION_SCRIPT_NAME}' not found.")
        print("Please ensure this script is in the same directory.")
        return

    # --- 2. PRE-CHECK: VALIDATE ALL MODEL PATHS BEFORE STARTING ---
    print("\n🔍 Performing pre-check of all model paths...")
    missing_models = []
    for model_name, paths in MODELS_TO_EVALUATE.items():
        if not os.path.exists(paths["model_path"]):
            missing_models.append((model_name, paths["model_path"]))

    if missing_models:
        print("\n❌ PRE-CHECK FAILED. The following model files could not be found:")
        for name, path in missing_models:
            print(f"   - Model '{name}': {path}")
        print("\nPlease correct the paths in the 'MODELS_TO_EVALUATE' dictionary and try again.")
        return  # Exit the script immediately

    print("✅ Pre-check successful! All model files found.")

    # --- 3. EXECUTE EVALUATIONS ---
    for model_name, paths in MODELS_TO_EVALUATE.items():
        model_path = paths["model_path"]
        log_path = paths["log_path"]

        print(f"\n{'='*60}")
        print(f"▶️  Now Evaluating: {model_name}")
        print(f"   - Model Path: {model_path}")
        print(f"   - Log File:   {log_path}")
        print(f"{'='*60}")

        # Create the directory for the log file if it doesn't exist
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # Construct the command to run the evaluation script
        command = [
            sys.executable,  # Use the same python interpreter
            EVALUATION_SCRIPT_NAME,
            "--model_path", model_path,
            "--log_file", log_path
        ]

        try:
            # Execute the command
            subprocess.run(command, check=True, text=True)
            print(f"✅ Successfully evaluated '{model_name}'.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error evaluating '{model_name}'. The script exited with an error.")
            print(f"   Return code: {e.returncode}")
        except FileNotFoundError:
            print(f"❌ Error: Could not find the Python interpreter '{sys.executable}'.")

if __name__ == "__main__":
    main()
    print("\n🏁 All model evaluations complete.")