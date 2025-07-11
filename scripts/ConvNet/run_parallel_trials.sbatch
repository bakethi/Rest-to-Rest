#!/bin/bash

#SBATCH --job-name=optuna_parallel_rl
#SBATCH --output=slurm_logs/worker_%A_%a.out  # Log file for each worker
#SBATCH --error=slurm_logs/worker_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4  # CPUs needed for one training run
#SBATCH --mem=16G             # Memory needed for one training run
#SBATCH --time=24:00:00       # Max runtime for each worker

# --- 💡 KEY CONFIGURATION ---
# Define how many parallel workers you want.
# --array=0-19 will launch 20 workers.
#SBATCH --array=0-19

# Define the study name and the location of the shared database file.
# This file will be created in your project directory.
STUDY_NAME="IntruderAvoidance-CNN-Parallel"
STORAGE_URL="sqlite:///shared_study.db"


# --- SCRIPT EXECUTION ---
echo "--- Starting Optuna Worker ${SLURM_ARRAY_TASK_ID} for study '${STUDY_NAME}' ---"

# Create directories for logs and the database if they don't exist
mkdir -p slurm_logs

# Prepare your Python environment (adjust if needed)
module load python/3.10
source .venv/bin/activate

# Run the python script for one worker.
# With 20 workers and 10 trials each, you will run 200 total trials.
python scripts/train_optuna_manager.py \
    --study_name "$STUDY_NAME" \
    --storage_url "$STORAGE_URL" \
    --n_trials_per_worker 10

echo "--- Worker ${SLURM_ARRAY_TASK_ID} finished. ---"