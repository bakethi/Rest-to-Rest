#!/bin/bash

#SBATCH --job-name=optuna_rl_workers
#SBATCH --output=slurm_logs/worker_%A_%a.out
#SBATCH --error=slurm_logs/worker_%A_%a.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G             # Kept at 16G, increase if you run out of memory
#SBATCH --time=24:00:00

# --- ✅ HPC-SPECIFIC DIRECTIVES ---
# These are required for your cluster
#SBATCH --partition=debugging
#SBATCH --qos=debugging
#SBATCH --account=debugging

# --- 💡 KEY CONFIGURATION ---
# Launch 20 parallel workers
#SBATCH --array=0-19

# Define the study name and the shared database file
STUDY_NAME="IntruderAvoidance-CNN-Parallel"
STORAGE_URL="sqlite:///shared_study.db"


# --- SCRIPT EXECUTION ---
echo "--- Starting Optuna Worker ${SLURM_ARRAY_TASK_ID} for study '${STUDY_NAME}' ---"

# Create log directory if it doesn't exist
mkdir -p slurm_logs

# Prepare your Python environment
# This assumes you have already created '.venv' and installed packages
source .venv/bin/activate

# Run the python script for one worker
python scripts/ConvNet/train_optuna_manager.py \
    --study_name "$STUDY_NAME" \
    --storage_url "$STORAGE_URL" \
    --n_trials_per_worker 10

echo "--- Worker ${SLURM_ARRAY_TASK_ID} finished. ---"