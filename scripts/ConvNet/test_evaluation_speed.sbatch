#!/bin/bash

#SBATCH --job-name=eval_speed_test
#SBATCH --output=slurm_logs/eval_test_%j.out
#SBATCH --error=slurm_logs/eval_test_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00 # 1 hour is more than enough for this test

# --- ✅ Use the ULTIMATE partition ---
#SBATCH --partition=ultimate
#SBATCH --qos=ultimate
#SBATCH --account=ultimate

echo "--- Starting Evaluation Speed Test on ULTIMATE partition ---"

# Prepare your Python environment
source .venv/bin/activate

# Run the same evaluation command
python -u scripts/ConvNet/evaluate_intruder_for_optuna.py --model_path /home/bake/Rest-to-Rest/optuna_trials/IntruderAvoidance-CNN-Parallel/trial_535/agent.zip

echo "--- Evaluation Speed Test Finished ---"