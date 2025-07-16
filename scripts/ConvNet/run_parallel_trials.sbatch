#!/bin/bash

#SBATCH --job-name=optuna_rl_workers
#SBATCH --output=slurm_logs/worker_%A_%a.out
#SBATCH --error=slurm_logs/worker_%A_%a.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:7g.79gb:1
#SBATCH --partition=ultimate
#SBATCH --qos=ultimate
#SBATCH --account=ultimate
#SBATCH --array=0-19

# --- ✅ 1. SETUP FAST STORAGE ---
# Define the path to your permanent project directory on the shared filesystem
PERMANENT_STORAGE_DIR="/home/bake/Rest-to-Rest"

# Create a unique job directory on the local super-fast storage
# This directory will be deleted at the end of the job
LOCAL_JOB_DIR="/super_fast_storage/bake/job_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p $LOCAL_JOB_DIR

echo "--- Staging project to fast local storage: ${LOCAL_JOB_DIR} ---"

# --- ✅ 2. STAGE-IN ---
# Copy your project files to the fast local directory.
# We exclude the venv and other artifacts to keep the copy fast.
rsync -a --exclude '.venv' --exclude 'slurm_logs' --exclude '*.db' --exclude 'optuna_trials' $PERMANENT_STORAGE_DIR/ $LOCAL_JOB_DIR

# Change the current directory to the job's local storage directory
cd $LOCAL_JOB_DIR

# Define the study name and the path to the database on the fast storage
STUDY_NAME="IntruderAvoidance-CNN-Parallel"
STORAGE_URL="sqlite:///shared_study.db"


# --- ✅ 3. RUN THE JOB ---
echo "--- Starting Optuna Worker ${SLURM_ARRAY_TASK_ID} in ${PWD} ---"

# Prepare your Python environment (the path is now relative to the new CWD)
source .venv/bin/activate

# Run the python script. It will now save models and the DB to fast storage.
python -u scripts/ConvNet/train_optuna_manager.py \
    --study_name "$STUDY_NAME" \
    --storage_url "$STORAGE_URL" \
    --n_trials_per_worker 10

echo "--- Worker finished. Staging results back to permanent storage. ---"

# --- ✅ 4. STAGE-OUT ---
# Copy the results (database and trials) back to your permanent home directory
rsync -a optuna_trials/ $PERMANENT_STORAGE_DIR/optuna_trials/
rsync -a shared_study.db $PERMANENT_STORAGE_DIR/

# --- ✅ 5. CLEANUP ---
# Remove the temporary directory from the fast storage
rm -rf $LOCAL_JOB_DIR

echo "--- Job ${SLURM_ARRAY_TASK_ID} fully complete. ---"