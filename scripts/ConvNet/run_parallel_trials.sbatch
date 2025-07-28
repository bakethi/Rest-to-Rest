#!/bin/bash

#SBATCH --job-name=cnn-optuna-study
#SBATCH --output=slurm_logs/cnn_%A_%a.out
#SBATCH --error=slurm_logs/cnn_%A_%a.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:7g.79gb:1
#SBATCH --partition=ultimate
#SBATCH --qos=ultimate
#SBATCH --account=ultimate
#SBATCH --array=0-19

# --- ✅ 1. SETUP STORAGE PATHS ---
# Path to your permanent project directory on the shared filesystem
PERMANENT_STORAGE_DIR="/home/bake/Rest-to-Rest"

# --- ✨ NEW: Define the study name and the SHARED storage path ---
# This log file MUST be on the permanent, shared filesystem.
STUDY_NAME="IntruderAvoidance-CNN-JournalStorage"
SHARED_STORAGE_PATH="${PERMANENT_STORAGE_DIR}/${STUDY_NAME}.log"

# Create a unique job directory on the local super-fast storage for the project files
LOCAL_JOB_DIR="/super_fast_storage/bake/job_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p $LOCAL_JOB_DIR

echo "--- Staging project to fast local storage: ${LOCAL_JOB_DIR} ---"
echo "--- Shared Optuna study log located at: ${SHARED_STORAGE_PATH} ---"


# --- ✅ 2. STAGE-IN ---
# Copy your project files to the fast local directory.
# Exclusions are the same, ensuring a quick copy.
rsync -a --exclude '.venv' --exclude 'slurm_logs' --exclude '*.db' --exclude '*.log' --exclude 'optuna_trials' $PERMANENT_STORAGE_DIR/ $LOCAL_JOB_DIR

# Change the current directory to the job's local storage directory
cd $LOCAL_JOB_DIR


# --- ✅ 3. RUN THE JOB ---
echo "--- Starting Optuna Worker ${SLURM_ARRAY_TASK_ID} in ${PWD} ---"

# Prepare your Python environment
source .venv/bin/activate

# --- ✨ MODIFIED: Run the python script with the new argument ---
# We now point directly to the shared log file on the permanent storage.
python -u scripts/ConvNet/train_optuna_manager.py \
    --study_name "$STUDY_NAME" \
    --storage_path "$SHARED_STORAGE_PATH" \
    --n_trials_per_worker 10

echo "--- Worker finished. Staging results back to permanent storage. ---"


# --- ✅ 4. STAGE-OUT ---
# --- ✨ MODIFIED: Copy ONLY the trial artifacts back. ---
# The study log file is already in the correct permanent location.
# This prevents race conditions and ensures each worker saves its own results.
rsync -a optuna_trials/ $PERMANENT_STORAGE_DIR/optuna_trials/


# --- ✅ 5. CLEANUP ---
# Remove the temporary directory from the fast storage
rm -rf $LOCAL_JOB_DIR

echo "--- Job ${SLURM_ARRAY_TASK_ID} fully complete. ---"