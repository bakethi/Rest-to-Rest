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
PERMANENT_STORAGE_DIR="/home/bake/Rest-to-Rest"
STUDY_NAME="IntruderAvoidance-CNN-JournalStorage"
SHARED_STORAGE_PATH="${PERMANENT_STORAGE_DIR}/${STUDY_NAME}.log"
LOCAL_JOB_DIR="/super_fast_storage/bake/job_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

mkdir -p $LOCAL_JOB_DIR

# --- ✨ MODIFIED: Added inline timestamps to echo statements ---
echo "--- Staging project to fast local storage: ${LOCAL_JOB_DIR} --- $(date +"%d.%B %Y %H:%M:%S")"
echo "--- Shared Optuna study log located at: ${SHARED_STORAGE_PATH} ---"


# --- ✅ 2. STAGE-IN ---
# Copy the full project including the virtual environment
rsync -a --exclude 'slurm_logs' --exclude '*.db' --exclude '*.log' --exclude 'optuna_trials' $PERMANENT_STORAGE_DIR/ $LOCAL_JOB_DIR
cd $LOCAL_JOB_DIR


# --- ✅ 3. RUN THE JOB ---
echo "--- Starting Optuna Worker ${SLURM_ARRAY_TASK_ID} in ${PWD} --- $(date +"%d.%B %Y %H:%M:%S")"

# Prepare your Python environment
source .venv/bin/activate

# Run the python script
python -u scripts/ConvNet/train_optuna_manager.py \
    --study_name "$STUDY_NAME" \
    --storage_path "$SHARED_STORAGE_PATH" \
    --n_trials_per_worker 10

echo "--- Worker finished. Staging results back to permanent storage. --- $(date +"%d.%B %Y %H:%M:%S")"


# --- ✅ 4. STAGE-OUT ---
# Check if the directory exists before copying to prevent errors
if [ -d "optuna_trials" ]; then
    rsync -a optuna_trials/ $PERMANENT_STORAGE_DIR/optuna_trials/
else
    echo "Warning: 'optuna_trials' directory not found. Nothing to stage-out. --- $(date +"%d.%B %Y %H:%M:%S")"
fi


# --- ✅ 5. CLEANUP ---
rm -rf $LOCAL_JOB_DIR

echo "--- Job ${SLURM_ARRAY_TASK_ID} fully complete. --- $(date +"%d.%B %Y %H:%M:%S")"