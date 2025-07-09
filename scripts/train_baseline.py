import subprocess
import os

# --- 1. Define the Fixed Hyperparameters ---
# These are the "average" values you calculated.
average_params = {
    "d_safe": 25.0,
    "k_bubble": 105.0,
    "C_collision": 1250.0,
    "k_pos": 0.01,
    "k_action": 0.01,
    "w_safe": 0.5,
    "w_pos": 0.5
}

# --- 2. Define Output Paths ---
# Create a dedicated folder for this new baseline model
model_dir = "models/baseline_average_params/"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "agent_avg.zip")

# --- 3. Build and Run the Training Command ---
# This command calls your existing training script with the fixed parameters.
train_command = [
    'python', 'scripts/train_optuna.py',
    '--save_path', model_path,
    '--total_timesteps', '2000000', # You can adjust this as needed
]

# Dynamically add the hyperparameter arguments from the dictionary
for key, value in average_params.items():
    train_command.append(f'--{key}')
    train_command.append(str(value))

print("--- Training New 'Average Parameter' Baseline ---")
print(f"Saving model to: {model_path}")
print(f"Using parameters: {average_params}")
print("\nExecuting command:")
print(' '.join(train_command))
print("\n" + "="*50 + "\n")

try:
    # Run the training script. The output will stream live to your console.
    subprocess.run(train_command, check=True)
    print("\n" + "="*50)
    print("✅ Successfully trained the new baseline model.")
    print(f"Model saved at: {model_path}")

except subprocess.CalledProcessError as e:
    print(f"\n❌ An error occurred during training.")
    print(f"STDOUT: {e.stdout}")
    print(f"STDERR: {e.stderr}")