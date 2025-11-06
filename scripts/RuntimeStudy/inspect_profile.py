import pstats
import sys

# --- Configuration ---
PROFILE_FILE_NAME = 'scripts/RuntimeStudy/profiling_run_1Mil_steps.prof'

print(f"--- Inspecting functions in: {PROFILE_FILE_NAME} ---")

try:
    stats = pstats.Stats(PROFILE_FILE_NAME)
except Exception as e:
    print(f"Error opening file: {e}")
    print("Please make sure the .prof file is in the same directory as this script.")
    sys.exit()

# This will print a list of every function that was profiled.
# The format is: (filename, line_number, function_name)
for key in stats.stats:
    print(key)

print("\n--- Inspection Complete ---")
print("Look through the list above for the main training loop function.")
print("It will likely be a 'learn' or 'train' method from a stable_baselines3 file.")