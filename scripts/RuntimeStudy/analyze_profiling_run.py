import pstats

# --- Configuration ---
PROFILE_FILE_NAME = 'scripts/RuntimeStudy/Results/profiling_run_1Mil_steps.prof'
TOTAL_STEPS = 1_000_000

# --- Load the profiling data ---
stats = pstats.Stats(PROFILE_FILE_NAME)

# --- Define the exact keys for major components ---
learn_method_key = ('/home/bake/Projects/Rest-to-Rest/.venv/lib/python3.10/site-packages/stable_baselines3/ppo/ppo.py', 302, 'learn')
simulation_step_key = ('/home/bake/Projects/Rest-to-Rest/gym_pathfinding/envs/intruder_avoidance_env.py', 137, 'step')
nn_train_key = ('/home/bake/Projects/Rest-to-Rest/.venv/lib/python3.10/site-packages/stable_baselines3/ppo/ppo.py', 184, 'train')
collect_rollouts_key = ('/home/bake/Projects/Rest-to-Rest/.venv/lib/python3.10/site-packages/stable_baselines3/common/on_policy_algorithm.py', 162, 'collect_rollouts')

# --- Extract timing data ---
try:
    total_runtime = stats.stats[learn_method_key][3]
    simulation_time = stats.stats[simulation_step_key][3]
    training_time = stats.stats[nn_train_key][3]
    collect_rollouts_time = stats.stats[collect_rollouts_key][3]
except KeyError as e:
    print(f"ERROR: A key was not found: {e}")
    exit()

# --- Deeper analysis of overhead ---
# The time in `collect_rollouts` *includes* the simulation time (`env.step`).
# The true overhead of data collection is the time spent in `collect_rollouts` MINUS the time spent in `env.step`.
data_collection_overhead = collect_rollouts_time - simulation_time

# The remaining overhead is everything else.
other_overhead = total_runtime - training_time - collect_rollouts_time

# --- Calculate Percentages ---
sim_percentage = (simulation_time / total_runtime) * 100
train_percentage = (training_time / total_runtime) * 100
data_collection_percentage = (data_collection_overhead / total_runtime) * 100
other_percentage = (other_overhead / total_runtime) * 100

# --- Print the Results ---
print("\n--- Overall Performance ---")
print(f"Total Runtime for {TOTAL_STEPS:,} steps: {total_runtime:.2f} seconds")
print(f"Steps Per Second (SPS): {(TOTAL_STEPS / total_runtime):.2f}")
print("-" * 30)

print("--- Detailed Time Breakdown ---")
print(f"1. Environment Simulation (env.step):         {simulation_time:.2f}s ({sim_percentage:.2f}%)")
print(f"2. Data Collection & Buffering (Overhead):    {data_collection_overhead:.2f}s ({data_collection_percentage:.2f}%)")
print(f"3. Neural Network Training (model.train):       {training_time:.2f}s ({train_percentage:.2f}%)")
print(f"4. Other (Logging, Callbacks, etc.):        {other_overhead:.2f}s ({other_percentage:.2f}%)")
print("-" * 30)