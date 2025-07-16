import time
import numpy as np
import gymnasium as gym

# --- Make sure your custom environment is importable ---
try:
    from gym_pathfinding.envs.intruder_avoidance_env import IntruderAvoidanceEnv
    print("✅ Successfully imported IntruderAvoidanceEnv.")
except ImportError:
    print("❌ Error: Could not import IntruderAvoidanceEnv.")
    print("Please make sure the gym_pathfinding package is installed correctly.")
    exit()

# --- CONFIGURATION ---
# We use a high number of intruders to test a performance-intensive scenario
BENCHMARK_PARAMS = {
    'number_of_intruders': 40,
    'intruder_size': 6,
    'max_intruder_speed': 2,
    'change_direction_interval': 9
}
NUM_BENCHMARK_RUNS = 1000

def benchmark_environment_step():
    """
    Initializes the environment and measures the average time for a single step.
    """
    print("\n--- Initializing Environment for Benchmarking ---")
    try:
        env = IntruderAvoidanceEnv(**BENCHMARK_PARAMS)
        print("✅ Environment created successfully.")
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        return

    # 1. Reset the environment and get the initial state
    obs, _ = env.reset()
    print("Environment reset complete.")

    # 2. Get a random action to pass to the step function
    action = env.action_space.sample()

    # 3. Perform a single "warm-up" step
    #    This ensures any one-time setup costs aren't part of the measurement.
    print("Performing a warm-up step...")
    _ = env.step(action)
    print("Warm-up complete.")

    # 4. Run the benchmark
    print(f"\n--- Running benchmark ({NUM_BENCHMARK_RUNS} iterations) ---")
    start_time = time.time()
    for _ in range(NUM_BENCHMARK_RUNS):
        # This is the core operation being measured
        obs, _, done, truncated, info = env.step(action)
        # Reset the environment if an episode finishes to keep the benchmark going
        if done or truncated:
            env.reset()
    end_time = time.time()

    # 5. Calculate and report the results
    total_time = end_time - start_time
    avg_step_time_ms = (total_time / NUM_BENCHMARK_RUNS) * 1000 # Convert to milliseconds

    print("\n--- Environment Step Benchmark Results ---")
    print(f"Total time for {NUM_BENCHMARK_RUNS} steps: {total_time:.4f} seconds")
    print(f"Average time per env.step(): {avg_step_time_ms:.4f} ms")

if __name__ == "__main__":
    benchmark_environment_step()