import time
import os
from stable_baselines3 import PPO

# --- Make sure your custom network is importable ---
from LidarCNN import LidarCNN

# --- CONFIGURATION ---
# Use the absolute path to a model file on the HPC
MODEL_PATH = "/home/bake/Rest-to-Rest/models/trial_535_agent.zip"
NUM_BENCHMARK_RUNS = 20 # Loading is slow, so we use fewer runs than for inference

def benchmark_io_load(model_path: str):
    """
    Loads a model file repeatedly to measure the average I/O time.
    """
    print(f"--- Benchmarking I/O for model loading ---")
    print(f"Model Path: {model_path}")

    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        return

    # 1. Define the custom objects needed to load the LidarCNN model
    custom_objects = {
        "policy": {
            "features_extractor_class": LidarCNN,
            "features_extractor_kwargs": {
                # We need a dummy value here for the class constructor
                "num_lidar_scans": 24
            }
        }
    }

    load_times = []
    print(f"\n--- Running benchmark ({NUM_BENCHMARK_RUNS} iterations) ---")
    for i in range(NUM_BENCHMARK_RUNS):
        print(f"Starting load iteration {i + 1}/{NUM_BENCHMARK_RUNS}...")
        start_time = time.time()

        # This is the core I/O operation being measured
        _ = PPO.load(model_path, custom_objects=custom_objects, device='cpu')

        end_time = time.time()
        duration = end_time - start_time
        load_times.append(duration)
        print(f"  > Iteration {i + 1} took {duration:.4f} seconds.")

    # 3. Calculate and report the results
    avg_load_time = sum(load_times) / len(load_times)

    print("\n--- I/O Benchmark Results ---")
    print(f"Total time for {NUM_BENCHMARK_RUNS} loads: {sum(load_times):.4f} seconds")
    print(f"Average time per model load: {avg_load_time:.4f} seconds")


if __name__ == "__main__":
    benchmark_io_load(MODEL_PATH)
