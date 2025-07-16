import gymnasium as gym
import numpy as np
import time
import torch
from stable_baselines3 import PPO

# --- Make sure your custom network is importable ---
from LidarCNN import LidarCNN

# --- CONFIGURATION ---
MODEL_PATH = "models/trial_535_agent.zip"
NUM_LIDAR_SCANS = 24
OTHER_STATE_VARIABLES = 4 # (e.g., velocity x, velocity y, target distance, target angle)
TOTAL_OBS_DIM = NUM_LIDAR_SCANS + OTHER_STATE_VARIABLES
NUM_BENCHMARK_RUNS = 1000

def benchmark_inference(model_path: str):
    """
    Loads a trained model and measures its average inference time.
    """
    print(f"--- Loading model for benchmarking ---")
    print(f"Model Path: {model_path}")

    # 1. Define the custom objects to correctly load the LidarCNN model
    custom_objects = {
        "policy": {
            "features_extractor_class": LidarCNN,
            "features_extractor_kwargs": {
                "num_lidar_scans": NUM_LIDAR_SCANS
            }
        }
    }

    # 2. Load the model onto the CPU
    try:
        model = PPO.load(model_path, custom_objects=custom_objects, device='cpu')
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 3. Create a dummy observation state
    #    This simulates a single state reading from the environment.
    dummy_observation = np.random.rand(TOTAL_OBS_DIM).astype(np.float32)
    print(f"\nCreated a dummy observation with shape: {dummy_observation.shape}")

    # 4. Perform a single "warm-up" run
    #    This helps ensure that any one-time setup costs are not part of the measurement.
    print("Performing a warm-up inference run...")
    _ = model.predict(dummy_observation, deterministic=True)
    print("Warm-up complete.")

    # 5. Run the benchmark
    print(f"\n--- Running benchmark ({NUM_BENCHMARK_RUNS} iterations) ---")
    start_time = time.time()
    for _ in range(NUM_BENCHMARK_RUNS):
        # This is the core operation being measured: one forward pass
        action, _ = model.predict(dummy_observation, deterministic=True)
    end_time = time.time()

    # 6. Calculate and report the results
    total_time = end_time - start_time
    avg_inference_time_ms = (total_time / NUM_BENCHMARK_RUNS) * 1000 # Convert to milliseconds

    print("\n--- Benchmark Results ---")
    print(f"Total time for {NUM_BENCHMARK_RUNS} runs: {total_time:.4f} seconds")
    print(f"Average inference time per pass: {avg_inference_time_ms:.4f} ms")


if __name__ == "__main__":
    benchmark_inference(MODEL_PATH)