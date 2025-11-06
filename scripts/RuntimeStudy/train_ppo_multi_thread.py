import time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from gym_pathfinding.envs.intruder_avoidance_env import IntruderAvoidanceEnv
import os
import torch
import psutil
import logging
import csv
import threading
from datetime import datetime

# --- BENCHMARK CONFIGURATION ---
CPU_COUNTS_TO_TEST = [1] + [2**i for i in range(1, (os.cpu_count() or 1).bit_length()) if 2**i <= (os.cpu_count() or 1)]
TOTAL_TIMESTEPS = 1_000_000
RESULTS_FILE = "scripts/RuntimeStudy/Results/multi_threading_scaling_results.csv"
MONITORING_INTERVAL_SEC = 5 # How often to log hardware stats

# --- LOGGING SETUP ---
log_file = "scripts/RuntimeStudy/Results/multi_threading_scaling_benchmark_log.txt"
if os.path.exists(log_file):
    os.remove(log_file)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)

class SystemMonitor(threading.Thread):
    """A thread that monitors and logs system utilization."""
    def __init__(self, interval=5):
        super().__init__()
        self.interval = interval
        self.daemon = True  # Allows main program to exit even if this thread is running
        self._stop_event = threading.Event()
        self.cpu_history = []
        self.gpu_history = []

    def run(self):
        logging.info("System monitor started.")
        while not self._stop_event.is_set():
            # CPU Utilization
            cpu_percent = psutil.cpu_percent(percpu=True)
            self.cpu_history.append(psutil.cpu_percent(percpu=False)) # Overall CPU
            logging.info(f"[UTILIZATION] CPU: {cpu_percent}%")
            
            # RAM Utilization
            ram = psutil.virtual_memory()
            logging.info(f"[UTILIZATION] RAM: {ram.percent}% used")

            # GPU Utilization
            if torch.cuda.is_available():
                # This requires the 'nvsmi' library: pip install nvsmi
                try:
                    import nvsmi
                    gpu_util = [gpu.gpu_util for gpu in nvsmi.get_gpus()][0]
                    self.gpu_history.append(gpu_util)
                    logging.info(f"[UTILIZATION] GPU: {gpu_util}%")
                except (ImportError, FileNotFoundError):
                    logging.warning("NVIDIA-SMI not found, cannot log GPU utilization.")
                    self.gpu_history.append(0) # Default to 0 if not found
            
            time.sleep(self.interval)

    def stop(self):
        self._stop_event.set()
        logging.info("System monitor stopped.")

    def get_average_utilization(self):
        avg_cpu = sum(self.cpu_history) / len(self.cpu_history) if self.cpu_history else 0
        avg_gpu = sum(self.gpu_history) / len(self.gpu_history) if self.gpu_history else 0
        return avg_cpu, avg_gpu

def snapshot_workstation_config():
    """Logs the hardware configuration of the machine."""
    logging.info("-" * 30)
    logging.info("--- Workstation Configuration ---")
    logging.info(f"CPU: {psutil.cpu_count(logical=True)} Cores")
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    logging.info(f"RAM: {ram_gb:.2f} GB")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_ram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logging.info(f"GPU: {gpu_name} ({gpu_ram_gb:.2f} GB)")
    else:
        logging.info("GPU: Not Available")
    logging.info("-" * 30)

def make_env():
    env = IntruderAvoidanceEnv()
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env

if __name__ == '__main__':
    snapshot_workstation_config()

    with open(RESULTS_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['num_cpu', 'total_runtime_sec', 'steps_per_second', 'avg_cpu_utilization_percent', 'avg_gpu_utilization_percent'])

    for num_cpu in CPU_COUNTS_TO_TEST:
        logging.info(f"\n{'='*15} STARTING RUN FOR {num_cpu} CPU CORE(S) {'='*15}")
        
        monitor = SystemMonitor(interval=MONITORING_INTERVAL_SEC)
        monitor.start()

        train_env = make_vec_env(make_env, n_envs=num_cpu, vec_env_cls=SubprocVecEnv if num_cpu > 1 else None)
        model = PPO("MlpPolicy", train_env, verbose=0)

        logging.info(f"--- Training with {num_cpu} core(s) for {TOTAL_TIMESTEPS:,} timesteps ---")
        start_time = time.time()
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        end_time = time.time()
        
        monitor.stop()
        monitor.join()

        total_runtime = end_time - start_time
        sps = TOTAL_TIMESTEPS / total_runtime
        avg_cpu, avg_gpu = monitor.get_average_utilization()

        logging.info(f"--- RESULTS FOR {num_cpu} CORE(S) ---")
        logging.info(f"Total Runtime: {total_runtime:.2f} seconds")
        logging.info(f"Steps Per Second (SPS): {sps:.2f}")
        logging.info(f"Average CPU Utilization: {avg_cpu:.2f}%")
        logging.info(f"Average GPU Utilization: {avg_gpu:.2f}%")

        with open(RESULTS_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([num_cpu, f"{total_runtime:.2f}", f"{sps:.2f}", f"{avg_cpu:.2f}", f"{avg_gpu:.2f}"])
            
        logging.info(f"{'='*15} FINISHED RUN FOR {num_cpu} CPU CORE(S) {'='*15}")

    logging.info("\n✅ All benchmark runs complete!")