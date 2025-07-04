import pandas as pd
import matplotlib.pyplot as plt
import os
import io

# The CSV data you provided is stored in this string
csv_data = "results/run_2025-06-20_09-54-43_model_ppo_intruder_24_50_simple_reward.csv"

# Use pandas to read the data from the string
df = pd.read_csv(csv_data)

# Create a directory to save the plots
plot_dir = "./plots/intruder_plots/Training_1"
os.makedirs(plot_dir, exist_ok=True)

# --- Plot 1: Environment Size vs. Avg Collisions per Step ---
plt.figure(figsize=(10, 6))
for intruder_percent in df["Intruder %"].unique():
    subset = df[df["Intruder %"] == intruder_percent]
    plt.plot(subset["Environment Size"], subset["Avg Collisions per Step"], marker="o", label=f"{intruder_percent}% Intruders")

plt.xlabel("Environment Size")
plt.ylabel("Avg Collisions per Step")
plt.title("Environment Size vs. Avg Collisions per Step")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "collisions_per_step.png"))
plt.close()

# --- Plot 2: Environment Size vs. Avg Deviation ---
plt.figure(figsize=(10, 6))
for intruder_percent in df["Intruder %"].unique():
    subset = df[df["Intruder %"] == intruder_percent]
    plt.plot(subset["Environment Size"], subset["Avg Deviation"], marker="o", label=f"{intruder_percent}% Intruders")

plt.xlabel("Environment Size")
plt.ylabel("Avg Deviation")
plt.title("Environment Size vs. Avg Deviation")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "avg_deviation.png"))
plt.close()

# --- Plot 3: Environment Size vs. Avg of Max Deviations ---
plt.figure(figsize=(10, 6))
for intruder_percent in df["Intruder %"].unique():
    subset = df[df["Intruder %"] == intruder_percent]
    plt.plot(subset["Environment Size"], subset["Avg of Max Deviations"], marker="o", label=f"{intruder_percent}% Intruders")

plt.xlabel("Environment Size")
plt.ylabel("Avg of Max Deviations")
plt.title("Environment Size vs. Avg of Max Deviations")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "max_deviations.png"))
plt.close()

print("Script finished and plots saved.")