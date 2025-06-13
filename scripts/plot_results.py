import pandas as pd
import matplotlib.pyplot as plt
import os

# 🔹 Load the most recent results file automatically
results_dir = "./results/"
files = sorted(
    [f for f in os.listdir(results_dir) if f.startswith("run_") and f.endswith(".csv")],
    reverse=True
)

if not files:
    print("❌ No results found in the ./results/ directory.")
    exit()

latest_file = os.path.join(results_dir, files[0])
print(f"📂 Loading results from: {latest_file}")

# 🔹 Read results from CSV
df = pd.read_csv(latest_file)

# 🔹 Ensure required columns exist
required_columns = {"Environment Size", "Obstacle %", "Normalized Steps", "Success Rate", "Avg Collisions"}
if not required_columns.issubset(df.columns):
    print("❌ CSV file does not contain required columns. Exiting.")
    exit()

# 🔹 Create the output directory for plots
plot_dir = "./plots/"
os.makedirs(plot_dir, exist_ok=True)

# 🔹 Plot 1: Environment Size vs. Normalized Steps
plt.figure(figsize=(10, 6))
for obstacle_percent in df["Obstacle %"].unique():
    subset = df[df["Obstacle %"] == obstacle_percent]
    plt.plot(subset["Environment Size"], subset["Normalized Steps"], marker="o", label=f"{obstacle_percent}% Obstacles")

plt.xlabel("Environment Size")
plt.ylabel("Normalized Steps (Steps per Unit Size)")
plt.title("Environment Size vs. Normalized Steps Required")
plt.legend()
plt.grid()
plot_path_1 = os.path.join(plot_dir, "normalized_steps.png")
plt.savefig(plot_path_1)
print(f"📊 Saved plot: {plot_path_1}")

# 🔹 Plot 2: Environment Size vs. Success Rate
plt.figure(figsize=(10, 6))
for obstacle_percent in df["Obstacle %"].unique():
    subset = df[df["Obstacle %"] == obstacle_percent]
    plt.plot(subset["Environment Size"], subset["Success Rate"], marker="o", label=f"{obstacle_percent}% Obstacles")

plt.xlabel("Environment Size")
plt.ylabel("Success Rate (%)")
plt.title("Environment Size vs. Success Rate")
plt.legend()
plt.grid()
plot_path_2 = os.path.join(plot_dir, "success_rate.png")
plt.savefig(plot_path_2)
print(f"📊 Saved plot: {plot_path_2}")

# 🔹 Plot 3: Environment Size vs. Collisions
plt.figure(figsize=(10, 6))
for obstacle_percent in df["Obstacle %"].unique():
    subset = df[df["Obstacle %"] == obstacle_percent]
    plt.plot(subset["Environment Size"], subset["Avg Collisions"], marker="o", label=f"{obstacle_percent}% Obstacles")

plt.xlabel("Environment Size")
plt.ylabel("Avg Collisions per Episode")
plt.title("Environment Size vs. Collisions")
plt.legend()
plt.grid()
plot_path_3 = os.path.join(plot_dir, "collisions.png")
plt.savefig(plot_path_3)
print(f"📊 Saved plot: {plot_path_3}")

print("\n✅ All plots saved successfully!")
