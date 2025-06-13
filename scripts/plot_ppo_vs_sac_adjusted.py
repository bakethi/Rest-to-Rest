import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm
import numpy as np

'''
Here's how the 'Adjusted Collisions' metric is calculated:

    It combines the Avg Collisions from successful runs with a penalty for truncated runs.
    The penalty for each truncated run is set to 1.5 times the maximum Avg Collisions observed across all successful runs in both datasets.
    This penalty is then weighted by the proportion of truncated runs (i.e., 1−Success Rate/100), and the Avg Collisions is weighted by the Success Rate.
'''
# Create the output directory for plots
plot_dir = "./plots/sac_vs_ppo"
os.makedirs(plot_dir, exist_ok=True)

# Load the PPO and SAC results files
ppo_file = "run_2025-05-22_16-37-14_model_ppo_pathfinding_2025-03-17_13-58-44_penalty_for_standing_still.csv"
sac_file = "run_2025-06-10_12-37-57_model_sac_pathfinding_2025-06-10_07-43-11_24_50.csv"

df_ppo = pd.read_csv(ppo_file)
df_sac = pd.read_csv(sac_file)

# Ensure required columns exist
required_columns = {"Environment Size", "Obstacle %", "Normalized Steps", "Success Rate", "Avg Collisions"}
if not required_columns.issubset(df_ppo.columns) or not required_columns.issubset(df_sac.columns):
    print("❌ CSV file does not contain required columns. Exiting.")
    exit()

def plot_combined_data_matched_colors(df_ppo, df_sac, y_column, y_label, title, filename):
    plt.figure(figsize=(10, 6))

    # Get all unique obstacle percentages from both dataframes
    all_obstacle_percents = sorted(list(set(df_ppo["Obstacle %"].unique()) | set(df_sac["Obstacle %"].unique())))

    # Create a colormap
    colors = cm.get_cmap('viridis', len(all_obstacle_percents))

    for i, obstacle_percent in enumerate(all_obstacle_percents):
        color = colors(i)

        # Plot PPO data
        subset_ppo = df_ppo[df_ppo["Obstacle %"] == obstacle_percent]
        if not subset_ppo.empty:
            plt.plot(subset_ppo["Environment Size"], subset_ppo[y_column], marker="o", linestyle="-", color=color,
                     label=f"PPO - {obstacle_percent}% Obstacles")

        # Plot SAC data
        subset_sac = df_sac[df_sac["Obstacle %"] == obstacle_percent]
        if not subset_sac.empty:
            plt.plot(subset_sac["Environment Size"], subset_sac[y_column], marker="x", linestyle="--", color=color,
                     label=f"SAC - {obstacle_percent}% Obstacles")

    plt.xlabel("Environment Size")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid()
    plot_path = os.path.join(plot_dir, filename)
    plt.savefig(plot_path)
    print(f"📊 Saved plot: {plot_path}")

# Plot 1: Environment Size vs. Normalized Steps
plot_combined_data_matched_colors(df_ppo, df_sac, "Normalized Steps", "Normalized Steps (Steps per Unit Size)",
                   "Environment Size vs. Normalized Steps Required", "combined_normalized_steps_matched_colors.png")

# Plot 2: Environment Size vs. Success Rate
plot_combined_data_matched_colors(df_ppo, df_sac, "Success Rate", "Success Rate (%)",
                   "Environment Size vs. Success Rate", "combined_success_rate_matched_colors.png")

# Plot 3: Environment Size vs. Collisions
plot_combined_data_matched_colors(df_ppo, df_sac, "Avg Collisions", "Average Collisions",
                   "Environment Size vs. Average Collisions", "combined_avg_collisions_matched_colors.png")