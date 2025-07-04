import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import re
import os # Recommended for handling paths

# --- Configuration ---
csv_file_path = "results/evaluation_run_2025-06-24_07-48-46.csv"
save_dir = "plots/intruder_plots/Training_3"

# Create the save directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# --- Data Loading ---
# Load the data directly from the CSV file
df = pd.read_csv(csv_file_path)

# (Good Practice) Clean up column names to remove leading/trailing whitespace
df.columns = df.columns.str.strip()


# --- Pre-processing: Extract training steps from model name for easier plotting ---
def extract_steps(model_name):
    # Use regex to find the number before "_steps.zip"
    match = re.search(r'_(\d+)_steps\.zip$', model_name)
    if match:
        return int(match.group(1))
    return "New Model"

df['Training Steps'] = df['Model Checkpoint'].apply(extract_steps)

# --- Plotting ---
sns.set_theme(style="whitegrid")

# Plot 1: Collisions vs. Number of Intruders, grouped by Intruder Speed
plt.figure(figsize=(12, 8))
g = sns.FacetGrid(df, col="Intruder Size", hue="Intruder Speed", palette="viridis", height=5)
g.map(sns.lineplot, "Number of Intruders", "Avg Collisions per Step", marker="o", alpha=0.9)
g.add_legend()
g.fig.suptitle("Performance vs. Number of Intruders", y=1.03)
g.set_axis_labels("Number of Intruders", "Avg Collisions per Step")
plt.savefig(f"{save_dir}/plot1_collisions_vs_num_intruders.png", bbox_inches='tight')
plt.close()

# Plot 2: Deviation vs. Intruder Speed, grouped by Number of Intruders
plt.figure(figsize=(12, 8))
g = sns.FacetGrid(df, col="Intruder Size", hue="Number of Intruders", palette="plasma", height=5)
g.map(sns.lineplot, "Intruder Speed", "Avg Deviation", marker="o", alpha=0.9)
g.add_legend()
g.fig.suptitle("Path Deviation vs. Intruder Speed", y=1.03)
g.set_axis_labels("Intruder Speed", "Avg Deviation from Target")
plt.savefig(f"{save_dir}/plot2_deviation_vs_speed.png", bbox_inches='tight')
plt.close()

# Plot 3: Heatmap of Collisions vs. Speed and Number of Intruders
g = sns.FacetGrid(df, col="Training Steps", col_wrap=3, height=5)
def draw_heatmap(*args, **kwargs):
    data = kwargs.pop('data')
    pivot_table = data.pivot_table(index="Number of Intruders", columns="Intruder Speed", values="Avg Collisions per Step")
    sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="rocket", **kwargs)

g.map_dataframe(draw_heatmap)
g.fig.suptitle("Heatmap of Avg Collisions per Step", y=1.02)
g.set_axis_labels("Intruder Speed", "Number of Intruders")
plt.savefig(f"{save_dir}/plot3_collision_heatmap.png", bbox_inches='tight')
plt.close()

# Plot 4: Bar plot comparing models
# 1. Get the unique step values from the column
step_order = sorted(df['Training Steps'].unique())

# 2. Create a copy of the dataframe for this specific plot
df_plot4 = df.copy()

# 3. Convert the 'Training Steps' column to an ordered categorical type
# This ensures the legend and the bars are sorted numerically.
df_plot4['Training Steps'] = pd.Categorical(
    df_plot4['Training Steps'],
    categories=step_order,
    ordered=True
)


plt.figure(figsize=(15, 8))
sns.barplot(
    data=df_plot4, # Use the dataframe with the ordered category
    x="Number of Intruders",
    y="Avg Collisions per Step",
    hue="Training Steps", # Seaborn will now respect the order
    palette="coolwarm"
)
plt.title("Model Comparison: Collisions vs. Number of Intruders")
plt.xlabel("Number of Intruders")
plt.ylabel("Avg Collisions per Step")
plt.legend(title="Training Steps")
plt.savefig(f"{save_dir}/plot4_model_comparison.png", bbox_inches='tight') # Changed filename
plt.close()

# Plot 5: Scatter plot for Safety vs. Deviation trade-off
df_plot5 = df.copy()
df_plot5['Intruder Speed'] = df_plot5['Intruder Speed'].astype('category')
df_plot5['Number of Intruders'] = df_plot5['Number of Intruders'].astype('category')


plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=df_plot5,
    x="Avg Deviation",
    y="Avg Collisions per Step",
    hue="Number of Intruders",  # Color will represent the number of intruders
    style="Intruder Speed",     # Marker shape will represent the speed
    palette="flare",            # A nice, visually distinct color palette
    s=150,                      # Increase the base marker size to make shapes clear
    alpha=0.8
)
plt.title("Safety vs. Deviation Trade-off")
plt.xlabel("Avg Deviation from Target (Lower is Better)")
plt.ylabel("Avg Collisions per Step (Lower is Better)")

# Seaborn automatically creates a comprehensive legend for both hue and style
plt.legend(title="Intruder Properties")
plt.savefig(f"{save_dir}/plot5_tradeoff_scatter.png", bbox_inches='tight')
plt.close()

print("Generated 5 plot files: ")
print(f"1. {save_dir}/plot1_collisions_vs_num_intruders.png")
print(f"2. {save_dir}/plot2_deviation_vs_speed.png")
print(f"3. {save_dir}/plot3_collision_heatmap.png")
print(f"4. {save_dir}/plot4_model_comparison.png")
print(f"5. {save_dir}/plot5_tradeoff_scatter.png")