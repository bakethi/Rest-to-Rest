import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

base_model_csv_path = "results/evaluation_run_2025-06-21_15-49-21.csv"
new_model_csv_path = "results/evaluation_run_2025-06-24_07-48-46.csv"
training_number = "Training_3"
save_dir = f"plots/intruder_plots/{training_number}"

# Create the save directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# --- Data Loading ---
df_base = pd.read_csv(base_model_csv_path)
df_new = pd.read_csv(new_model_csv_path)

# (Good Practice) Clean up column names to remove leading/trailing whitespace
df_base.columns = df_base.columns.str.strip()
df_new.columns = df_new.columns.str.strip()


# --- Pre-processing ---
# 1. Filter the base model dataframe to only include the 500k steps model
df_500k = df_base[df_base['Model Checkpoint'].str.contains('_500000_steps.zip')]

# 2. Add a 'Model' column to each dataframe to identify the models
df_500k['Model'] = '500k Steps Baseline'
df_new['Model'] = 'Best Model'

# 3. Concatenate the two dataframes
df_comparison = pd.concat([df_500k, df_new], ignore_index=True)


# --- Plotting ---
sns.set_theme(style="whitegrid")

# Plot 1: Collisions vs. Intruder Speed, comparing models
plt.figure(figsize=(12, 8))
g = sns.FacetGrid(df_comparison, col="Intruder Size", hue="Model", palette="viridis", height=5)
g.map(sns.lineplot, "Intruder Speed", "Avg Collisions per Step", marker="o", alpha=0.9)
g.add_legend()
g.fig.suptitle("Model Comparison: Collisions vs. Intruder Speed", y=1.03)
g.set_axis_labels("Intruder Speed", "Avg Collisions per Step")
plt.savefig(f"{save_dir}/plot1_collisions_vs_speed_comparison.png", bbox_inches='tight')
plt.close()

# Plot 2: Deviation vs. Intruder Speed, comparing models
plt.figure(figsize=(12, 8))
g = sns.FacetGrid(df_comparison, col="Intruder Size", hue="Model", palette="plasma", height=5)
g.map(sns.lineplot, "Intruder Speed", "Avg Deviation", marker="o", alpha=0.9)
g.add_legend()
g.fig.suptitle("Model Comparison: Path Deviation vs. Intruder Speed", y=1.03)
g.set_axis_labels("Intruder Speed", "Avg Deviation from Target")
plt.savefig(f"{save_dir}/plot2_deviation_vs_speed_comparison.png", bbox_inches='tight')
plt.close()


# Plot 3: Bar plot comparing models based on Intruder Speed
plt.figure(figsize=(15, 8))
sns.barplot(
    data=df_comparison,
    x="Intruder Speed",
    y="Avg Collisions per Step",
    hue="Model",
    palette="coolwarm"
)
plt.title("Model Comparison: Collisions vs. Intruder Speed")
plt.xlabel("Intruder Speed")
plt.ylabel("Avg Collisions per Step")
plt.legend(title="Model")
plt.savefig(f"{save_dir}/plot3_model_comparison_bar.png", bbox_inches='tight')
plt.close()

# Plot 4: Scatter plot for Safety vs. Deviation trade-off, comparing models
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=df_comparison,
    x="Avg Deviation",
    y="Avg Collisions per Step",
    hue="Model",
    style="Intruder Speed",
    palette="flare",
    s=150,
    alpha=0.8
)
plt.title("Safety vs. Deviation Trade-off by Model")
plt.xlabel("Avg Deviation from Target (Lower is Better)")
plt.ylabel("Avg Collisions per Step (Lower is Better)")
plt.legend(title="Model and Intruder Speed")
plt.savefig(f"{save_dir}/plot4_tradeoff_scatter_comparison.png", bbox_inches='tight')
plt.close()


print("Generated 4 plot files: ")
print(f"1. {save_dir}/plot1_collisions_vs_speed_comparison.png")
print(f"2. {save_dir}/plot2_deviation_vs_speed_comparison.png")
print(f"3. {save_dir}/plot3_model_comparison_bar.png")
print(f"4. {save_dir}/plot4_tradeoff_scatter_comparison.png")