import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- 1. DEFINE ALL MODEL PATHS ---

# --- Baseline Model ---
base_model_csv_path = "results/baseline_average_params_evaluation_2.csv"

# --- Single-Objective Model ---
single_objective_model_csv_path = "results/single_objective_evaluation.csv"

# --- Multi-Objective Models from the Pareto Front ---
multi_obj_base_path = "results/"

# *** CORRECTED: Swapped paths to match your analysis ***
# Trial 155 is the safest (lowest collision rate)
# Trial 20 is the most efficient (lowest deviation)
mo_model_safe_csv_path = os.path.join(multi_obj_base_path, "safest_model_full_evaluation.csv")
mo_model_balanced_csv_path = os.path.join(multi_obj_base_path, "balanced_model_full_evaluation.csv") 
mo_model_efficient_csv_path = os.path.join(multi_obj_base_path, "most_efficient_model_full_evaluation.csv")


# --- Output Configuration ---
training_number = "Training_5/full_eval" # Updated version number
save_dir = f"plots/intruder_plots/{training_number}"
os.makedirs(save_dir, exist_ok=True)


# --- 2. DATA LOADING AND PRE-PROCESSING ---
try:
    # Load all data sources
    df_base = pd.read_csv(base_model_csv_path)
    df_single_obj = pd.read_csv(single_objective_model_csv_path)
    df_mo_safe = pd.read_csv(mo_model_safe_csv_path)
    df_mo_balanced = pd.read_csv(mo_model_balanced_csv_path)
    df_mo_efficient = pd.read_csv(mo_model_efficient_csv_path)

    all_dfs = [df_base, df_single_obj, df_mo_safe, df_mo_balanced, df_mo_efficient]
    
    # Clean column names
    for df in all_dfs:
        df.columns = df.columns.str.strip()

    # Add a descriptive 'Model' column to each DataFrame
    # *** This section is now correct because the DataFrames were loaded correctly above ***
    df_base['Model'] = 'Baseline (Avg. Params)'
    df_single_obj['Model'] = 'Single-Objective Optuna'
    df_mo_safe['Model'] = 'Multi-Obj (Safest)'
    df_mo_balanced['Model'] = 'Multi-Obj (Balanced)'
    df_mo_efficient['Model'] = 'Multi-Obj (Most Efficient)'

    df_comparison = pd.concat(all_dfs, ignore_index=True)

    print("✅ Successfully loaded and combined data for 5 models with corrected labels.")

except FileNotFoundError as e:
    print(f"❌ Error: Could not find a CSV file. Please check the paths.")
    print(f"Missing file: {e.filename}")
    exit()

model_order = [
    'Baseline (Avg. Params)',
    'Single-Objective Optuna',
    'Multi-Obj (Safest)',
    'Multi-Obj (Balanced)',
    'Multi-Obj (Most Efficient)'
]

# Create a color dictionary mapping each model to a color
# We use a built-in seaborn palette with enough distinct colors
colors = sns.color_palette("viridis", len(model_order))
color_palette = dict(zip(model_order, colors))

# --- 3. PLOTTING (This section remains unchanged) ---
# The rest of the script is correct. It will now generate plots with the right data associated with each label.

sns.set_theme(style="whitegrid")

# Plot 1: Collisions vs. Intruder Speed
print("Generating Plot 1: Collisions vs. Speed...")
plt.figure(figsize=(12, 8))
g = sns.FacetGrid(df_comparison, col="Intruder Size", hue="Model", palette="tab10", height=5)
g.map(sns.lineplot, "Intruder Speed", "Avg Collisions per Step", marker="o", alpha=0.9)
g.add_legend()
g.fig.suptitle("Model Comparison: Collisions vs. Intruder Speed", y=1.03)
g.set_axis_labels("Intruder Speed", "Avg Collisions per Step")
plt.savefig(f"{save_dir}/plot1_collisions_vs_speed_comparison.png", bbox_inches='tight')
plt.close()

# Plot 2: Deviation vs. Intruder Speed
print("Generating Plot 2: Deviation vs. Speed...")
plt.figure(figsize=(12, 8))
g = sns.FacetGrid(df_comparison, col="Intruder Size", hue="Model", palette="tab10", height=5)
g.map(sns.lineplot, "Intruder Speed", "Avg Deviation", marker="o", alpha=0.9)
g.add_legend()
g.fig.suptitle("Model Comparison: Path Deviation vs. Intruder Speed", y=1.03)
g.set_axis_labels("Intruder Speed", "Avg Deviation from Target")
plt.savefig(f"{save_dir}/plot2_deviation_vs_speed_comparison.png", bbox_inches='tight')
plt.close()

# ... (Plots 3, 4, and 5 also remain the same) ...
# Plot 3: Bar plot comparing models
print("Generating Plot 3: Bar Plot Comparison...")
plt.figure(figsize=(15, 8))
sns.barplot(
    data=df_comparison,
    x="Intruder Speed",
    y="Avg Collisions per Step",
    hue="Model",
    palette=color_palette
)
plt.title("Model Comparison: Collisions vs. Intruder Speed")
plt.xlabel("Intruder Speed")
plt.ylabel("Avg Collisions per Step")
plt.legend(title="Model")
plt.savefig(f"{save_dir}/plot3_model_comparison_bar.png", bbox_inches='tight')
plt.close()

# Plot 4: Scatter plot for Safety vs. Deviation trade-off
print("Generating Plot 4: Trade-off Scatter Plot...")
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=df_comparison,
    x="Avg Deviation",
    y="Avg Collisions per Step",
    hue="Model",
    style="Intruder Speed",
    palette="deep",
    s=150,
    alpha=0.8
)
plt.title("Safety vs. Deviation Trade-off by Model")
plt.xlabel("Avg Deviation from Target (Lower is Better)")
plt.ylabel("Avg Collisions per Step (Lower is Better)")
plt.legend(title="Model and Intruder Speed")
plt.savefig(f"{save_dir}/plot4_tradeoff_scatter_comparison.png", bbox_inches='tight')
plt.close()

# Plot 5 - Aggregated Trade-off Scatter Plot
print("Aggregating data for the summary trade-off plot...")
df_aggregated = df_comparison.groupby('Model').agg({
    'Avg Collisions per Step': 'mean',
    'Avg Deviation': 'mean'
}).reset_index()

print("Aggregated Data:")
print(df_aggregated)

print("Generating Plot 5: Aggregated Trade-off Scatter Plot...")
plt.figure(figsize=(10, 8))
ax = sns.scatterplot(
    data=df_aggregated,
    x="Avg Deviation",
    y="Avg Collisions per Step",
    hue="Model",
    palette=color_palette,
    s=250,
    alpha=0.9,
    legend='full'
)

for i in range(df_aggregated.shape[0]):
    plt.text(
        x=df_aggregated['Avg Deviation'][i] + 0.3,
        y=df_aggregated['Avg Collisions per Step'][i],
        s=df_aggregated['Model'][i].replace('Multi-Obj ', ''),
        fontdict=dict(color='black', size=10)
    )

plt.title("Overall Safety vs. Efficiency Trade-off (Aggregated)")
plt.xlabel("Overall Avg. Deviation from Target (Lower is Better)")
plt.ylabel("Overall Avg. Collisions per Step (Lower is Better)")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, title="Model Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig(f"{save_dir}/plot5_tradeoff_scatter_AGGREGATED.png", bbox_inches='tight')
plt.close()


print("\n✅ Successfully generated 5 plot files with corrected model labels.")