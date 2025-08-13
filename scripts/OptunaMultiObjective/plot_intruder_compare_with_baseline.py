import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- 1. DEFINE ALL MODEL PATHS ---

# --- Baseline Model ---
base_model_csv_path = "results/baseline_average_params_evaluation.csv"

# --- Single-Objective Model ---
single_objective_model_csv_path = "results/single_objective_evaluation.csv"

# --- Multi-Objective NSGAII Models from the Pareto Front ---
multi_obj_base_path = "results/Training_7"

# *** CORRECTED: Swapped paths to match your analysis ***
# Trial 155 is the safest (lowest collision rate)
# Trial 20 is the most efficient (lowest deviation)
NSGAII_model_safe_csv_path = os.path.join(multi_obj_base_path, "evaluation_safest.csv")
NSGAII_model_balanced_csv_path = os.path.join(multi_obj_base_path, "evaluation_balanced.csv") 
NSGAII_model_efficient_csv_path = os.path.join(multi_obj_base_path, "evaluation_most_efficient.csv")

# Multi-Obj Random Sampler
random_sampler_base_path = "models/best_model_24_50_PBRS_Random_Sampler"

random_sampler_safest_path = os.path.join(random_sampler_base_path, "evaluation_safest.csv")
random_sampler_balanced_path = os.path.join(random_sampler_base_path, "evaluation_balanced.csv")
random_sampler_most_efficient_path = os.path.join(random_sampler_base_path, "evaluation_most_efficient.csv")

# Multi-Obj TPE
MO_TPE_base_path = "models/best_model_24_50_PBRS_MO_TPE"

MO_TPE_safest_path = os.path.join(MO_TPE_base_path, "evaluation_safest.csv")
MO_TPE_balanced_path = os.path.join(MO_TPE_base_path, "evaluation_balanced.csv")
MO_TPE_most_efficient_path = os.path.join(MO_TPE_base_path, "evaluation_most_efficient.csv")

# --- Output Configuration ---
training_number = "MO_TPE" # Updated version number
save_dir = f"plots/intruder_plots/{training_number}"
os.makedirs(save_dir, exist_ok=True)


# --- 2. DATA LOADING AND PRE-PROCESSING ---
try:
    # Load all data sources
    df_base = pd.read_csv(base_model_csv_path)
    df_single_obj = pd.read_csv(single_objective_model_csv_path)
    df_NSGAII_safe = pd.read_csv(NSGAII_model_safe_csv_path)
    df_NSGAII_balanced = pd.read_csv(NSGAII_model_balanced_csv_path)
    df_NSGAII_efficient = pd.read_csv(NSGAII_model_efficient_csv_path)
    df_rs_safe = pd.read_csv(random_sampler_safest_path)
    df_rs_balanced = pd.read_csv(random_sampler_balanced_path)
    df_rs_efficient = pd.read_csv(random_sampler_most_efficient_path)
    df_MO_TPE_safe = pd.read_csv(MO_TPE_safest_path)
    df_MO_TPE_balanced = pd.read_csv(MO_TPE_balanced_path)
    df_MO_TPE_efficient = pd.read_csv(MO_TPE_most_efficient_path)

    all_dfs = [df_base,
            df_single_obj,
            df_NSGAII_safe, df_NSGAII_balanced, df_NSGAII_efficient,
            df_rs_safe, df_rs_balanced, df_rs_efficient,
            df_MO_TPE_safe, df_MO_TPE_balanced, df_MO_TPE_efficient]
    
    # Clean column names
    for df in all_dfs:
        df.columns = df.columns.str.strip()

    # Add a descriptive 'Model' column to each DataFrame
    # *** This section is now correct because the DataFrames were loaded correctly above ***
    df_base['Model'] = 'Baseline (Hand-Crafted)'
    df_single_obj['Model'] = 'Single-Objective TPE'
    df_NSGAII_safe['Model'] = 'Multi-Obj NSGAII(Safest)'
    df_NSGAII_balanced['Model'] = 'Multi-Obj NSGAII(Balanced)'
    df_NSGAII_efficient['Model'] = 'Multi-Obj NSGAII(Most Efficient)'
    df_rs_safe['Model'] = 'Multi-Obj RS(Safest)'
    df_rs_balanced['Model'] = 'Multi-Obj RS(Balanced)'
    df_rs_efficient['Model'] = 'Multi-Obj RS(Most Efficient)'
    df_MO_TPE_safe['Model'] = 'Multi-Obj TPE(Safest)'
    df_MO_TPE_balanced['Model'] = 'Multi-Obj TPE(Balanced)'
    df_MO_TPE_efficient['Model'] = 'Multi-Obj TPE(Most Efficient)'

    df_comparison = pd.concat(all_dfs, ignore_index=True)

    print("✅ Successfully loaded and combined data for 8 models with corrected labels.")

except FileNotFoundError as e:
    print(f"❌ Error: Could not find a CSV file. Please check the paths.")
    print(f"Missing file: {e.filename}")
    exit()

model_order = [
    'Baseline (Hand-Crafted)',
    'Single-Objective TPE',
    'Multi-Obj NSGAII(Safest)',
    'Multi-Obj NSGAII(Balanced)',
    'Multi-Obj NSGAII(Most Efficient)',
    'Multi-Obj RS(Safest)',
    'Multi-Obj RS(Balanced)',
    'Multi-Obj RS(Most Efficient)',
    'Multi-Obj TPE(Safest)',
    'Multi-Obj TPE(Balanced)',
    'Multi-Obj TPE(Most Efficient)',
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