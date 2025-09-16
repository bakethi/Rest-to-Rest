import pandas as pd
import os

# --- 1. DEFINE ALL MODEL PATHS ---

# --- Baseline Model ---
base_model_csv_path = "results/PBRS-Full-Eval/baseline/baseline_average_params_evaluation.csv"

# --- Single-Objective TPE Model ---
single_objective_model_csv_path = "results/PBRS-Full-Eval/SO_TPE/single_objective_TPE_evaluation.csv"

# --- Multi-Objective NSGAII Models from the Pareto Front ---
multi_obj_base_path = "results/PBRS-Full-Eval/MO_NSGAII"

NSGAII_model_safe_csv_path = os.path.join(multi_obj_base_path, "evaluation_safest.csv")
NSGAII_model_balanced_csv_path = os.path.join(multi_obj_base_path, "evaluation_balanced.csv")
NSGAII_model_efficient_csv_path = os.path.join(multi_obj_base_path, "evaluation_most_efficient.csv")

# Multi-Obj Random Sampler
random_sampler_base_path = "results/PBRS-Full-Eval/MO_Random_Sampler"

random_sampler_safest_path = os.path.join(random_sampler_base_path, "evaluation_safest.csv")
random_sampler_balanced_path = os.path.join(random_sampler_base_path, "evaluation_balanced.csv")
random_sampler_most_efficient_path = os.path.join(random_sampler_base_path, "evaluation_most_efficient.csv")

# Multi-Obj TPE
MO_TPE_base_path = "results/PBRS-Full-Eval/MO_TPE"

MO_TPE_safest_path = os.path.join(MO_TPE_base_path, "evaluation_safest.csv")
MO_TPE_balanced_path = os.path.join(MO_TPE_base_path, "evaluation_balanced.csv")
MO_TPE_most_efficient_path = os.path.join(MO_TPE_base_path, "evaluation_most_efficient.csv")

# Multi-Obj NSGAIII
MO_NSGAIII_base_path = "results/PBRS-Full-Eval/MO_NSGAIII"

MO_NSGAIII_safest_path = os.path.join(MO_NSGAIII_base_path, "evaluation_safest.csv")
MO_NSGAIII_balanced_path = os.path.join(MO_NSGAIII_base_path, "evaluation_balanced.csv")
MO_NSGAIII_most_efficient_path = os.path.join(MO_NSGAIII_base_path, "evaluation_most_efficient.csv")

# SO CmaEs
SO_CmaEs_csv_path = "results/PBRS-Full-Eval/SO_CmaEs/evaluation_SO-CmaEs.csv"

# SO GPSampler
SO_GPSampler_csv_path = "results/PBRS-Full-Eval/SO_GPSampler/evaluation_SO-GPSampler.csv"


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
    df_MO_NSGAIII_safe = pd.read_csv(MO_NSGAIII_safest_path)
    df_MO_NSGAIII_balanced = pd.read_csv(MO_NSGAIII_balanced_path)
    df_MO_NSGAIII_efficient = pd.read_csv(MO_NSGAIII_most_efficient_path)
    df_SO_CmaEs = pd.read_csv(SO_CmaEs_csv_path)
    df_SO_GPSampler = pd.read_csv(SO_GPSampler_csv_path)

    all_dfs = [df_base,
            df_single_obj,
            df_NSGAII_safe, df_NSGAII_balanced, df_NSGAII_efficient,
            df_rs_safe, df_rs_balanced, df_rs_efficient,
            df_MO_TPE_safe, df_MO_TPE_balanced, df_MO_TPE_efficient,
            df_MO_NSGAIII_safe, df_MO_NSGAIII_balanced, df_MO_NSGAIII_efficient,
            df_SO_CmaEs,
            df_SO_GPSampler,
            ]
    
    # Clean column names
    for df in all_dfs:
        df.columns = df.columns.str.strip()

    # Add a descriptive 'Model' column to each DataFrame
    df_base['Model'] = 'Baseline (Hand-Crafted)'
    df_single_obj['Model'] = 'SO TPE'
    df_NSGAII_safe['Model'] = 'MO NSGAII(Safest)'
    df_NSGAII_balanced['Model'] = 'MO NSGAII(Balanced)'
    df_NSGAII_efficient['Model'] = 'MO NSGAII(Most Efficient)'
    df_rs_safe['Model'] = 'MO RS(Safest)'
    df_rs_balanced['Model'] = 'MO RS(Balanced)'
    df_rs_efficient['Model'] = 'MO RS(Most Efficient)'
    df_MO_TPE_safe['Model'] = 'MO TPE(Safest)'
    df_MO_TPE_balanced['Model'] = 'MO TPE(Balanced)'
    df_MO_TPE_efficient['Model'] = 'MO TPE(Most Efficient)'
    df_MO_NSGAIII_safe['Model'] = 'MO NSGAIII(Safest)'
    df_MO_NSGAIII_balanced['Model'] = 'MO NSGAIII(Balanced)'
    df_MO_NSGAIII_efficient['Model'] = 'MO NSGAIII(Most Efficient)'
    df_SO_CmaEs['Model'] = 'SO CmaEs'
    df_SO_GPSampler['Model'] = 'SO GPSampler'

    df_comparison = pd.concat(all_dfs, ignore_index=True)

    print("✅ Successfully loaded and combined data for all models.")

except FileNotFoundError as e:
    print(f"❌ Error: Could not find a CSV file. Please check the paths.")
    print(f"Missing file: {e.filename}")
    exit()

# --- 3. CREATE AND SAVE THE SUMMARY TABLE ---
print("Aggregating data for the summary table...")
df_aggregated = df_comparison.groupby('Model').agg({
    'Avg Collisions per Step': 'mean',
    'Avg Deviation': 'mean'
}).reset_index()

# --- 4. SORT THE TABLE AS REQUESTED ---
# Separate the baseline model
baseline_df = df_aggregated[df_aggregated['Model'] == 'Baseline (Hand-Crafted)']
other_models_df = df_aggregated[df_aggregated['Model'] != 'Baseline (Hand-Crafted)']

# Sort the other models by performance (lowest collisions first, then lowest deviation)
sorted_other_models_df = other_models_df.sort_values(
    by=['Avg Collisions per Step', 'Avg Deviation'],
    ascending=[True, True]
)

# Combine the baseline and the sorted models
df_final_sorted = pd.concat([baseline_df, sorted_other_models_df], ignore_index=True)


# --- 5. CALCULATE PERCENTAGE IMPROVEMENT AGAINST BASELINE ---
print("\nCalculating percentage improvement over baseline...")

# Get the baseline performance values
baseline_row = df_final_sorted[df_final_sorted['Model'] == 'Baseline (Hand-Crafted)']
baseline_collisions = baseline_row['Avg Collisions per Step'].iloc[0]
baseline_deviation = baseline_row['Avg Deviation'].iloc[0]

# Calculate the improvement. A higher percentage is better.
# Formula: ((Baseline - Model_Value) / Baseline) * 100
df_final_sorted['Collision Improvement (%)'] = (
    (baseline_collisions - df_final_sorted['Avg Collisions per Step']) / baseline_collisions
) * 100

df_final_sorted['Deviation Improvement (%)'] = (
    (baseline_deviation - df_final_sorted['Avg Deviation']) / baseline_deviation
) * 100


# --- 6. PRINT AND SAVE THE FINAL TABLE ---
# Print the final table with the new columns
print("\n--- Final Model Performance Summary with Improvement ---")
print(df_final_sorted.round(2).to_markdown(index=False)) # Rounding for cleaner display


# Save the sorted data to a new CSV file
output_csv_path = "plots/intruder_plots/Full-Eval-second-Try/model_comparison_summary_sorted.csv"
df_final_sorted.to_csv(output_csv_path, index=False)

print(f"\n✅ Successfully generated and saved the sorted summary table to: {output_csv_path}")