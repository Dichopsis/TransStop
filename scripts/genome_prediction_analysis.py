import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import os
from scipy import stats
from tqdm import tqdm
import plotly.express as px


# --- Configuration ---
RESULTS_DIR = "../results/"
FINAL_PREDS_PATH = os.path.join(RESULTS_DIR, "our_genome_wide_predictions_full.parquet")

# Define drug names (without "Untreated" for treatability analysis)
OUR_PREDS_COLS = [
    'our_preds_CC90009', 'our_preds_Clitocine', 'our_preds_DAP', 'our_preds_FUr',
    'our_preds_G418', 'our_preds_Gentamicin', 'our_preds_SJ6986', 'our_preds_SRI'
]
TOLEDANO_PREDS_COLS = [
    'predictions_CC90009', 'predictions_Clitocine', 'predictions_dap', 'predictions_fur',
    'predictions_G418', 'predictions_Gentamicin', 'predictions_SJ6986', 'predictions_sri'
]
# Map column names for comparison
TOLEDANO_MAP = {
    'predictions_dap': 'DAP', 'predictions_fur': 'FUr', 'predictions_sri': 'SRI',
    'predictions_Clitocine': 'Clitocine', 'predictions_Gentamicin': 'Gentamicin',
    'predictions_SJ6986': 'SJ6986', 'predictions_CC90009': 'CC90009', 'predictions_G418': 'G418'
}
OUR_MAP = {
    'our_preds_DAP': 'DAP', 'our_preds_FUr': 'FUr', 'our_preds_SRI': 'SRI',
    'our_preds_Clitocine': 'Clitocine', 'our_preds_Gentamicin': 'Gentamicin',
    'our_preds_SJ6986': 'SJ6986', 'our_preds_CC90009': 'CC90009', 'our_preds_G418': 'G418'
}


# Create a consistent color palette for the drugs
drug_list_for_palette = sorted(list(OUR_MAP.values()))
colors_rgb = sns.color_palette('tab20', n_colors=len(drug_list_for_palette))
# Convert RGB colors to HEX format for Plotly compatibility
colors_hex = [mcolors.to_hex(c) for c in colors_rgb]
drug_color_map = dict(zip(drug_list_for_palette, colors_hex))
print("Color palette for drugs created (HEX format).")

# --- Data Loading ---
print(f"Loading genome-wide predictions file from: {FINAL_PREDS_PATH}")
try:
    df = pd.read_parquet(FINAL_PREDS_PATH)
except FileNotFoundError:
    print("Predictions file not found. Please run the inference script first.")
    exit()
print("Loading complete. DataFrame size:", df.shape)


# --- ANALYSIS 2: COMPARISON OF BEST DRUG PREDICTIONS ---
print("\n--- Starting Analysis 2: Model Comparison ---")

# 1. Identify the best drug for each PTC according to each model
print("Identifying the best drug for each PTC...")
# For our model
our_preds_df = df[OUR_PREDS_COLS].rename(columns=OUR_MAP)
df['our_best_drug'] = our_preds_df.idxmax(axis=1)

# For Toledano's model
toledano_preds_df = df[TOLEDANO_PREDS_COLS].rename(columns=TOLEDANO_MAP)
df['toledano_best_drug'] = toledano_preds_df.idxmax(axis=1)

# 2. Calculate the agreement rate
agreement = (df['our_best_drug'] == df['toledano_best_drug'])
agreement_rate = agreement.mean()
print(f"Overall agreement rate on the best drug: {agreement_rate:.2%}")

# 3. Analyze disagreements: create a confusion matrix
print("Generating the disagreement confusion matrix...")
confusion_matrix = pd.crosstab(df['toledano_best_drug'], df['our_best_drug'], normalize='index')

plt.figure(figsize=(12, 10))
sns.heatmap(
    confusion_matrix, 
    annot=True, 
    fmt='.2f', 
    cmap='viridis',
    linewidths=.5
)
#plt.title('Agreement of Predicted Best Drug', fontsize=20, pad=20)
plt.xlabel('Best Drug (TransStop)', fontsize=16)
plt.ylabel('Best Drug (Toledano et al. Model)', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "best_drug_confusion_matrix.png"), format='png', dpi=600)
plt.close()
print("Confusion matrix saved.")


# 4. Deeper and improved analysis: When does our model significantly change its mind?
print("\n--- In-depth analysis of significant disagreement cases ---")

# Calculate the best prediction for each model
df['our_best_pred_val'] = df[OUR_PREDS_COLS].max(axis=1)

# Find our model's prediction for the drug Toledano chose
# We need to dynamically build the column name of our prediction
# corresponding to the drug chosen by Toledano
# For example, if toledano_best_drug is 'DAP', we want the value of 'our_preds_DAP'
print("Calculating TransStop's prediction for Toledano et al.'s choice...")

# 1. Create a DataFrame containing only our predictions, with simple column names
our_preds_df_simple = df[OUR_PREDS_COLS].rename(columns=OUR_MAP)

# 2. Get the underlying NumPy values for maximum performance
our_preds_values = our_preds_df_simple.values

# 3. Get the ordered list of columns (drugs)
column_names = our_preds_df_simple.columns.tolist()

# 4. Create a mapping from drug name to its column index
col_indexer = {name: i for i, name in enumerate(column_names)}

# 5. Create an array of column indices corresponding to Toledano's choice for each row
# df['toledano_best_drug'].map(col_indexer) will create a series where each value is the column index
# of the drug chosen by Toledano.
col_indices = df['toledano_best_drug'].map(col_indexer).values

# 6. Use advanced NumPy indexing to extract all values at once
# This is equivalent to doing `our_preds_values[0, col_indices[0]]`, `our_preds_values[1, col_indices[1]]`, etc.
# for all rows. It's extremely fast.
num_rows = len(df)
row_indices = np.arange(num_rows)
df['our_pred_for_toledano_choice'] = our_preds_values[row_indices, col_indices]

print("Calculation finished.")

# Calculate the "gain" of our model when it changes its mind
df['our_gain'] = df['our_best_pred_val'] - df['our_pred_for_toledano_choice']

# Filter for disagreement cases
disagreement_df = df[df['our_best_drug'] != df['toledano_best_drug']].copy()


# --- Improvement 3.1: Generate graph (logarithmic) ---

# Graph: Logarithmic Scale
print("Generating gain distribution plot (log scale)...")
plt.figure(figsize=(12, 7))
sns.histplot(disagreement_df['our_gain'], bins=50, kde=False) # kde=False is often better with log scale
plt.yscale('log')
#plt.title('Distribution of Predicted Performance Gain (Log Scale)', fontsize=16)
plt.xlabel('RT Gain (Our Best Drug vs. Toledano\'s Choice)', fontsize=16)
plt.ylabel('Number of PTCs (Log Scale)', fontsize=16)
plt.xlim(left=0)  # Ensure the x-axis starts at 0
#plt.axvline(x=0, color='red', linestyle='--')
plt.grid(True, which='major', linestyle='--', color='#cccccc', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "disagreement_gain_distribution_log.png"), format='png', dpi=600)
plt.close()
print("Gain distribution plots saved.")


# --- Improvement 3.2: Quantify the Distribution ---
print("\n--- Statistics on Performance Gains ---")

total_disagreements = len(disagreement_df)
positive_gain_cases = (disagreement_df['our_gain'] > 0).sum()
positive_gain_percentage = (positive_gain_cases / total_disagreements) * 100

gain_threshold_0_5 = (disagreement_df['our_gain'] > 0.5).sum()
gain_threshold_1_0 = (disagreement_df['our_gain'] > 1.0).sum()

print(f"Total number of PTCs where models disagree: {total_disagreements:,}")
print(f"Percentage of disagreement cases where our model predicts a positive gain: {positive_gain_percentage:.2f}%")
print(f"Number of PTCs where the gain is greater than 0.5 RT: {gain_threshold_0_5:,}")
print(f"Number of PTCs where the gain is greater than 1.0 RT: {gain_threshold_1_0:,}")


# --- Improvement 3.3 (Revisited): Analyze High-Impact Disagreement Cases ---
print("\n--- Analysis of High-Impact Disagreement Cases (RT Gain > 1.0) ---")

# 1. Define the threshold and filter the DataFrame
gain_threshold = 1.0
high_gain_df = disagreement_df[disagreement_df['our_gain'] > gain_threshold].copy()
num_high_gain_cases = len(high_gain_df)

if num_high_gain_cases > 0:
    print(f"Number of PTCs with a predicted performance gain > {gain_threshold}: {num_high_gain_cases:,}")

    # Save all these cases to a CSV file for deeper exploration
    high_gainers_path = os.path.join(RESULTS_DIR, f"high_impact_disagreements_gain_gt_{gain_threshold}.csv")
    columns_to_save = [
        'gene', 'stop_type', 'extracted_context', 'our_best_drug', 'our_best_pred_val', 
        'toledano_best_drug', 'our_pred_for_toledano_choice', 'our_gain'
    ]
    high_gain_df[columns_to_save].to_csv(high_gainers_path, index=False)
    print(f"Log of high-impact disagreements saved to: {high_gainers_path}")

    # --- Analysis 3.3a: Overrepresentation of stop type ---
    print("\nAnalysis of stop type distribution in high-impact cases...")

    # --- Processing for the "High-Impact Cases" group ---
    # 1. Calculate proportions
    stop_type_high_gain_s = high_gain_df['stop_type'].value_counts(normalize=True)
    # 2. Convert the Series to a DataFrame. The index becomes a column.
    stop_type_high_gain_df = stop_type_high_gain_s.reset_index()
    # 3. Rename columns explicitly and robustly, regardless of their default names
    stop_type_high_gain_df.columns = ['Stop_Type', 'Proportion']
    # 4. Add the group column
    stop_type_high_gain_df['Group'] = 'High-Impact Cases (Gain > 1.0)'


    # --- Processing for the "Baseline" group ---
    # 1. Calculate proportions
    stop_type_baseline_s = disagreement_df['stop_type'].value_counts(normalize=True)
    # 2. Convert the Series to a DataFrame
    stop_type_baseline_df = stop_type_baseline_s.reset_index()
    # 3. Rename columns explicitly
    stop_type_baseline_df.columns = ['Stop_Type', 'Proportion']
    # 4. Add the group column
    stop_type_baseline_df['Group'] = 'All Disagreement Cases (Baseline)'


    # --- Final Concatenation ---
    comparison_df_melted = pd.concat([stop_type_high_gain_df, stop_type_baseline_df], ignore_index=True)

    # Debugging: Display the first 5 rows of the final DataFrame for verification
    print("\n--- Preview of the final DataFrame for the plot ---")
    print(comparison_df_melted.head())
    print("\nAvailable columns:", comparison_df_melted.columns)
    
    # The rest of the visualization code remains the same and should work now
    plt.figure(figsize=(10, 7))
    barplot = sns.barplot(data=comparison_df_melted, x='Stop_Type', y='Proportion', hue='Group', palette='pastel')
    #plt.title('Stop Type Distribution: High-Impact Cases vs. Baseline', fontsize=16)
    plt.ylabel('Proportion of Cases', fontsize=16)
    plt.xlabel('Stop Codon Type', fontsize=16)
    current_labels = barplot.get_xticklabels()
    uppercase_labels = [label.get_text().upper() for label in current_labels]
    barplot.set_xticklabels(uppercase_labels)
    plt.grid(axis='y', linestyle='--')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y))) # Format Y-axis as percentages
    for p in barplot.patches:
        if p.get_height() > 0:
            barplot.annotate(format(p.get_height(), '.1%'), 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "high_gain_stop_type_analysis.png"), format='png', dpi=600)
    plt.close()
    print("Stop type analysis plot saved.")


    # --- Analysis 3.3b: Most frequent "change of mind" pairs ---
    print("\nAnalysis of drug pairs in changes of mind...")
    
    # Create a column combining the old and new choice
    high_gain_df['change_pair'] = high_gain_df['toledano_best_drug'] + ' -> ' + high_gain_df['our_best_drug']
    
    # Count the most frequent pairs
    change_pair_counts = high_gain_df['change_pair'].value_counts().nlargest(10) # We take the 10 most frequent
    plt.figure(figsize=(12, 10))
    barplot_pairs = sns.barplot(x=change_pair_counts.values, y=change_pair_counts.index, palette='viridis_r', hue=change_pair_counts.index, dodge=False, legend=False)
    #plt.title('Top 10 High-Impact "Changes of Mind" (Gain > 1.0)', fontsize=18)
    plt.xlabel('Number of PTCs', fontsize=16)
    plt.ylabel('Drug Change (Toledano -> TransStop)', fontsize=16)
    plt.grid(axis='x', linestyle='--')
    
    max_count = change_pair_counts.values.max()
    plt.xlim(right=max_count * 1.1) 
    
    # Add counts on the bars
    for i, (count, pair) in enumerate(zip(change_pair_counts.values, change_pair_counts.index)):
        barplot_pairs.text(count, i, f' {count:,}', va='center', ha='left')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "high_gain_drug_switch_analysis.png"), format='png', dpi=600)
    plt.close()
    print("Drug switch analysis plot saved.")

else:
    print(f"No disagreement cases found with a performance gain > {gain_threshold}.")

# --- ANALYSIS 3 (Revisited): DRUG EFFICACY PROFILE AND HIERARCHY ---
print("\n--- Starting Analysis 3: Drug Efficacy Profile and Hierarchy ---")

# 1. Identify the best drug for each PTC (necessary for both analyses)
print("Identifying the best drug for each PTC...")
df['our_best_drug'] = df[OUR_PREDS_COLS].rename(columns=OUR_MAP).idxmax(axis=1)



# --- Visualization 1 (Revisited): Inverted Sunburst Plot (Hierarchy Drug -> Stop Type) ---
print("Generating the Inverted Sunburst Plot...")

if 'stop_type' in df.columns:
    sunburst_data = df.groupby(['our_best_drug', 'stop_type']).size().reset_index(name='ptc_count')
    sunburst_data['stop_type'] = sunburst_data['stop_type'].str.upper()

    # Create the interactive figure with the inverted order in 'path'
    fig = px.sunburst(
        sunburst_data,
        path=['our_best_drug', 'stop_type'], # <-- INVERTED ORDER HERE
        values='ptc_count',
        color='our_best_drug', # Coloring by drug makes more sense now
        color_discrete_map=drug_color_map,
        #title='Drug Specialization Profile by Stop Codon Type',
    )

    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25), font_size=16, title_font_size=22)
    fig.update_traces(texttemplate="%{label}<br>%{percentParent:.1%}") # '% parent' shows the internal composition of each drug
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25), font_size=16, title_font_size=22)
    sunburst_path = os.path.join(RESULTS_DIR, "best_drug_sunburst_inverted.png")
    fig.write_image(sunburst_path, width=1200, height=1200, scale=2)
    print(f"Inverted sunburst plot saved to: {sunburst_path}")
else:
    print("Column 'stop_type' missing, the Sunburst plot is skipped.")

# --- Alternative Visualization: Raincloud Plot (Violin Plots) ---
print("Generating the Raincloud Plot to characterize drug styles...")

# Prepare the data: switch from wide to long format
print("Formatting data (melt)...")
df_melted = df[OUR_PREDS_COLS].melt(var_name='drug_col', value_name='predicted_rt')
df_melted['drug'] = df_melted['drug_col'].map(OUR_MAP)
df_melted['log_rt'] = np.log1p(df_melted['predicted_rt'])

print("Sampling data for visualization...")
# For the "rain", a smaller sample is more readable.
if len(df_melted) > 100_000:
    df_sample = df_melted.sample(n=100_000, random_state=42)
else:
    df_sample = df_melted

# Sort drugs by median performance for a logical visual order
drug_order = df_sample.groupby('drug')['predicted_rt'].median().sort_values(ascending=True).index

# --- Manual Construction of the Raincloud Plot ---
fig, ax = plt.subplots(figsize=(16, 12))

# Define vertical offsets to separate the elements
# Each drug will have its "level" centered on an integer (0, 1, 2...).
CLOUD_OFFSET = 0   # The cloud will be above the baseline
RAIN_OFFSET = -0.15   # The rain will be below
BOX_OFFSET = -0.15   # The boxplot will be even lower

for i, drug_name in enumerate(drug_order):
    # 1. --- Prepare data for the current drug ---
    drug_data = df_sample[df_sample['drug'] == drug_name]
    drug_log_rt = drug_data['log_rt']
    color = drug_color_map[drug_name]
    
    # 2. --- Layer 1: The "Cloud" (Manual Half-Violin) ---
    # Calculate the kernel density estimate (KDE)
    initial_kde = stats.gaussian_kde(drug_log_rt, bw_method='scott')
    default_bandwidth = initial_kde.factor

    adjusted_bandwidth = default_bandwidth * 0.5

    kde = stats.gaussian_kde(drug_log_rt, bw_method=adjusted_bandwidth)
    data_range = drug_log_rt.max() - drug_log_rt.min()
    padding = data_range * 0.05
    x_min = max(0, drug_log_rt.min() - padding) 
    x_max = drug_log_rt.max() + padding
    x_range = np.linspace(x_min, x_max, 200)
    density = kde(x_range)
    
    # Normalize the cloud height to be aesthetically pleasing
    scaled_density = density / density.max() * 0.4
    
    # Draw the filled cloud
    ax.fill_between(x_range, i + CLOUD_OFFSET, i + CLOUD_OFFSET + scaled_density, 
                    color=color, alpha=0.5, zorder=1)
    # Draw the cloud outline
    ax.plot(x_range, i + CLOUD_OFFSET + scaled_density, color=color, lw=1.5, zorder=2)
    # Draw the cloud baseline
    ax.plot(x_range, np.full_like(x_range, i + CLOUD_OFFSET), color=color, lw=1.5, zorder=2)


    # 3. --- Layer 2: The "Rain" (Manual Stripplot) ---
    # Create vertical jitter
    jitter = np.random.uniform(-0.15, 0.15, size=len(drug_log_rt))
    y_rain = np.full_like(drug_log_rt, i + RAIN_OFFSET) + jitter
    
    ax.scatter(drug_log_rt, y_rain, color=color, s=0.5, alpha=0.5, zorder=3, linewidths=0)


    # 4. --- Layer 3: The Boxplot ---
    # Define styles
    boxprops = {'facecolor': 'none', 'edgecolor': 'black', 'linewidth': 1.5, 'zorder': 4}
    medianprops = {'color': 'black', 'linewidth': 2, 'zorder': 5}
    whiskerprops = {'color': 'black', 'linewidth': 1.5, 'zorder': 4}
    capprops = {'color': 'black', 'linewidth': 1.5, 'zorder': 4}
    
    ax.boxplot(drug_log_rt, vert=False, positions=[i + BOX_OFFSET],
               showfliers=False, showcaps=True,
               patch_artist=True, # Essential for `facecolor`
               boxprops=boxprops, medianprops=medianprops,
               whiskerprops=whiskerprops, capprops=capprops,
               widths=0.30)


# --- Finalization and Aesthetics ---
# Configure the Y-axis to display drug names
ax.set_yticks(np.arange(len(drug_order)))
ax.set_yticklabels(drug_order)
ax.tick_params(axis='y', length=0) # Hide the small tick marks on the Y-axis

# Configure the X-axis to be readable in original RT scale
original_ticks = [0, 0.5, 1, 2, 3, 5, 7]
log_ticks = np.log1p(original_ticks)
ax.set_xticks(log_ticks)
ax.set_xticklabels(labels=original_ticks)
ax.set_xlabel("Predicted Readthrough (RT) - Log-transformed Scale", fontsize=16)
ax.set_ylabel("") # No need for a "Drug" label here

#ax.set_title("Drug Performance Profile (Raincloud Plot)", fontsize=22, pad=20)
ax.tick_params(axis='x', which='major', labelsize=14)
ax.grid(True, axis='x', linestyle='--', alpha=0.6)

sns.despine(left=True, bottom=True, trim=True)
plt.tight_layout()

raincloud_path = os.path.join(RESULTS_DIR, "drug_profile_raincloud_plot_custom.png")
plt.savefig(raincloud_path, format='png', dpi=600)
plt.close()
print(f"Custom raincloud plot saved to: {raincloud_path}")


# --- ANALYSIS 4 (Final Version): PREDICTIVE USE CASE ON THE CFTR GENE ---
print("\n--- Starting Analysis 4: Predictive Use Case on the CFTR Gene ---")

# ... (the loading and filtering code for cftr_df remains the same) ...
if 'gene' not in df.columns:
    print("The 'gene' column is missing. CFTR analysis is skipped.")
else:
    cftr_df = df[df['gene'] == 'CFTR'].copy()

    if cftr_df.empty:
        print("No data found for the CFTR gene. Analysis skipped.")
    else:
        # --- Analysis 1: Therapeutic Profile by Mutation (Heatmaps with Fixed Order) ---
        print("Generating therapeutic profile heatmaps for key CFTR mutations...")

        mutations_of_interest = {
            'G542X': 542,
            'R553X': 553,
            'R1162X': 1162,
            'W1282X': 1282,
        }
        
        positions_to_analyze = list(mutations_of_interest.values())
        cftr_mutations_df = cftr_df[cftr_df['position_PTC'].isin(positions_to_analyze)].copy()

        cftr_melted = cftr_mutations_df.melt(
            id_vars=['position_PTC', 'stop_type'],
            value_vars=OUR_PREDS_COLS,
            var_name='drug_col',
            value_name='predicted_rt'
        )
        cftr_melted['drug'] = cftr_melted['drug_col'].map(OUR_MAP)
        pos_to_name_map = {v: k for k, v in mutations_of_interest.items()}
        cftr_melted['mutation_name'] = cftr_melted['position_PTC'].map(pos_to_name_map)
        
        # --- MAIN CORRECTION: DEFINE A GLOBAL AND FIXED ORDER FOR DRUGS ---
        # 1. Calculate the average performance of each drug across all selected CFTR mutations.
        #    This gives us a global ranking of their relevance for this gene.
        global_drug_order = cftr_melted.groupby('drug')['predicted_rt'].mean().sort_values(ascending=False).index
        print("Global drug order for heatmaps (based on average performance):")
        print(global_drug_order)
        # --- END OF CORRECTION ---

        num_mutations = len(mutations_of_interest)
        fig, axes = plt.subplots(1, num_mutations, figsize=(5.5 * num_mutations, 10), sharey=True)
        if num_mutations == 1: axes = [axes]

        vmax = cftr_melted['predicted_rt'].max()

        for i, (name, pos) in enumerate(mutations_of_interest.items()):
            ax = axes[i]
            mutation_data = cftr_melted[cftr_melted['position_PTC'] == pos]
            if not mutation_data.empty:
                pivot_df = mutation_data.pivot_table(index='drug', columns='stop_type', values='predicted_rt')
                
                # --- APPLY THE GLOBAL AND FIXED ORDER ---
                # Re-index the DataFrame to follow our predefined order.
                # .reindex() will handle cases where a drug might not have data.
                pivot_df = pivot_df.reindex(global_drug_order)
                # --- END OF APPLICATION ---
                
                sns.heatmap(
                    pivot_df, ax=ax, annot=True, fmt=".2f", cmap='viridis',
                    linewidths=.5, vmin=0, vmax=vmax, cbar=(i == num_mutations - 1)
                )
                ax.set_title(name, fontsize=18, pad=15)
                ax.set_xlabel("")
                ax.set_ylabel("Drug" if i == 0 else "", fontsize=16)
                ax.tick_params(axis='x', labelsize=12)
                ax.tick_params(axis='y', labelsize=12)
                current_labels = ax.get_xticklabels()
                uppercase_labels = [label.get_text().upper() for label in current_labels]
                ax.set_xticklabels(uppercase_labels)
                
                if i > 0:
                    ax.tick_params(axis='y', length=0)

        fig.text(0.5, 0.04, 'Stop Codon Type', ha='center', va='center', fontsize=16)
        #fig.suptitle("Predicted Therapeutic Profile for Key CFTR Mutations", fontsize=22, y=0.98)
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])

        plt.savefig(os.path.join(RESULTS_DIR, "cftr_therapeutic_profiles_heatmap.png"), format='png', dpi=600)
        plt.close()
        print("CFTR therapeutic profile heatmaps saved.")

print("\n--- All final analyses are complete. ---")
