import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Path to your results file
# Make sure this path is correct
results_dir = "./results/"
file_path = f"{results_dir}/optuna_trials_report.csv"

# Load the data
try:
    df_trials = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"File {file_path} not found. Please check the path.")
    # If the script doesn't find the file, stop here.
    exit()

# Display the first few lines and important columns
print("Available columns:", df_trials.columns.tolist())
print("\nData preview:")
print(df_trials.head())

# Rename parameter columns for clarity (optional but recommended)
# Optuna names the columns 'params_PARAMETER_NAME'
df_trials = df_trials.rename(columns={
    "value": "r2_score",  # 'value' is the metric you are optimizing (the R²)
    "params_learning_rate": "learning_rate",
    "params_batch_size": "batch_size",
    "params_weight_decay": "weight_decay",
    "params_warmup_ratio": "warmup_ratio",
    "params_lr_scheduler_type": "lr_scheduler_type",
    "params_head_hidden_size": "head_hidden_size",
    "params_drug_embedding_size": "drug_embedding_size",
    "params_dropout_rate": "dropout_rate"
})

# Ensure trials are sorted by number
df_trials = df_trials.sort_values("number").reset_index(drop=True)

# Calculate the best cumulative score after each trial
df_trials['best_r2_so_far'] = df_trials['r2_score'].cummax()

# Create the plot
plt.figure(figsize=(12, 7))
sns.lineplot(data=df_trials, x='number', y='best_r2_so_far', marker='o', label='Best R² Found')
plt.title("Optimization History (Convergence Curve)", fontsize=16)
plt.xlabel("Trial Number", fontsize=12)
plt.ylabel("Best R² Score (cumulative)", fontsize=12)
plt.grid(True, which='both', linestyle='--')
plt.legend()
plt.show()
# To save the image
plt.savefig(f"{results_dir}/convergence_plot.png")

# List of numerical hyperparameters to analyze
numeric_params = [
    'learning_rate', 'weight_decay', 'warmup_ratio',
    'dropout_rate', 'head_hidden_size', 'drug_embedding_size'
]

# Create scatter plots for each hyperparameter
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
axes = axes.flatten() # Flatten the grid of axes for an easy loop

for i, param in enumerate(numeric_params):
    sns.scatterplot(data=df_trials, x=param, y='r2_score', ax=axes[i])
    axes[i].set_title(f"R² Score vs. {param}", fontsize=12)
    axes[i].set_xlabel(param)
    axes[i].set_ylabel("R² Score")
    # Set x-axis to log scale for learning rate for better visualization
    if param == 'learning_rate':
        axes[i].set_xscale('log')

plt.tight_layout()
plt.show()
# To save the image
plt.savefig(f"{results_dir}/hyperparameter_scatters.png")

### Analysis 3: Checking Search Ranges

# Select the top 5 trials
top_5_trials = df_trials.sort_values("r2_score", ascending=False).head(5)

print("--- Analysis of the Top 5 Trials ---")
# Display the parameters of the best trials
print(top_5_trials[[
    'number', 'r2_score', 'learning_rate', 'weight_decay',
    'warmup_ratio', 'dropout_rate', 'head_hidden_size'
]].round(6))