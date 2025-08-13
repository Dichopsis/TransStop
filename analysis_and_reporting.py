#!/home2020/home/icube/nhaas/.conda/envs/TransStop/bin/python

#SBATCH -p publicgpu
#SBATCH -N 1
#SBATCH -x hpc-n932
#SBATCH --gres=gpu:2
#SBATCH --constraint="gpuh100|gpua100|gpul40s|gpua40|gpurtx6000"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nicolas.haas3@etu.unistra.fr
#SBATCH --job-name=analysis_and_reporting
#SBATCH --output=analysis_and_reporting_%j.out

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, default_data_collator
from sklearn.metrics import r2_score
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from tqdm import tqdm
from safetensors.torch import load_file
import logomaker
from itertools import combinations

# --- Configuration and Artifact Loading ---
print("--- PART 3: DEEP MODEL INTERPRETATION AND INSIGHT GENERATION (Corrected) ---")

RESULTS_DIR = "./results/"
MODELS_DIR = "./models/"
PROCESSED_DATA_DIR = "./processed_data/"
PROD_MODEL_PATH = os.path.join(MODELS_DIR, "production_model")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Load necessary configurations and data
try:
    with open(os.path.join(RESULTS_DIR, "best_hyperparams.json"), 'r') as f:
        best_hyperparams = json.load(f)
    best_config_df = pd.read_csv(os.path.join(RESULTS_DIR, "systematic_evaluation_log.csv"))
    best_config = best_config_df.iloc[0].to_dict()
    test_df_original = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "test_df.csv"))
except FileNotFoundError as e:
    print(f"File loading error: {e}")
    print("Please ensure that scripts 01 and 02b have been executed successfully.")
    exit()

# *** FINAL CORRECTION: LOADING THE MAPPING FROM THE SOURCE OF TRUTH ***
# Instead of reconstructing the mapping, we load it from the JSON file
# that was saved or reconstructed identically.
# This is the only method that guarantees perfect synchronization with the trained model.
try:
    map_path = os.path.join(PROD_MODEL_PATH, "drug_map.json")
    with open(map_path, 'r') as f:
        drug_to_id = json.load(f)
    
    # Create the inverse mapping
    id_to_drug = {i: drug for drug, i in drug_to_id.items()}
    NUM_DRUGS = len(drug_to_id)
    print("Drug mapping loaded successfully from the source of truth:", drug_to_id)

    # Create a consistent color palette for the drugs
    drug_list_for_palette = sorted(drug_to_id.keys())
    colors = sns.color_palette('tab20', n_colors=len(drug_list_for_palette))
    drug_color_map = dict(zip(drug_list_for_palette, colors))
    print("Color palette for drugs created.")

except FileNotFoundError:
    print(f"CRITICAL ERROR: The file 'drug_map.json' was not found in {PROD_MODEL_PATH}.")
    print("This file is essential. Please run the 'reconstruct_and_save_map.py' script to create it.")
    exit()
# *** END OF CORRECTION ***


# Apply the loaded and consistent mapping to the test set
test_df_original['drug_id'] = test_df_original['drug'].map(drug_to_id)
# Check if any drugs from the test set were not in the mapping (which would be a pipeline error)
if test_df_original['drug_id'].isnull().any():
    missing_drugs = test_df_original[test_df_original['drug_id'].isnull()]['drug'].unique()
    print(f"WARNING: The following drugs from the test set were not found in the mapping: {missing_drugs}")
    print("The corresponding rows will be deleted.")
    test_df_original.dropna(subset=['drug_id'], inplace=True)
    test_df_original['drug_id'] = test_df_original['drug_id'].astype(int)


# Define global variables that might be used later (e.g., for UMAP)
SEED = 42 # Make sure SEED is defined if you use it in UMAP or .sample()
MODEL_HF_NAME = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
context_col = best_config['context_column']

# --- Recreate the model class ---
class PTCDataset(Dataset):
    def __init__(self, dataframe, tokenizer, context_col):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.context_col = context_col
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sequence = row[self.context_col].replace('U', 'T')
        encoding = self.tokenizer(sequence, truncation=True, padding='longest', return_tensors='pt')
        return {
            "input_ids": encoding['input_ids'].flatten(),
            "attention_mask": encoding['attention_mask'].flatten(),
            "drug_id": torch.tensor(row['drug_id'], dtype=torch.long),
            "labels": torch.tensor(float(row['RT_transformed']), dtype=torch.float)
        }

class PanDrugTransformerForTrainer(torch.nn.Module):
    def __init__(self, model_name, num_drugs, head_hidden_size=256, drug_embed_dim=64, num_attention_heads=8, dropout_rate=0.1, **kwargs):
        super().__init__()
        full_model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True, **kwargs)
        self.base_model = full_model.base_model
        self.config = self.base_model.config
        
        base_model_hidden_size = self.config.hidden_size
        
        # Drug embedding and projection to match sequence embedding dimension
        self.drug_embedding = torch.nn.Embedding(num_drugs, drug_embed_dim)
        self.query_projection = torch.nn.Linear(drug_embed_dim, base_model_hidden_size)
        
        # Cross-Attention Layer where drug query attends to sequence key/values
        self.cross_attention = torch.nn.MultiheadAttention(
            embed_dim=base_model_hidden_size,
            num_heads=num_attention_heads,
            batch_first=True
        )
        
        # Regression Head
        self.reg_head = torch.nn.Sequential(
            torch.nn.Linear(base_model_hidden_size, head_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(head_hidden_size, 1)
        )

    def forward(self, input_ids, attention_mask, drug_id, labels=None, **kwargs):
        output_hidden_states = kwargs.get("output_hidden_states", False)
        # Get sequence embeddings from the base transformer
        sequence_outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        
        # Get drug embeddings and project to query dimension
        drug_embeds = self.drug_embedding(drug_id)
        query = self.query_projection(drug_embeds).unsqueeze(1) # Shape: (batch, 1, hidden_size)
        
        # Perform cross-attention: drug embedding queries the sequence embeddings
        # attn_output shape: (batch, 1, hidden_size)
        attn_output, attn_weights = self.cross_attention(
            query=query,
            key=sequence_outputs,
            value=sequence_outputs
        )
        
        # The attended output vector is the input to the regression head
        context_vector = attn_output.squeeze(1) # Shape: (batch, hidden_size)
        
        logits = self.reg_head(context_vector).squeeze(-1)
        
        if output_hidden_states:
            return logits, context_vector

        loss = None
        if labels is not None:
            loss = torch.nn.MSELoss()(logits, labels)
            return (loss, logits)

        return logits

# --- Loading the Production Model ---
print("Loading the production model...")
tokenizer = AutoTokenizer.from_pretrained(PROD_MODEL_PATH, trust_remote_code=True)
model = PanDrugTransformerForTrainer(
    MODEL_HF_NAME, NUM_DRUGS,
    head_hidden_size=best_hyperparams['head_hidden_size'],
    drug_embed_dim=best_hyperparams['drug_embed_dim'],
    num_attention_heads=best_hyperparams['num_attention_heads'],
    dropout_rate=best_hyperparams['dropout_rate']
)
weights_path_safetensors = os.path.join(PROD_MODEL_PATH, 'model.safetensors')
weights_path_bin = os.path.join(PROD_MODEL_PATH, 'pytorch_model.bin')
if os.path.exists(weights_path_safetensors):
    print("Loading weights from model.safetensors...")
    state_dict = load_file(weights_path_safetensors, device=DEVICE)
    model.load_state_dict(state_dict)
elif os.path.exists(weights_path_bin):
    print("Loading weights from pytorch_model.bin...")
    model.load_state_dict(torch.load(weights_path_bin, map_location=torch.device(DEVICE)))
else:
    raise FileNotFoundError(f"No weight file ('model.safetensors' or 'pytorch_model.bin') found in {PROD_MODEL_PATH}")
model.to(DEVICE)
print(model)
model.eval()
print("Model loaded successfully.")


# --- SECTION 3.0: Performance Evaluation per Drug ---
print("\n--- 3.0. Performance Evaluation per Drug on the Test Set ---")

test_df = test_df_original.copy().reset_index(drop=True)
test_dataset = PTCDataset(test_df, tokenizer, context_col)
test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=default_data_collator, shuffle=False)

all_preds_transformed = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predictions on the test set"):
        # Move only tensors to the GPU
        batch = {k: v.to(DEVICE) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        _, preds = model(**batch)
        all_preds_transformed.extend(preds.cpu().numpy())

test_df['preds_transformed'] = all_preds_transformed
test_df['preds'] = np.expm1(test_df['preds_transformed'])
test_df['preds'] = test_df['preds'].clip(lower=0)

r2_per_drug = {}
for drug_name, group_df in test_df.groupby('drug'):
    r2 = r2_score(group_df['RT'], group_df['preds'])
    r2_per_drug[drug_name] = r2
    print(f"R² for {drug_name}: {r2:.4f}")

r2_per_drug_df = pd.DataFrame(list(r2_per_drug.items()), columns=['Drug', 'R2_Score']).sort_values('R2_Score', ascending=False)
r2_global = r2_score(test_df['RT'], test_df['preds'])
print(f"\n--- Global R² on the test set: {r2_global:.4f} ---")

print("Generating the global correlation plot with coloring by drug...")

plt.figure(figsize=(12, 12))

# 1. Create the scatter plot with coloring by drug
# 'hue' colors the points based on the 'drug' column
# 'alpha' adds transparency to better see dense areas
# 's' controls the size of the points
sns.scatterplot(
    data=test_df,
    x='RT',
    y='preds',
    hue='drug',
    palette=drug_color_map,
    alpha=0.7,
    s=50,
    edgecolor='k', # Adds a slight black outline to the points for readability
    linewidth=0.5
)

# 2. Determine the plot limits to draw a perfect line
min_val = min(test_df['RT'].min(), test_df['preds'].min())
max_val = max(test_df['RT'].max(), test_df['preds'].max())
# Add a small margin
min_val -= (max_val - min_val) * 0.05
max_val += (max_val - min_val) * 0.05

# 3. Draw the perfection line (y=x) in dashed red
# This is the line on which all points would lie if the predictions were perfect
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction (y=x)')

# 4. Add titles, labels, and grid
#plt.title('Predictions vs. Actual Values on the Test Set', fontsize=20, pad=20)
plt.xlabel('Actual Readthrough Value (RT)', fontsize=16)
plt.ylabel('Predicted Readthrough Value (RT)', fontsize=16)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(title='Drug', fontsize=12, title_fontsize=14)

# 5. Add the global R² on the plot for context
plt.text(
    x=min_val, 
    y=max_val * 0.95, # Position the text in the top left
    s=f'Global R² = {r2_global:.4f}',
    fontdict={'size': 16, 'weight': 'bold', 'color': 'white'},
    bbox=dict(facecolor='black', alpha=0.6) # Background box for readability
)

# 6. Ensure a square aspect ratio so that the y=x line is at 45 degrees
plt.axis('equal')
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)

# 7. Save the figure
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "global_correlation_plot.png"), dpi=300)
plt.close()

print("Global correlation plot saved in 'global_correlation_plot.png'.")

# Insert this block after generating the global correlation plot.

print("Generating the grid of correlation plots by drug...")

# 1. Get the list of drugs and prepare the plot grid
drug_list = sorted(test_df['drug'].unique())
num_drugs = len(drug_list)
# Dynamically calculate the number of rows and columns
n_cols = 3 
n_rows = (num_drugs + n_cols - 1) // n_cols # Calculates the number of rows needed

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows), sharex=False, sharey=False)
axes = axes.flatten() # Flatten the 2D grid into a 1D list for easy iteration

# The palette is now defined globally with drug_color_map

# 2. Loop over each drug and create its own plot
for i, drug_name in enumerate(drug_list):
    ax = axes[i] # Select the current subplot
    
    # Filter the data for the current drug
    drug_df = test_df[test_df['drug'] == drug_name]
    
    # Retrieve the already calculated R² score
    r2_value = r2_per_drug[drug_name]
    
    # Draw the scatter plot on the subplot
    sns.scatterplot(
        data=drug_df,
        x='RT',
        y='preds',
        ax=ax,
        color=drug_color_map[drug_name],
        alpha=0.8,
        s=40,
        edgecolor='k',
        linewidth=0.5
    )
    
    # Determine the limits for the perfect prediction line (specific to this plot)
    min_val = min(drug_df['RT'].min(), drug_df['preds'].min())
    max_val = max(drug_df['RT'].max(), drug_df['preds'].max())
    margin = (max_val - min_val) * 0.05
    min_val -= margin
    max_val += margin
    
    # Draw the perfection line (y=x)
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5)
    
    # Add the title and R²
    ax.set_title(f'{drug_name}', fontsize=14, weight='bold')
    ax.text(
        x=min_val,
        y=max_val * 0.9,
        s=f'R² = {r2_value:.4f}',
        fontdict={'size': 12, 'weight': 'bold', 'color': 'black'},
        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3')
    )
    
    # Customize the axes
    ax.set_xlabel('Actual Value (RT)', fontsize=10)
    ax.set_ylabel('Predicted Value (RT)', fontsize=10)
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.axis('equal') # Ensure a 1:1 ratio
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

# 3. Hide unused subplots if any
for j in range(num_drugs, len(axes)):
    axes[j].set_visible(False)

# 4. Add a main title to the figure
#fig.suptitle('Predictions vs. Actual Values by Drug', fontsize=22, y=1.02)

# 5. Adjust the layout and save
fig.tight_layout(rect=[0, 0.03, 1, 0.98]) # rect leaves space for the suptitle
plt.savefig(os.path.join(RESULTS_DIR, "per_drug_correlation_grid.png"), dpi=300)
plt.close()

print("Grid of plots by drug saved in 'per_drug_correlation_grid.png'.")


print("Generating violin plot of predicted RT distributions per drug...")

# 1. Determine the order of drugs based on the median of their predictions
median_preds = test_df.groupby('drug')['preds'].median().sort_values(ascending=False)
drug_order = median_preds.index.tolist()

# 2. Create the plot
plt.figure(figsize=(18, 10))

# Use seaborn's violinplot
sns.violinplot(
    data=test_df,
    x='drug',
    y='preds',
    order=drug_order,
    palette=drug_color_map,
    inner='box'  # Display a boxplot inside the violin
)

# 3. Customize the plot
#plt.title('Distribution of Predicted Readthrough (RT) by Drug', fontsize=20, pad=20)
plt.ylabel('Predicted Readthrough Value (RT)', fontsize=16)
plt.xlabel('Drug', fontsize=16)
plt.xticks(rotation=45, ha='right') # Rotate labels for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# 4. Save the figure
plt.savefig(os.path.join(RESULTS_DIR, "drug_profile_violin_plot.png"), dpi=300)
plt.close()

print("Violin plot saved as 'drug_profile_violin_plot.png'.")


toledano_r2 = {
    'Pan-drug': 0.83, 'CC90009': 0.55, 'Clitocine': 0.89, 'DAP': 0.87,
    'G418': 0.76, 'SJ6986': 0.71, 'SRI': 0.76, 'FUr': 0.37, 'Gentamicin': 0.38, 'Untreated': 0.02,
}

# Your results (R² per drug and global R²)
# r2_per_drug is a dictionary you have already calculated
# r2_global is the variable you have already calculated
our_r2 = r2_per_drug.copy()
our_r2['Pan-drug'] = r2_global

# Create a DataFrame for comparison
comparison_data = []
for drug, r2_val in our_r2.items():
    if drug in toledano_r2:
        comparison_data.append({'Drug': drug, 'R2_Score': r2_val, 'Model': 'Our Transformer'})
        comparison_data.append({'Drug': drug, 'R2_Score': toledano_r2[drug], 'Model': 'Toledano et al.'})

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values(by='R2_Score', ascending=False)

# Create the comparison plot
plt.figure(figsize=(14, 10))
barplot = sns.barplot(
    data=comparison_df,
    x='R2_Score',
    y='Drug',
    hue='Model',
    palette={'Our Transformer': 'deepskyblue', 'Toledano et al.': 'lightgray'},
    dodge=True
)

#plt.title('Model Performance Comparison (R²)', fontsize=20, pad=20)
plt.xlabel('R² Score', fontsize=16)
plt.ylabel('Drug / Condition', fontsize=16)
plt.xlim(0, 1.05)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.legend(title='Model', fontsize=12, title_fontsize=14)

# Add values on the bars
for p in barplot.patches:
    width = p.get_width()
    plt.text(width + 0.01, p.get_y() + p.get_height() / 2,
             f'{width:.2f}',
             ha='left', va='center', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "r2_comparison_barplot.png"), dpi=300)
plt.close()
print("R² comparison bar plot saved.")
    

print("\n--- 4.0. Generation of Sequence Logos ---")

def generate_sequence_logos_for_drug(drug_name, drug_df, context_col, n_seqs=100):
    # Sort sequences by predicted performance
    best_df = drug_df.nlargest(n_seqs, 'preds')
    worst_df = drug_df.nsmallest(n_seqs, 'preds')

    best_seqs = best_df[context_col].tolist()
    worst_seqs = worst_df[context_col].tolist()

    # Create count matrices
    best_counts_df = logomaker.alignment_to_matrix(best_seqs)
    worst_counts_df = logomaker.alignment_to_matrix(worst_seqs)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 8))
    
    # Logo for the best sequences
    logomaker.Logo(best_counts_df, ax=ax1, color_scheme='classic')
    ax1.set_title(f"Top Performing Sequences (Top {n_seqs}) for {drug_name}", fontsize=16)
    ax1.set_ylabel("Bits")
    
    # Logo for the worst sequences
    logomaker.Logo(worst_counts_df, ax=ax2, color_scheme='classic')
    ax2.set_title(f"Least Performing Sequences (Bottom {n_seqs}) for {drug_name}", fontsize=16)
    ax2.set_ylabel("Bits")
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"sequence_logo_{drug_name}.png"), dpi=300)
    plt.close()
    print(f"Sequence logo saved for {drug_name}.")

# Generate logos for each drug
for drug_name, group_df in test_df.groupby('drug'):
    # Ensure there are enough sequences for the analysis
    if len(group_df) >= 200:
        generate_sequence_logos_for_drug(drug_name, group_df, context_col)
    else:
        print(f"Not enough data for {drug_name} to generate sequence logos.")


# --- SECTION 4.0: PREPARATION AND UTILITY FUNCTION ---
print("\n--- 4.0. Preparation for Interpretability Analysis ---")

def predict_batch(sequences, drug_ids, tokenizer, model, device):
    """
    Utility function to predict a batch of sequences for given drugs.
    Takes a list of sequences and a list of corresponding drug IDs.
    """
    # Handle the case of empty lists to avoid errors
    if not sequences:
        return np.array([])
        
    inputs = tokenizer(sequences, return_tensors='pt', padding=True, truncation=True)
    batch = {k: v.to(device) for k, v in inputs.items()}
    batch['drug_id'] = torch.tensor(drug_ids, dtype=torch.long).to(device)
    
    with torch.no_grad():
        preds_transformed = model(**batch)
        
    return np.expm1(preds_transformed.cpu().numpy())

# --- SECTION 4.1: FUNCTIONAL SIMILARITY OF PREDICTION PROFILES ---
print("\n--- 4.1. Analysis of the Similarity of Drug Prediction Profiles ---")

# 1. Use a common set of sequences for comparison
unique_sequences = test_df[context_col].unique().tolist()
print(f"Generating in-silico predictions for {len(unique_sequences)} unique sequences across all drugs...")

# 2. Generate predictions for each drug on this common set
all_drug_preds = {}
for drug_name, drug_id in tqdm(drug_to_id.items(), desc="Profiling drugs"):
    drug_ids_batch = [drug_id] * len(unique_sequences)
    preds = predict_batch(unique_sequences, drug_ids_batch, tokenizer, model, DEVICE)
    all_drug_preds[drug_name] = preds

# 3. Create the dense DataFrame and calculate the correlation matrix
drug_profiles_df = pd.DataFrame(all_drug_preds, index=unique_sequences)
drug_similarity_matrix = drug_profiles_df.corr(method='pearson')

# 4. Visualize with a clustermap
print("Generating the similarity clustermap...")
try:
    cluster_map = sns.clustermap(
        drug_similarity_matrix,
        annot=True,
        fmt=".2f",
        cmap='viridis',
        linewidths=.5,
        vmin=0, vmax=1
    )
    plt.setp(cluster_map.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
    plt.setp(cluster_map.ax_heatmap.get_yticklabels(), rotation=0)
    #cluster_map.fig.suptitle('Functional Similarity of Response Profiles', fontsize=20, y=1.02)
    plt.savefig(os.path.join(RESULTS_DIR, "drug_similarity_clustermap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Drug similarity clustermap saved.")
except Exception as e:
    print(f"Error during clustermap generation: {e}. Step skipped.")

# --- SECTION 6.0: VISUALIZATION OF THE SEQUENCE EMBEDDING SPACE ---
print("\n--- 6.0. Visualization of the Sequence Embedding Space ---")

def get_sequence_embeddings(dataframe, tokenizer, model, device, context_col, batch_size=64):
    """
    Extracts the drug-conditioned context vectors for all sequences in a DataFrame.
    
    The context vector is the output of the cross-attention layer, representing
    the sequence as conditioned by the drug. This is the representation we will visualize.
    """
    model.eval()
    embeddings = []
    
    # Create a DataLoader to extract embeddings
    class EmbeddingDataset(Dataset):
        def __init__(self, dataframe, tokenizer, context_col):
            self.df = dataframe
            self.tokenizer = tokenizer
            self.context_col = context_col
        def __len__(self):
            return len(self.df)
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            sequence = row[self.context_col].replace('U', 'T')
            encoding = self.tokenizer(sequence, truncation=True, padding='longest', return_tensors='pt')
            return {
                "input_ids": encoding['input_ids'].flatten(),
                "attention_mask": encoding['attention_mask'].flatten(),
                "drug_id": torch.tensor(row['drug_id'], dtype=torch.long),
            }

    embedding_dataset = EmbeddingDataset(dataframe, tokenizer, context_col)
    embedding_loader = DataLoader(embedding_dataset, batch_size=batch_size, collate_fn=default_data_collator, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(embedding_loader, desc="Extracting sequence embeddings"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            drug_id = batch['drug_id'].to(device)
            
            # The model's forward pass must return the context vectors
            _, context_vectors = model(input_ids=input_ids, attention_mask=attention_mask, drug_id=drug_id, output_hidden_states=True)
            embeddings.append(context_vectors.cpu().numpy())
            
    return np.vstack(embeddings)

# --- Step 1: Data Preparation for UMAP ---
# For a comprehensive analysis, we use the entire test dataset.
# Note: This can be computationally and memory-intensive, especially
# for the UMAP step. The resulting plots may also suffer from
# overplotting, making the visualization denser.
print("Preparing the entire test set for UMAP analysis...")
sample_df_for_umap = test_df.copy().reset_index(drop=True)

# --- Step 2: Extraction of Sequence Embeddings ---
# We use the model to convert each sequence from the sample into a high-dimensional
# numerical vector (the embedding). This is the model's interpretation of the sequence.
print(f"Extracting embeddings for {len(sample_df_for_umap)} sequences for UMAP...")
sequence_embeddings = get_sequence_embeddings(sample_df_for_umap, tokenizer, model, DEVICE, context_col)

# --- Step 3: Dimensionality Reduction with UMAP ---
# The embeddings have a high dimension (often > 768). To visualize them on a
# 2D plot, we use UMAP (Uniform Manifold Approximation and Projection).
# UMAP is an algorithm that reduces dimensionality while trying to preserve
# the global structure and neighborhood relationships of the original data as much as possible.
# In other words, points that are close in the high-dimensional space will remain so in 2D.
print("Applying UMAP for dimensionality reduction...")
umap_reducer = UMAP(n_components=2, random_state=SEED)
reduced_embeddings = umap_reducer.fit_transform(sequence_embeddings)

sample_df_for_umap['umap_x'] = reduced_embeddings[:, 0]
sample_df_for_umap['umap_y'] = reduced_embeddings[:, 1]

# --- Step 4: Visualization and Interpretation ---
# We create several UMAP plots by coloring the points according to different
# characteristics. This helps us understand how the model organizes
# information in its latent space.
print("Generating UMAP plots...")

# Plot 1: Coloring by actual RT value
# Objective: To see if the spatial organization of embeddings correlates with the prediction target.
# Expected interpretation: We hope to see a color gradient, indicating that the model
# places sequences with low and high readthrough efficiency in distinct regions.
plt.figure(figsize=(12, 10))
sns.scatterplot(
    data=sample_df_for_umap,
    x='umap_x',
    y='umap_y',
    hue='RT', # Color by the actual RT value
    palette='viridis',
    s=10,
    alpha=0.7
)
#plt.title('UMAP of Sequence Embeddings (Colored by Actual RT)', fontsize=18)
plt.xlabel('UMAP Dimension 1', fontsize=14)
plt.ylabel('UMAP Dimension 2', fontsize=14)
plt.legend(title='Actual RT', fontsize=10, title_fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "umap_embeddings_by_rt.png"), dpi=300)
plt.close()
print("UMAP by RT saved.")

# Plot 2: Coloring by stop codon type
# Objective: To check if the model has learned fundamental and obvious biological
# characteristics of the sequences in an unsupervised manner.
# Expected interpretation: Very clear and separate clusters for each stop codon type (UAA, UAG, UGA)
# would be striking proof that the model has captured this essential information.
if 'stop_type' in sample_df_for_umap.columns:
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        data=sample_df_for_umap,
        x='umap_x',
        y='umap_y',
        hue='stop_type', # Color by stop codon type
        palette='tab10', # A discrete palette for categories
        s=10,
        alpha=0.7
    )
    #plt.title('UMAP of Sequence Embeddings (Colored by Stop Codon Type)', fontsize=18)
    plt.xlabel('UMAP Dimension 1', fontsize=14)
    plt.ylabel('UMAP Dimension 2', fontsize=14)
    plt.legend(title='Stop Codon Type', fontsize=10, title_fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "umap_embeddings_by_stop_type.png"), dpi=300)
    plt.close()
    print("UMAP by stop codon type saved.")
else:
    print("The 'stop_type' column is not present in the DataFrame for UMAP visualization.")

# Plot 3: Coloring by drug
# Objective: To understand if the sequence embedding is universal or specific to a drug.
# Expected interpretation: Since the visualized vector is the output of the cross-attention between the sequence and the drug, it is now drug-conditioned. We expect to see distinct clusters for different drugs or groups of drugs with similar mechanisms of action. This would demonstrate that the cross-attention mechanism successfully created specialized representations.
plt.figure(figsize=(12, 10))
sns.scatterplot(
    data=sample_df_for_umap,
    x='umap_x',
    y='umap_y',
    hue='drug', # Color by drug
    palette=drug_color_map,
    s=10,
    alpha=0.7
)
#plt.title('UMAP of Sequence Embeddings (Colored by Drug)', fontsize=18)
plt.xlabel('UMAP Dimension 1', fontsize=14)
plt.ylabel('UMAP Dimension 2', fontsize=14)
plt.legend(title='Drug', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=10, title_fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "umap_embeddings_by_drug.png"), dpi=300)
plt.close()
print("UMAP by drug saved.")



# --- SECTION 7.0: IN-SILICO SATURATION MUTAGENESIS ---
print("\n--- 7.0. In Silico Saturation Mutagenesis Analysis ---")

def perform_saturation_mutagenesis(sequence, drug_id, model, tokenizer, device):
    """
    Performs saturation mutagenesis on a given sequence for a specific drug.
    Calculates the impact of each possible mutation outside the stop codon.
    """
    nucleotides = ['A', 'C', 'G', 'T']
    mutagenesis_results = []

    # 1. Get the prediction for the reference sequence (wild-type)
    wt_pred = predict_batch([sequence], [drug_id], tokenizer, model, device)[0]
    
    # Handle the case where the base prediction is zero to avoid division by zero
    if wt_pred == 0:
        wt_pred = 1e-9 # Small value to avoid division by zero
    
    # 2. Determine the positions of the stop codon to ignore
    # For a sequence of type 'NNN...STOP...NNN', the stop is in the center.
    n_context = (len(sequence) - 3) // 2
    stop_start_index = n_context
    
    # 3. Iterate over each position and each possible mutation
    for position in tqdm(range(len(sequence)), desc=f"Mutating sequence for drug_id {drug_id}"):
        # Ignore the positions of the stop codon
        if stop_start_index <= position < stop_start_index + 3:
            continue
            
        original_nucleotide = sequence[position]
        
        for mutated_nucleotide in nucleotides:
            # No need to test the "mutation" to the same nucleotide
            if original_nucleotide == mutated_nucleotide:
                log2_fold_change = 0.0
            else:
                # Create the mutated sequence
                mutated_sequence = list(sequence)
                mutated_sequence[position] = mutated_nucleotide
                mutated_sequence = "".join(mutated_sequence)
                
                # Get the prediction for the mutated sequence
                mutant_pred = predict_batch([mutated_sequence], [drug_id], tokenizer, model, device)[0]
                
                # Calculate the log2 fold change
                log2_fold_change = np.log2(mutant_pred / wt_pred)

            mutagenesis_results.append({
                'position': position - n_context, # Center position 0 on the stop codon
                'original_nucleotide': original_nucleotide,
                'mutated_nucleotide': mutated_nucleotide,
                'log2_fold_change': log2_fold_change
            })
            
    return pd.DataFrame(mutagenesis_results)

def plot_mutagenesis_heatmap(df, title, filename, reference_sequence):
    """
    Generates and saves a heatmap from the mutagenesis results.

    The color scale represents the log2 fold change of readthrough efficiency (RT):
    - 0 (white): No impact.
    - > 0 (red): Increased efficiency (e.g., +1 = 2x more efficient).
    - < 0 (blue): Decreased efficiency (e.g., -1 = 2x less efficient).
    """
    heatmap_data = df.pivot_table(
        index='mutated_nucleotide',
        columns='position',
        values='log2_fold_change'
    )
    
    # --- SORTING BUG FIX ---
    # 1. Convert columns (positions) to integers for numerical sorting.
    numeric_columns = sorted([int(c) for c in heatmap_data.columns])
    
    # 2. Reindex the DataFrame to force the correct numerical order.
    heatmap_data = heatmap_data.reindex(columns=numeric_columns)
    # --- END OF FIX ---

    # Ensure canonical order of nucleotides on the Y-axis
    heatmap_data = heatmap_data.reindex(['A', 'C', 'G', 'T'])
    
    plt.figure(figsize=(20, 6))
    heatmap = sns.heatmap(
        heatmap_data,
        cmap='coolwarm', # Diverging palette: blue (negative), white (neutral), red (positive)
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=.5
    )
    heatmap.collections[0].colorbar.set_label("log2 Fold Change", rotation=270, labelpad=20)
    full_title = f"Reference sequence: {reference_sequence}"
    plt.title(full_title, fontsize=16, pad=20)
    plt.xlabel("Position (relative to the start of the stop codon)", fontsize=12)
    plt.ylabel("Mutation to", fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Mutagenesis heatmap saved in '{filename}'.")

# --- Main analysis logic ---
# Define the drugs and stop codon types to analyze
drugs_to_analyze = ["FUr", "Gentamicin", "CC90009", "G418", "Clitocine", "DAP", "SJ6986", "SRI", "Untreated"]
stop_types_to_analyze = ['uga', 'uag', 'uaa']

for drug_name in drugs_to_analyze:
    if drug_name not in drug_to_id:
        print(f"Drug '{drug_name}' not found, skipped.")
        continue
        
    drug_id = drug_to_id[drug_name]
    drug_df = test_df[test_df['drug'] == drug_name]
    
    for stop_type in stop_types_to_analyze:
        # 1. Select the best performing reference sequence for this combo
        reference_df = drug_df[drug_df['stop_type'] == stop_type]
        if reference_df.empty:
            print(f"No sequence found for {drug_name} with stop codon {stop_type}. Skipped.")
            continue
        
        # Sort by prediction to find the best sequence
        reference_sequence = reference_df.loc[reference_df['preds'].idxmax()][context_col]
        
        print(f"\nMutagenesis analysis for {drug_name} on codon {stop_type}...")
        print(f"Reference sequence: {reference_sequence}")
        
        # 2. Perform mutagenesis
        mutagenesis_df = perform_saturation_mutagenesis(reference_sequence, drug_id, model, tokenizer, DEVICE)
        
        # 3. Generate the heatmap
        plot_title = f"Mutational Impact around the {stop_type} codon for {drug_name}"
        output_filename = os.path.join(RESULTS_DIR, f"saturation_mutagenesis_heatmap_{drug_name}_{stop_type}.png")
        plot_mutagenesis_heatmap(mutagenesis_df, plot_title, output_filename, reference_sequence)

print("\n--- Mutagenesis analysis completed ---")

# --- SECTION 8.0: EPISTASIS ANALYSIS BY DOUBLE MUTAGENESIS ---
print("\n--- 8.0. Epistasis Analysis by In Silico Double Mutagenesis ---")

def calculate_epistasis(sequence, drug_id, model, tokenizer, device):
    """
    Calculates epistasis scores for pairs of mutations in a given sequence.
    """
    nucleotides = ['A', 'C', 'G', 'T']
    
    # 1. Calculate the base prediction (WT)
    wt_pred = predict_batch([sequence], [drug_id], tokenizer, model, device)[0]
    if wt_pred == 0: wt_pred = 1e-9
    log_wt_pred = np.log2(wt_pred)

    # 2. Calculate the effects of all single mutations
    single_mutant_effects = {}
    n_context = (len(sequence) - 3) // 2
    stop_start_index = n_context
    
    context_indices = [i for i in range(len(sequence)) if not (stop_start_index <= i < stop_start_index + 3)]

    for pos in tqdm(context_indices, desc="Calculating single mutations"):
        original_nuc = sequence[pos]
        for new_nuc in nucleotides:
            if original_nuc == new_nuc: continue
            
            mut_seq = list(sequence)
            mut_seq[pos] = new_nuc
            mut_pred = predict_batch(["".join(mut_seq)], [drug_id], tokenizer, model, device)[0]
            if mut_pred == 0: mut_pred = 1e-9
            
            effect = np.log2(mut_pred) - log_wt_pred
            single_mutant_effects[(pos, new_nuc)] = effect

    # 3. Calculate the effects of double mutations and epistasis
    epistasis_results = []
    
    # Create unique pairs of positions
    position_pairs = list(combinations(context_indices, 2))

    for pos1, pos2 in tqdm(position_pairs, desc="Calculating double mutations"):
        original_nuc1 = sequence[pos1]
        original_nuc2 = sequence[pos2]

        for new_nuc1 in nucleotides:
            if original_nuc1 == new_nuc1: continue
            for new_nuc2 in nucleotides:
                if original_nuc2 == new_nuc2: continue

                # Create the doubly mutated sequence
                double_mut_seq = list(sequence)
                double_mut_seq[pos1] = new_nuc1
                double_mut_seq[pos2] = new_nuc2
                
                double_mut_pred = predict_batch(["".join(double_mut_seq)], [drug_id], tokenizer, model, device)[0]
                if double_mut_pred == 0: double_mut_pred = 1e-9
                
                # Observed effect of the double mutant
                observed_effect = np.log2(double_mut_pred) - log_wt_pred
                
                # Expected (additive) effect
                effect1 = single_mutant_effects.get((pos1, new_nuc1), 0)
                effect2 = single_mutant_effects.get((pos2, new_nuc2), 0)
                expected_effect = effect1 + effect2
                
                # Epistasis score
                epistasis_score = observed_effect - expected_effect
                
                epistasis_results.append({
                    'mutation1': f"{pos1-n_context}:{original_nuc1}>{new_nuc1}",
                    'mutation2': f"{pos2-n_context}:{original_nuc2}>{new_nuc2}",
                    'epistasis_score': epistasis_score
                })

    return pd.DataFrame(epistasis_results)

def plot_epistasis_heatmap(df, title, filename, reference_sequence):
    """
    Generates a heatmap of epistasis scores, ensuring that the axes are
    sorted numerically by mutation position.
    """
    if df.empty:
        print("The epistasis DataFrame is empty. Cannot generate the heatmap.")
        return
        
    # --- SORTING BUG FIX ---
    # 1. Utility function to extract the numerical position from the label.
    def get_pos_from_label(label):
        try:
            # Extracts the part before the ':' and converts it to an integer.
            return int(label.split(':')[0])
        except (ValueError, IndexError):
            # Returns a large value for malformed labels to sort them at the end.
            return float('inf')

    # 2. Get all unique mutation labels and sort them numerically.
    all_labels = pd.unique(df[['mutation1', 'mutation2']].values.ravel('K'))
    # Filter out potentially null or malformed labels to avoid errors.
    all_labels = [label for label in all_labels if isinstance(label, str) and ':' in label]
    sorted_labels = sorted(all_labels, key=get_pos_from_label)
    
    # 3. Create the pivot table and reindex it with the sorted labels to force the correct order.
    epistasis_matrix = df.pivot_table(index='mutation1', columns='mutation2', values='epistasis_score')
    epistasis_matrix = epistasis_matrix.reindex(index=sorted_labels, columns=sorted_labels)
    # --- END OF FIX ---

    # Make the matrix symmetric for better visualization.
    # combine_first fills the NaNs of one matrix with the values of the other.
    epistasis_matrix = epistasis_matrix.combine_first(epistasis_matrix.T)
    
    # Fill the diagonal with 0 because a mutation does not interact with itself in this context.
    np.fill_diagonal(epistasis_matrix.values, 0)

    plt.figure(figsize=(20, 18))
    heatmap = sns.heatmap(
        epistasis_matrix,
        cmap='coolwarm',
        center=0,
        annot=False, # Annotation would completely overload the plot.
        square=True, # Ensure cells are square for better readability.
        linewidths=.1
    )
    heatmap.collections[0].colorbar.set_label("Epistasis Score", rotation=270, labelpad=20)
    full_title = f"Reference sequence: {reference_sequence}"
    plt.title(full_title, fontsize=20, pad=20)
    plt.xlabel("Mutation", fontsize=16)
    plt.ylabel("Mutation", fontsize=16)
    
    # Improve readability of axis labels.
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Epistasis heatmap saved in '{filename}'.")

# --- Main logic of the generalized epistasis analysis ---
print("\nStarting generalized epistasis analysis...")

# Define the stop codon types to analyze
stop_types_to_analyze = ['uaa', 'uag', 'uga']

# Loop over each drug and each stop codon type
for drug_name in drug_to_id.keys():
    drug_id = drug_to_id[drug_name]
    drug_df = test_df[test_df['drug'] == drug_name]
    
    for stop_type in stop_types_to_analyze:
        # Select the best performing reference sequence for this combo
        reference_df = drug_df[drug_df['stop_type'] == stop_type]
        
        if reference_df.empty:
            print(f"No sequence found for {drug_name} with stop codon {stop_type}. Epistasis analysis skipped.")
            continue
        
        # Sort by prediction to find the best sequence
        reference_sequence = reference_df.loc[reference_df['preds'].idxmax()][context_col]
        
        print(f"\nEpistasis analysis for {drug_name} on codon {stop_type}...")
        print(f"Reference sequence: {reference_sequence}")
        
        # Perform epistasis analysis
        epistasis_df = calculate_epistasis(reference_sequence, drug_id, model, tokenizer, DEVICE)
        
        # Generate the heatmap
        if not epistasis_df.empty:
            plot_title = f"Epistasis Analysis for {drug_name} (Stop: {stop_type})"
            output_filename = os.path.join(RESULTS_DIR, f"epistasis_heatmap_{drug_name}_{stop_type}.png")
            plot_epistasis_heatmap(epistasis_df, plot_title, output_filename, reference_sequence)
        else:
            print(f"The epistasis DataFrame is empty for {drug_name} / {stop_type}. No heatmap generated.")

print("\n--- Generalized epistasis analysis completed ---")
