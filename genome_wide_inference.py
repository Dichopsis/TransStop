#!/home2020/home/icube/nhaas/.conda/envs/TransStop/bin/python

#SBATCH -p publicgpu
#SBATCH -N 1
#SBATCH -x hpc-n932
#SBATCH --gres=gpu:4
#SBATCH --constraint="gpuh100"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nicolas.haas3@etu.unistra.fr
#SBATCH --job-name=genome_wide_inference
#SBATCH --output=genome_wide_inference_%j.out

import pandas as pd
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorWithPadding
import json
import os
from tqdm import tqdm
import pyreadr
import re
from safetensors.torch import load_file
import time

# --- Global Configuration ---
# These variables will be used by all processes
RESULTS_DIR = "../results/"
MODELS_DIR = "../models/"
# UPDATE THIS PATH
DATA_DIR = "../data/" 
PROD_MODEL_PATH = os.path.join(MODELS_DIR, "production_model")
# UPDATE THIS COLUMN
csv_path = os.path.join(RESULTS_DIR, "systematic_evaluation_log.csv")  # Remplace par le vrai nom de fichier
df_context = pd.read_csv(csv_path)

# Stocker la valeur de la première ligne de 'context_column' dans la variable
CONTEXT_COL_NAME_FROM_MODEL = df_context["context_column"].iloc[0] 
NUM_GPUS = torch.cuda.device_count()

# --- Classes (must be defined at the global level for multiprocessing) ---

class PanDrugTransformer(torch.nn.Module):
    def __init__(self, model_name, num_drugs, head_hidden_size=256, drug_embed_dim=64, num_attention_heads=8, dropout_rate=0.1, **kwargs):
        super().__init__()
        full_model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True, **kwargs)
        self.base_model = full_model.base_model
        self.config = self.base_model.config
        
        base_model_hidden_size = self.config.hidden_size
        
        self.drug_embedding = torch.nn.Embedding(num_drugs, drug_embed_dim)
        self.query_projection = torch.nn.Linear(drug_embed_dim, base_model_hidden_size)
        
        self.cross_attention = torch.nn.MultiheadAttention(
            embed_dim=base_model_hidden_size,
            num_heads=num_attention_heads,
            batch_first=True
        )
        
        self.reg_head = torch.nn.Sequential(
            torch.nn.Linear(base_model_hidden_size, head_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(head_hidden_size, 1)
        )

    def forward(self, input_ids, attention_mask, drug_id, labels=None, **kwargs):
        sequence_outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        
        drug_embeds = self.drug_embedding(drug_id)
        query = self.query_projection(drug_embeds).unsqueeze(1)
        
        attn_output, _ = self.cross_attention(
            query=query,
            key=sequence_outputs,
            value=sequence_outputs
        )
        
        context_vector = attn_output.squeeze(1)
        logits = self.reg_head(context_vector).squeeze(-1)
        
        if labels is not None:
            loss = torch.nn.MSELoss()(logits, labels)
            return (loss, logits)
        return logits

class InferenceDataset(Dataset):
    """
    Complete definition of the Dataset class for inference.
    """
    def __init__(self, dataframe, tokenizer, context_col):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.context_col = context_col
    def __len__(self): 
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # The context is already formatted, we pass it directly
        sequence = row[self.context_col].upper().replace('U', 'T')
        encoding = self.tokenizer(sequence, truncation=True, padding='longest', return_tensors='pt')
        return {"input_ids": encoding['input_ids'].flatten(), "attention_mask": encoding['attention_mask'].flatten()}

def run_inference_worker(gpu_id, data_chunk, drug_to_id, best_hyperparams):
    """
    Function executed by each worker process on its assigned GPU.
    """
    device = torch.device(f"cuda:{gpu_id}")
    worker_pid = os.getpid()
    print(f"[GPU {gpu_id}, PID {worker_pid}]: Starting worker.")

    # Each worker loads its own copy of the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(PROD_MODEL_PATH, trust_remote_code=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = PanDrugTransformer(
        "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
        num_drugs=len(drug_to_id),
        head_hidden_size=best_hyperparams['head_hidden_size'],
        drug_embed_dim=best_hyperparams['drug_embed_dim'],
        num_attention_heads=best_hyperparams['num_attention_heads'],
        dropout_rate=best_hyperparams['dropout_rate']
    )
    # Loading weights
    weights_path_safetensors = os.path.join(PROD_MODEL_PATH, 'model.safetensors')
    weights_path_bin = os.path.join(PROD_MODEL_PATH, 'pytorch_model.bin')

    if os.path.exists(weights_path_safetensors):
        state_dict = load_file(weights_path_safetensors, device='cpu')
        model.load_state_dict(state_dict)
    elif os.path.exists(weights_path_bin):
        model.load_state_dict(torch.load(weights_path_bin, map_location='cpu'))
    else:
        raise FileNotFoundError(f"No weights file found in {PROD_MODEL_PATH}")

    model.to(device)
    model.eval()
    
    print(f"[GPU {gpu_id}, PID {worker_pid}]: Model loaded on device.")

    # DataFrame to store the predictions of this worker
    predictions_df_chunk = data_chunk.copy()

    for drug_name, drug_id in drug_to_id.items():
        print(f"[GPU {gpu_id}, PID {worker_pid}]: Starting inference for {drug_name}.")
        dataset = InferenceDataset(data_chunk, tokenizer, 'extracted_context')
        # num_workers=0 is safer inside a multiprocessing process
        loader = DataLoader(dataset, batch_size=2048, shuffle=False, num_workers=0, collate_fn=data_collator)
        
        all_preds_transformed = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                batch['drug_id'] = torch.tensor([drug_id] * len(batch['input_ids']), dtype=torch.long).to(device)
                preds = model(**batch)
                all_preds_transformed.extend(preds.cpu().numpy())
                
        predictions_df_chunk[f'our_preds_{drug_name}'] = np.expm1(all_preds_transformed)

    # Save the results of this chunk to a temporary file
    chunk_output_path = os.path.join(RESULTS_DIR, f"temp_predictions_gpu_{gpu_id}.parquet")
    predictions_df_chunk.to_parquet(chunk_output_path, index=False)
    print(f"[GPU {gpu_id}, PID {worker_pid}]: Work finished. Results saved in {chunk_output_path}.")


def extract_context(row, n):
    """
    Extracts a context of 'n' nucleotides around the uppercase stop codon.
    Handles edge cases with 'N' padding.
    """
    seq = row['nt_seq']
    # Use a regular expression to find the uppercase stop codon
    match = re.search(r'[A-Z]{3}', seq)
    if not match:
        return None
    
    start_pos = match.start()
    end_pos = match.end() # <-- CORRECTION HERE
    
    # Extract the context
    upstream = seq[max(0, start_pos - n):start_pos]
    downstream = seq[end_pos:end_pos + n]
    
    # Handle padding
    pad_left = 'N' * (n - len(upstream))
    pad_right = 'N' * (n - len(downstream))
    
    # Reconstruct the context sequence expected by the model, with the stop codon in lowercase
    return pad_left + upstream.lower() + match.group(0).lower() + downstream.lower()


# --- Main Process ---
if __name__ == "__main__":
    # 'spawn' is recommended for compatibility with CUDA
    mp.set_start_method('spawn', force=True)
    start_time = time.time()

    print(f"--- STARTING MAIN PROCESS ---")
    print(f"Detected {NUM_GPUS} available GPUs.")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Loading configs
    with open(os.path.join(RESULTS_DIR, "best_hyperparams.json"), 'r') as f:
        best_hyperparams = json.load(f)
    with open(os.path.join(PROD_MODEL_PATH, "drug_map.json"), 'r') as f:
        drug_to_id = json.load(f)

    # 2. Loading and preparing data (CPU-bound, done once)
    rds_path = os.path.join(DATA_DIR, "list2_dtbl.rds")
    if not os.path.exists(rds_path):
        raise FileNotFoundError(f"The data file {rds_path} was not found. Please update the path in the DATA_DIR variable.")
        
    print(f"Loading R file: {rds_path}...")
    result_r = pyreadr.read_r(rds_path)
    genome_df = result_r[list(result_r.keys())[0]]
    print(f"File loaded. Size: {genome_df.shape}")

    N_CONTEXT = int(''.join(filter(str.isdigit, CONTEXT_COL_NAME_FROM_MODEL))) // 2
    print(f"Extracting a context of n={N_CONTEXT} nucleotides...")
    tqdm.pandas(desc="Extracting contexts")
    genome_df['extracted_context'] = genome_df.progress_apply(lambda row: extract_context(row, N_CONTEXT), axis=1)
    genome_df.dropna(subset=['extracted_context'], inplace=True)
    
    unique_contexts_df = genome_df[['extracted_context']].drop_duplicates().reset_index(drop=True)
    print(f"Number of unique contexts to process: {len(unique_contexts_df)}")

    # 3. Splitting data for workers
    data_chunks = np.array_split(unique_contexts_df, NUM_GPUS)
    print(f"Data split into {len(data_chunks)} chunks for {NUM_GPUS} GPUs.")

    # 4. Launching worker processes
    processes = []
    for gpu_id in range(NUM_GPUS):
        p = mp.Process(target=run_inference_worker, args=(gpu_id, data_chunks[gpu_id], drug_to_id, best_hyperparams))
        p.start()
        processes.append(p)

    # 5. Wait for all workers to finish
    for p in processes:
        p.join()

    print("\n--- ALL WORKERS HAVE FINISHED ---")
    print("--- Result Aggregation Step ---")

    # 6. Merge partial results
    all_predictions_dfs = []
    for gpu_id in range(NUM_GPUS):
        chunk_path = os.path.join(RESULTS_DIR, f"temp_predictions_gpu_{gpu_id}.parquet")
        if os.path.exists(chunk_path):
            df_chunk = pd.read_parquet(chunk_path)
            all_predictions_dfs.append(df_chunk)
            os.remove(chunk_path) # Delete the temporary file
    
    if not all_predictions_dfs:
        raise Exception("No partial prediction file was generated. Check the worker logs.")

    predictions_df = pd.concat(all_predictions_dfs, ignore_index=True)
    print(f"Partial results merged. Total size of unique predictions: {predictions_df.shape}")
    
    # 7. Final join and save
    print("Final join with genomic data...")
    # To be sure that the join works, we only keep the necessary columns from genome_df
    # to avoid conflicts of types or column names
    cols_to_keep = [col for col in genome_df.columns if 'our_preds_' not in col]
    final_genome_wide_df = pd.merge(genome_df[cols_to_keep], predictions_df, on='extracted_context', how='left')
    
    output_final_path = os.path.join(RESULTS_DIR, "our_genome_wide_predictions_full.parquet")
    print(f"Saving the final complete file in {output_final_path}...")
    final_genome_wide_df.to_parquet(output_final_path, index=False)
    
    end_time = time.time()
    print(f"\n--- MULTI-GPU GENOME-WIDE ANALYSIS FINISHED ---")
    print(f"Total execution time: {(end_time - start_time) / 3600:.2f} hours.")