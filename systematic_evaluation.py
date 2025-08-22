#!/home2020/home/icube/nhaas/.conda/envs/TransStop/bin/python

#SBATCH -p publicgpu
#SBATCH -N 1
#SBATCH -x hpc-n932
#SBATCH --gres=gpu:4
#SBATCH --constraint="gpuh100|gpua100|gpul40s|gpua40|gpurtx6000"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nicolas.haas3@etu.unistra.fr
#SBATCH --job-name=systematic_evaluation
#SBATCH --output=systematic_evaluation_%j.out

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments, default_data_collator, set_seed
from sklearn.metrics import r2_score
import os
import json

os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_PROJECT"] = "ptc-context-evaluation"

SEED = 42
set_seed(SEED)

print("--- PART 2a: CONTEXT EVALUATION FOR NUCTRANSFORMER ---")

# --- Configuration & Data Loading ---
PROCESSED_DATA_DIR = "../processed_data/"
RESULTS_DIR = "../results/"
TEMP_MODEL_DIR = "../temp_models_context_search/"
for dir_path in [RESULTS_DIR, TEMP_MODEL_DIR]: os.makedirs(dir_path, exist_ok=True)

train_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "train_df.csv"))
val_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "val_df.csv"))
drug_to_id = {drug: i for i, drug in enumerate(train_df['drug'].unique())}
NUM_DRUGS = len(drug_to_id)
train_df['drug_id'] = train_df['drug'].map(drug_to_id)
val_df['drug_id'] = val_df['drug'].map(drug_to_id)

# --- Re-usable Classes and Functions ---
class PTCDataset(Dataset):
    def __init__(self, dataframe, tokenizer, context_col):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.context_col = context_col
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sequence = row[self.context_col].replace('U', 'T')
        encoding = self.tokenizer(sequence, truncation=True, padding='longest', return_tensors='pt')
        return {"input_ids": encoding['input_ids'].flatten(), "attention_mask": encoding['attention_mask'].flatten(), "drug_id": torch.tensor(row['drug_id'], dtype=torch.long), "labels": torch.tensor(float(row['RT_transformed']), dtype=torch.float)}

class PanDrugTransformerForTrainer(torch.nn.Module):
    def __init__(self, model_name, num_drugs, head_hidden_size=256, drug_embed_dim=64, num_attention_heads=8, **kwargs):
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
            torch.nn.Dropout(0.1),
            torch.nn.Linear(head_hidden_size, 1)
        )

    def forward(self, input_ids, attention_mask, drug_id, labels=None, **kwargs):
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
        
        loss = None
        if labels is not None:
            loss = torch.nn.MSELoss()(logits, labels)
            
        return (loss, logits) if loss is not None else logits

def compute_metrics_global(eval_pred):
    predictions, labels = eval_pred; predictions_inv = np.expm1(predictions); labels_inv = np.expm1(labels); predictions_inv[predictions_inv < 0] = 0
    return {"r2_score": r2_score(labels_inv, predictions_inv)}

# --- Sequential Context Evaluation ---
MODEL_KEY = "NucTransformer"
MODEL_HF_NAME = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
# *** UPDATE CONTEXT LIST ***
CONTEXTS_TO_TEST = ['seq_context_144', 'seq_context_42', 'seq_context_18', 'seq_context_12', 'seq_context_6', 'seq_context_0']
evaluation_log = []

print(f"\n--- Evaluating contexts for model: {MODEL_KEY} ---")
context_scores = {}
for context in CONTEXTS_TO_TEST:
    run_name = f"{MODEL_KEY}-{context}-full_finetune"
    print(f"\n--- Testing context: {context} ---")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_HF_NAME, trust_remote_code=True)
    train_dataset = PTCDataset(train_df, tokenizer, context)
    val_dataset = PTCDataset(val_df, tokenizer, context)
    model = PanDrugTransformerForTrainer(MODEL_HF_NAME, NUM_DRUGS)
    
    try: model.base_model.gradient_checkpointing_enable()
    except ValueError: print(f"Warning: Gradient checkpointing not supported. Continuing without.")

    training_args = TrainingArguments(
        output_dir=os.path.join(TEMP_MODEL_DIR, run_name),
        num_train_epochs=20, per_device_train_batch_size=32,
        per_device_eval_batch_size=32, learning_rate=2e-5, warmup_steps=500, weight_decay=0.01,
        logging_dir='./logs', logging_steps=200, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="r2_score", greater_is_better=True,
        save_total_limit=1, report_to="wandb", fp16=True, dataloader_num_workers=4, seed=SEED
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset, data_collator=default_data_collator, compute_metrics=compute_metrics_global)
    
    try:
        trainer.train()
        best_val_r2 = max([log['eval_r2_score'] for log in trainer.state.log_history if 'eval_r2_score' in log])
    except Exception as e:
        print(f"ERROR on {run_name}: {e}. Setting score to -inf.")
        best_val_r2 = -float('inf')
        
    context_scores[context] = best_val_r2
    evaluation_log.append({"model": MODEL_KEY, "context": context, "strategy": "full_finetune", "r2": best_val_r2})
    del model, trainer, tokenizer
    torch.cuda.empty_cache()

# --- Finalization and Saving ---
valid_context_scores = {k: v for k, v in context_scores.items() if v > -float('inf')}
if not valid_context_scores:
    raise RuntimeError("All context evaluations failed.")

best_r2_score = max(valid_context_scores.values())
TOLERANCE = 0.01

# Filter for contexts within the tolerance
candidate_contexts = {
    context: r2 for context, r2 in valid_context_scores.items()
    if r2 >= best_r2_score - TOLERANCE
}

# Select the context with the smallest size from the candidates
best_context = min(
    candidate_contexts.keys(),
    key=lambda context: int(context.split('_')[-1])
)

print(f"\n===> BEST CONTEXT FOUND: {best_context} (R2={valid_context_scores[best_context]:.4f})")

final_best_config = {
    "model_name": MODEL_KEY,
    "context_column": best_context,
    "tuning_strategy": "full_finetune",
    "best_validation_R2": valid_context_scores[best_context]
}

print("\n--- FINAL OPTIMAL CONFIGURATION ---")
print(json.dumps(final_best_config, indent=2))
log_df = pd.DataFrame(evaluation_log)
log_df.to_csv(os.path.join(RESULTS_DIR, "context_evaluation_log.csv"), index=False)
final_log_df = pd.DataFrame([final_best_config])
final_log_df.to_csv(os.path.join(RESULTS_DIR, "systematic_evaluation_log.csv"), index=False)
print("\nContext search finished. The best configuration has been saved for the next step.")
print("--- END OF PART 2a ---")