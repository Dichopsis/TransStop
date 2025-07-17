#!/home2020/home/icube/nhaas/.conda/envs/TransStop/bin/python

#SBATCH -p publicgpu
#SBATCH -N 1
#SBATCH -x hpc-n932
#SBATCH --gres=gpu:2
#SBATCH --constraint="gpuh100|gpua100|gpul40s|gpua40|gpurtx6000"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nicolas.haas3@etu.unistra.fr

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments, default_data_collator, set_seed
from sklearn.metrics import r2_score, mean_absolute_error
import optuna
import os
import json
import wandb

# --- Configuration Initiale ---
# Activer le mode hors ligne pour la compatibilité avec les clusters sans accès internet direct
os.environ["WANDB_MODE"] = "offline"
# Définir le projet W&B pour cette étape
os.environ["WANDB_PROJECT"] = "ptc-hyperparameter-tuning"

SEED = 42
set_seed(SEED)

print("--- PART 2b: HYPERPARAMETER OPTIMIZATION AND FINAL TRAINING ---")

# --- Chemins et Répertoires ---
PROCESSED_DATA_DIR = "./processed_data/"
RESULTS_DIR = "./results/"
MODELS_DIR = "./models/"
TEMP_MODEL_DIR_HPARAM = "./temp_models_hparam/"

for dir_path in [MODELS_DIR, TEMP_MODEL_DIR_HPARAM]: os.makedirs(dir_path, exist_ok=True)

# --- Chargement de la meilleure configuration de l'étape 2a ---
try:
    best_config_df = pd.read_csv(os.path.join(RESULTS_DIR, "systematic_evaluation_log.csv"))
    best_config = best_config_df.iloc[0].to_dict()
except FileNotFoundError:
    raise FileNotFoundError("Le fichier 'systematic_evaluation_log.csv' de l'étape 2a est introuvable.")

print(f"Meilleure Configuration Chargée: {best_config}")

# --- Chargement des Données ---
train_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "train_df.csv"))
val_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "val_df.csv"))
test_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "test_df.csv"))

drug_to_id = {drug: i for i, drug in enumerate(train_df['drug'].unique())}
NUM_DRUGS = len(drug_to_id)
train_df['drug_id'] = train_df['drug'].map(drug_to_id)
val_df['drug_id'] = val_df['drug'].map(drug_to_id)
test_df['drug_id'] = test_df['drug'].map(drug_to_id)

MODEL_HF_NAME = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
context_col = best_config['context_column']

# --- Classes et Fonctions Réutilisables ---
class PTCDataset(Dataset):
    def __init__(self, dataframe, tokenizer, context_col):
        self.df = dataframe; self.tokenizer = tokenizer; self.context_col = context_col
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]; sequence = row[self.context_col].replace('U', 'T')
        encoding = self.tokenizer(sequence, truncation=True, padding='longest', return_tensors='pt')
        return {"input_ids": encoding['input_ids'].flatten(), "attention_mask": encoding['attention_mask'].flatten(), "drug_id": torch.tensor(row['drug_id'], dtype=torch.long), "labels": torch.tensor(float(row['RT_transformed']), dtype=torch.float)}

class PanDrugTransformerForTrainer(torch.nn.Module):
    def __init__(self, model_name, num_drugs, head_hidden_size=256, drug_embedding_size=16, dropout_rate=0.1, **kwargs):
        super().__init__()
        full_model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True, **kwargs)
        self.base_model = full_model.base_model
        self.config = self.base_model.config
        self.drug_embedding = torch.nn.Embedding(num_drugs, drug_embedding_size)
        base_model_hidden_size = self.base_model.config.hidden_size
        self.reg_head = torch.nn.Sequential(
            torch.nn.Linear(base_model_hidden_size + drug_embedding_size, head_hidden_size),
            torch.nn.ReLU(), torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(head_hidden_size, 1)
        )
    def forward(self, input_ids, attention_mask, drug_id, labels=None, **kwargs):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.reg_head(torch.cat([outputs.last_hidden_state[:, 0], self.drug_embedding(drug_id)], dim=1)).squeeze(-1)
        loss = None
        if labels is not None: loss = torch.nn.MSELoss()(logits, labels)
        return (loss, logits) if loss is not None else logits

def compute_metrics_global(eval_pred):
    predictions, labels = eval_pred; predictions_inv = np.expm1(predictions); labels_inv = np.expm1(labels); predictions_inv[predictions_inv < 0] = 0
    return {"r2_score": r2_score(labels_inv, predictions_inv), "mae": mean_absolute_error(labels_inv, predictions_inv)}

# --- 2.3. Optimisation des Hyperparamètres avec Optuna et W&B ---
print("\n--- 2.3. Lancement de l'Optimisation des Hyperparamètres ---")
tokenizer = AutoTokenizer.from_pretrained(MODEL_HF_NAME, trust_remote_code=True)
train_dataset = PTCDataset(train_df, tokenizer, context_col)
val_dataset = PTCDataset(val_df, tokenizer, context_col)

def objective(trial):
    run_name = f"optuna-trial-{trial.number}"
    
    # Définition de l'espace de recherche des hyperparamètres
    training_args = TrainingArguments(
        output_dir=os.path.join(TEMP_MODEL_DIR_HPARAM, run_name),
        run_name=run_name,
        learning_rate=trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True),
        per_device_train_batch_size=trial.suggest_categorical("batch_size", [16, 32]),
        num_train_epochs=8,
        weight_decay=trial.suggest_float("weight_decay", 0.0, 0.1),
        warmup_ratio=trial.suggest_float("warmup_ratio", 0.0, 0.2),
        #adam_beta2=trial.suggest_float("adam_beta2", 0.98, 0.999),
        lr_scheduler_type=trial.suggest_categorical("lr_scheduler_type", ["linear", "cosine"]),
        eval_strategy="epoch",
        logging_steps=200,
        save_strategy="no",
        report_to="wandb",
        fp16=True,
        seed=SEED
    )
    
    head_hidden_size = trial.suggest_categorical("head_hidden_size", [128, 256, 512])
    drug_embedding_size = trial.suggest_categorical("drug_embedding_size", [8, 16, 32])
    dropout_rate = trial.suggest_float("dropout_rate", 0.05, 0.3)
    
    model = PanDrugTransformerForTrainer(
        MODEL_HF_NAME, NUM_DRUGS, 
        head_hidden_size=head_hidden_size,
        drug_embedding_size=drug_embedding_size,
        dropout_rate=dropout_rate
    )
    
    try: model.base_model.gradient_checkpointing_enable()
    except ValueError: pass

    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=val_dataset, data_collator=default_data_collator,
        compute_metrics=compute_metrics_global
    )
    
    try:
        trainer.train()
        r2 = max([log['eval_r2_score'] for log in trainer.state.log_history if 'eval_r2_score' in log])
    except (torch.cuda.OutOfMemoryError, ValueError) as e:
        print(f"ERREUR durant l'essai HPO {trial.number}: {e}. Essai abandonné (pruned).")
        raise optuna.exceptions.TrialPruned()
    finally:
        wandb.finish() # S'assurer que chaque exécution W&B se termine

    return r2

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
best_hyperparams = study.best_params
print(f"Meilleurs Hyperparamètres trouvés : {best_hyperparams}")

# Sauvegarde des résultats de l'étude Optuna
print("Sauvegarde du rapport d'étude Optuna en CSV...")
trials_df = study.trials_dataframe()
trials_df.to_csv(os.path.join(RESULTS_DIR, "optuna_trials_report.csv"), index=False)
print("Rapport Optuna sauvegardé.")

with open(os.path.join(RESULTS_DIR, "best_hyperparams.json"), 'w') as f:
    json.dump(best_hyperparams, f)

# --- 2.4. Entraînement du Modèle Final de Production ---
print("\n--- 2.4. Entraînement du Modèle Final de Production ---")
FINAL_EPOCHS = 20
final_train_df = pd.concat([train_df, val_df], ignore_index=True)

final_train_dataset = PTCDataset(final_train_df, tokenizer, context_col)
test_dataset = PTCDataset(test_df, tokenizer, context_col)

# Initialiser une exécution W&B pour l'entraînement final
wandb.init(project="ptc-final-training", name="production-run", config=best_hyperparams, job_type="train")

final_training_args = TrainingArguments(
    output_dir=os.path.join(MODELS_DIR, "production_model_checkpoints"),
    run_name="production-run",
    num_train_epochs=FINAL_EPOCHS,
    learning_rate=best_hyperparams['learning_rate'],
    per_device_train_batch_size=best_hyperparams['batch_size'],
    per_device_eval_batch_size=best_hyperparams['batch_size'] * 2,
    weight_decay=best_hyperparams['weight_decay'],
    warmup_ratio=best_hyperparams['warmup_ratio'],
    #adam_beta2=best_hyperparams['adam_beta2'],
    lr_scheduler_type=best_hyperparams['lr_scheduler_type'],
    eval_strategy="no",
    save_strategy="epoch",
    save_total_limit=1,
    logging_steps=100,
    report_to="wandb",
    fp16=True,
    seed=SEED
)

model = PanDrugTransformerForTrainer(
    MODEL_HF_NAME, NUM_DRUGS, 
    head_hidden_size=best_hyperparams['head_hidden_size'],
    drug_embedding_size=best_hyperparams['drug_embedding_size'],
    dropout_rate=best_hyperparams['dropout_rate']
)
try: model.base_model.gradient_checkpointing_enable()
except ValueError: pass

final_trainer = Trainer(
    model=model, args=final_training_args, train_dataset=final_train_dataset,
    data_collator=default_data_collator, compute_metrics=compute_metrics_global
)

print("Lancement de l'entraînement final...")
final_trainer.train()

print("Évaluation du modèle final sur le jeu de test...")
test_results = final_trainer.evaluate(test_dataset)

# Log des métriques finales sur W&B
wandb.log({"final_test_metrics": test_results})

print("\n--- PERFORMANCE FINALE DU MODÈLE SUR LE JEU DE TEST ---")
print(f"Test R-squared: {test_results['eval_r2_score']:.4f}")
print(f"Test MAE: {test_results['eval_mae']:.4f}")

production_model_path = os.path.join(MODELS_DIR, "production_model")
final_trainer.save_model(production_model_path)
tokenizer.save_pretrained(production_model_path)
print(f"Modèle de production sauvegardé dans '{production_model_path}'.")

# Log du modèle final comme un "artefact" sur W&B
artifact = wandb.Artifact('production-model', type='model')
artifact.add_dir(production_model_path)
wandb.log_artifact(artifact)

with open(os.path.join(RESULTS_DIR, "final_test_metrics.json"), 'w') as f:
    json.dump(test_results, f)

wandb.finish()
print("--- FIN DE LA PARTIE 2b ---")