#!/home2020/home/icube/nhaas/.conda/envs/TransStop/bin/python

#SBATCH -p publicgpu
#SBATCH -N 1
#SBATCH -x hpc-n932
#SBATCH --gres=gpu:4
#SBATCH --constraint="gpuh100|gpua100|gpul40s"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nicolas.haas3@etu.unistra.fr
#SBATCH --job-name=PTC_Inference
#SBATCH --output=ptc_inference_%j.out

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

# --- Configuration Globale ---
# Ces variables seront utilisées par tous les processus
RESULTS_DIR = "./results/"
MODELS_DIR = "./models/"
# METTRE À JOUR CE CHEMIN
DATA_DIR = "./data/" 
PROD_MODEL_PATH = os.path.join(MODELS_DIR, "production_model")
# METTRE À JOUR CETTE COLONNE
CONTEXT_COL_NAME_FROM_MODEL = "seq_context_18" 
NUM_GPUS = torch.cuda.device_count()

# --- Classes (doivent être définies au niveau global pour le multiprocessing) ---

class PanDrugTransformerForTrainer(torch.nn.Module):
    """
    Définition complète de la classe du modèle.
    """
    def __init__(self, model_name, num_drugs, head_hidden_size=256, drug_embedding_size=16, dropout_rate=0.1, **kwargs):
        super().__init__()
        full_model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True, **kwargs)
        self.base_model = full_model.base_model
        self.config = self.base_model.config
        self.drug_embedding = torch.nn.Embedding(num_drugs, drug_embedding_size)
        base_model_hidden_size = self.base_model.config.hidden_size
        self.reg_head = torch.nn.Sequential(
            torch.nn.Linear(base_model_hidden_size + drug_embedding_size, head_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(head_hidden_size, 1)
        )
    
    def forward(self, input_ids, attention_mask, drug_id, labels=None, **kwargs):
        output_attentions = kwargs.get("output_attentions", False)
        base_model_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )
        cls_embedding = base_model_outputs.last_hidden_state[:, 0]
        drug_emb = self.drug_embedding(drug_id)
        combined_embedding = torch.cat([cls_embedding, drug_emb], dim=1)
        logits = self.reg_head(combined_embedding).squeeze(-1)
        
        if output_attentions:
            return logits, base_model_outputs.attentions
        return logits

class InferenceDataset(Dataset):
    """
    Définition complète de la classe Dataset pour l'inférence.
    """
    def __init__(self, dataframe, tokenizer, context_col):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.context_col = context_col
    def __len__(self): 
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Le contexte est déjà formaté, on le passe directement
        sequence = row[self.context_col].upper().replace('U', 'T')
        encoding = self.tokenizer(sequence, truncation=True, padding='longest', return_tensors='pt')
        return {"input_ids": encoding['input_ids'].flatten(), "attention_mask": encoding['attention_mask'].flatten()}

def run_inference_worker(gpu_id, data_chunk, drug_to_id, best_hyperparams):
    """
    Fonction exécutée par chaque processus travailleur sur son GPU assigné.
    """
    device = torch.device(f"cuda:{gpu_id}")
    worker_pid = os.getpid()
    print(f"[GPU {gpu_id}, PID {worker_pid}]: Démarrage du worker.")

    # Chaque worker charge sa propre copie du modèle et du tokenizer
    tokenizer = AutoTokenizer.from_pretrained(PROD_MODEL_PATH, trust_remote_code=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = PanDrugTransformerForTrainer(
        "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", len(drug_to_id), 
        head_hidden_size=best_hyperparams['head_hidden_size'],
        drug_embedding_size=best_hyperparams['drug_embedding_size'],
        dropout_rate=best_hyperparams['dropout_rate']
    )
    # Chargement des poids
    weights_path_safetensors = os.path.join(PROD_MODEL_PATH, 'model.safetensors')
    weights_path_bin = os.path.join(PROD_MODEL_PATH, 'pytorch_model.bin')

    if os.path.exists(weights_path_safetensors):
        state_dict = load_file(weights_path_safetensors, device='cpu')
        model.load_state_dict(state_dict)
    elif os.path.exists(weights_path_bin):
        model.load_state_dict(torch.load(weights_path_bin, map_location='cpu'))
    else:
        raise FileNotFoundError(f"Aucun fichier de poids trouvé dans {PROD_MODEL_PATH}")

    model.to(device)
    model.eval()
    
    print(f"[GPU {gpu_id}, PID {worker_pid}]: Modèle chargé sur le device.")

    # DataFrame pour stocker les prédictions de ce worker
    predictions_df_chunk = data_chunk.copy()

    for drug_name, drug_id in drug_to_id.items():
        print(f"[GPU {gpu_id}, PID {worker_pid}]: Début de l'inférence pour {drug_name}.")
        dataset = InferenceDataset(data_chunk, tokenizer, 'extracted_context')
        # num_workers=0 est plus sûr à l'intérieur d'un processus multiprocessing
        loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0, collate_fn=data_collator)
        
        all_preds_transformed = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                batch['drug_id'] = torch.tensor([drug_id] * len(batch['input_ids']), dtype=torch.long).to(device)
                preds = model(**batch)
                all_preds_transformed.extend(preds.cpu().numpy())
                
        predictions_df_chunk[f'our_preds_{drug_name}'] = np.expm1(all_preds_transformed)

    # Sauvegarder les résultats de ce chunk dans un fichier temporaire
    chunk_output_path = os.path.join(RESULTS_DIR, f"temp_predictions_gpu_{gpu_id}.parquet")
    predictions_df_chunk.to_parquet(chunk_output_path, index=False)
    print(f"[GPU {gpu_id}, PID {worker_pid}]: Travail terminé. Résultats sauvegardés dans {chunk_output_path}.")


def extract_context(row, n):
    """
    Extrait un contexte de 'n' nucléotides autour du codon stop en majuscules.
    Gère les cas de bord avec du padding 'N'.
    """
    seq = row['nt_seq']
    # Utiliser une expression régulière pour trouver le stop en majuscules
    match = re.search(r'[A-Z]{3}', seq)
    if not match:
        return None
    
    start_pos = match.start()
    end_pos = match.end() # <-- CORRECTION ICI
    
    # Extraire le contexte
    upstream = seq[max(0, start_pos - n):start_pos]
    downstream = seq[end_pos:end_pos + n]
    
    # Gérer le padding
    pad_left = 'N' * (n - len(upstream))
    pad_right = 'N' * (n - len(downstream))
    
    # Reconstruire la séquence de contexte attendue par le modèle, avec le stop en minuscules
    return pad_left + upstream.lower() + match.group(0).lower() + downstream.lower()


# --- Processus Principal ---
if __name__ == "__main__":
    # 'spawn' est recommandé pour la compatibilité avec CUDA
    mp.set_start_method('spawn', force=True)
    start_time = time.time()

    print(f"--- DÉMARRAGE DU PROCESSUS PRINCIPAL ---")
    print(f"Détecté {NUM_GPUS} GPUs disponibles.")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Chargement des configs
    with open(os.path.join(RESULTS_DIR, "best_hyperparams.json"), 'r') as f:
        best_hyperparams = json.load(f)
    with open(os.path.join(PROD_MODEL_PATH, "drug_map.json"), 'r') as f:
        drug_to_id = json.load(f)

    # 2. Chargement et préparation des données (CPU-bound, fait une seule fois)
    rds_path = os.path.join(DATA_DIR, "list2_dtbl.rds")
    if not os.path.exists(rds_path):
        raise FileNotFoundError(f"Le fichier de données {rds_path} est introuvable. Veuillez mettre à jour le chemin dans la variable DATA_DIR.")
        
    print(f"Chargement du fichier R : {rds_path}...")
    result_r = pyreadr.read_r(rds_path)
    genome_df = result_r[list(result_r.keys())[0]]
    print(f"Fichier chargé. Taille: {genome_df.shape}")

    N_CONTEXT = int(''.join(filter(str.isdigit, CONTEXT_COL_NAME_FROM_MODEL))) // 2
    print(f"Extraction d'un contexte de n={N_CONTEXT} nucléotides...")
    tqdm.pandas(desc="Extraction des contextes")
    genome_df['extracted_context'] = genome_df.progress_apply(lambda row: extract_context(row, N_CONTEXT), axis=1)
    genome_df.dropna(subset=['extracted_context'], inplace=True)
    
    unique_contexts_df = genome_df[['extracted_context']].drop_duplicates().reset_index(drop=True)
    print(f"Nombre de contextes uniques à traiter : {len(unique_contexts_df)}")

    # 3. Division des données pour les workers
    data_chunks = np.array_split(unique_contexts_df, NUM_GPUS)
    print(f"Données divisées en {len(data_chunks)} chunks pour {NUM_GPUS} GPUs.")

    # 4. Lancement des processus workers
    processes = []
    for gpu_id in range(NUM_GPUS):
        p = mp.Process(target=run_inference_worker, args=(gpu_id, data_chunks[gpu_id], drug_to_id, best_hyperparams))
        p.start()
        processes.append(p)

    # 5. Attendre la fin de tous les workers
    for p in processes:
        p.join()

    print("\n--- TOUS LES WORKERS ONT TERMINÉ ---")
    print("--- Étape d'Agrégation des Résultats ---")

    # 6. Fusionner les résultats partiels
    all_predictions_dfs = []
    for gpu_id in range(NUM_GPUS):
        chunk_path = os.path.join(RESULTS_DIR, f"temp_predictions_gpu_{gpu_id}.parquet")
        if os.path.exists(chunk_path):
            df_chunk = pd.read_parquet(chunk_path)
            all_predictions_dfs.append(df_chunk)
            os.remove(chunk_path) # Supprimer le fichier temporaire
    
    if not all_predictions_dfs:
        raise Exception("Aucun fichier de prédiction partiel n'a été généré. Vérifiez les logs des workers.")

    predictions_df = pd.concat(all_predictions_dfs, ignore_index=True)
    print(f"Résultats partiels fusionnés. Taille totale des prédictions uniques: {predictions_df.shape}")
    
    # 7. Jointure finale et sauvegarde
    print("Jointure finale avec les données génomiques...")
    # Pour être sûr que la jointure fonctionne, on ne garde que les colonnes nécessaires de genome_df
    # pour éviter les conflits de types ou de noms de colonnes
    cols_to_keep = [col for col in genome_df.columns if 'our_preds_' not in col]
    final_genome_wide_df = pd.merge(genome_df[cols_to_keep], predictions_df, on='extracted_context', how='left')
    
    output_final_path = os.path.join(RESULTS_DIR, "our_genome_wide_predictions_full.parquet")
    print(f"Sauvegarde du fichier final complet dans {output_final_path}...")
    final_genome_wide_df.to_parquet(output_final_path, index=False)
    
    end_time = time.time()
    print(f"\n--- ANALYSE GENOME-WIDE MULTI-GPU TERMINÉE ---")
    print(f"Temps total d'exécution : {(end_time - start_time) / 3600:.2f} heures.")