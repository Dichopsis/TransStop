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

# --- Configuration et Chargement des Artefacts ---
print("--- PART 3: DEEP MODEL INTERPRETATION AND INSIGHT GENERATION (Corrected) ---")

RESULTS_DIR = "./results/"
MODELS_DIR = "./models/"
PROCESSED_DATA_DIR = "./processed_data/"
PROD_MODEL_PATH = os.path.join(MODELS_DIR, "production_model")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Utilisation du device : {DEVICE}")

# Charger les configurations et les données nécessaires
try:
    with open(os.path.join(RESULTS_DIR, "best_hyperparams.json"), 'r') as f:
        best_hyperparams = json.load(f)
    best_config_df = pd.read_csv(os.path.join(RESULTS_DIR, "systematic_evaluation_log.csv"))
    best_config = best_config_df.iloc[0].to_dict()
    test_df_original = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "test_df.csv"))
except FileNotFoundError as e:
    print(f"Erreur de chargement de fichier : {e}")
    print("Veuillez vous assurer que les scripts 01 et 02b ont été exécutés avec succès.")
    exit()

# *** CORRECTION DÉFINITIVE : CHARGEMENT DU MAPPAGE DEPUIS LA SOURCE DE VÉRITÉ ***
# Au lieu de reconstruire le mappage, nous le chargeons depuis le fichier JSON 
# qui a été sauvegardé ou reconstruit à l'identique.
# C'est la seule méthode qui garantit une synchronisation parfaite avec le modèle entraîné.
try:
    map_path = os.path.join(PROD_MODEL_PATH, "drug_map.json")
    with open(map_path, 'r') as f:
        drug_to_id = json.load(f)
    
    # Créer le mappage inverse
    id_to_drug = {i: drug for drug, i in drug_to_id.items()}
    NUM_DRUGS = len(drug_to_id)
    print("Mappage des médicaments chargé avec succès depuis la source de vérité :", drug_to_id)

    # Créer une palette de couleurs cohérente pour les médicaments
    drug_list_for_palette = sorted(drug_to_id.keys())
    colors = sns.color_palette('tab20', n_colors=len(drug_list_for_palette))
    drug_color_map = dict(zip(drug_list_for_palette, colors))
    print("Palette de couleurs pour les médicaments créée.")

except FileNotFoundError:
    print(f"ERREUR CRITIQUE : Le fichier 'drug_map.json' est introuvable dans {PROD_MODEL_PATH}.")
    print("Ce fichier est essentiel. Veuillez exécuter le script 'reconstruct_and_save_map.py' pour le créer.")
    exit()
# *** FIN DE LA CORRECTION ***


# Appliquer le mappage chargé et cohérent au jeu de test
test_df_original['drug_id'] = test_df_original['drug'].map(drug_to_id)
# Vérifier si des médicaments du test n'étaient pas dans le mappage (ce qui serait une erreur de pipeline)
if test_df_original['drug_id'].isnull().any():
    missing_drugs = test_df_original[test_df_original['drug_id'].isnull()]['drug'].unique()
    print(f"ATTENTION : Les médicaments suivants du jeu de test n'ont pas été trouvés dans le mappage : {missing_drugs}")
    print("Les lignes correspondantes seront supprimées.")
    test_df_original.dropna(subset=['drug_id'], inplace=True)
    test_df_original['drug_id'] = test_df_original['drug_id'].astype(int)


# Définir des variables globales qui pourraient être utilisées plus tard (ex: pour UMAP)
SEED = 42 # Assurez-vous que SEED est défini si vous l'utilisez dans UMAP ou .sample()
MODEL_HF_NAME = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
context_col = best_config['context_column']

# --- Recréer la classe du modèle ---
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
        output_hidden_states = kwargs.get("output_hidden_states", False) # Ajout pour les embeddings
        
        base_model_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states # Passer l'argument
        )
        # Utiliser le CLS token [:, 0] pour la représentation de la séquence
        cls_embedding = base_model_outputs.last_hidden_state[:, 0]
        # Obtenir l'embedding du médicament
        drug_emb = self.drug_embedding(drug_id)
        # Concaténer et passer dans la tête de régression
        combined_embedding = torch.cat([cls_embedding, drug_emb], dim=1)
        logits = self.reg_head(combined_embedding).squeeze(-1)
        
        # Retourner les logits, les attentions (si demandé) et les embeddings CLS (si demandé)
        if output_attentions and output_hidden_states:
            return logits, base_model_outputs.attentions, cls_embedding
        elif output_attentions:
            return logits, base_model_outputs.attentions
        elif output_hidden_states:
            return logits, cls_embedding
        return logits

# --- Chargement du Modèle de Production ---
print("Chargement du modèle de production...")
tokenizer = AutoTokenizer.from_pretrained(PROD_MODEL_PATH, trust_remote_code=True)
model = PanDrugTransformerForTrainer(
    MODEL_HF_NAME, NUM_DRUGS, 
    head_hidden_size=best_hyperparams['head_hidden_size'],
    drug_embedding_size=best_hyperparams['drug_embedding_size'],
    dropout_rate=best_hyperparams['dropout_rate']
)
weights_path_safetensors = os.path.join(PROD_MODEL_PATH, 'model.safetensors')
weights_path_bin = os.path.join(PROD_MODEL_PATH, 'pytorch_model.bin')
if os.path.exists(weights_path_safetensors):
    print("Chargement des poids depuis model.safetensors...")
    state_dict = load_file(weights_path_safetensors, device=DEVICE)
    model.load_state_dict(state_dict)
elif os.path.exists(weights_path_bin):
    print("Chargement des poids depuis pytorch_model.bin...")
    model.load_state_dict(torch.load(weights_path_bin, map_location=torch.device(DEVICE)))
else:
    raise FileNotFoundError(f"Aucun fichier de poids ('model.safetensors' ou 'pytorch_model.bin') trouvé dans {PROD_MODEL_PATH}")
model.to(DEVICE)
model.eval()
print("Modèle chargé avec succès.")


# --- SECTION 3.0: Évaluation des Performances par Médicament ---
print("\n--- 3.0. Évaluation des Performances par Médicament sur le jeu de Test ---")

test_df = test_df_original.copy().reset_index(drop=True)
test_dataset = PTCDataset(test_df, tokenizer, context_col)
test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=default_data_collator, shuffle=False)

all_preds_transformed = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Prédictions sur le jeu de test"):
        # Déplacer seulement les tenseurs vers le GPU
        batch = {k: v.to(DEVICE) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        preds = model(**batch)
        all_preds_transformed.extend(preds.cpu().numpy())

test_df['preds_transformed'] = all_preds_transformed
test_df['preds'] = np.expm1(test_df['preds_transformed'])
test_df['preds'] = test_df['preds'].clip(lower=0)

r2_per_drug = {}
for drug_name, group_df in test_df.groupby('drug'):
    r2 = r2_score(group_df['RT'], group_df['preds'])
    r2_per_drug[drug_name] = r2
    print(f"R² pour {drug_name}: {r2:.4f}")

r2_per_drug_df = pd.DataFrame(list(r2_per_drug.items()), columns=['Drug', 'R2_Score']).sort_values('R2_Score', ascending=False)
r2_global = r2_score(test_df['RT'], test_df['preds'])
print(f"\n--- R² Global sur le jeu de test : {r2_global:.4f} ---")

print("Génération du graphique de corrélation global avec coloration par médicament...")

plt.figure(figsize=(12, 12))

# 1. Créer le nuage de points avec coloration par médicament
# 'hue' colore les points en fonction de la colonne 'drug'
# 'alpha' ajoute de la transparence pour mieux voir les zones denses
# 's' contrôle la taille des points
sns.scatterplot(
    data=test_df,
    x='RT',
    y='preds',
    hue='drug',
    palette=drug_color_map,
    alpha=0.7,
    s=50,
    edgecolor='k', # Ajoute un léger contour noir aux points pour la lisibilité
    linewidth=0.5
)

# 2. Déterminer les limites du graphique pour tracer une ligne parfaite
min_val = min(test_df['RT'].min(), test_df['preds'].min())
max_val = max(test_df['RT'].max(), test_df['preds'].max())
# Ajouter une petite marge
min_val -= (max_val - min_val) * 0.05
max_val += (max_val - min_val) * 0.05

# 3. Tracer la ligne de perfection (y=x) en pointillés rouges
# C'est la ligne sur laquelle tous les points se trouveraient si les prédictions étaient parfaites
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Prédiction Parfaite (y=x)')

# 4. Ajouter les titres, les labels et la grille
plt.title('Prédictions vs. Valeurs Réelles sur le Jeu de Test', fontsize=20, pad=20)
plt.xlabel('Valeur Réelle de Readthrough (RT)', fontsize=16)
plt.ylabel('Valeur Prédite de Readthrough (RT)', fontsize=16)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(title='Médicament', fontsize=12, title_fontsize=14)

# 5. Ajouter le R² global sur le graphique pour le contexte
plt.text(
    x=min_val, 
    y=max_val * 0.95, # Positionner le texte en haut à gauche
    s=f'R² Global = {r2_global:.4f}',
    fontdict={'size': 16, 'weight': 'bold', 'color': 'white'},
    bbox=dict(facecolor='black', alpha=0.6) # Boîte de fond pour la lisibilité
)

# 6. Assurer un ratio d'aspect carré pour que la ligne y=x soit bien à 45 degrés
plt.axis('equal')
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)

# 7. Sauvegarder la figure
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "global_correlation_plot.png"), dpi=300)
plt.close()

print("Graphique de corrélation global sauvegardé dans 'global_correlation_plot.png'.")

# Insérez ce bloc après la génération du graphique de corrélation global.

print("Génération de la grille de graphiques de corrélation par médicament...")

# 1. Obtenir la liste des drogues et préparer la grille de graphiques
drug_list = sorted(test_df['drug'].unique())
num_drugs = len(drug_list)
# Calculer dynamiquement le nombre de lignes et de colonnes
n_cols = 3 
n_rows = (num_drugs + n_cols - 1) // n_cols # Calcule le nombre de lignes nécessaires

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows), sharex=False, sharey=False)
axes = axes.flatten() # Aplatir la grille 2D en une liste 1D pour une itération facile

# La palette est maintenant définie globalement avec drug_color_map

# 2. Boucler sur chaque drogue et créer son propre graphique
for i, drug_name in enumerate(drug_list):
    ax = axes[i] # Sélectionner le sous-graphique courant
    
    # Filtrer les données pour la drogue actuelle
    drug_df = test_df[test_df['drug'] == drug_name]
    
    # Récupérer le score R² déjà calculé
    r2_value = r2_per_drug[drug_name]
    
    # Dessiner le nuage de points sur le sous-graphique
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
    
    # Déterminer les limites pour la ligne de prédiction parfaite (spécifique à ce graphique)
    min_val = min(drug_df['RT'].min(), drug_df['preds'].min())
    max_val = max(drug_df['RT'].max(), drug_df['preds'].max())
    margin = (max_val - min_val) * 0.05
    min_val -= margin
    max_val += margin
    
    # Tracer la ligne de perfection (y=x)
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5)
    
    # Ajouter le titre et le R²
    ax.set_title(f'{drug_name}', fontsize=14, weight='bold')
    ax.text(
        x=min_val,
        y=max_val * 0.9,
        s=f'R² = {r2_value:.4f}',
        fontdict={'size': 12, 'weight': 'bold', 'color': 'black'},
        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3')
    )
    
    # Personnaliser les axes
    ax.set_xlabel('Valeur Réelle (RT)', fontsize=10)
    ax.set_ylabel('Valeur Prédite (RT)', fontsize=10)
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.axis('equal') # Assurer un ratio 1:1
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

# 3. Masquer les sous-graphiques inutilisés s'il y en a
for j in range(num_drugs, len(axes)):
    axes[j].set_visible(False)

# 4. Ajouter un titre principal à la figure
fig.suptitle('Prédictions vs. Valeurs Réelles par Médicament', fontsize=22, y=1.02)

# 5. Ajuster la mise en page et sauvegarder
fig.tight_layout(rect=[0, 0.03, 1, 0.98]) # rect laisse de la place pour le suptitle
plt.savefig(os.path.join(RESULTS_DIR, "per_drug_correlation_grid.png"), dpi=300)
plt.close()

print("Grille de graphiques par médicament sauvegardée dans 'per_drug_correlation_grid.png'.")

toledano_r2 = {
    'Pan-drug': 0.83, 'CC90009': 0.55, 'Clitocine': 0.89, 'DAP': 0.87,
    'G418': 0.76, 'SJ6986': 0.71, 'SRI': 0.76, 'FUr': 0.37, 'Gentamicin': 0.38, 'Untreated': 0.02,
}

# Vos résultats (R² par drogue et R² global)
# r2_per_drug est un dictionnaire que vous avez déjà calculé
# r2_global est la variable que vous avez déjà calculée
our_r2 = r2_per_drug.copy()
our_r2['Pan-drug'] = r2_global

# Créer un DataFrame pour la comparaison
comparison_data = []
for drug, r2_val in our_r2.items():
    if drug in toledano_r2:
        comparison_data.append({'Drug': drug, 'R2_Score': r2_val, 'Model': 'Notre Transformer'})
        comparison_data.append({'Drug': drug, 'R2_Score': toledano_r2[drug], 'Model': 'Toledano et al.'})

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values(by='R2_Score', ascending=False)

# Créer le graphique de comparaison
plt.figure(figsize=(14, 10))
barplot = sns.barplot(
    data=comparison_df,
    x='R2_Score',
    y='Drug',
    hue='Model',
    palette={'Notre Transformer': 'deepskyblue', 'Toledano et al.': 'lightgray'},
    dodge=True
)

plt.title('Comparaison des Performances de Modèle (R²)', fontsize=20, pad=20)
plt.xlabel('R² Score', fontsize=16)
plt.ylabel('Médicament / Condition', fontsize=16)
plt.xlim(0, 1.05)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.legend(title='Modèle', fontsize=12, title_fontsize=14)

# Ajouter les valeurs sur les barres
for p in barplot.patches:
    width = p.get_width()
    plt.text(width + 0.01, p.get_y() + p.get_height() / 2,
             f'{width:.2f}',
             ha='left', va='center', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "r2_comparison_barplot.png"), dpi=300)
plt.close()
print("Graphique de comparaison des R² sauvegardé.")
    

print("\n--- 4.0. Génération des Logos de Séquence ---")

def generate_sequence_logos_for_drug(drug_name, drug_df, context_col, n_seqs=100):
    # Trier les séquences par performance prédite
    best_df = drug_df.nlargest(n_seqs, 'preds')
    worst_df = drug_df.nsmallest(n_seqs, 'preds')

    best_seqs = best_df[context_col].tolist()
    worst_seqs = worst_df[context_col].tolist()

    # Créer les matrices de comptage
    best_counts_df = logomaker.alignment_to_matrix(best_seqs)
    worst_counts_df = logomaker.alignment_to_matrix(worst_seqs)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 8))
    
    # Logo pour les meilleures séquences
    logomaker.Logo(best_counts_df, ax=ax1, color_scheme='classic')
    ax1.set_title(f"Séquences les plus performantes (Top {n_seqs}) pour {drug_name}", fontsize=16)
    ax1.set_ylabel("Bits")
    
    # Logo pour les pires séquences
    logomaker.Logo(worst_counts_df, ax=ax2, color_scheme='classic')
    ax2.set_title(f"Séquences les moins performantes (Bottom {n_seqs}) pour {drug_name}", fontsize=16)
    ax2.set_ylabel("Bits")
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"sequence_logo_{drug_name}.png"), dpi=300)
    plt.close()
    print(f"Logo de séquence sauvegardé pour {drug_name}.")

# Générer les logos pour chaque médicament
for drug_name, group_df in test_df.groupby('drug'):
    # S'assurer qu'il y a assez de séquences pour l'analyse
    if len(group_df) >= 200:
        generate_sequence_logos_for_drug(drug_name, group_df, context_col)
    else:
        print(f"Pas assez de données pour {drug_name} pour générer les logos de séquence.")


# --- SECTION 4.0: PRÉPARATION ET FONCTION UTILITAIRE ---
print("\n--- 4.0. Préparation pour l'Analyse d'Interprétabilité ---")

def predict_batch(sequences, drug_ids, tokenizer, model, device):
    """
    Fonction utilitaire pour prédire un batch de séquences pour des drogues données.
    Prend une liste de séquences et une liste d'IDs de drogues correspondants.
    """
    # Gérer le cas de listes vides pour éviter les erreurs
    if not sequences:
        return np.array([])
        
    inputs = tokenizer(sequences, return_tensors='pt', padding=True, truncation=True)
    batch = {k: v.to(device) for k, v in inputs.items()}
    batch['drug_id'] = torch.tensor(drug_ids, dtype=torch.long).to(device)
    
    with torch.no_grad():
        preds_transformed = model(**batch)
        
    return np.expm1(preds_transformed.cpu().numpy())

# --- SECTION 4.1: SIMILARITÉ FONCTIONNELLE DES PROFILS DE PRÉDICTION ---
print("\n--- 4.1. Analyse de la Similarité des Profils de Prédiction des Médicaments ---")

# 1. Utiliser un ensemble commun de séquences pour la comparaison
unique_sequences = test_df[context_col].unique().tolist()
print(f"Génération de prédictions in-silico pour {len(unique_sequences)} séquences uniques sur toutes les drogues...")

# 2. Générer les prédictions pour chaque drogue sur cet ensemble commun
all_drug_preds = {}
for drug_name, drug_id in tqdm(drug_to_id.items(), desc="Profilage des drogues"):
    drug_ids_batch = [drug_id] * len(unique_sequences)
    preds = predict_batch(unique_sequences, drug_ids_batch, tokenizer, model, DEVICE)
    all_drug_preds[drug_name] = preds

# 3. Créer le DataFrame dense et calculer la matrice de corrélation
drug_profiles_df = pd.DataFrame(all_drug_preds, index=unique_sequences)
drug_similarity_matrix = drug_profiles_df.corr(method='pearson')

# 4. Visualiser avec un clustermap
print("Génération du clustermap de similarité...")
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
    cluster_map.fig.suptitle('Similarité Fonctionnelle des Profils de Réponse', fontsize=20, y=1.02)
    plt.savefig(os.path.join(RESULTS_DIR, "drug_similarity_clustermap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Clustermap de similarité des drogues sauvegardé.")
except Exception as e:
    print(f"Erreur lors de la génération du clustermap : {e}. Étape ignorée.")

# --- SECTION 6.0: VISUALISATION DE L'ESPACE D'EMBEDDING DES SÉQUENCES ---
print("\n--- 6.0. Visualisation de l'Espace d'Embedding des Séquences ---")

def get_sequence_embeddings(dataframe, tokenizer, model, device, context_col, batch_size=64):
    """
    Extrait les embeddings du token CLS pour toutes les séquences d'un DataFrame.
    
    L'embedding du token CLS ([CLS]) est une représentation numérique de la séquence entière,
    capturée par le modèle Transformer. C'est cette représentation que nous allons visualiser.
    """
    model.eval()
    embeddings = []
    
    # Créer un DataLoader pour extraire les embeddings
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
        for batch in tqdm(embedding_loader, desc="Extraction des embeddings de séquence"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            drug_id = batch['drug_id'].to(device)
            
            # Le forward du modèle doit retourner les embeddings CLS
            _, cls_embeddings = model(input_ids=input_ids, attention_mask=attention_mask, drug_id=drug_id, output_hidden_states=True)
            embeddings.append(cls_embeddings.cpu().numpy())
            
    return np.vstack(embeddings)

# --- Étape 1: Préparation des données pour UMAP ---
# Pour une analyse exhaustive, nous utilisons l'intégralité du jeu de données de test.
# Note : Cela peut être coûteux en temps de calcul et en mémoire, en particulier
# pour l'étape UMAP. Les graphiques résultants peuvent également souffrir de
# sur-impression ("overplotting"), rendant la visualisation plus dense.
print("Préparation de l'intégralité du jeu de test pour l'analyse UMAP...")
sample_df_for_umap = test_df.copy().reset_index(drop=True)

# --- Étape 2: Extraction des Embeddings de Séquence ---
# Nous utilisons le modèle pour convertir chaque séquence de l'échantillon en un vecteur
# numérique de haute dimension (l'embedding). C'est l'interprétation de la séquence par le modèle.
print(f"Extraction des embeddings pour {len(sample_df_for_umap)} séquences pour UMAP...")
sequence_embeddings = get_sequence_embeddings(sample_df_for_umap, tokenizer, model, DEVICE, context_col)

# --- Étape 3: Réduction de Dimensionnalité avec UMAP ---
# Les embeddings ont une dimension élevée (souvent > 768). Pour les visualiser sur un
# graphique 2D, nous utilisons UMAP (Uniform Manifold Approximation and Projection).
# UMAP est un algorithme qui réduit la dimensionnalité tout en essayant de préserver
# au mieux la structure globale et les relations de voisinage des données originales.
# En d'autres termes, des points proches dans l'espace de haute dimension le resteront en 2D.
print("Application de UMAP pour la réduction de dimensionnalité...")
umap_reducer = UMAP(n_components=2, random_state=SEED)
reduced_embeddings = umap_reducer.fit_transform(sequence_embeddings)

sample_df_for_umap['umap_x'] = reduced_embeddings[:, 0]
sample_df_for_umap['umap_y'] = reduced_embeddings[:, 1]

# --- Étape 4: Visualisation et Interprétation ---
# Nous créons plusieurs graphiques UMAP en colorant les points selon différentes
# caractéristiques. Cela nous aide à comprendre comment le modèle organise
# l'information dans son espace latent.
print("Génération des graphiques UMAP...")

# Plot 1: Coloration par valeur réelle de RT
# Objectif : Voir si l'organisation spatiale des embeddings est corrélée à la cible de prédiction.
# Interprétation attendue : On espère voir un gradient de couleur, indiquant que le modèle
# place les séquences à faible et haute efficacité de readthrough dans des régions distinctes.
plt.figure(figsize=(12, 10))
sns.scatterplot(
    data=sample_df_for_umap,
    x='umap_x',
    y='umap_y',
    hue='RT', # Colorer par la valeur réelle de RT
    palette='viridis',
    s=10,
    alpha=0.7
)
plt.title('UMAP des Embeddings de Séquence (Coloré par RT Réel)', fontsize=18)
plt.xlabel('UMAP Dimension 1', fontsize=14)
plt.ylabel('UMAP Dimension 2', fontsize=14)
plt.legend(title='RT Réel', fontsize=10, title_fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "umap_embeddings_by_rt.png"), dpi=300)
plt.close()
print("UMAP par RT sauvegardé.")

# Plot 2: Coloration par type de codon stop
# Objectif : Vérifier si le modèle a appris de manière non-supervisée des caractéristiques
# biologiques fondamentales et évidentes des séquences.
# Interprétation attendue : Des clusters très nets et séparés pour chaque type de codon stop (UAA, UAG, UGA)
# seraient une preuve éclatante que le modèle a capturé cette information essentielle.
if 'stop_type' in sample_df_for_umap.columns:
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        data=sample_df_for_umap,
        x='umap_x',
        y='umap_y',
        hue='stop_type', # Colorer par type de codon stop
        palette='tab10', # Une palette discrète pour les catégories
        s=10,
        alpha=0.7
    )
    plt.title('UMAP des Embeddings de Séquence (Coloré par Type de Codon Stop)', fontsize=18)
    plt.xlabel('UMAP Dimension 1', fontsize=14)
    plt.ylabel('UMAP Dimension 2', fontsize=14)
    plt.legend(title='Type de Codon Stop', fontsize=10, title_fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "umap_embeddings_by_stop_type.png"), dpi=300)
    plt.close()
    print("UMAP par type de codon stop sauvegardé.")
else:
    print("La colonne 'stop_type' n'est pas présente dans le DataFrame pour la visualisation UMAP.")

# Plot 3: Coloration par médicament
# Objectif : Comprendre si l'embedding de la séquence est universel ou spécifique à un médicament.
# Interprétation attendue : Un mélange des couleurs (médicaments) est un bon signe.
# Cela signifierait que le modèle apprend une représentation générale de la séquence (sa "lisibilité"),
# indépendamment du médicament, qui est ensuite combinée à l'embedding du médicament dans
# la tête de régression pour la prédiction finale.
plt.figure(figsize=(12, 10))
sns.scatterplot(
    data=sample_df_for_umap,
    x='umap_x',
    y='umap_y',
    hue='drug', # Colorer par médicament
    palette=drug_color_map,
    s=10,
    alpha=0.7
)
plt.title('UMAP des Embeddings de Séquence (Coloré par Médicament)', fontsize=18)
plt.xlabel('UMAP Dimension 1', fontsize=14)
plt.ylabel('UMAP Dimension 2', fontsize=14)
plt.legend(title='Médicament', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=10, title_fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "umap_embeddings_by_drug.png"), dpi=300)
plt.close()
print("UMAP par médicament sauvegardé.")


# --- SECTION 6.1: VISUALISATION DE L'ESPACE D'EMBEDDING DES MÉDICAMENTS ---
print("\n--- 6.1. Visualisation de l'Espace d'Embedding des Médicaments ---")

# --- Étape 1: Extraction des Embeddings de Médicaments ---
# Les embeddings sont les poids de la couche `drug_embedding` du modèle.
# Chaque ligne de cette matrice de poids est le vecteur appris pour un médicament.
print("Extraction des embeddings de médicaments depuis la couche du modèle...")
drug_embeddings = model.drug_embedding.weight.detach().cpu().numpy()

# --- Étape 2: Réduction de Dimensionnalité avec UMAP ---
# Nous appliquons UMAP sur ces embeddings pour les projeter en 2D.
print("Application de UMAP pour la réduction de dimensionnalité des embeddings de médicaments...")
# Ajustement des paramètres UMAP pour un petit nombre de points (médicaments)
# n_neighbors doit être inférieur au nombre de points.
umap_reducer_drugs = UMAP(n_components=2, random_state=SEED, n_neighbors=min(NUM_DRUGS - 1, 15), min_dist=0.1)
reduced_drug_embeddings = umap_reducer_drugs.fit_transform(drug_embeddings)

# --- Étape 3: Création d'un DataFrame pour la Visualisation ---
# Nous assemblons les résultats dans un DataFrame pour une manipulation facile avec Seaborn/Matplotlib.
drug_umap_df = pd.DataFrame({
    'umap_x': reduced_drug_embeddings[:, 0],
    'umap_y': reduced_drug_embeddings[:, 1],
    'drug_name': [id_to_drug[i] for i in range(NUM_DRUGS)] # Utiliser le mappage id -> nom
})

# --- Étape 4: Visualisation et Annotation ---
# Création du nuage de points. Chaque point est un médicament.
# Nous annotons chaque point avec son nom pour l'identification.
print("Génération du graphique UMAP pour l'espace d'embedding des médicaments...")
plt.figure(figsize=(14, 12))
colors_for_plot = drug_umap_df['drug_name'].map(drug_color_map)
sns.scatterplot(
    data=drug_umap_df,
    x='umap_x',
    y='umap_y',
    c=colors_for_plot,
    s=200, # Taille des points plus grande pour la lisibilité
    alpha=0.8,
    edgecolor='k',
    linewidth=1
)

# Ajout des annotations (noms des médicaments)
for i, row in drug_umap_df.iterrows():
    plt.text(row['umap_x'] + 0.05, row['umap_y'], row['drug_name'], fontsize=12, weight='bold')

plt.title("Espace d'Embedding des Médicaments (visualisé avec UMAP)", fontsize=20, pad=20)
plt.xlabel("UMAP Dimension 1", fontsize=16)
plt.ylabel("UMAP Dimension 2", fontsize=16)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "drug_embedding_space_umap.png"), dpi=300)
plt.close()
print("Graphique UMAP de l'espace d'embedding des médicaments sauvegardé.")


# --- SECTION 7.0: IN-SILICO SATURATION MUTAGENESIS ---
print("\n--- 7.0. Analyse par Mutagénèse Saturationnelle In Silico ---")

def perform_saturation_mutagenesis(sequence, drug_id, model, tokenizer, device):
    """
    Effectue une mutagénèse saturationnelle sur une séquence donnée pour une drogue spécifique.
    Calcule l'impact de chaque mutation possible en dehors du codon stop.
    """
    nucleotides = ['A', 'C', 'G', 'T']
    mutagenesis_results = []

    # 1. Obtenir la prédiction pour la séquence de référence (wild-type)
    wt_pred = predict_batch([sequence], [drug_id], tokenizer, model, device)[0]
    
    # Gérer le cas où la prédiction de base est nulle pour éviter la division par zéro
    if wt_pred == 0:
        wt_pred = 1e-9 # Petite valeur pour éviter la division par zéro

    # 2. Déterminer les positions du codon stop à ignorer
    # Pour une séquence de type 'NNN...STOP...NNN', le stop est au centre.
    n_context = (len(sequence) - 3) // 2
    stop_start_index = n_context
    
    # 3. Itérer sur chaque position et chaque mutation possible
    for position in tqdm(range(len(sequence)), desc=f"Mutating sequence for drug_id {drug_id}"):
        # Ignorer les positions du codon stop
        if stop_start_index <= position < stop_start_index + 3:
            continue
            
        original_nucleotide = sequence[position]
        
        for mutated_nucleotide in nucleotides:
            # Pas besoin de tester la "mutation" vers le même nucléotide
            if original_nucleotide == mutated_nucleotide:
                log2_fold_change = 0.0
            else:
                # Créer la séquence mutée
                mutated_sequence = list(sequence)
                mutated_sequence[position] = mutated_nucleotide
                mutated_sequence = "".join(mutated_sequence)
                
                # Obtenir la prédiction pour la séquence mutée
                mutant_pred = predict_batch([mutated_sequence], [drug_id], tokenizer, model, device)[0]
                
                # Calculer le log2 fold change
                log2_fold_change = np.log2(mutant_pred / wt_pred)

            mutagenesis_results.append({
                'position': position - n_context, # Centrer la position 0 sur le codon stop
                'original_nucleotide': original_nucleotide,
                'mutated_nucleotide': mutated_nucleotide,
                'log2_fold_change': log2_fold_change
            })
            
    return pd.DataFrame(mutagenesis_results)

def plot_mutagenesis_heatmap(df, title, filename, reference_sequence):
    """
    Génère et sauvegarde une heatmap à partir des résultats de la mutagénèse.

    L'échelle de couleur représente le log2 fold change de l'efficacité du readthrough (RT) :
    - 0 (blanc) : Aucun impact.
    - > 0 (rouge) : Augmentation de l'efficacité (ex: +1 = 2x plus efficace).
    - < 0 (bleu) : Diminution de l'efficacité (ex: -1 = 2x moins efficace).
    """
    heatmap_data = df.pivot_table(
        index='mutated_nucleotide',
        columns='position',
        values='log2_fold_change'
    )
    
    # --- CORRECTION DU BUG DE TRI ---
    # 1. Convertir les colonnes (positions) en entiers pour un tri numérique.
    numeric_columns = sorted([int(c) for c in heatmap_data.columns])
    
    # 2. Réindexer le DataFrame pour forcer l'ordre numérique correct.
    heatmap_data = heatmap_data.reindex(columns=numeric_columns)
    # --- FIN DE LA CORRECTION ---

    # S'assurer de l'ordre canonique des nucléotides sur l'axe Y
    heatmap_data = heatmap_data.reindex(['A', 'C', 'G', 'T'])
    
    plt.figure(figsize=(20, 6))
    heatmap = sns.heatmap(
        heatmap_data,
        cmap='coolwarm', # Palette divergente: bleu (négatif), blanc (neutre), rouge (positif)
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=.5
    )
    heatmap.collections[0].colorbar.set_label("log2 Fold Change", rotation=270, labelpad=20)
    full_title = f"{title}\nSéquence de référence: {reference_sequence}"
    plt.title(full_title, fontsize=16, pad=20)
    plt.xlabel("Position (relative au début du codon stop)", fontsize=12)
    plt.ylabel("Mutation vers", fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Heatmap de mutagénèse sauvegardée dans '{filename}'.")

# --- Logique principale de l'analyse ---
# Définir les drogues et les types de codons stop à analyser
drugs_to_analyze = ['Gentamicin', 'G418', 'DAP']
stop_types_to_analyze = ['uga', 'uag', 'uaa']

for drug_name in drugs_to_analyze:
    if drug_name not in drug_to_id:
        print(f"Médicament '{drug_name}' non trouvé, ignoré.")
        continue
        
    drug_id = drug_to_id[drug_name]
    drug_df = test_df[test_df['drug'] == drug_name]
    
    for stop_type in stop_types_to_analyze:
        # 1. Sélectionner la séquence de référence la plus performante pour ce combo
        reference_df = drug_df[drug_df['stop_type'] == stop_type]
        if reference_df.empty:
            print(f"Aucune séquence trouvée pour {drug_name} avec codon stop {stop_type}. Ignoré.")
            continue
        
        # Trier par prédiction pour trouver la meilleure séquence
        reference_sequence = reference_df.loc[reference_df['preds'].idxmax()][context_col]
        
        print(f"\nAnalyse de mutagénèse pour {drug_name} sur le codon {stop_type}...")
        print(f"Séquence de référence : {reference_sequence}")
        
        # 2. Effectuer la mutagénèse
        mutagenesis_df = perform_saturation_mutagenesis(reference_sequence, drug_id, model, tokenizer, DEVICE)
        
        # 3. Générer la heatmap
        plot_title = f"Impact Mutationnel autour du codon {stop_type} pour {drug_name}"
        output_filename = os.path.join(RESULTS_DIR, f"saturation_mutagenesis_heatmap_{drug_name}_{stop_type}.png")
        plot_mutagenesis_heatmap(mutagenesis_df, plot_title, output_filename, reference_sequence)

print("\n--- Analyse de mutagénèse terminée ---")

# --- SECTION 8.0: ANALYSE D'ÉPISTASIE PAR DOUBLE MUTAGENÈSE ---
print("\n--- 8.0. Analyse d'Épistasie par Double Mutagenèse In Silico ---")

def calculate_epistasis(sequence, drug_id, model, tokenizer, device):
    """
    Calcule les scores d'épistasie pour les paires de mutations dans une séquence donnée.
    """
    nucleotides = ['A', 'C', 'G', 'T']
    
    # 1. Calculer la prédiction de base (WT)
    wt_pred = predict_batch([sequence], [drug_id], tokenizer, model, device)[0]
    if wt_pred == 0: wt_pred = 1e-9
    log_wt_pred = np.log2(wt_pred)

    # 2. Calculer les effets de toutes les mutations simples
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

    # 3. Calculer les effets des doubles mutations et l'épistasie
    epistasis_results = []
    
    # Créer des paires de positions uniques
    position_pairs = list(combinations(context_indices, 2))

    for pos1, pos2 in tqdm(position_pairs, desc="Calculating double mutations"):
        original_nuc1 = sequence[pos1]
        original_nuc2 = sequence[pos2]

        for new_nuc1 in nucleotides:
            if original_nuc1 == new_nuc1: continue
            for new_nuc2 in nucleotides:
                if original_nuc2 == new_nuc2: continue

                # Créer la séquence doublement mutée
                double_mut_seq = list(sequence)
                double_mut_seq[pos1] = new_nuc1
                double_mut_seq[pos2] = new_nuc2
                
                double_mut_pred = predict_batch(["".join(double_mut_seq)], [drug_id], tokenizer, model, device)[0]
                if double_mut_pred == 0: double_mut_pred = 1e-9
                
                # Effet observé du double mutant
                observed_effect = np.log2(double_mut_pred) - log_wt_pred
                
                # Effet attendu (additif)
                effect1 = single_mutant_effects.get((pos1, new_nuc1), 0)
                effect2 = single_mutant_effects.get((pos2, new_nuc2), 0)
                expected_effect = effect1 + effect2
                
                # Score d'épistasie
                epistasis_score = observed_effect - expected_effect
                
                epistasis_results.append({
                    'mutation1': f"{pos1-n_context}:{original_nuc1}>{new_nuc1}",
                    'mutation2': f"{pos2-n_context}:{original_nuc2}>{new_nuc2}",
                    'epistasis_score': epistasis_score
                })

    return pd.DataFrame(epistasis_results)

def plot_epistasis_heatmap(df, title, filename, reference_sequence):
    """
    Génère une heatmap des scores d'épistasie, en s'assurant que les axes sont
    triés numériquement par position de mutation.
    """
    if df.empty:
        print("Le DataFrame d'épistasie est vide. Impossible de générer la heatmap.")
        return
        
    # --- CORRECTION DU BUG DE TRI ---
    # 1. Fonction utilitaire pour extraire la position numérique du label.
    def get_pos_from_label(label):
        try:
            # Extrait la partie avant le ':' et la convertit en entier.
            return int(label.split(':')[0])
        except (ValueError, IndexError):
            # Retourne une grande valeur pour les labels mal formés pour les trier à la fin.
            return float('inf')

    # 2. Obtenir tous les labels de mutation uniques et les trier numériquement.
    all_labels = pd.unique(df[['mutation1', 'mutation2']].values.ravel('K'))
    # Filtrer les labels potentiellement nuls ou mal formés pour éviter les erreurs.
    all_labels = [label for label in all_labels if isinstance(label, str) and ':' in label]
    sorted_labels = sorted(all_labels, key=get_pos_from_label)
    
    # 3. Créer la table pivot et la réindexer avec les labels triés pour forcer le bon ordre.
    epistasis_matrix = df.pivot_table(index='mutation1', columns='mutation2', values='epistasis_score')
    epistasis_matrix = epistasis_matrix.reindex(index=sorted_labels, columns=sorted_labels)
    # --- FIN DE LA CORRECTION ---

    # Rendre la matrice symétrique pour une meilleure visualisation.
    # combine_first remplit les NaN d'une matrice avec les valeurs de l'autre.
    epistasis_matrix = epistasis_matrix.combine_first(epistasis_matrix.T)
    
    # Remplir la diagonale avec 0 car une mutation n'interagit pas avec elle-même dans ce contexte.
    np.fill_diagonal(epistasis_matrix.values, 0)

    plt.figure(figsize=(20, 18))
    heatmap = sns.heatmap(
        epistasis_matrix,
        cmap='coolwarm',
        center=0,
        annot=False, # L'annotation surchargerait complètement le graphique.
        square=True, # Assurer que les cellules sont carrées pour une meilleure lisibilité.
        linewidths=.1
    )
    heatmap.collections[0].colorbar.set_label("Epistasis Score", rotation=270, labelpad=20)
    full_title = f"{title}\nSéquence de référence: {reference_sequence}"
    plt.title(full_title, fontsize=20, pad=20)
    plt.xlabel("Mutation", fontsize=16)
    plt.ylabel("Mutation", fontsize=16)
    
    # Améliorer la lisibilité des labels des axes.
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Heatmap d'épistasie sauvegardée dans '{filename}'.")

# --- Logique principale de l'analyse d'épistasie (généralisée) ---
print("\nLancement de l'analyse d'épistasie généralisée...")

# Définir les types de codons stop à analyser
stop_types_to_analyze = ['uaa', 'uag', 'uga']

# Boucler sur chaque médicament et chaque type de codon stop
for drug_name in drug_to_id.keys():
    drug_id = drug_to_id[drug_name]
    drug_df = test_df[test_df['drug'] == drug_name]
    
    for stop_type in stop_types_to_analyze:
        # Sélectionner la séquence de référence la plus performante pour ce combo
        reference_df = drug_df[drug_df['stop_type'] == stop_type]
        
        if reference_df.empty:
            print(f"Aucune séquence trouvée pour {drug_name} avec codon stop {stop_type}. Analyse d'épistasie ignorée.")
            continue
        
        # Trier par prédiction pour trouver la meilleure séquence
        reference_sequence = reference_df.loc[reference_df['preds'].idxmax()][context_col]
        
        print(f"\nAnalyse d'épistasie pour {drug_name} sur le codon {stop_type}...")
        print(f"Séquence de référence : {reference_sequence}")
        
        # Effectuer l'analyse d'épistasie
        epistasis_df = calculate_epistasis(reference_sequence, drug_id, model, tokenizer, DEVICE)
        
        # Générer la heatmap
        if not epistasis_df.empty:
            plot_title = f"Analyse d'Épistasie pour {drug_name} (Stop: {stop_type})"
            output_filename = os.path.join(RESULTS_DIR, f"epistasis_heatmap_{drug_name}_{stop_type}.png")
            plot_epistasis_heatmap(epistasis_df, plot_title, output_filename, reference_sequence)
        else:
            print(f"Le DataFrame d'épistasie est vide pour {drug_name} / {stop_type}. Aucune heatmap générée.")

print("\n--- Analyse d'épistasie généralisée terminée ---")
