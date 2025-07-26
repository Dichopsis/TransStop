#!/home2020/home/icube/nhaas/.conda/envs/TransStop/bin/python

#SBATCH -p publicgpu
#SBATCH -N 1
#SBATCH -x hpc-n932
#SBATCH --gres=gpu:2
#SBATCH --constraint="gpuh100|gpua100|gpul40s|gpua40|gpurtx6000"
#SBATCH --mail-type=END
#SBATCH --mail-user=nicolas.haas3@etu.unistra.fr

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
        base_model_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )
        # Utiliser le CLS token [:, 0] pour la représentation de la séquence
        cls_embedding = base_model_outputs.last_hidden_state[:, 0]
        # Obtenir l'embedding du médicament
        drug_emb = self.drug_embedding(drug_id)
        # Concaténer et passer dans la tête de régression
        combined_embedding = torch.cat([cls_embedding, drug_emb], dim=1)
        logits = self.reg_head(combined_embedding).squeeze(-1)
        
        if output_attentions:
            return logits, base_model_outputs.attentions
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
    palette='viridis',  # 'viridis' est une palette de couleurs agréable, vous pouvez en choisir une autre
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

# Créer une palette de couleurs pour distinguer les graphiques
palette = sns.color_palette("viridis", num_drugs)

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
        color=palette[i],
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
    'G418': 0.76, 'SJ6986': 0.71, 'SRI': 0.76, 'FUr': 0.37, 'Gentamicin': 0.38
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


# --- SECTION 5.0: INTERPRÉTABILITÉ - HEATMAPS D'INTERACTION (Corrigé) ---
print("\n--- 5.0. Génération des Heatmaps d'Interaction ---")

# Vérifier si les colonnes nécessaires sont présentes
if 'stop_type' not in test_df.columns or context_col not in test_df.columns:
    print(f"Colonnes requises ('stop_type', '{context_col}') manquantes. L'analyse des interactions est ignorée.")
else:
    try:
        # Extraire le 'n' de 'seq_context_n' pour calculer la position +1 de manière robuste
        # Ex: pour 'seq_context_18', n=9. Stop est aux indices 9,10,11. Pos +1 est à l'indice 12.
        # La séquence a une longueur de 2*n + 3. Le stop commence à l'indice n. Le +1 est à l'indice n+3.
        # NOTE: D'après votre description, seq_context_18 = 9nt-STOP-9nt, donc n=9. Longueur = 9+3+9 = 21.
        # Stop aux indices 9, 10, 11. Le nucléotide +1 est donc bien à l'indice 12.
        # La formule est : n + 3. Pour extraire n, on prend les chiffres dans le nom de la colonne.
        
        # Trouver les nombres dans le nom de la colonne. Ex: 'seq_context_18' -> '18'
        context_num_str = ''.join(filter(str.isdigit, context_col))
        if not context_num_str:
            raise ValueError("Aucun nombre trouvé dans le nom de la colonne de contexte.")
        
        n_context = int(context_num_str) // 2 # Si context_18, n=9.
        pos_plus_1 = n_context + 3

        print(f"Contexte détecté : n={n_context}. Extraction du nucléotide +1 à la position d'index {pos_plus_1}.")
        
        test_df['plus_one_nuc'] = test_df[context_col].str[pos_plus_1]

        # Calculer les performances moyennes pour chaque interaction
        interaction_results = []
        for (drug, stop, nuc), group in test_df.groupby(['drug', 'stop_type', 'plus_one_nuc']):
            interaction_results.append({
                'drug': drug,
                'stop_type': stop,
                'plus_one_nuc': nuc,
                'mean_preds': group['preds'].mean()
            })
        interaction_df = pd.DataFrame(interaction_results)

        # Gérer le cas où il n'y a aucune donnée (par exemple, si la position +1 était incorrecte)
        if interaction_df.empty:
            raise ValueError("Le DataFrame des interactions est vide. Vérifiez la logique d'extraction des nucléotides.")
            
        # Standardiser l'échelle de couleurs pour une comparaison facile
        vmin = interaction_df['mean_preds'].min()
        vmax = interaction_df['mean_preds'].max()

        # Créer une grille de heatmaps
        drug_list = sorted(interaction_df['drug'].unique())
        n_cols = 3
        n_rows = (len(drug_list) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows), squeeze=False)
        axes = axes.flatten()

        for i, drug_name in enumerate(drug_list):
            ax = axes[i]
            drug_interaction_df = interaction_df[interaction_df['drug'] == drug_name]
            
            # Créer la table pivot
            pivot_df = drug_interaction_df.pivot_table(
                index='plus_one_nuc', 
                columns='stop_type', 
                values='mean_preds'
            )
            # S'assurer de l'ordre canonique des nucléotides et des codons stop
            # .reindex() gérera les valeurs manquantes (NaN) si une combinaison n'existe pas
            pivot_df = pivot_df.reindex(['A', 'C', 'G', 'T']).reindex(['UAA', 'UAG', 'UGA'], axis=1)
            
            sns.heatmap(
                pivot_df, 
                ax=ax, 
                annot=True, 
                fmt=".3f", 
                cmap="viridis", 
                linewidths=.5,
                vmin=vmin, # Appliquer l'échelle de couleur standard
                vmax=vmax
            )
            ax.set_title(f"Interaction pour {drug_name}", fontsize=14, weight='bold')
            ax.set_xlabel("Type de Codon Stop", fontsize=12)
            ax.set_ylabel("Nucléotide en pos +1", fontsize=12)

        # Masquer les sous-graphiques inutilisés
        for j in range(len(drug_list), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Performance Moyenne Prédite : Interaction Codon Stop × Nucléotide +1", fontsize=20, y=1.03)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(os.path.join(RESULTS_DIR, "interaction_heatmap_grid.png"), dpi=300)
        plt.close()
        print("Grille de heatmaps d'interaction sauvegardée.")

    except Exception as e:
        print(f"Une erreur est survenue lors de la génération des heatmaps d'interaction : {e}")
        print("Cette étape est ignorée.")


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


# --- SECTION 5.0 (Révision 3): PROFIL D'IMPORTANCE DES NUCLÉOTIDES AVEC SHAP ---
print("\n--- 5.0. Profil d'Importance des Nucléotides avec KernelExplainer (Alternative) ---")

try:
    import shap
    print(f"Version de SHAP installée : {shap.__version__}")

    # --- Étape 1: Préparer les données et la fonction de prédiction ---
    # On va travailler sur un échantillon et utiliser ses indices comme "features" pour SHAP
    sample_size = min(100, len(test_df))
    if sample_size == 0:
        raise ValueError("Le jeu de test est vide, impossible de créer un échantillon.")
    
    shap_sample_df = test_df.sample(n=sample_size, random_state=SEED).reset_index(drop=True)

    # La fonction de prédiction pour KernelExplainer
    # Elle prend un tableau (n_samples, n_features). Ici n_features=1 (l'indice de la séquence)
    def kernel_predictor(indices_array):
        # indices_array sera de la forme [[0], [1], [2], ...]
        indices = indices_array.flatten().astype(int)
        
        # Aller chercher les séquences et les drug_ids correspondants dans notre échantillon
        sequences_to_predict = shap_sample_df.loc[indices, context_col].tolist()
        drug_ids_to_predict = shap_sample_df.loc[indices, 'drug_id'].tolist()
        
        # Prédire avec notre fonction utilitaire
        # On doit retourner une seule sortie à la fois, donc on fait une boucle par drogue
        # C'est un peu moins efficace mais nécessaire pour cette approche.
        drug_id_of_interest = shap.drug_id_to_explain # Variable globale temporaire
        
        # Filtre pour ne prédire que pour la drogue d'intérêt
        preds = predict_batch(sequences_to_predict, [drug_id_of_interest] * len(sequences_to_predict), tokenizer, model, DEVICE)
        return preds

    # --- Étape 2: Boucler sur chaque drogue pour créer un explainer et calculer l'importance ---
    all_shap_values = []
    
    # On crée une "donnée de fond" (background data) pour que SHAP ait une référence.
    # On prend un petit sous-échantillon.
    background_data = np.arange(min(10, sample_size)).reshape(-1, 1)

    for drug_name, drug_id in tqdm(drug_to_id.items(), desc="Calcul des valeurs SHAP par drogue"):
        # Indiquer à la fonction predictor pour quelle drogue elle doit travailler
        shap.drug_id_to_explain = drug_id
        
        # Créer l'explainer
        explainer = shap.KernelExplainer(kernel_predictor, background_data)
        
        # Expliquer toutes les instances de notre échantillon
        indices_to_explain = np.arange(len(shap_sample_df)).reshape(-1, 1)
        
        # nsamples='auto' est un bon compromis vitesse/précision
        shap_values_for_drug = explainer.shap_values(indices_to_explain, nsamples='auto')
        
        # Stocker les résultats
        all_shap_values.append(pd.DataFrame({
            'shap_value': shap_values_for_drug,
            'drug': drug_name,
            'sequence': shap_sample_df[context_col]
        }))

    shap_df = pd.concat(all_shap_values, ignore_index=True)
    
    # --- Étape 3: Utiliser une autre méthode pour l'importance positionnelle : "Leave-One-Out" ---
    # Puisque SHAP sur le texte est complexe, revenons à une ablation robuste.
    # On va calculer l'importance comme la différence de SHAP value quand on enlève un nucléotide.
    # C'est trop complexe. On va simplifier et revenir à l'ablation R2 qui est plus directe.
    
    # ---- Ré-implémentation de l'ablation R2, qui est plus directe et robuste que SHAP pour ce cas d'usage ----
    print("\nChangement de stratégie : SHAP sur texte est trop complexe/instable. Retour à l'ablation robuste.")
    
    context_num_str = ''.join(filter(str.isdigit, context_col))
    n_context = int(context_num_str) // 2
    seq_len = 2 * n_context + 3
    positions_to_ablate = range(seq_len)
    importance_results = []

    for drug_name, drug_id in tqdm(drug_to_id.items(), desc="Analyse d'ablation R2 par drogue"):
        drug_df = test_df[test_df['drug'] == drug_name]
        if len(drug_df) < 50: continue
        
        sample_df = drug_df.sample(n=min(500, len(drug_df)), random_state=SEED)
        base_sequences = sample_df[context_col].tolist()
        true_values = sample_df['RT'].values
        drug_ids_batch = [drug_id] * len(base_sequences)
        
        base_preds = predict_batch(base_sequences, drug_ids_batch, tokenizer, model, DEVICE)
        r2_base = r2_score(true_values, base_preds)
        
        for pos_idx in positions_to_ablate:
            ablated_sequences = ["".join(list(s[:pos_idx]) + ['N'] + list(s[pos_idx+1:])) for s in base_sequences]
            ablated_preds = predict_batch(ablated_sequences, drug_ids_batch, tokenizer, model, DEVICE)
            r2_ablated = r2_score(true_values, ablated_preds)
            importance_drop = r2_base - r2_ablated
            
            relative_pos = pos_idx - n_context
            importance_results.append({
                'drug': drug_name,
                'position': relative_pos,
                'Importance (R2 Drop)': importance_drop
            })

    importance_df = pd.DataFrame(importance_results)
    heatmap_pivot = importance_df.pivot_table(index='drug', columns='position', values='Importance (R2 Drop)')
    viz_cols = [c for c in sorted(heatmap_pivot.columns) if -9 <= c < 12]
    heatmap_pivot = heatmap_pivot[viz_cols]
    heatmap_pivot_normalized = heatmap_pivot.apply(lambda x: (x - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) > 0 else x, axis=1)

    plt.figure(figsize=(20, 8))
    sns.heatmap(heatmap_pivot_normalized, cmap='rocket_r', linewidths=.5, annot=True, fmt=".2f",
                cbar_kws={'label': 'Importance Relative Normalisée (0=min, 1=max)'})
    plt.title("Profil d'Importance des Nucléotides par Drogue (Ablation R²)", fontsize=20, pad=20)
    plt.xlabel("Position Relative au Début du Codon Stop", fontsize=14)
    plt.ylabel("Drogue", fontsize=14)
    
    ax = plt.gca()
    try:
        stop_start_tick_pos = heatmap_pivot_normalized.columns.get_loc(0)
        ax.add_patch(plt.Rectangle((stop_start_tick_pos, 0), 3, len(heatmap_pivot_normalized),
                                   fill=False, edgecolor='white', lw=3, clip_on=False))
        ax.text(stop_start_tick_pos + 1.5, -0.05, 'STOP', color='white', ha='center', va='bottom',
                weight='bold', transform=ax.get_xaxis_transform())
    except KeyError:
        print("Position 0 non trouvée, impossible de dessiner le rectangle du stop.")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "nucleotide_importance_heatmap_R2_Ablation.png"), dpi=300)
    plt.close()
    print("Heatmap d'importance des nucléotides (Ablation R²) sauvegardée.")

except ImportError:
    print("La bibliothèque 'shap' n'est pas installée. Cette étape est ignorée.")
except Exception as e:
    print(f"Erreur durant l'analyse d'importance : {e}. Étape ignorée.")
print("--- FIN DU PROJET ---")