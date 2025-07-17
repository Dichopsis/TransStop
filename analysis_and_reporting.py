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


# --- SECTION 3.1: Analyse des Représentations Apprises des Médicaments ---
print("\n--- 3.1. Analyse des Embeddings de Médicaments avec Clustering Hiérarchique ---")

from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.manifold import TSNE # t-SNE est souvent meilleur pour visualiser les clusters

drug_embeddings = model.drug_embedding.weight.data.cpu().numpy()

# 1. Utiliser t-SNE pour une meilleure séparation des clusters
tsne = TSNE(n_components=2, perplexity=min(NUM_DRUGS - 2, 5), random_state=SEED, n_iter=1000)
projected_embeddings = tsne.fit_transform(drug_embeddings)

# 2. Effectuer un clustering hiérarchique sur les embeddings originaux
Z = linkage(drug_embeddings, method='ward')

# 3. Créer la visualisation combinée
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 3]) # Une colonne pour le dendrogramme, une pour le scatter plot

# Dendrogramme sur la gauche
ax_dendro = fig.add_subplot(gs[0, 0])
dendrogram(Z, labels=list(id_to_drug.values()), orientation='left', leaf_font_size=14, ax=ax_dendro)
ax_dendro.set_title("Similarité des Mécanismes Appris", fontsize=16)
ax_dendro.spines['top'].set_visible(False)
ax_dendro.spines['right'].set_visible(False)
ax_dendro.spines['bottom'].set_visible(False)
ax_dendro.spines['left'].set_visible(False)

# Scatter plot t-SNE sur la droite
ax_scatter = fig.add_subplot(gs[0, 1])
scatter = ax_scatter.scatter(projected_embeddings[:, 0], projected_embeddings[:, 1], 
                             s=500, c=np.arange(NUM_DRUGS), cmap='viridis', alpha=0.9)
for i, drug_name in id_to_drug.items():
    ax_scatter.text(projected_embeddings[i, 0] * 1.05, projected_embeddings[i, 1], drug_name, 
                    fontsize=16, fontweight='bold', ha='left', va='center')

ax_scatter.set_title("Projection t-SNE des Embeddings de Médicaments", fontsize=18)
ax_scatter.set_xlabel("Dimension t-SNE 1", fontsize=14)
ax_scatter.set_ylabel("Dimension t-SNE 2", fontsize=14)
ax_scatter.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "drug_embeddings_tsne_clustered.png"), dpi=300)
plt.close()
print("Graphique t-SNE et dendrogramme des embeddings sauvegardé.")

# --- NOUVELLE SECTION 3.2: ANALYSE DE LA SPÉCIFICITÉ DES MÉDICAMENTS (Volcano Plot) ---
print("\n--- 3.2. Analyse de la Spécificité des Médicaments ---")

# Comparons deux drogues intéressantes, par exemple DAP (UGA-spécifique) et Clitocine (UAA-spécifique)
drug1_name = 'DAP'
drug2_name = 'Clitocine'

# Assurez-vous d'avoir des prédictions pour ces deux drogues
if drug1_name in test_df['drug'].unique() and drug2_name in test_df['drug'].unique():
    # Pivoter le DataFrame pour avoir une colonne par drogue
    preds_pivot = test_df.pivot_table(index=context_col, columns='drug', values='preds')

    # Garder uniquement les séquences testées avec les deux drogues
    preds_pivot.dropna(subset=[drug1_name, drug2_name], inplace=True)
    
    # Calculer la différence de performance (log-ratio pour la symétrie)
    preds_pivot['log2_fold_change'] = np.log2(preds_pivot[drug1_name] / preds_pivot[drug2_name])
    
    # Calculer la performance moyenne (pour l'axe Y)
    preds_pivot['mean_performance'] = preds_pivot[[drug1_name, drug2_name]].mean(axis=1)

    # Créer le Volcano Plot
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        data=preds_pivot,
        x='log2_fold_change',
        y='mean_performance',
        alpha=0.5
    )
    
    # Mettre en évidence les séquences les plus spécifiques
    top_specific_d1 = preds_pivot.nlargest(5, 'log2_fold_change')
    top_specific_d2 = preds_pivot.nsmallest(5, 'log2_fold_change')
    
    for idx, row in top_specific_d1.iterrows():
        plt.text(row['log2_fold_change'], row['mean_performance'], idx[:10]+"...", color='red', fontsize=10)
    for idx, row in top_specific_d2.iterrows():
        plt.text(row['log2_fold_change'], row['mean_performance'], idx[:10]+"...", color='blue', fontsize=10)

    plt.axvline(0, color='grey', linestyle='--')
    plt.title(f'Analyse Différentielle : {drug1_name} vs. {drug2_name}', fontsize=18)
    plt.xlabel(f'Log2 ( Performance {drug1_name} / Performance {drug2_name} )', fontsize=14)
    plt.ylabel('Performance Moyenne Prédite', fontsize=14)
    plt.grid(True, linestyle='--')
    
    plt.text(0.1, plt.ylim()[1]*0.9, f'Séquences préférées par {drug1_name} ->', 
             ha='left', va='center', fontsize=12, color='red', weight='bold')
    plt.text(-0.1, plt.ylim()[1]*0.9, f'<- Séquences préférées par {drug2_name}', 
             ha='right', va='center', fontsize=12, color='blue', weight='bold')
    
    plt.savefig(os.path.join(RESULTS_DIR, "differential_volcano_plot.png"), dpi=300)
    plt.close()
    print("Volcano plot différentiel sauvegardé.")
else:
    print("Drogues pour l'analyse différentielle non trouvées. Étape ignorée.")

# --- SECTION 3.3: Visualisation de l'Importance Dynamique via l'Attention ---
# print("\n--- 3.3. Visualisation des Poids d'Attention ---")
# if len(test_df) > 20:
#     sample1 = test_df.iloc[10]
#     sample2_base = test_df.iloc[20] # Utiliser un autre échantillon pour plus de diversité
#     seq1 = sample1[context_col]
    
#     # Créer une mutation dans la séquence 2 pour la comparaison
#     seq2_list = list(sample2_base[context_col])
#     mutation_pos = len(seq2_list) // 2 - 2
#     original_nuc = seq2_list[mutation_pos]
#     new_nuc = 'G' if original_nuc != 'G' else 'A'
#     seq2_list[mutation_pos] = new_nuc
#     seq2 = "".join(seq2_list)
    
#     drug_id1 = sample1['drug_id']
#     drug_id2 = sample2_base['drug_id']

#     def get_attention_map(sequence_str, drug_id_int):
#         sequence = sequence_str.replace('U', 'T')
#         encoding = tokenizer(sequence, return_tensors='pt')
#         batch = {k: v.to(DEVICE) for k, v in encoding.items()}
#         batch['drug_id'] = torch.tensor([drug_id_int], dtype=torch.long).to(DEVICE)
#         with torch.no_grad():
#             _, attentions = model(**batch, output_attentions=True)
#         # Moyenne des têtes d'attention pour une couche spécifique (ex: couche 6)
#         attention_layer = attentions[6] # Choisir une couche pertinente
#         cls_attention = attention_layer[0, :, 0, :].mean(dim=0).cpu().numpy()
#         tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
#         return cls_attention, tokens

#     try:
#         att1, tokens1 = get_attention_map(seq1, drug_id1)
#         att2, tokens2 = get_attention_map(seq2, drug_id2)
#         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 6))
        
#         pred_rt1 = predict_rt(seq1, drug_id1)
#         pred_rt2 = predict_rt(seq2, drug_id2)
        
#         sns.heatmap([att1], xticklabels=tokens1, yticklabels=False, cmap="viridis", ax=ax1, cbar=False)
#         ax1.set_title(f"Attention Map for Sample 1 (Drug: {id_to_drug[drug_id1]}, Pred RT: {pred_rt1:.3f})")
        
#         sns.heatmap([att2], xticklabels=tokens2, yticklabels=False, cmap="viridis", ax=ax2, cbar=False)
#         ax2.set_title(f"Attention Map for Mutated Sample 2 (Drug: {id_to_drug[drug_id2]}, Pred RT: {pred_rt2:.3f})")
        
#         plt.tight_layout()
#         plt.savefig(os.path.join(RESULTS_DIR, "attention_map_comparison.png"), dpi=300)
#         plt.close()
#         print("Graphique de comparaison des cartes d'attention sauvegardé.")
#     except Exception as e:
#         print(f"Erreur lors de la génération des cartes d'attention : {e}. Cette étape est ignorée.")
# else:
#     print("Pas assez de données dans le jeu de test pour la visualisation de l'attention. Étape ignorée.")
    
    
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



print("\n--- Analyse terminée. Tous les artefacts sont dans le répertoire 'results/' ---")
print("--- FIN DU PROJET ---")