# --- SCRIPT: final_analysis.py ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

# --- Configuration ---
RESULTS_DIR = "./results/"
FINAL_PREDS_PATH = os.path.join(RESULTS_DIR, "our_genome_wide_predictions_full.parquet")

# Définir les noms de drogues (sans "Untreated" pour l'analyse de traitabilité)
OUR_PREDS_COLS = [
    'our_preds_CC90009', 'our_preds_Clitocine', 'our_preds_DAP', 'our_preds_FUr',
    'our_preds_G418', 'our_preds_Gentamicin', 'our_preds_SJ6986', 'our_preds_SRI'
]
TOLEDANO_PREDS_COLS = [
    'predictions_CC90009', 'predictions_Clitocine', 'predictions_dap', 'predictions_fur',
    'predictions_G418', 'predictions_Gentamicin', 'predictions_SJ6986', 'predictions_sri'
]
# Mapper les noms de colonnes pour la comparaison
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


# --- Chargement des données ---
print(f"Chargement du fichier de prédictions genome-wide depuis : {FINAL_PREDS_PATH}")
try:
    df = pd.read_parquet(FINAL_PREDS_PATH)
except FileNotFoundError:
    print("Fichier de prédictions introuvable. Veuillez d'abord lancer le script d'inférence.")
    exit()
print("Chargement terminé. Taille du DataFrame :", df.shape)


# --- ANALYSE 1: PAYSAGE DE TRAITABILITÉ ---
print("\n--- Début de l'Analyse 1: Paysage de Traitabilité ---")

# 1. Calculer les métriques pour chaque PTC
print("Calcul des métriques de traitabilité pour chaque PTC...")
df['our_max_pred'] = df[OUR_PREDS_COLS].max(axis=1)
# Utiliser l'écart-type comme mesure de spécificité
df['our_specificity'] = df[OUR_PREDS_COLS].std(axis=1)

# 2. Agréger par gène
print("Agrégation des métriques par gène...")
# On filtre les gènes avec un nombre minimum de PTCs pour la robustesse
gene_counts = df['gene'].value_counts()
genes_to_keep = gene_counts[gene_counts >= 20].index
df_filtered_genes = df[df['gene'].isin(genes_to_keep)]

gene_treatability = df_filtered_genes.groupby('gene').agg(
    mean_max_rt=('our_max_pred', 'mean'),
    mean_specificity=('our_specificity', 'mean'),
    ptc_count=('gene', 'size')
).reset_index()

# 3. Créer la visualisation du paysage
print("Génération du graphique 'Paysage de Traitabilité'...")
plt.figure(figsize=(16, 12))
scatter = sns.scatterplot(
    data=gene_treatability,
    x='mean_max_rt',
    y='mean_specificity',
    size='ptc_count',
    hue='ptc_count',
    palette='viridis',
    sizes=(50, 2000),
    alpha=0.7,
    edgecolor='k',
    linewidth=0.5
)

plt.title('Paysage de Traitabilité des Gènes Humains', fontsize=22, pad=20)
plt.xlabel('Traitabilité Moyenne (Meilleur RT Prédit)', fontsize=16)
plt.ylabel('Spécificité Requise (Écart-type des RT Prédits)', fontsize=16)
plt.grid(True, linestyle='--')

# Ajouter des annotations pour les quadrants
xlim = plt.xlim()
ylim = plt.ylim()
x_mid = sum(xlim) / 2
y_mid = sum(ylim) / 2
plt.text(xlim[0] + 0.05*abs(xlim[0]), ylim[0] + 0.05*abs(ylim[0]), 'Déserts Thérapeutiques\n(Faible RT, Faible Spécificité)', fontsize=14, color='red', alpha=0.8, ha='left')
plt.text(xlim[1] - 0.05*abs(xlim[1]), ylim[1] - 0.05*abs(ylim[1]), 'Oasis Spécifiques\n(Haut RT, Haute Spécificité)', fontsize=14, color='green', alpha=0.8, ha='right')
plt.text(xlim[1] - 0.05*abs(xlim[1]), ylim[0] + 0.05*abs(ylim[0]), 'Terrains Faciles\n(Haut RT, Faible Spécificité)', fontsize=14, color='blue', alpha=0.8, ha='right')

# Annoter quelques gènes importants
genes_to_annotate = ['TP53', 'APC', 'BRCA1', 'BRCA2', 'DMD', 'CFTR', 'PTEN']
for gene in genes_to_annotate:
    if gene in gene_treatability['gene'].values:
        gene_data = gene_treatability[gene_treatability['gene'] == gene]
        plt.text(gene_data['mean_max_rt'].values[0], gene_data['mean_specificity'].values[0], gene, 
                 fontdict={'weight': 'bold', 'size': 12, 'color': 'black'})

plt.legend(title='Nombre de PTCs', loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "treatability_landscape.png"), dpi=300)
plt.close()
print("Graphique du paysage de traitabilité sauvegardé.")


# --- ANALYSE 2: COMPARAISON DES PRÉDICTIONS DE LA MEILLEURE DROGUE ---
print("\n--- Début de l'Analyse 2: Comparaison des Modèles ---")

# 1. Identifier la meilleure drogue pour chaque PTC selon chaque modèle
print("Identification de la meilleure drogue pour chaque PTC...")
# Pour notre modèle
our_preds_df = df[OUR_PREDS_COLS].rename(columns=OUR_MAP)
df['our_best_drug'] = our_preds_df.idxmax(axis=1)

# Pour le modèle de Toledano
toledano_preds_df = df[TOLEDANO_PREDS_COLS].rename(columns=TOLEDANO_MAP)
df['toledano_best_drug'] = toledano_preds_df.idxmax(axis=1)

# 2. Calculer le taux de concordance
agreement = (df['our_best_drug'] == df['toledano_best_drug'])
agreement_rate = agreement.mean()
print(f"Taux de concordance global sur la meilleure drogue : {agreement_rate:.2%}")

# 3. Analyser les discordances : créer une matrice de confusion
print("Génération de la matrice de confusion des discordances...")
confusion_matrix = pd.crosstab(df['toledano_best_drug'], df['our_best_drug'], normalize='index')

plt.figure(figsize=(12, 10))
sns.heatmap(
    confusion_matrix, 
    annot=True, 
    fmt='.2f', 
    cmap='Blues',
    linewidths=.5
)
plt.title('Concordance de la Meilleure Drogue Prédite', fontsize=20, pad=20)
plt.xlabel('Meilleure Drogue (Notre Modèle)', fontsize=16)
plt.ylabel('Meilleure Drogue (Modèle Toledano et al.)', fontsize=16)
plt.text(0.5, 1.05, f'Concordance globale (diagonale) = {agreement_rate:.2%}', 
         ha='center', va='bottom', transform=plt.gca().transAxes, fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "best_drug_confusion_matrix.png"), dpi=300)
plt.close()
print("Matrice de confusion sauvegardée.")


# 4. Analyse plus fine et améliorée : Quand notre modèle change-t-il d'avis de manière significative ?
print("\n--- Analyse approfondie des cas de discordance significative ---")

# Calculer la meilleure prédiction pour chaque modèle
df['our_best_pred_val'] = df[OUR_PREDS_COLS].max(axis=1)

# Trouver la prédiction de notre modèle pour la drogue que Toledano a choisie
# Il faut construire dynamiquement le nom de la colonne de notre prédiction
# correspondant à la drogue choisie par Toledano
# Par exemple, si toledano_best_drug est 'DAP', on veut la valeur de 'our_preds_DAP'
print("Calcul de la prédiction de notre modèle pour le choix de Toledano...")

# 1. Créer un DataFrame contenant uniquement nos prédictions, avec des noms de colonnes simples
our_preds_df_simple = df[OUR_PREDS_COLS].rename(columns=OUR_MAP)

# 2. Obtenir les valeurs NumPy sous-jacentes pour une performance maximale
our_preds_values = our_preds_df_simple.values

# 3. Obtenir la liste ordonnée des colonnes (drogues)
column_names = our_preds_df_simple.columns.tolist()

# 4. Créer un mappage de nom de drogue vers son indice de colonne
col_indexer = {name: i for i, name in enumerate(column_names)}

# 5. Créer un tableau d'indices de colonnes correspondant au choix de Toledano pour chaque ligne
# df['toledano_best_drug'].map(col_indexer) va créer une série où chaque valeur est l'index de la colonne
# de la drogue choisie par Toledano.
col_indices = df['toledano_best_drug'].map(col_indexer).values

# 6. Utiliser l'indexation avancée de NumPy pour extraire toutes les valeurs en une seule fois
# C'est l'équivalent de faire `our_preds_values[0, col_indices[0]]`, `our_preds_values[1, col_indices[1]]`, etc.
# pour toutes les lignes. C'est extrêmement rapide.
num_rows = len(df)
row_indices = np.arange(num_rows)
df['our_pred_for_toledano_choice'] = our_preds_values[row_indices, col_indices]

print("Calcul terminé.")

# Calculer le "gain" de notre modèle quand il change d'avis
df['our_gain'] = df['our_best_pred_val'] - df['our_pred_for_toledano_choice']

# Filtrer les cas de discordance
disagreement_df = df[df['our_best_drug'] != df['toledano_best_drug']].copy()


# --- Amélioration 3.1: Générer deux graphiques (linéaire et logarithmique) ---

# Graphique 1: Échelle Linéaire (comme avant)
print("Génération du graphique de distribution du gain (échelle linéaire)...")
plt.figure(figsize=(12, 7))
sns.histplot(disagreement_df['our_gain'], bins=50, kde=True)
plt.title('Distribution du Gain de Performance Prédit (Cas de Désaccord)', fontsize=16)
plt.xlabel('Gain de RT (Notre Meilleure Drogue vs. Choix de Toledano)', fontsize=14)
plt.ylabel('Nombre de PTCs', fontsize=14)
plt.axvline(x=0, color='red', linestyle='--')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "disagreement_gain_distribution_linear.png"), dpi=300)
plt.close()

# Graphique 2: Échelle Logarithmique
print("Génération du graphique de distribution du gain (échelle logarithmique)...")
plt.figure(figsize=(12, 7))
sns.histplot(disagreement_df['our_gain'], bins=50, kde=False) # kde=False est souvent mieux avec l'échelle log
plt.yscale('log')
plt.title('Distribution du Gain de Performance Prédit (Échelle Log)', fontsize=16)
plt.xlabel('Gain de RT (Notre Meilleure Drogue vs. Choix de Toledano)', fontsize=14)
plt.ylabel('Nombre de PTCs (Échelle Log)', fontsize=14)
plt.axvline(x=0, color='red', linestyle='--')
plt.grid(True, which='both', linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "disagreement_gain_distribution_log.png"), dpi=300)
plt.close()
print("Graphiques de distribution du gain sauvegardés.")


# --- Amélioration 3.2: Quantifier la Distribution ---
print("\n--- Statistiques sur les Gains de Performance ---")

total_disagreements = len(disagreement_df)
positive_gain_cases = (disagreement_df['our_gain'] > 0).sum()
positive_gain_percentage = (positive_gain_cases / total_disagreements) * 100

gain_threshold_0_5 = (disagreement_df['our_gain'] > 0.5).sum()
gain_threshold_1_0 = (disagreement_df['our_gain'] > 1.0).sum()

print(f"Nombre total de PTCs où les modèles sont en désaccord : {total_disagreements:,}")
print(f"Pourcentage de cas de désaccord où notre modèle prédit un gain positif : {positive_gain_percentage:.2f}%")
print(f"Nombre de PTCs où le gain est supérieur à 0.5 RT : {gain_threshold_0_5:,}")
print(f"Nombre de PTCs où le gain est supérieur à 1.0 RT : {gain_threshold_1_0:,}")


# --- Amélioration 3.3 (Revisité): Analyser les Cas de Désaccord à Fort Impact ---
print("\n--- Analyse des Cas de Désaccord à Fort Impact (Gain de RT > 2.0) ---")

# 1. Définir le seuil et filtrer le DataFrame
gain_threshold = 2.0
high_gain_df = disagreement_df[disagreement_df['our_gain'] > gain_threshold].copy()
num_high_gain_cases = len(high_gain_df)

if num_high_gain_cases > 0:
    print(f"Nombre de PTCs avec un gain de performance prédit > {gain_threshold}: {num_high_gain_cases:,}")

    # Sauvegarder tous ces cas dans un fichier CSV pour une exploration plus approfondie
    high_gainers_path = os.path.join(RESULTS_DIR, f"high_impact_disagreements_gain_gt_{gain_threshold}.csv")
    columns_to_save = [
        'gene', 'stop_type', 'extracted_context', 'our_best_drug', 'our_best_pred_val', 
        'toledano_best_drug', 'our_pred_for_toledano_choice', 'our_gain'
    ]
    high_gain_df[columns_to_save].to_csv(high_gainers_path, index=False)
    print(f"Log des désaccords à fort impact sauvegardé dans : {high_gainers_path}")

    # --- Analyse 3.3a: Surreprésentation du type de stop ---
    print("\nAnalyse de la distribution des types de stop dans les cas à fort impact...")

    # --- Traitement pour le groupe "Cas à Fort Impact" ---
    # 1. Calculer les proportions
    stop_type_high_gain_s = high_gain_df['stop_type'].value_counts(normalize=True)
    # 2. Convertir la Series en DataFrame. L'index devient une colonne.
    stop_type_high_gain_df = stop_type_high_gain_s.reset_index()
    # 3. Renommer les colonnes de manière explicite et robuste, quels que soient leurs noms par défaut
    stop_type_high_gain_df.columns = ['Stop_Type', 'Proportion']
    # 4. Ajouter la colonne de groupe
    stop_type_high_gain_df['Groupe'] = 'Cas à Fort Impact (Gain > 2.0)'


    # --- Traitement pour le groupe "Baseline" ---
    # 1. Calculer les proportions
    stop_type_baseline_s = disagreement_df['stop_type'].value_counts(normalize=True)
    # 2. Convertir la Series en DataFrame
    stop_type_baseline_df = stop_type_baseline_s.reset_index()
    # 3. Renommer les colonnes de manière explicite
    stop_type_baseline_df.columns = ['Stop_Type', 'Proportion']
    # 4. Ajouter la colonne de groupe
    stop_type_baseline_df['Groupe'] = 'Tous les Cas de Désaccord (Baseline)'


    # --- Concaténation finale ---
    comparison_df_melted = pd.concat([stop_type_high_gain_df, stop_type_baseline_df], ignore_index=True)

    # Debugging: Afficher les 5 premières lignes du DataFrame final pour vérifier
    print("\n--- Aperçu du DataFrame final pour le graphique ---")
    print(comparison_df_melted.head())
    print("\nColonnes disponibles:", comparison_df_melted.columns)
    
    # La suite du code de visualisation reste la même et devrait fonctionner maintenant
    plt.figure(figsize=(10, 7))
    barplot = sns.barplot(data=comparison_df_melted, x='Stop_Type', y='Proportion', hue='Groupe', palette='pastel')
    plt.title('Distribution des Types de Stop : Cas à Fort Impact vs. Baseline', fontsize=16)
    plt.ylabel('Proportion des Cas', fontsize=14)
    plt.xlabel('Type de Codon Stop', fontsize=14)
    plt.grid(axis='y', linestyle='--')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y))) # Formatter l'axe Y en pourcentages
    for p in barplot.patches:
        barplot.annotate(format(p.get_height(), '.1%'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "high_gain_stop_type_analysis.png"), dpi=300)
    plt.close()
    print("Graphique d'analyse des types de stop sauvegardé.")


    # --- Analyse 3.3b: Paires de "changement d'avis" les plus fréquentes ---
    print("\nAnalyse des paires de drogues dans les changements d'avis...")
    
    # Créer une colonne combinant l'ancien et le nouveau choix
    high_gain_df['change_pair'] = high_gain_df['toledano_best_drug'] + ' -> ' + high_gain_df['our_best_drug']
    
    # Compter les paires les plus fréquentes
    change_pair_counts = high_gain_df['change_pair'].value_counts().nlargest(15) # On prend les 15 plus fréquentes

    plt.figure(figsize=(12, 10))
    barplot_pairs = sns.barplot(x=change_pair_counts.values, y=change_pair_counts.index, palette='viridis', hue=change_pair_counts.index, dodge=False, legend=False)
    plt.title('Top 15 des "Changements d\'Avis" à Fort Impact (Gain > 2.0)', fontsize=18)
    plt.xlabel('Nombre de PTCs', fontsize=14)
    plt.ylabel('Changement de Drogue (Toledano -> Notre Modèle)', fontsize=14)
    plt.grid(axis='x', linestyle='--')
    
    # Ajouter les comptes sur les barres
    for i, (count, pair) in enumerate(zip(change_pair_counts.values, change_pair_counts.index)):
        barplot_pairs.text(count, i, f' {count:,}', va='center', ha='left')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "high_gain_drug_switch_analysis.png"), dpi=300)
    plt.close()
    print("Graphique d'analyse des changements de drogues sauvegardé.")

else:
    print(f"Aucun cas de désaccord trouvé avec un gain de performance > {gain_threshold}.")

print("\n--- Toutes les analyses finales sont terminées. ---")