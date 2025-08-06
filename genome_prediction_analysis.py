#!/home2020/home/icube/nhaas/.conda/envs/TransStop/bin/python


#SBATCH -N 1
#SBATCH --time=01:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=nicolas.haas3@etu.unistra.fr
#SBATCH --job-name=genome_prediction_analysis
#SBATCH --output=genome_prediction_analysis_%j.out


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import os
from scipy import stats
from tqdm import tqdm
import plotly.express as px


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


# Créer une palette de couleurs cohérente pour les médicaments
drug_list_for_palette = sorted(list(OUR_MAP.values()))
colors_rgb = sns.color_palette('tab20', n_colors=len(drug_list_for_palette))
# Convertir les couleurs RGB en format HEX pour une compatibilité avec Plotly
colors_hex = [mcolors.to_hex(c) for c in colors_rgb]
drug_color_map = dict(zip(drug_list_for_palette, colors_hex))
print("Palette de couleurs pour les médicaments créée (format HEX).")

# --- Chargement des données ---
print(f"Chargement du fichier de prédictions genome-wide depuis : {FINAL_PREDS_PATH}")
try:
    df = pd.read_parquet(FINAL_PREDS_PATH)
except FileNotFoundError:
    print("Fichier de prédictions introuvable. Veuillez d'abord lancer le script d'inférence.")
    exit()
print("Chargement terminé. Taille du DataFrame :", df.shape)


# # --- ANALYSE 1: PAYSAGE DE TRAITABILITÉ ---
# print("\n--- Début de l'Analyse 1: Paysage de Traitabilité ---")

# # 1. Calculer les métriques pour chaque PTC
# print("Calcul des métriques de traitabilité pour chaque PTC...")
# df['our_max_pred'] = df[OUR_PREDS_COLS].max(axis=1)
# # Utiliser l'écart-type comme mesure de spécificité
# df['our_specificity'] = df[OUR_PREDS_COLS].std(axis=1)

# # 2. Agréger par gène
# print("Agrégation des métriques par gène...")
# # On filtre les gènes avec un nombre minimum de PTCs pour la robustesse
# gene_counts = df['gene'].value_counts()
# genes_to_keep = gene_counts[gene_counts >= 20].index
# df_filtered_genes = df[df['gene'].isin(genes_to_keep)]

# gene_treatability = df_filtered_genes.groupby('gene').agg(
#     mean_max_rt=('our_max_pred', 'mean'),
#     mean_specificity=('our_specificity', 'mean'),
#     ptc_count=('gene', 'size')
# ).reset_index()

# # 3. Créer la visualisation du paysage
# print("Génération du graphique 'Paysage de Traitabilité'...")
# plt.figure(figsize=(16, 12))
# scatter = sns.scatterplot(
#     data=gene_treatability,
#     x='mean_max_rt',
#     y='mean_specificity',
#     size='ptc_count',
#     hue='ptc_count',
#     palette='viridis',
#     sizes=(50, 2000),
#     alpha=0.7,
#     edgecolor='k',
#     linewidth=0.5
# )

# plt.title('Paysage de Traitabilité des Gènes Humains', fontsize=22, pad=20)
# plt.xlabel('Traitabilité Moyenne (Meilleur RT Prédit)', fontsize=16)
# plt.ylabel('Spécificité Requise (Écart-type des RT Prédits)', fontsize=16)
# plt.grid(True, linestyle='--')

# # Ajouter des annotations pour les quadrants
# xlim = plt.xlim()
# ylim = plt.ylim()
# x_mid = sum(xlim) / 2
# y_mid = sum(ylim) / 2
# plt.text(xlim[0] + 0.05*abs(xlim[0]), ylim[0] + 0.05*abs(ylim[0]), 'Déserts Thérapeutiques\n(Faible RT, Faible Spécificité)', fontsize=14, color='red', alpha=0.8, ha='left')
# plt.text(xlim[1] - 0.05*abs(xlim[1]), ylim[1] - 0.05*abs(ylim[1]), 'Oasis Spécifiques\n(Haut RT, Haute Spécificité)', fontsize=14, color='green', alpha=0.8, ha='right')
# plt.text(xlim[1] - 0.05*abs(xlim[1]), ylim[0] + 0.05*abs(ylim[0]), 'Terrains Faciles\n(Haut RT, Faible Spécificité)', fontsize=14, color='blue', alpha=0.8, ha='right')

# # Annoter quelques gènes importants
# genes_to_annotate = ['TP53', 'APC', 'BRCA1', 'BRCA2', 'DMD', 'CFTR', 'PTEN']
# for gene in genes_to_annotate:
#     if gene in gene_treatability['gene'].values:
#         gene_data = gene_treatability[gene_treatability['gene'] == gene]
#         plt.text(gene_data['mean_max_rt'].values[0], gene_data['mean_specificity'].values[0], gene, 
#                  fontdict={'weight': 'bold', 'size': 12, 'color': 'black'})

# plt.legend(title='Nombre de PTCs', loc='upper left', bbox_to_anchor=(1, 1))
# plt.tight_layout()
# plt.savefig(os.path.join(RESULTS_DIR, "treatability_landscape.png"), dpi=300)
# plt.close()
# print("Graphique du paysage de traitabilité sauvegardé.")


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
    cmap='viridis',
    linewidths=.5
)
plt.title('Concordance de la Meilleure Drogue Prédite', fontsize=20, pad=20)
plt.xlabel('Meilleure Drogue (Notre Modèle)', fontsize=16)
plt.ylabel('Meilleure Drogue (Modèle Toledano et al.)', fontsize=16)
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


# --- Amélioration 3.1: Générer graphique (logarithmique) ---

# Graphique: Échelle Logarithmique
print("Génération du graphique de distribution du gain (échelle logarithmique)...")
plt.figure(figsize=(12, 7))
sns.histplot(disagreement_df['our_gain'], bins=50, kde=False) # kde=False est souvent mieux avec l'échelle log
plt.yscale('log')
plt.title('Distribution du Gain de Performance Prédit (Échelle Log)', fontsize=16)
plt.xlabel('Gain de RT (Notre Meilleure Drogue vs. Choix de Toledano)', fontsize=14)
plt.ylabel('Nombre de PTCs (Échelle Log)', fontsize=14)
plt.axvline(x=0, color='red', linestyle='--')
plt.grid(True, which='major', linestyle='--')
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
print("\n--- Analyse des Cas de Désaccord à Fort Impact (Gain de RT > 1.0) ---")

# 1. Définir le seuil et filtrer le DataFrame
gain_threshold = 1.0
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
    stop_type_high_gain_df['Groupe'] = 'Cas à Fort Impact (Gain > 1.0)'


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
    change_pair_counts = high_gain_df['change_pair'].value_counts().nlargest(10) # On prend les 10 plus fréquentes

    plt.figure(figsize=(12, 10))
    barplot_pairs = sns.barplot(x=change_pair_counts.values, y=change_pair_counts.index, palette='viridis_r', hue=change_pair_counts.index, dodge=False, legend=False)
    plt.title('Top 10 des "Changements d\'Avis" à Fort Impact (Gain > 1.0)', fontsize=18)
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

# --- ANALYSE 3 (Revisité): PROFIL ET HIÉRARCHIE D'EFFICACITÉ DES DROGUES ---
print("\n--- Début de l'Analyse 3: Profil et Hiérarchie d'Efficacité des Drogues ---")

# 1. Identifier la meilleure drogue pour chaque PTC (nécessaire pour les deux analyses)
print("Identification de la meilleure drogue pour chaque PTC...")
df['our_best_drug'] = df[OUR_PREDS_COLS].rename(columns=OUR_MAP).idxmax(axis=1)



# --- Visualisation 1 (Revisité): Sunburst Plot Inversé (Hiérarchie Drogue -> Stop Type) ---
print("Génération du Sunburst Plot Inversé...")

if 'stop_type' in df.columns:
    sunburst_data = df.groupby(['our_best_drug', 'stop_type']).size().reset_index(name='ptc_count')

    # Créer la figure interactive avec l'ordre inversé dans 'path'
    fig = px.sunburst(
        sunburst_data,
        path=['our_best_drug', 'stop_type'], # <-- ORDRE INVERSÉ ICI
        values='ptc_count',
        color='our_best_drug', # Colorer par drogue est plus logique maintenant
        color_discrete_map=drug_color_map,
        title='Profil de Spécialisation des Drogues par Type de Codon Stop',
    )

    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25), font_size=16, title_font_size=22)
    fig.update_traces(textinfo="label+percent parent") # '% parent' montre la composition interne de chaque drogue

    sunburst_path = os.path.join(RESULTS_DIR, "best_drug_sunburst_inverted.png")
    fig.write_image(sunburst_path, width=1200, height=1200, scale=2)
    print(f"Sunburst plot inversé sauvegardé dans : {sunburst_path}")
else:
    print("Colonne 'stop_type' manquante, le Sunburst plot est ignoré.")

# --- Visualisation Alternative: Raincloud Plot (Violin Plots) ---
print("Génération du Raincloud Plot pour caractériser le style des drogues...")

# Préparer les données : passer du format large au format long
print("Mise en forme des données (melt)...")
df_melted = df[OUR_PREDS_COLS].melt(var_name='drug_col', value_name='predicted_rt')
df_melted['drug'] = df_melted['drug_col'].map(OUR_MAP)
df_melted['log_rt'] = np.log1p(df_melted['predicted_rt'])

print("Échantillonnage des données pour la visualisation...")
# Pour la "pluie", un échantillon plus petit est plus lisible.
if len(df_melted) > 100_000:
    df_sample = df_melted.sample(n=100_000, random_state=42)
else:
    df_sample = df_melted

# Trier les drogues par performance médiane pour un ordre visuel logique
drug_order = df_sample.groupby('drug')['predicted_rt'].median().sort_values(ascending=True).index

# --- Construction Manuelle du Raincloud Plot ---
fig, ax = plt.subplots(figsize=(16, 12))

# Définir les offsets verticaux pour séparer les éléments
# Chaque drogue aura son "étage" centré sur un entier (0, 1, 2...).
CLOUD_OFFSET = 0   # Le nuage sera au-dessus de la ligne de base
RAIN_OFFSET = -0.15   # La pluie sera en dessous
BOX_OFFSET = -0.15   # Le boxplot sera encore plus bas

for i, drug_name in enumerate(drug_order):
    # 1. --- Préparer les données pour la drogue actuelle ---
    drug_data = df_sample[df_sample['drug'] == drug_name]
    drug_log_rt = drug_data['log_rt']
    color = drug_color_map[drug_name]
    
    # 2. --- Couche 1: Le "Nuage" (Half-Violin manuel) ---
    # Calculer l'estimation de la densité par noyau (KDE)
    kde = stats.gaussian_kde(drug_log_rt, bw_method='scott')
    x_range = np.linspace(drug_log_rt.min(), drug_log_rt.max(), 100)
    density = kde(x_range)
    
    # Normaliser la hauteur du nuage pour qu'il soit esthétique
    scaled_density = density / density.max() * 0.4
    
    # Dessiner le nuage rempli
    ax.fill_between(x_range, i + CLOUD_OFFSET, i + CLOUD_OFFSET + scaled_density, 
                    color=color, alpha=0.5, zorder=1)
    # Dessiner le contour du nuage
    ax.plot(x_range, i + CLOUD_OFFSET + scaled_density, color=color, lw=1.5, zorder=2)
    # Dessiner la ligne de base du nuage
    ax.plot(x_range, np.full_like(x_range, i + CLOUD_OFFSET), color=color, lw=1.5, zorder=2)


    # 3. --- Couche 2: La "Pluie" (Stripplot manuel) ---
    # Créer un jitter vertical
    jitter = np.random.uniform(-0.15, 0.15, size=len(drug_log_rt))
    y_rain = np.full_like(drug_log_rt, i + RAIN_OFFSET) + jitter
    
    ax.scatter(drug_log_rt, y_rain, color=color, s=2, alpha=0.1, zorder=3)


    # 4. --- Couche 3: Le Boxplot ---
    # Définir les styles
    boxprops = {'facecolor': 'none', 'edgecolor': 'black', 'linewidth': 1.5, 'zorder': 4}
    medianprops = {'color': 'black', 'linewidth': 2, 'zorder': 5}
    whiskerprops = {'color': 'black', 'linewidth': 1.5, 'zorder': 4}
    capprops = {'color': 'black', 'linewidth': 1.5, 'zorder': 4}
    
    ax.boxplot(drug_log_rt, vert=False, positions=[i + BOX_OFFSET],
               showfliers=False, showcaps=True,
               patch_artist=True, # Indispensable pour `facecolor`
               boxprops=boxprops, medianprops=medianprops,
               whiskerprops=whiskerprops, capprops=capprops,
               widths=0.30)


# --- Finalisation et Esthétique ---
# Configurer l'axe Y pour afficher les noms des drogues
ax.set_yticks(np.arange(len(drug_order)))
ax.set_yticklabels(drug_order)
ax.tick_params(axis='y', length=0) # Cacher les petites barres de graduation sur l'axe Y

# Configurer l'axe X pour être lisible en échelle RT originale
original_ticks = [0, 0.5, 1, 2, 3, 5, 7]
log_ticks = np.log1p(original_ticks)
ax.set_xticks(log_ticks)
ax.set_xticklabels(labels=original_ticks)
ax.set_xlabel("Readthrough Prédit (RT) - Échelle log-transformée", fontsize=16)
ax.set_ylabel("") # Pas besoin de label "Drogue" ici

ax.set_title("Profil de Performance des Drogues (Raincloud Plot)", fontsize=22, pad=20)
ax.tick_params(axis='x', which='major', labelsize=14)
ax.grid(True, axis='x', linestyle='--', alpha=0.6)

sns.despine(left=True, bottom=True, trim=True)
plt.tight_layout()

raincloud_path = os.path.join(RESULTS_DIR, "drug_profile_raincloud_plot_custom.png")
plt.savefig(raincloud_path, dpi=300)
plt.close()
print(f"Raincloud plot personnalisé sauvegardé dans : {raincloud_path}")


# --- ANALYSE 4 (Version Finale): CAS D'USAGE PRÉDICTIF SUR LE GÈNE CFTR ---
print("\n--- Début de l'Analyse 4: Cas d'Usage Prédictif sur le Gène CFTR ---")

# ... (le code de chargement et de filtrage de cftr_df reste identique) ...
if 'gene' not in df.columns:
    print("La colonne 'gene' est manquante. L'analyse CFTR est ignorée.")
else:
    cftr_df = df[df['gene'] == 'CFTR'].copy()

    if cftr_df.empty:
        print("Aucune donnée trouvée pour le gène CFTR. Analyse ignorée.")
    else:
        # --- Analyse 1: Profil Thérapeutique par Mutation (Heatmaps avec Ordre Fixe) ---
        print("Génération des heatmaps de profil thérapeutique pour les mutations CFTR clés...")

        mutations_of_interest = {
            'G542X': 542,
            'R553X': 553,
            'R1162X': 1162,
            'W1282X': 1282,
        }
        
        positions_to_analyze = list(mutations_of_interest.values())
        cftr_mutations_df = cftr_df[cftr_df['position_PTC'].isin(positions_to_analyze)].copy()

        cftr_melted = cftr_mutations_df.melt(
            id_vars=['position_PTC', 'stop_type'],
            value_vars=OUR_PREDS_COLS,
            var_name='drug_col',
            value_name='predicted_rt'
        )
        cftr_melted['drug'] = cftr_melted['drug_col'].map(OUR_MAP)
        pos_to_name_map = {v: k for k, v in mutations_of_interest.items()}
        cftr_melted['mutation_name'] = cftr_melted['position_PTC'].map(pos_to_name_map)
        
        # --- CORRECTION PRINCIPALE : DÉFINIR UN ORDRE GLOBAL ET FIXE POUR LES DROGUES ---
        # 1. Calculer la performance moyenne de chaque drogue sur l'ensemble des mutations CFTR sélectionnées.
        #    Ceci nous donne un classement global de leur pertinence pour ce gène.
        global_drug_order = cftr_melted.groupby('drug')['predicted_rt'].mean().sort_values(ascending=False).index
        print("Ordre global des drogues pour les heatmaps (basé sur la performance moyenne) :")
        print(global_drug_order)
        # --- FIN DE LA CORRECTION ---

        num_mutations = len(mutations_of_interest)
        fig, axes = plt.subplots(1, num_mutations, figsize=(5.5 * num_mutations, 10), sharey=True)
        if num_mutations == 1: axes = [axes]

        vmax = cftr_melted['predicted_rt'].max()

        for i, (name, pos) in enumerate(mutations_of_interest.items()):
            ax = axes[i]
            mutation_data = cftr_melted[cftr_melted['position_PTC'] == pos]
            if not mutation_data.empty:
                pivot_df = mutation_data.pivot_table(index='drug', columns='stop_type', values='predicted_rt')
                
                # --- APPLIQUER L'ORDRE GLOBAL ET FIXE ---
                # Ré-indexer le DataFrame pour qu'il suive notre ordre prédéfini.
                # .reindex() gérera les cas où une drogue n'aurait pas de données.
                pivot_df = pivot_df.reindex(global_drug_order)
                # --- FIN DE L'APPLICATION ---
                
                sns.heatmap(
                    pivot_df, ax=ax, annot=True, fmt=".2f", cmap='viridis',
                    linewidths=.5, vmin=0, vmax=vmax, cbar=(i == num_mutations - 1)
                )
                ax.set_title(name, fontsize=18, pad=15)
                ax.set_xlabel("")
                ax.set_ylabel("Drogue" if i == 0 else "", fontsize=14)
                ax.tick_params(axis='x', labelsize=12)
                ax.tick_params(axis='y', labelsize=12)

        fig.text(0.5, 0.04, 'Type de Codon Stop', ha='center', va='center', fontsize=16)
        fig.suptitle("Profil Thérapeutique Prédit pour des Mutations CFTR Clés", fontsize=22, y=0.98)
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])

        plt.savefig(os.path.join(RESULTS_DIR, "cftr_therapeutic_profiles_heatmap.png"), dpi=300)
        plt.close()
        print("Heatmaps des profils thérapeutiques pour CFTR sauvegardées.")

print("\n--- Toutes les analyses finales sont terminées. ---")

