import pandas as pd
import json
import os

print("--- Début de la reconstruction du mappage drug_to_id ---")

# --- Configurez ces chemins pour qu'ils correspondent à votre structure ---
PROCESSED_DATA_DIR = "./processed_data/"
MODELS_DIR = "./models/"
# -------------------------------------------------------------------

# Le chemin vers le fichier qui a servi à créer le mappage original
train_df_path = os.path.join(PROCESSED_DATA_DIR, "train_df.csv")

# Le chemin où nous allons sauvegarder l'artefact reconstruit
production_model_path = os.path.join(MODELS_DIR, "production_model")
map_save_path = os.path.join(production_model_path, "drug_map.json")

# S'assurer que le dossier du modèle de production existe
os.makedirs(production_model_path, exist_ok=True)

# 1. Charger le DataFrame d'entraînement original
print(f"Chargement de {train_df_path}...")
train_df = pd.read_csv(train_df_path)

# 2. Recréer le mappage en utilisant LA LIGNE DE CODE EXACTE du script d'entraînement
#    pd.unique() préserve l'ordre d'apparition, ce qui est crucial ici.
#    N'utilisez PAS sorted() !
print("Reconstruction du mappage en utilisant l'ordre d'apparition original...")
drug_to_id = {drug: i for i, drug in enumerate(train_df['drug'].unique())}

# 3. Sauvegarder ce dictionnaire dans un fichier JSON
print(f"Sauvegarde du mappage reconstruit dans {map_save_path}...")
with open(map_save_path, 'w') as f:
    json.dump(drug_to_id, f, indent=4)

print("\n--- Opération terminée avec succès ! ---")
print("Le mappage reconstruit est :")
print(drug_to_id)
print(f"\nLe fichier 'drug_map.json' a été créé. Vous pouvez maintenant lancer votre script d'évaluation.")