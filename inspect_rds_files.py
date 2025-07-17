import pyreadr
import pandas as pd
import os

# --- Configuration ---
# Remplacez ceci par le chemin où vous avez téléchargé les fichiers .rds
DATA_DIR = "./data/" 
# --------------------

# Noms des fichiers à inspecter
file1_name = "list2.rds"
file2_name = "list2_dtbl.rds"

path1 = os.path.join(DATA_DIR, file1_name)
path2 = os.path.join(DATA_DIR, file2_name)

# Augmenter la largeur d'affichage de pandas pour voir toutes les colonnes
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("="*80)
print(f"INSPECTION DU FICHIER : {file1_name}")
print("="*80)

if os.path.exists(path1):
    try:
        # pyreadr lit les fichiers .rds dans un dictionnaire
        result1_dict = pyreadr.read_r(path1)
        
        # Le code R suggère que l'objet principal est une liste.
        # Le dictionnaire contiendra un seul élément dont la valeur est la liste R.
        # On extrait cette liste.
        list_key = list(result1_dict.keys())[0]
        list2 = result1_dict[list_key]
        
        print(f"Type de l'objet '{list_key}' chargé : {type(list2)}")
        print(f"Nombre d'éléments dans la liste (nombre de gènes/transcrits) : {len(list2)}")
        print("\n--- Contenu du premier élément de la liste (généralement un DataFrame) ---")
        
        # Le code R indique que chaque élément de la liste est un DataFrame (ou data.table)
        first_element_df = list2[0]
        print(f"Type du premier élément : {type(first_element_df)}")
        print(f"Nombre de lignes dans le premier DataFrame : {len(first_element_df)}")
        print("Colonnes :", first_element_df.columns.tolist())
        
        print("\n--- Affichage des 5 premières lignes du premier DataFrame ---")
        print(first_element_df.head(5))
        
    except Exception as e:
        print(f"Une erreur est survenue lors de la lecture ou de l'analyse de {file1_name}: {e}")
else:
    print(f"Fichier non trouvé : {path1}")

print("\n\n" + "="*80)
print(f"INSPECTION DU FICHIER : {file2_name}")
print("="*80)

if os.path.exists(path2):
    try:
        # Lire le fichier. `list2_dtbl` devrait être un seul grand DataFrame.
        result2_dict = pyreadr.read_r(path2)
        
        # Extraire le DataFrame du dictionnaire
        dtbl_key = list(result2_dict.keys())[0]
        list2_dtbl = result2_dict[dtbl_key]
        
        print(f"Type de l'objet '{dtbl_key}' chargé : {type(list2_dtbl)}")
        print(f"Dimensions du DataFrame (lignes, colonnes) : {list2_dtbl.shape}")
        print("Colonnes :", list2_dtbl.columns.tolist())
        
        print("\n--- Affichage des 5 premières lignes du DataFrame ---")
        print(list2_dtbl.head(5))

    except Exception as e:
        print(f"Une erreur est survenue lors de la lecture ou de l'analyse de {file2_name}: {e}")
else:
    print(f"Fichier non trouvé : {path2}")

print("\n" + "="*80)
print("Inspection terminée.")