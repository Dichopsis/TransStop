import pyreadr  # Pour lire le fichier .rds
import pandas as pd
import numpy as np

file_path = "./data/PTC.rds"
result = pyreadr.read_r(file_path)  # Lire le fichier

df = list(result.values())[0]

drugs = ["FUr", "Gentamicin", "CC90009", "G418", "Clitocine", "DAP", "SJ6986", "SRI", "Untreated"]
for name in drugs:
    df_filtered = df[
        (df["treatment"] == name) &  # Sélection du traitement spécifique
        (df["replicate"] == 2) &  # Réplicat doit être 2
        (df["reads_allbins"] > 15) &  # Nombre de lectures > 15
        (df["viral"] == "no") &  # Exclure les entrées virales
        (df["RT"].notna())  # Exclure les valeurs manquantes de RT
    ]
    print(df_filtered.columns)
    df_filtered = df_filtered[['nt_seq','stop_type','GENEINFO','mutation_identifier','Ref_allele','Mutant_allele','RT']]

    # Mettre les séquences nucléotidiques en majuscule
    df_filtered["nt_seq"] = df_filtered["nt_seq"].str.upper()

    # Exporter les dataframes en fichiers CSV
    df_path = f'./data/PTC_{name}.csv'

    df_filtered.to_csv(df_path, index=False)
    

file_path = "./data/NTC.rds"
result = pyreadr.read_r(file_path)  # Lire le fichier

df = list(result.values())[0]

drugs = ["G418", "Clitocine", "DAP", "SJ6986", "SRI"]
for name in drugs:
    df_filtered = df[
        (df["treatment"] == name) &  # Sélection du traitement spécifique
        (df["replicate"] == 2) &  # Réplicat doit être 2
        (df["reads_allbins"] > 15) &  # Nombre de lectures > 15
        (df["RT"].notna())  # Exclure les valeurs manquantes de RT
    ]
    df_filtered = df_filtered[['nt_seq','stop_type','GENEINFO','Transcript_ID','RT']]

    # Mettre les séquences nucléotidiques en majuscule
    df_filtered["nt_seq"] = df_filtered["nt_seq"].str.upper()

    # Exporter les dataframes en fichiers CSV
    df_path = f'./data/NTC_{name}.csv'

    df_filtered.to_csv(df_path, index=False)