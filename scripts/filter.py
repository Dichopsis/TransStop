import pyreadr  # To read the .rds file
import pandas as pd
import numpy as np

file_path = "../data/PTC.rds"
result = pyreadr.read_r(file_path)  # Read the file

df = list(result.values())[0]

drugs = ["FUr", "Gentamicin", "CC90009", "G418", "Clitocine", "DAP", "SJ6986", "SRI", "Untreated"]
for name in drugs:
    df_filtered = df[
        (df["treatment"] == name) &  # Select the specific treatment
        (df["replicate"] == 2) &  # Replicate must be 2
        (df["reads_allbins"] > 15) &  # Number of reads > 15
        (df["viral"] == "no") &  # Exclude viral entries
        (df["RT"].notna())  # Exclude missing RT values
    ]
    print(df_filtered.columns)
    print(f"Number of entries for {name}: {len(df_filtered)}")
    df_filtered = df_filtered[['nt_seq','stop_type','GENEINFO','mutation_identifier','Ref_allele','Mutant_allele','RT']]

    # Convert nucleotide sequences to uppercase
    df_filtered["nt_seq"] = df_filtered["nt_seq"].str.upper()

    # Export dataframes to CSV files
    df_path = f'../data/PTC_{name}.csv'

    df_filtered.to_csv(df_path, index=False)
    