import pandas as pd
import json
import os

print("--- Starting reconstruction of the drug_to_id map ---")

# --- Configure these paths to match your structure ---
PROCESSED_DATA_DIR = "./processed_data/"
MODELS_DIR = "./models/"
# -------------------------------------------------------------------

# The path to the file that was used to create the original map
train_df_path = os.path.join(PROCESSED_DATA_DIR, "train_df.csv")

# The path where we will save the reconstructed artifact
production_model_path = os.path.join(MODELS_DIR, "production_model")
map_save_path = os.path.join(production_model_path, "drug_map.json")

# Ensure the production model directory exists
os.makedirs(production_model_path, exist_ok=True)

# 1. Load the original training DataFrame
print(f"Loading {train_df_path}...")
train_df = pd.read_csv(train_df_path)

# 2. Recreate the map using THE EXACT LINE OF CODE from the training script
#    pd.unique() preserves the order of appearance, which is crucial here.
#    Do NOT use sorted()!
print("Reconstructing the map using the original order of appearance...")
drug_to_id = {drug: i for i, drug in enumerate(train_df['drug'].unique())}

# 3. Save this dictionary to a JSON file
print(f"Saving the reconstructed map to {map_save_path}...")
with open(map_save_path, 'w') as f:
    json.dump(drug_to_id, f, indent=4)

print("\n--- Operation completed successfully! ---")
print("The reconstructed map is:")
print(drug_to_id)
print(f"\nThe 'drug_map.json' file has been created. You can now run your evaluation script.")