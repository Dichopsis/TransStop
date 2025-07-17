import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import glob
from tqdm import tqdm

print("--- PART 1: DATA INGESTION, PREPARATION, AND STRATEGY DEFINITION ---")

# --- 1.1. Data Ingestion and Scoping ---
DATA_DIR = "./data/"
OUTPUT_DIR = "./processed_data/"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Find all PTC data files with the new format "PTC_{drug}.csv"
ptc_files = glob.glob(os.path.join(DATA_DIR, "PTC_*.csv"))
if not ptc_files:
    raise FileNotFoundError(f"No PTC data files found in '{DATA_DIR}'. Please check the path and filenames (expected format: PTC_drugname.csv).")

print(f"Found {len(ptc_files)} PTC data files.")

# Ingest all PTC files into a single master DataFrame
df_list = []
for f in ptc_files:
    drug_name = os.path.basename(f).replace('PTC_', '').replace('.csv', '')
    temp_df = pd.read_csv(f)
    temp_df['drug'] = drug_name
    df_list.append(temp_df)

master_df = pd.concat(df_list, ignore_index=True)
print(f"Master DataFrame created with {len(master_df)} rows.")

# --- 1.2. Sequence Context Engineering ---
print("Engineering sequence contexts...")

def create_contexts(sequence):
    """Slices a sequence to create contexts of different lengths."""
    center_start_idx = 72
    
    seq_144 = sequence[center_start_idx - 72 : center_start_idx + 72 + 3]
    seq_42 = sequence[center_start_idx - 21 : center_start_idx + 21 + 3]
    seq_18 = sequence[center_start_idx - 9 : center_start_idx + 9 + 3]
    seq_12 = sequence[center_start_idx - 6 : center_start_idx + 6 + 3]
    seq_6 = sequence[center_start_idx - 3 : center_start_idx + 3 + 3]
    
    if len(seq_144) != 147 or len(seq_42) != 45 or len(seq_18) != 21 or len(seq_12) != 15 or len(seq_6) != 9:
        return None, None, None, None, None

    return seq_144, seq_42, seq_18, seq_12, seq_6

tqdm.pandas(desc="Creating sequence contexts")
contexts = master_df['nt_seq'].progress_apply(create_contexts)

master_df[['seq_context_144', 'seq_context_42', 'seq_context_18', 'seq_context_12','seq_context_6']] = pd.DataFrame(contexts.tolist(), index=master_df.index)

master_df.dropna(subset=['seq_context_144'], inplace=True)
print(f"Context engineering complete. DataFrame size: {len(master_df)} rows.")


# --- 1.3. Target Variable Transformation ---
print("Applying log1p transformation to RT column...")
master_df['RT_transformed'] = np.log1p(master_df['RT'])


# --- 1.4. Final Dataset Splitting and Archiving ---
print("Splitting data into train, validation, and test sets (80/10/10)...")
train_val_df, test_df = train_test_split(
    master_df,
    test_size=0.1,
    random_state=42,
    stratify=master_df['drug']
)

train_df, val_df = train_test_split(
    train_val_df,
    test_size=1/9,
    random_state=42,
    stratify=train_val_df['drug']
)

print(f"Train set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")

train_df.to_csv(os.path.join(OUTPUT_DIR, "train_df.csv"), index=False)
val_df.to_csv(os.path.join(OUTPUT_DIR, "val_df.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, "test_df.csv"), index=False)

print(f"Data preparation complete. Processed datasets saved to '{OUTPUT_DIR}'.")
print("--- END OF PART 1 ---")