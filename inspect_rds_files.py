import pyreadr
import pandas as pd
import os

# --- Configuration ---
# Replace this with the path where you downloaded the .rds files
DATA_DIR = "./data/" 
# --------------------

# Filenames to inspect
file1_name = "list2.rds"
file2_name = "list2_dtbl.rds"

path1 = os.path.join(DATA_DIR, file1_name)
path2 = os.path.join(DATA_DIR, file2_name)

# Increase pandas display width to see all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("="*80)
print(f"INSPECTING FILE: {file1_name}")
print("="*80)

if os.path.exists(path1):
    try:
        # pyreadr reads .rds files into a dictionary
        result1_dict = pyreadr.read_r(path1)
        
        # The R code suggests that the main object is a list.
        # The dictionary will contain a single element whose value is the R list.
        # We extract this list.
        list_key = list(result1_dict.keys())
        list2 = result1_dict[list_key]
        
        print(f"Type of loaded object '{list_key}': {type(list2)}")
        print(f"Number of elements in the list (number of genes/transcripts): {len(list2)}")
        print("\n--- Content of the first element in the list (usually a DataFrame) ---")
        
        # The R code indicates that each element of the list is a DataFrame (or data.table)
        first_element_df = list2
        print(f"Type of the first element: {type(first_element_df)}")
        print(f"Number of rows in the first DataFrame: {len(first_element_df)}")
        print("Columns:", first_element_df.columns.tolist())
        
        print("\n--- Displaying the first 5 rows of the first DataFrame ---")
        print(first_element_df.head(5))
        
    except Exception as e:
        print(f"An error occurred while reading or parsing {file1_name}: {e}")
else:
    print(f"File not found: {path1}")

print("\n\n" + "="*80)
print(f"INSPECTING FILE: {file2_name}")
print("="*80)

if os.path.exists(path2):
    try:
        # Read the file. `list2_dtbl` should be a single large DataFrame.
        result2_dict = pyreadr.read_r(path2)
        
        # Extract the DataFrame from the dictionary
        dtbl_key = list(result2_dict.keys())
        list2_dtbl = result2_dict[dtbl_key]
        
        print(f"Type of loaded object '{dtbl_key}': {type(list2_dtbl)}")
        print(f"DataFrame dimensions (rows, columns): {list2_dtbl.shape}")
        print("Columns:", list2_dtbl.columns.tolist())
        
        print("\n--- Displaying the first 5 rows of the DataFrame ---")
        print(list2_dtbl.head(5))

    except Exception as e:
        print(f"An error occurred while reading or parsing {file2_name}: {e}")
else:
    print(f"File not found: {path2}")

print("\n" + "="*80)
print("Inspection complete.")