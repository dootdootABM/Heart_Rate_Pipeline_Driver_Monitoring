import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split

# --- CONFIGURATION -----------------------------------------------------------
# --- Edit the values below to control the script ---

# 1. INPUT FILE
#    Path to your original, large CSV file.
INPUT_FILE_PATH = r"F:\Users\Aryan\Documents\IAE-M\THI_Notes_Files\Group Project\Source Code\Data_Files\drowsiness_dataset.csv"

# 2. OUTPUT FOLDER
#    Where do you want to save the new split files?
OUTPUT_FOLDER = r"F:\Users\Aryan\Documents\IAE-M\THI_Notes_Files\Group Project\Source Code\Data_Files\new_dataset"

# 3. COLUMNS TO KEEP
#    List of column names you want to KEEP. This makes the new files smaller.
#    (Set to None or [] to keep all columns)
COLUMNS_TO_KEEP = ['heartRate','ppgGreen', 'drowsiness']

# 4. LABEL_COLUMN
#    Which column should be used for stratification?
#    This ensures all three new files have the same class distribution.
LABEL_COLUMN = 'drowsiness'

# 5. SPLIT SIZES
#    The percentages must add up to 1.0
TRAIN_SIZE = 0.7  # 70%
TEST_SIZE = 0.2   # 20%
VAL_SIZE = 0.1    # 10% (the "untested" set)

# 6. RANDOM_STATE
#    Set to any number to ensure your split is reproducible.
RANDOM_STATE = 42

# --- END CONFIGURATION -------------------------------------------------------


def split_data():
    """
    Loads the full dataset, splits it into train, test, and validation sets,
    and saves them to the output folder.
    """
    print(f"[INFO] Starting data splitting for: {INPUT_FILE_PATH}")

    # --- 1. Validate config and paths ---
    total_size = TRAIN_SIZE + TEST_SIZE + VAL_SIZE
    if round(total_size, 9) != 1.0:
        print(f"[ERROR] Split sizes do not add up to 1.0 (got {total_size})")
        return

    if not os.path.exists(INPUT_FILE_PATH):
        print(f"[ERROR] Input file not found: {INPUT_FILE_PATH}")
        return

    if not os.path.exists(OUTPUT_FOLDER):
        print(f"[INFO] Output folder not found. Creating: {OUTPUT_FOLDER}")
        os.makedirs(OUTPUT_FOLDER)

    use_cols = COLUMNS_TO_KEEP if (COLUMNS_TO_KEEP and len(COLUMNS_TO_KEEP) > 0) else None

    # --- 2. Load Full Dataset ---
    print(f"[INFO] Loading full dataset into memory...")
    try:
        df = pd.read_csv(INPUT_FILE_PATH, usecols=use_cols)
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        print("        Ensure the file path is correct and 'COLUMNS_TO_KEEP' match columns in the file.")
        return
        
    print(f"[INFO] Data loaded successfully. Shape: {df.shape}")

    if LABEL_COLUMN not in df.columns:
        print(f"[ERROR] Label column '{LABEL_COLUMN}' not found in the loaded data.")
        print(f"        Available columns: {df.columns.tolist()}")
        return

    print(f"[INFO] Original class distribution:\n{df[LABEL_COLUMN].value_counts(normalize=True).sort_index()}")

    # --- 3. First Split: Train (70%) and Temp (30%) ---
    print(f"[INFO] Performing first split (Train: {TRAIN_SIZE}, Temp: {1 - TRAIN_SIZE})...")
    train_df, temp_df = train_test_split(
        df,
        train_size=TRAIN_SIZE,
        stratify=df[LABEL_COLUMN],
        random_state=RANDOM_STATE
    )

    # --- 4. Second Split: Test (20%) and Validation (10%) from Temp (30%) ---
    # We need to calculate the new split ratio for the temp_df
    # val_size_of_temp = 0.1 / (0.2 + 0.1) = 0.333...
    validation_size_of_temp = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
    
    print(f"[INFO] Performing second split (Test and Validation from Temp)...")
    test_df, val_df = train_test_split(
        temp_df,
        test_size=validation_size_of_temp,
        stratify=temp_df[LABEL_COLUMN],
        random_state=RANDOM_STATE
    )

    # --- 5. Verification and Saving ---
    print("\n--- Verification ---")
    print(f"Total rows:   {len(df)}")
    print(f"Training set: {len(train_df)} rows ({len(train_df)/len(df):.1%})")
    print(f"Testing set:  {len(test_df)} rows ({len(test_df)/len(df):.1%})")
    print(f"Valid. set:   {len(val_df)} rows ({len(val_df)/len(df):.1%})")

    print("\n[INFO] Saving files...")
    
    # Define output paths
    train_path = os.path.join(OUTPUT_FOLDER, "train_dataset.csv")
    test_path = os.path.join(OUTPUT_FOLDER, "test_dataset.csv")
    val_path = os.path.join(OUTPUT_FOLDER, "validation_dataset.csv")
    
    # Save files
    train_df.to_csv(train_path, index=False)
    print(f"  -> Saved training set to:   {train_path}")
    
    test_df.to_csv(test_path, index=False)
    print(f"  -> Saved testing set to:    {test_path}")
    
    val_df.to_csv(val_path, index=False)
    print(f"  -> Saved validation set to: {val_path}")

    print("\n[SUCCESS] Data splitting finished.")


if __name__ == "__main__":
    split_data()