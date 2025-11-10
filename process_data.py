import pandas as pd
import os
import sys

# --- CONFIGURATION -----------------------------------------------------------
# --- Edit the values below to control the script ---

# 1. INPUT FILE
#    Path to your HUGE CSV file.
INPUT_FILE_PATH = r"F:\Users\Aryan\Documents\IAE-M\THI_Notes_Files\Group Project\Source Code\Data_Files\drowsiness_dataset.csv"

# 2. OUTPUT FOLDER
#    Where do you want to save the new, smaller files?
OUTPUT_FOLDER = r"F:\Users\Aryan\Documents\IAE-M\THI_Notes_Files\Group Project\Source Code\Data_Files\Processed"

# 3. TRIM COLUMNS
#    List of column names you want to KEEP. All others will be dropped.
#    This saves a lot of memory.
#    (Set to None or [] to keep all columns)
COLUMNS_TO_KEEP = ['ppgGreen', 'ppgRed', 'ppgIR', 'drowsiness']

# 4. TRIM ROWS
#    How do you want to reduce the number of rows?
#    Choose ONE method:
#    - 'first_n': Keep only the first N rows (e.g., for a quick test).
#    - 'every_n': Subsample the file, keeping 1 row for every N (e.g., 10 for 10% of data).
#    - 'all':     Keep all rows (use this if you only want to segregate).
TRIM_ROWS_METHOD = 'all' # 'first_n', 'every_n', or 'all'

# 5. TRIM ROWS VALUE (The 'N' for the method above)
#    - If method is 'first_n', this is the total number of rows (e.g., 100000).
#    - If method is 'every_n', this is the step (e.g., 10).
#    - (Ignored if method is 'all')
TRIM_ROWS_VALUE = 10 

# 6. SEGREGATE
#    Do you want to split the final data into separate files?
#    - Set to a column name (e.g., 'drowsiness') to split by that column.
#    - Set to None to save the trimmed data to a single file.
SEGREGATE_COLUMN = 'drowsiness' # or None

# 7. CHUNK SIZE
#    How many rows to read into memory at a time.
#    Lower this if you have very little RAM.
CHUNK_SIZE = 50000

# --- END CONFIGURATION -------------------------------------------------------


def process_huge_csv():
    """
    Main function to load, trim, and segregate the CSV based on CONFIG.
    """
    print(f"[INFO] Starting data processing for: {INPUT_FILE_PATH}")
    
    # --- 1. Validate paths and config ---
    if not os.path.exists(INPUT_FILE_PATH):
        print(f"[ERROR] Input file not found: {INPUT_FILE_PATH}")
        return

    if not os.path.exists(OUTPUT_FOLDER):
        print(f"[INFO] Output folder not found. Creating: {OUTPUT_FOLDER}")
        os.makedirs(OUTPUT_FOLDER)

    # Use 'usecols' for memory efficiency
    use_cols = COLUMNS_TO_KEEP if (COLUMNS_TO_KEEP and len(COLUMNS_TO_KEEP) > 0) else None

    # --- 2. Load and Trim Data ---
    df = None
    
    try:
        if TRIM_ROWS_METHOD == 'first_n':
            # This is the most efficient way to get the first N rows
            print(f"[INFO] Loading first {TRIM_ROWS_VALUE} rows...")
            df = pd.read_csv(
                INPUT_FILE_PATH,
                nrows=TRIM_ROWS_VALUE,
                usecols=use_cols
            )
            
        elif TRIM_ROWS_METHOD == 'every_n':
            # Read in chunks and subsample each chunk
            print(f"[INFO] Loading every {TRIM_ROWS_VALUE}th row...")
            all_chunks = []
            with pd.read_csv(
                INPUT_FILE_PATH,
                chunksize=CHUNK_SIZE,
                usecols=use_cols
            ) as reader:
                for i, chunk in enumerate(reader):
                    print(f"  ...processing chunk {i+1}")
                    subsampled_chunk = chunk.iloc[::TRIM_ROWS_VALUE]
                    all_chunks.append(subsampled_chunk)
            
            if not all_chunks:
                print("[ERROR] No data loaded. Check file and config.")
                return
            df = pd.concat(all_chunks, ignore_index=True)

        elif TRIM_ROWS_METHOD == 'all':
            # Load all specified columns, but still in chunks for large files
            print("[INFO] Loading all rows (chunked)...")
            all_chunks = []
            with pd.read_csv(
                INPUT_FILE_PATH,
                chunksize=CHUNK_SIZE,
                usecols=use_cols
            ) as reader:
                for i, chunk in enumerate(reader):
                    print(f"  ...processing chunk {i+1}")
                    all_chunks.append(chunk)

            if not all_chunks:
                print("[ERROR] No data loaded. Check file and config.")
                return
            df = pd.concat(all_chunks, ignore_index=True)
            
        else:
            print(f"[ERROR] Unknown TRIM_ROWS_METHOD: '{TRIM_ROWS_METHOD}'")
            return

    except Exception as e:
        print(f"[ERROR] Failed to load or process data: {e}")
        return

    print(f"\n[INFO] Data successfully loaded and trimmed.")
    print(f"       Final DataFrame shape: {df.shape}")
    print(f"       Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")

    # --- 3. Segregate and Save Data ---
    if SEGREGATE_COLUMN:
        if SEGREGATE_COLUMN not in df.columns:
            print(f"[ERROR] Segregate column '{SEGREGATE_COLUMN}' not found in data.")
            print(f"        Available columns: {df.columns.tolist()}")
            return
            
        print(f"\n[INFO] Segregating data by column: '{SEGREGATE_COLUMN}'...")
        unique_values = df[SEGREGATE_COLUMN].unique()
        print(f"       Found {len(unique_values)} unique values: {unique_values}")
        
        for value in unique_values:
            # Create a subset for this value
            subset_df = df[df[SEGREGATE_COLUMN] == value]
            
            # Create a safe filename
            safe_value = str(value).replace('.', '_')
            output_filename = f"trimmed_segregated_{SEGREGATE_COLUMN}_{safe_value}.csv"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            
            print(f"  -> Saving {len(subset_df)} rows for value '{value}' to: {output_path}")
            subset_df.to_csv(output_path, index=False)
            
    else:
        # Save the trimmed data as one single file
        print(f"\n[INFO] No segregation requested. Saving to a single file...")
        output_filename = "data_trimmed.csv"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        print(f"  -> Saving {len(df)} rows to: {output_path}")
        df.to_csv(output_path, index=False)

    print("\n[SUCCESS] Data processing finished.")


if __name__ == "__main__":
    process_huge_csv()