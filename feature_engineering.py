import pandas as pd
import numpy as np
import neurokit2 as nk
import sys

# --- Import the load_data function from the previous module ---
# Make sure 'load_data.py' is in the same folder as this script
try:
    from load_data import load_data
except ImportError:
    print("[ERROR] Could not import 'load_data' function. Make sure 'load_data.py' is in the same directory.")
    sys.exit(1) # Exit if the import fails

# --- CONFIGURATION: YOU MUST VERIFY/ADJUST THESE! ---
# 1. SAMPLING_RATE: This is CRITICAL. It MUST match your sensor's recording frequency.
#    Based on previous attempts, we are using 25 Hz, but double-check this.
#    Incorrect rate will lead to incorrect HRV features or errors.
SAMPLING_RATE = 25  # Hz (Samples per second)

# 2. PPG_COLUMN: Choose the PPG signal column to analyze.
#    'ppgIR' or 'ppgGreen' are often good choices.
PPG_COLUMN = 'ppgGreen'

# 3. WINDOW_SECONDS: Duration of each analysis window in seconds.
#    60 seconds is a common starting point for HRV.
WINDOW_SECONDS = 1200 #changed from 60 to 300 to clear all warnings
# --- END CONFIGURATION ---

def engineer_features(df):
    """
    Applies sliding window and uses NeuroKit2 to extract HRV features from raw PPG.

    Args:
        df (pd.DataFrame): The DataFrame loaded by load_data.py.

    Returns:
        tuple (pd.DataFrame, np.array):
            - X: DataFrame containing the calculated HRV features for each window.
            - y: NumPy array containing the corresponding drowsiness label for each window.
            Returns (None, None) if feature engineering fails completely.
    """
    if df is None or df.empty:
        print("[ERROR] Input DataFrame is empty or None. Cannot engineer features.")
        return None, None

    if PPG_COLUMN not in df.columns:
        print(f"[ERROR] Specified PPG column '{PPG_COLUMN}' not found in the DataFrame.")
        print(f"Available columns: {df.columns.tolist()}")
        return None, None

    if 'drowsiness' not in df.columns:
        print("[ERROR] 'drowsiness' column not found in the DataFrame. Cannot determine labels.")
        return None, None

    # Calculate window size in samples
    window_size = SAMPLING_RATE * WINDOW_SECONDS
    # Overlap windows by 50% for more data points
    step_size = window_size // 2

    # Check if the dataset is long enough for at least one window
    if len(df) < window_size:
        print(f"[ERROR] Dataset length ({len(df)}) is shorter than window size ({window_size}). Cannot process.")
        return None, None

    features_list = []
    labels_list = []
    processed_windows = 0
    skipped_windows = 0
    max_errors_to_show = 5 # Limit detailed error printing

    print(f"\n[INFO] Starting feature engineering...")
    print(f"       - Sampling Rate: {SAMPLING_RATE} Hz")
    print(f"       - PPG Column: '{PPG_COLUMN}'")
    print(f"       - Window Duration: {WINDOW_SECONDS} seconds")
    print(f"       - Window Size: {window_size} samples")
    print(f"       - Step Size: {step_size} samples")

    # --- Sliding Window Loop ---
    for i in range(0, len(df) - window_size, step_size):
        window_df = df.iloc[i : i + window_size]
        ppg_signal_raw = window_df[PPG_COLUMN].values # Use numpy array for NK
        # Determine the label for the window (most frequent value)
        label = window_df['drowsiness'].mode()[0]

        try:
            # --- Perform NeuroKit2 PPG Analysis (Broken Down for Debugging) ---

            # Step 1: Clean the signal
            ppg_cleaned = nk.ppg_clean(ppg_signal_raw, sampling_rate=SAMPLING_RATE)

            # Step 2: Find peaks (heartbeats)
            peaks_info = nk.ppg_findpeaks(ppg_cleaned, sampling_rate=SAMPLING_RATE)
            peak_indices = peaks_info["PPG_Peaks"]

            # Step 3: Check if enough peaks were found
            min_peaks_required = 5 # Need at least a few peaks for HRV
            if len(peak_indices) < min_peaks_required:
                 raise ValueError(f"Too few peaks found ({len(peak_indices)} < {min_peaks_required}). Signal quality might be too low or sampling rate incorrect.")

            # Step 4: Calculate HRV features from the peak locations
            hrv_indices = nk.hrv(peak_indices, sampling_rate=SAMPLING_RATE, show=False)

            # Step 5: Check HRV results
            if hrv_indices is None or hrv_indices.empty:
                raise ValueError("HRV calculation returned no results (None or empty).")

            # Successfully extracted features for this window
            hrv_features = hrv_indices.iloc[0] # Get the feature row
            features_list.append(hrv_features)
            labels_list.append(label)
            processed_windows += 1

        except Exception as e:
            skipped_windows += 1
            if skipped_windows <= max_errors_to_show:
                error_msg = str(e)
                if len(error_msg) > 150: error_msg = error_msg[:150] + "..."
                print(f"[WARN] Skipping window at index {i}. Reason: {type(e).__name__}: {error_msg}")
            if skipped_windows == max_errors_to_show:
                print(f"[WARN] ... (suppressing further error messages for skipped windows)")

    # --- Post-Loop Processing ---
    print(f"\n[INFO] Feature engineering finished.")
    print(f"       - Successfully processed {processed_windows} windows.")
    print(f"       - Skipped {skipped_windows} windows due to errors.")

    if not features_list:
        print("[ERROR] No features were successfully extracted from any window.")
        return None, None # Return None if no features could be extracted

    # Convert the list of feature dictionaries (Series) into a DataFrame
    try:
        X = pd.DataFrame(features_list)
        y = np.array(labels_list)

        # --- Data Cleaning ---
        # 1. Drop columns that are entirely NaN (e.g., if DFA failed everywhere)
        cols_before = X.shape[1]
        X = X.dropna(axis=1, how='all')
        cols_after = X.shape[1]
        if cols_before > cols_after:
            print(f"[INFO] Dropped {cols_before - cols_after} columns that contained only NaN values.")

        # 2. Fill remaining NaN values with the mean of their column
        #    (Can happen if a feature fails only in *some* windows)
        if X.isnull().sum().sum() > 0:
             print("[INFO] Filling remaining NaN values with column means.")
             X = X.fillna(X.mean())

        # 3. Check for infinite values (can sometimes occur in calculations)
        if np.isinf(X.values).any():
            print("[WARN] Infinite values detected in features. Replacing with NaN and re-filling.")
            X.replace([np.inf, -np.inf], np.nan, inplace=True)
            X = X.fillna(X.mean()) # Re-fill potential new NaNs

        print(f"[INFO] Final feature set has shape: {X.shape}")
        print(f"[INFO] Corresponding labels array has shape: {y.shape}")

        return X, y

    except Exception as e:
        print(f"[ERROR] Failed to create or clean the final DataFrame from features: {e}")
        return None, None
    
def create_combined_dataset(features_df, labels_array):
    """
    Combines the feature DataFrame (X) and the labels array (y) into a
    single DataFrame for easy saving and analysis.

    Args:
        features_df (pd.DataFrame): The DataFrame of HRV features (X).
        labels_array (np.array): The NumPy array of corresponding labels (y).

    Returns:
        pd.DataFrame: A new, single DataFrame with the labels included
                      as a column named 'drowsiness_label'.
                      Returns None if inputs are invalid.
    """
    if features_df is None or labels_array is None:
        print("[ERROR] Cannot combine dataset: features or labels are None.")
        return None

    if len(features_df) != len(labels_array):
        print(f"[ERROR] Mismatch in length: Features have {len(features_df)} rows, "
              f"but labels have {len(labels_array)} entries.")
        return None

    try:
        # Create a copy to avoid modifying the original DataFrame in-place
        combined_df = features_df.copy()

        # Assign the labels array as a new column.
        # This is robust as it assigns by position.
        combined_df['drowsiness_label'] = labels_array

        print(f"[INFO] Successfully combined features and labels into new DataFrame with shape: {combined_df.shape}")
        return combined_df

    except Exception as e:
        print(f"[ERROR] An error occurred while combining features and labels: {e}")
        return None
    
# --- Main execution block for testing this module ---(function block for testing)
if __name__ == "__main__":
    print(f"NeuroKit2 version: {nk.__version__}")

    # --- Define File Paths ---
    input_csv_path = r'F:\Users\Aryan\Documents\IAE-M\THI_Notes_Files\Group Project\Source Code\Data_Files\drowsiness_dataset.csv'
    # --- [NEW] Define an output path for the clean, engineered file ---
    output_features_path = r'F:\Users\Aryan\Documents\IAE-M\THI_Notes_Files\Group Project\Source Code\Data_Files\engineered_features_dataset.csv'

    # 1. Load data using the function from the other file
    raw_data_df = load_data(input_csv_path)

    # 2. If data loaded, perform feature engineering
    if raw_data_df is not None:
        features_df, labels_array = engineer_features(raw_data_df)

        if features_df is not None and not features_df.empty:
            print("\n--- Feature Engineering Output ---")
            print("First 5 rows of features (X):")
            print(features_df.head())
            print("\nLabels (y) shape:", labels_array.shape)
            print("Unique labels found:", np.unique(labels_array, return_counts=True))

            # --- [NEW] Use the new function to combine X and y ---
            print("\n--- Combining Features and Labels ---")
            combined_dataset = create_combined_dataset(features_df, labels_array)

            if combined_dataset is not None:
                print("First 5 rows of *combined* dataset:")
                print(combined_dataset.head())

                # --- [NEW] Save the final, clean dataset to a new CSV ---
                try:
                    combined_dataset.to_csv(output_features_path, index=False)
                    print(f"\n[SUCCESS] Successfully saved engineered features to:")
                    print(f"{output_features_path}")
                except Exception as e:
                    print(f"\n[ERROR] Failed to save final dataset to CSV: {e}")

            else:
                print("\n[FAILURE] Could not create combined dataset.")

        else:
            print("\n[FAILURE] Feature engineering did not produce usable features.")
    else:
        print("\n[FAILURE] Skipping feature engineering due to data loading failure.")
