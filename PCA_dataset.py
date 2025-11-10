import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys

# --- Configuration ---

INPUT_FILE = r'F:\Users\Aryan\Documents\IAE-M\THI_Notes_Files\Group Project\Source Code\Data_Files\engineered_features_dataset.csv'

LABEL_COLUMN = 'drowsiness_label'
# --- End Configuration ---

def run_pca_analysis(file_path):
    """
    Loads the engineered feature dataset, scales it, runs PCA,
    and generates visualization plots.
    """
    print(f"[INFO] Loading data from: {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"[ERROR] File not found at: {file_path}")
        print("        Please make sure the file is in the same directory as this script.")
        return
    except Exception as e:
        print(f"[ERROR] Could not load data: {e}")
        return

    print(f"[INFO] Data loaded: {df.shape[0]} rows, {df.shape[1]} columns.")

    # --- 1. Clean Data ---
    # PCA cannot handle NaN (Not a Number) or Inf (Infinity) values.
    initial_rows = len(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    final_rows = len(df)
    
    if final_rows < initial_rows:
        print(f"[INFO] Removed {initial_rows - final_rows} rows containing NaN or Inf values.")

    if final_rows == 0:
        print("[ERROR] No data left after cleaning. Cannot proceed.")
        return

    # --- 2. Separate Features (X) and Label (y) ---
    if LABEL_COLUMN not in df.columns:
        print(f"[ERROR] The label column '{LABEL_COLUMN}' was not found in the CSV.")
        print(f"        Available columns are: {df.columns.tolist()}")
        return
        
    y = df[LABEL_COLUMN]
    X = df.drop(columns=[LABEL_COLUMN])
    
    # Ensure all feature columns are numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    X.dropna(axis=1, how='any', inplace=True) # Drop any columns that became all NaN
    
    print(f"[INFO] Separated label '{LABEL_COLUMN}' from {len(X.columns)} feature columns.")

    # --- 3. Standardize Features ---
    # This is CRITICAL for PCA. All features must be on the same scale.
    print("[INFO] Standardizing features (mean=0, std=1)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- 4. Run PCA ---
    print("[INFO] Fitting PCA to all components...")
    pca = PCA() # By default, fits to all components
    X_pca = pca.fit_transform(X_scaled)
    
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    print("\n--- PCA Results ---")
    print(f"Variance explained by 1st component:  {explained_variance[0]*100:.2f}%")
    print(f"Variance explained by 2nd component: {explained_variance[1]*100:.2f}%")
    print(f"Variance explained by 10 components: {cumulative_variance[9]*100:.2f}%")
    
    # Find how many components are needed for 95% variance
    components_for_95 = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"Components needed for 95% variance: {components_for_95}")
    print("--------------------")

    # --- 5. Generate Plot 1: Explained Variance (Scree Plot) ---
    print("[INFO] Saving 'pca_explained_variance.png'...")
    plt.figure(figsize=(12, 7))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
    plt.axhline(y=0.95, color='r', linestyle=':', label='95% Explained Variance')
    plt.axhline(y=1.0, color='g', linestyle=':', label='100% Explained Variance')
    plt.title('PCA Cumulative Explained Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Variance (%)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('pca_explained_variance.png')
    plt.close()

    # --- 6. Generate Plot 2: 2D Visualization ---
    print("[INFO] Saving 'pca_2d_visualization.png'...")
    
    # Create a DataFrame with the first two components and the label
    pca_df = pd.DataFrame(data={
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'label': y
    })
    
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        data=pca_df,
        x='PC1',
        y='PC2',
        hue='label',
        palette=sns.color_palette('bright', n_colors=len(y.unique())),
        alpha=0.7
    )
    plt.title('2D PCA Visualization of Drowsiness Features')
    plt.xlabel(f'Principal Component 1 ({explained_variance[0]*100:.2f}%)')
    plt.ylabel(f'Principal Component 2 ({explained_variance[1]*100:.2f}%)')
    plt.legend(title='Drowsiness Label')
    plt.grid(True)
    plt.savefig('pca_2d_visualization.png')
    plt.close()

    print("\n[SUCCESS] PCA analysis complete. Check the saved .png files.")

if __name__ == "__main__":
    # Make sure you have the required libraries
    # pip install pandas numpy matplotlib seaborn scikit-learn
    run_pca_analysis(INPUT_FILE)