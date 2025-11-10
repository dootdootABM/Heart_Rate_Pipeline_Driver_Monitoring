import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Import functions from your other scripts ---
try:
    from load_data import load_data
    from feature_engineering import engineer_features
except ImportError:
    print("[ERROR] Could not import from 'load_data.py' or 'feature_engineering.py'.")
    print("        Make sure all three scripts are in the same directory.")
    sys.exit(1)

# --- CONFIGURATION: SET THE FILE PATH ---
# This must match the path in your other scripts
CSV_FILE_PATH = r'F:\Users\Aryan\Documents\IAE-M\THI_Notes_Files\Group Project\Source Code\Data_Files\drowsiness_dataset.csv'

# --- 
# NOTE: We must use a smaller window for the sample file to get >1 window.
# For real analysis, you would change this back to 60 or 120.
# This value will be SET inside the feature_engineering.py file.
# Manually go into feature_engineering.py and set:
# WINDOW_SECONDS = 10 
# ---

# --- [NEW] VISUALIZATION FUNCTIONS FOR ENGINEERED DATA ---

def plot_label_distribution(y):
    """
    Plots a count plot of the final labels to check for class balance.
    """
    if y is None:
        print("[WARN] No labels (y) to plot.")
        return
        
    print("\n[INFO] Generating label distribution plot...")
    plt.figure(figsize=(8, 6))
    sns.countplot(x=y)
    plt.title('Distribution of Drowsiness Labels (after windowing)')
    plt.xlabel('Drowsiness Label')
    plt.ylabel('Number of Windows')
    try:
        # [MODIFIED] Save with 'rf_' prefix
        plt.savefig('rf_label_distribution.png')
        print("[INFO] Saved label distribution plot to 'rf_label_distribution.png'")
    except Exception as e:
        print(f"[ERROR] Could not save rf_label_distribution.png: {e}")
    plt.close() # Close plot to save memory

def plot_key_feature_distributions(X):
    """
    Plots histograms for a subset of key engineered features.
    """
    if X is None:
        print("[WARN] No features (X) to plot.")
        return

    print("\n[INFO] Generating key feature distribution plots...")
    # Select a few key features you've identified as important
    key_features = [
        'HRV_RMSSD', 'HRV_SDNN', 'HRV_HF', 'HRV_LFHF', 
        'HRV_DFA_alpha1', 'HRV_SampEn'
    ]
    # Filter to only features that actually exist in the dataframe
    features_to_plot = [f for f in key_features if f in X.columns]
    
    if not features_to_plot:
        print("[WARN] None of the pre-selected key features were found. Skipping distribution plots.")
        return

    num_plots = len(features_to_plot)
    num_cols = 3
    num_rows = (num_plots + num_cols - 1) // num_cols
    
    plt.figure(figsize=(15, 5 * num_rows))
    for i, feature in enumerate(features_to_plot):
        plt.subplot(num_rows, num_cols, i + 1)
        sns.histplot(X[feature], kde=True, bins=30)
        plt.title(f'Distribution of {feature}')
    
    plt.tight_layout()
    try:
        # [MODIFIED] Save with 'rf_' prefix
        plt.savefig('rf_key_feature_distributions.png')
        print("[INFO] Saved key feature distributions to 'rf_key_feature_distributions.png'")
    except Exception as e:
        print(f"[ERROR] Could not save rf_key_feature_distributions.png: {e}")
    plt.close()

# --- END OF NEW FUNCTIONS ---


def train_and_evaluate(X, y):
    """
    Trains a Random Forest classifier and prints evaluation metrics.
    """
    if X is None or y is None:
        print("[ERROR] No features (X) or labels (y) to train on. Exiting.")
        return

    print("\n[INFO] Starting model training and evaluation...")

    # Check for number of unique classes
    unique_classes = np.unique(y)
    print(f"       - Found {len(unique_classes)} unique classes: {unique_classes}")

    # Handle the case where the dataset is too small or only has one class
    if len(unique_classes) < 2:
        print(f"[ERROR] Cannot train a classifier. Need at least 2 different classes in the labels, but only found 1.")
        return

    # Split the data into training and testing sets
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
    except ValueError:
        print(f"[WARN] Could not stratify split. Dataset is likely too small.")
        print("       Proceeding with a non-stratified split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
    
    print(f"       - Training set size: {X_train.shape[0]} samples")
    print(f"       - Testing set size: {X_test.shape[0]} samples")

    # --- Initialize and Train the Model ---
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    
    print("[INFO] Training the Random Forest model...")
    model.fit(X_train, y_train)
    print("[INFO] Model training complete.")

    # --- Evaluate the Model ---
    y_pred = model.predict(X_test)

    print("\n--- 1. Classification Report ---")
    class_names = [str(c) for c in unique_classes]
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    print("\n--- 2. Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
    cm_df = pd.DataFrame(cm, index=[f"True {c}" for c in unique_classes], columns=[f"Pred {c}" for c in unique_classes])
    print(cm_df)

    # --- [NEW] Plot Confusion Matrix Heatmap ---
    print("\n[INFO] Generating confusion matrix heatmap...")
    plt.figure(figsize=(10, 7))
    # [MODIFIED] Using a different color map (Greens) to distinguish from XGBoost
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Greens') 
    plt.title('Random Forest Confusion Matrix Heatmap')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    try:
        # [MODIFIED] Save with 'rf_' prefix
        plt.savefig('rf_confusion_matrix_heatmap.png')
        print("[INFO] Saved confusion matrix heatmap to 'rf_confusion_matrix_heatmap.png'")
    except Exception as e:
        print(f"[ERROR] Could not save rf_confusion_matrix_heatmap.png: {e}")
    plt.close()
    # --- End of new plot ---


    # --- 3. Feature Importance ---
    print("\n--- 3. Top 10 Most Important Features ---")
    importances = model.feature_importances_
    feature_names = X.columns
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    
    print(feature_importance_df.head(10))

    # Plot feature importances
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20))
    # [MODIFIED] Title changed
    plt.title('Top 20 Random Forest Feature Importances') 
    plt.tight_layout()
    # [MODIFIED] Save with 'rf_' prefix
    plt.savefig('rf_feature_importances.png') 
    print("\n[INFO] Saved feature importance plot to 'rf_feature_importances.png'")
    plt.close()


# --- Main execution block ---
if __name__ == "__main__":
    print("==============================================")
    print("     Drowsiness Prediction Pipeline: START    ")
    print("==============================================")
    
    # --- Step 1: Load Data ---
    raw_data_df = load_data(CSV_FILE_PATH)

    # --- Step 2: Engineer Features ---
    if raw_data_df is not None:
        X_features, y_labels = engineer_features(raw_data_df)
        
        # --- [NEW] Step 2.5: Visualize Engineered Data ---
        if X_features is not None and y_labels is not None:
            print("\n--- Starting Engineered Data Visualization ---")
            plot_label_distribution(y_labels)
            plot_key_feature_distributions(X_features)
        
        # --- Step 3: Train Model ---
        if X_features is not None and y_labels is not None:
            train_and_evaluate(X_features, y_labels)
        else:
            print("\n[FAILURE] Feature engineering failed. Cannot train model.")
    else:
        print("\n[FAILURE] Data loading failed. Cannot continue pipeline.")
    
    print("\n==============================================")
    print("     Drowsiness Prediction Pipeline: END      ")
    print("==============================================")