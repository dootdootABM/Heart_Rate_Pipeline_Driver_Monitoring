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
    # test_size=0.3 means 30% of data is for testing, 70% for training
    # stratify=y ensures the class distribution (e.g., % of 0s, 1s, 2s) is the same in train and test sets
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
    # class_weight='balanced' automatically adjusts for the data imbalance
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    
    print("[INFO] Training the Random Forest model...")
    model.fit(X_train, y_train)
    print("[INFO] Model training complete.")

    # --- Evaluate the Model ---
    y_pred = model.predict(X_test)

    print("\n--- 1. Classification Report ---")
    # target_names should match your class labels [0, 1, 2]
    class_names = [str(c) for c in unique_classes]
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    print("\n--- 2. Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
    cm_df = pd.DataFrame(cm, index=[f"True {c}" for c in unique_classes], columns=[f"Pred {c}" for c in unique_classes])
    print(cm_df)

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
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importances.png')
    print("\n[INFO] Saved feature importance plot to 'feature_importances.png'")


# --- Main execution block ---
if __name__ == "__main__":
    print("==============================================")
    print("     Drowsiness Prediction Pipeline: START    ")
    print("==============================================")
    
    # --- Step 1: Load Data ---
    raw_data_df = load_data(CSV_FILE_PATH)

    # --- Step 2: Engineer Features ---
    # IMPORTANT: For this to work with your sample file,
    # you MUST edit feature_engineering.py and set:
    # WINDOW_SECONDS = 10 (or something small)
    if raw_data_df is not None:
        X_features, y_labels = engineer_features(raw_data_df)
        
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