import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
# --- CHANGED: Import XGBClassifier instead of RandomForestClassifier ---
from xgboost import XGBClassifier
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
    Trains an XGBoost classifier and prints evaluation metrics.
    """
    if X is None or y is None:
        print("[ERROR] No features (X) or labels (y) to train on. Exiting.")
        return

    print("\n[INFO] Starting model training and evaluation...")

    # Check for number of unique classes
    unique_classes = np.unique(y)
    print(f"        - Found {len(unique_classes)} unique classes: {unique_classes}")

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
        print("        Proceeding with a non-stratified split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
    
    print(f"        - Training set size: {X_train.shape[0]} samples")
    print(f"        - Testing set size: {X_test.shape[0]} samples")

    # --- Initialize and Train the Model (XGBoost) ---
    # XGBoost is not directly compatible with 'class_weight="balanced"'.
    # We use 'scale_pos_weight' or calculate 'sample_weight' for binary classification.
    # For multi-class (like 0, 1, 2), we focus on common parameters:
    
    # max_depth: Max depth of a tree.
    # learning_rate (eta): Step size shrinkage to prevent overfitting.
    # n_estimators: Number of boosting rounds (trees).
    # objective: 'multi:softmax' for multi-class classification, or 'binary:logistic' for binary.
    
    # Since y contains unique_classes [0, 1, 2], we use 'multi:softmax'
    num_classes = len(unique_classes)
    
    if num_classes > 2:
        xgb_objective = 'multi:softmax'
        # XGBoost requires num_class parameter for multi-class objective
        xgb_params = {'num_class': num_classes}
        print(f"[INFO] Using multi-class objective: {xgb_objective}")
    else: # Binary classification (0 and 1)
        xgb_objective = 'binary:logistic'
        xgb_params = {}
        print(f"[INFO] Using binary objective: {xgb_objective}")
        # To emulate Random Forest's 'balanced' weight, you could calculate scale_pos_weight
        # scale_pos_weight = (count of negative examples) / (count of positive examples)
        # However, for simplicity and a direct conversion, we'll omit it for now.
    
    model = XGBClassifier(
        n_estimators=100,
        random_state=42,
        objective=xgb_objective,
        use_label_encoder=False, # Suppress a common XGBoost warning
        eval_metric='logloss',   # Set a default evaluation metric
        **xgb_params             # Include num_class for multi-class
    )
    
    print("[INFO] Training the XGBoost model...")
    # NOTE: XGBoost often works best with numerical labels starting from 0.
    # Your code already seems to handle this with unique_classes [0, 1, 2].
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

    # --- 3. Feature Importance ---
    print("\n--- 3. Top 10 Most Important Features (Gain) ---")
    # XGBoost default is 'gain', which is highly effective
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
    plt.title('Top 20 XGBoost Feature Importances (Gain)')
    plt.tight_layout()
    plt.savefig('xgb_feature_importances.png')
    print("\n[INFO] Saved feature importance plot to 'xgb_feature_importances.png'")


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