"""
train_model_rf_combined.py

--------------------------------
Random Forest Pipeline for Combined/Selected Features Dataset.
Target Variable: 'drowsiness_label'
--------------------------------
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ================= Configuration =================

# Path to your combined dataset
CSV_FILE_NAME = r"src/drowsiness_detection_pkg/drowsiness_detection/heart_rate/Datasets/combined_training_dataset.csv"
CSV_FILE_PATH = os.path.abspath(CSV_FILE_NAME)

# Output directory for this specific model
OUTPUT_DIR = os.path.join(os.getcwd(), 'src', 'drowsiness_detection_pkg', 'drowsiness_detection', 'heart_rate', 'Results', 'RF_Results', 'Combined')

TARGET_COLUMN = 'drowsiness_label'

# ================= Helpers =================

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    print(f"[INFO] Output directory ready: {path}")

def savefig_safe(figpath: str):
    try:
        plt.savefig(figpath, bbox_inches='tight')
        print(f"[INFO] Saved figure to '{os.path.basename(figpath)}'")
    except Exception as e:
        print(f"[ERROR] Could not save figure: {e}")
    finally:
        plt.close()

# ================= Data Pipeline =================

def load_data(csv_path: str):
    print(f"[INFO] Loading data from {csv_path}...")
    if not os.path.exists(csv_path):
        # Fallback for local testing if relative path fails
        local_path = os.path.basename(csv_path)
        if os.path.exists(local_path):
            csv_path = local_path
        else:
            print(f"[ERROR] File not found at: {csv_path}")
            return None
    
    try:
        df = pd.read_csv(csv_path)
        print(f"[INFO] Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return None

def engineer_features(df: pd.DataFrame):
    # Remove metadata columns if they exist
    cols_to_remove = [TARGET_COLUMN, 'window_id', 'Timestamp', 'Time_Period_Label', 'Drowsiness_Label_Str']
    
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"TARGET_COLUMN '{TARGET_COLUMN}' not found in dataset. Available: {list(df.columns)}")

    y = df[TARGET_COLUMN]
    
    # Drop target and non-numeric metadata
    drop_actual = [c for c in cols_to_remove if c in df.columns]
    X = df.drop(columns=drop_actual)
    
    # Keep only numbers
    X = X.select_dtypes(include=[np.number])
    
    # Handle NaNs/Infs
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    print(f"[INFO] Features prepared. X: {X.shape}, y: {y.shape}")
    return X, y

# ================= Visualization =================

def plot_visuals(X: pd.DataFrame, y: pd.Series, output_dir: str):
    # 1. Label Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x=y)
    plt.title('Distribution of Drowsiness Labels')
    plt.xlabel('Label')
    plt.ylabel('Count')
    savefig_safe(os.path.join(output_dir, '01_label_distribution.png'))

    # 2. Key Feature Distributions
    # Check for HRV keys, otherwise plot top 3
    key_features = ['HRV_RMSSD', 'HRV_SDNN', 'raw_hr', 'Smooth_BPM']
    features_to_plot = [f for f in key_features if f in X.columns]
    if not features_to_plot:
        features_to_plot = X.columns[:3]

    if len(features_to_plot) > 0:
        plt.figure(figsize=(15, 5))
        for i, feature in enumerate(features_to_plot[:3]):
            plt.subplot(1, 3, i + 1)
            sns.histplot(X[feature], kde=True, bins=20)
            plt.title(f'Dist: {feature}')
        plt.tight_layout()
        savefig_safe(os.path.join(output_dir, '02_feature_distributions.png'))

# ================= Training =================

def train_and_evaluate(X: pd.DataFrame, y: pd.Series, output_dir: str):
    unique_classes = np.unique(y)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print('[INFO] Training Random Forest...')
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Save Predictions
    results_df = X_test.copy()
    results_df['Actual_Label'] = y_test
    results_df['Predicted_Label'] = y_pred
    results_df.to_csv(os.path.join(output_dir, 'test_set_predictions.csv'), index=False)
    
    # Feature Importance
    importances = model.feature_importances_
    fi_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    fi_df.to_csv(os.path.join(output_dir, 'feature_importances.csv'), index=False)
    
    # Save Report
    report = classification_report(y_test, y_pred, zero_division=0)
    with open(os.path.join(output_dir, '00_classification_report.txt'), 'w') as f:
        f.write(report)
        
    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=unique_classes, yticklabels=unique_classes)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    savefig_safe(os.path.join(output_dir, '03_confusion_matrix.png'))
    
    # Feature Importance Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=fi_df.head(10))
    plt.title('Top 10 Feature Importances')
    plt.tight_layout()
    savefig_safe(os.path.join(output_dir, '04_feature_importances.png'))
    
    # Save Model
    joblib.dump(model, os.path.join(output_dir, 'rf_model_combined.joblib'))
    print(f"[SUCCESS] Model saved to {output_dir}")

# ================= Main =================

if __name__ == '__main__':
    print("=== RF Training: Combined Dataset ===")
    _ensure_dir(OUTPUT_DIR)
    
    df = load_data(CSV_FILE_PATH)
    if df is not None:
        try:
            X, y = engineer_features(df)
            plot_visuals(X, y, OUTPUT_DIR)
            train_and_evaluate(X, y, OUTPUT_DIR)
        except Exception as e:
            print(f"[CRITICAL ERROR] {e}")
            import traceback
            traceback.print_exc()
