"""
train_model_xgboost.py
---------------------------------
XGBoost training pipeline that writes ALL .png diagrams and reports to a **hardcoded**
output directory. No CLI arguments required.

You can edit these constants at the top:
- CSV_FILE_PATH : path to your dataset
- OUTPUT_DIR    : folder where artifacts (.png, .txt, model) are saved

This script mirrors your original structure (load_data -> engineer_features -> visualize -> train) 
but changes all `plt.savefig(...)` calls to write into OUTPUT_DIR.
"""

import os
import sys
import numpy as np
import pandas as pd

# Headless backend so saving figures works without a display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

# ---------------------------
# HARD-CODED PATHS
# ---------------------------
CSV_FILE_PATH = r'C:\\Users\\devan\\Dropbox\\Aktif\\Germany\\TH Ingolstadt\\2. Semester\\PRJ Project\\My Project\\03 Team 2\\02 VScode\\drowsiness_dataset.csv'
OUTPUT_DIR    = r'C:\Users\devan\Dropbox\Aktif\Germany\TH Ingolstadt\2. Semester\PRJ Project\My Project\03 Team 2\02 VScode\artifacts 2\04_models xgb'  # <-- change to your preferred folder

# ---------------------------
# Try to import your project utilities; otherwise use fallbacks
# ---------------------------
try:
    from load_data import load_data as _load_data
    from feature_engineering import engineer_features as _engineer_features
except Exception:
    _load_data = None
    _engineer_features = None

# ---------------------------
# Helpers
# ---------------------------

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _savefig(path: str):
    try:
        _ensure_dir(os.path.dirname(path) or '.')
        plt.savefig(path, bbox_inches='tight')
        print(f"[INFO] Saved figure: {path}")
    except Exception as e:
        print(f"[ERROR] Could not save figure '{path}': {e}")
    finally:
        plt.close()


def _save_text(path: str, text: str):
    try:
        _ensure_dir(os.path.dirname(path) or '.')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"[INFO] Saved text file: {path}")
    except Exception as e:
        print(f"[ERROR] Could not save text file '{path}': {e}")

# ---------------------------
# Fallback implementations if your modules are unavailable
# ---------------------------

def load_data(csv_path: str):
    if _load_data is not None:
        return _load_data(csv_path)
    try:
        print(f"[INFO] Attempting to load data from: {csv_path}")
        df = pd.read_csv(csv_path)
        print("[INFO] Data loaded successfully.")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return None


def engineer_features(df: pd.DataFrame):
    if _engineer_features is not None:
        return _engineer_features(df)
    # Minimal fallback: expect a 'label' column
    target = 'label'
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found. Adjust your feature engineering or rename columns.")
    y = df[target]
    X = df.drop(columns=[target])
    return X, y

# ---------------------------
# Visualization (PNG outputs under OUTPUT_DIR)
# ---------------------------

def plot_label_distribution(y):
    if y is None:
        print("[WARN] No labels (y) to plot.")
        return
    print("\n[INFO] Generating label distribution plot...")
    plt.figure(figsize=(8, 6))
    sns.countplot(x=y)
    plt.title('Distribution of Drowsiness Labels (after windowing)')
    plt.xlabel('Drowsiness Label')
    plt.ylabel('Number of Windows')
    _savefig(os.path.join(OUTPUT_DIR, 'xgb_label_distribution.png'))


def plot_key_feature_distributions(X: pd.DataFrame):
    if X is None:
        print("[WARN] No features (X) to plot.")
        return
    print("\n[INFO] Generating key feature distribution plots...")
    key_features = [
        'HRV_RMSSD', 'HRV_SDNN', 'HRV_HF', 'HRV_LFHF',
        'HRV_DFA_alpha1', 'HRV_SampEn'
    ]
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
    _savefig(os.path.join(OUTPUT_DIR, 'xgb_key_feature_distributions.png'))

# ---------------------------
# Train & Evaluate (saves confusion matrix, feature importances, report)
# ---------------------------

def train_and_evaluate(X: pd.DataFrame, y: pd.Series):
    if X is None or y is None:
        print("[ERROR] No features (X) or labels (y) to train on. Exiting.")
        return

    print("\n[INFO] Starting model training and evaluation...")
    unique_classes = np.unique(y)
    print(f"- Found {len(unique_classes)} unique classes: {unique_classes}")
    if len(unique_classes) < 2:
        print("[ERROR] Cannot train a classifier. Need at least 2 classes.")
        return

    # Split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
    except ValueError:
        print("[WARN] Could not stratify split. Proceeding non-stratified.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

    # XGBoost config: binary vs multi-class
    num_classes = len(unique_classes)
    if num_classes > 2:
        model = XGBClassifier(
            n_estimators=200,
            random_state=42,
            eval_metric='mlogloss',
            tree_method='hist'
        )
        print("[INFO] Using multi-class setup (mlogloss).")
    else:
        model = XGBClassifier(
            n_estimators=200,
            random_state=42,
            eval_metric='logloss',
            tree_method='hist'
        )
        print("[INFO] Using binary setup (logloss).")

    print("[INFO] Training the XGBoost model...")
    model.fit(X_train, y_train)
    print("[INFO] Model training complete.")

    # 1) Classification report (save to OUTPUT_DIR)
    y_pred = model.predict(X_test)
    report_text = classification_report(y_test, y_pred, target_names=[str(c) for c in unique_classes], zero_division=0)
    print("\n--- 1. Classification Report ---\n" + report_text)
    _save_text(os.path.join(OUTPUT_DIR, 'xgb_classification_report.txt'), report_text)

    # 2) Confusion Matrix Heatmap
    print("\n[INFO] Generating confusion matrix heatmap...")
    cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
    cm_df = pd.DataFrame(cm, index=[f"True {c}" for c in unique_classes], columns=[f"Pred {c}" for c in unique_classes])
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('XGBoost Confusion Matrix Heatmap')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    _savefig(os.path.join(OUTPUT_DIR, 'xgb_confusion_matrix_heatmap.png'))

    # 3) Feature Importances
    print("\n--- 3. Top 10 Most Important Features (Gain) ---")
    try:
        importances = model.feature_importances_
    except AttributeError:
        importances = None
    if importances is not None:
        feature_importance_df = (
            pd.DataFrame({'feature': X.columns, 'importance': importances})
              .sort_values(by='importance', ascending=False)
        )
        print(feature_importance_df.head(10))
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20))
        plt.title('Top 20 XGBoost Feature Importances (Gain)')
        plt.tight_layout()
        _savefig(os.path.join(OUTPUT_DIR, 'xgb_feature_importances.png'))
    else:
        print("[WARN] Feature importances not available.")

    # Optionally save the trained model
    try:
        import joblib
        model_path = os.path.join(OUTPUT_DIR, 'xgb_model.joblib')
        joblib.dump(model, model_path)
        print(f"[INFO] Saved trained model: {model_path}")
    except Exception as e:
        print(f"[WARN] Could not save model: {e}")

# ---------------------------
# Main
# ---------------------------
if __name__ == '__main__':
    _ensure_dir(OUTPUT_DIR)

    print('=' * 8)
    print('Drowsiness Prediction Pipeline (XGBoost): START')
    print('=' * 8)

    raw_df = load_data(CSV_FILE_PATH)
    if raw_df is not None:
        try:
            X_features, y_labels = engineer_features(raw_df)
        except Exception as e:
            print(f"[ERROR] Feature engineering failed: {e}")
            print('[FAILURE] Cannot continue pipeline.')
            print('\n' + '=' * 8)
            print('Drowsiness Prediction Pipeline (XGBoost): END')
            print('=' * 8)
            sys.exit(1)

        # Visualize engineered data to PNGs in OUTPUT_DIR
        if X_features is not None and y_labels is not None:
            print("\n--- Engineered Data Visualization ---")
            plot_label_distribution(y_labels)
            plot_key_feature_distributions(X_features)

            # Train & evaluate
            train_and_evaluate(X_features, y_labels)
        else:
            print('[FAILURE] Feature engineering returned None. Cannot train model.')
    else:
        print('[FAILURE] Data loading failed. Cannot continue pipeline.')

    print('\n' + '=' * 8)
    print('Drowsiness Prediction Pipeline (XGBoost): END')
    print('=' * 8)
