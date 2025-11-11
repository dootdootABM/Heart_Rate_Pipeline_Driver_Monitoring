"""
train_model.py
----------------
Random Forest training pipeline with a **hardcoded output path**.
- Set `CSV_FILE_PATH` to your dataset
- Set `OUTPUT_DIR` to where you want all artifacts written
- Optionally adjust `TARGET_COLUMN`

No CLI args are used.
"""

import os
import sys
import numpy as np
import pandas as pd

# Use a non-interactive backend for safe saving without GUI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------
# HARD-CODED PATHS / SETTINGS
# ---------------------------
CSV_FILE_PATH = r'C:\\Users\\devan\\Dropbox\\Aktif\\Germany\\TH Ingolstadt\\2. Semester\\PRJ Project\\My Project\\03 Team 2\\02 VScode\\drowsiness_dataset.csv'
OUTPUT_DIR    = r'C:\Users\devan\Dropbox\Aktif\Germany\TH Ingolstadt\2. Semester\PRJ Project\My Project\03 Team 2\02 VScode\artifacts 2\03_model_rf'  # <— change this to your preferred folder
TARGET_COLUMN = 'label'                      # <— change if your target column is named differently

# If you prefer relative to project root, you can use:
# OUTPUT_DIR = os.path.join(os.getcwd(), 'outputs_rf')

# ---------------------------
# Imports from your project (optional)
# ---------------------------
# If you already have these modules, they will be used. Otherwise, the
# fallback implementations below will run.
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


def savefig_safe(figpath: str):
    """Safely save current Matplotlib figure to figpath, logging any errors."""
    try:
        _ensure_dir(os.path.dirname(figpath) or '.')
        plt.savefig(figpath, bbox_inches='tight')
        print(f"[INFO] Saved figure to '{figpath}'")
    except Exception as e:
        print(f"[ERROR] Could not save figure '{figpath}': {e}")
    finally:
        plt.close()

# ---------------------------
# Fallback data/feature functions (used if project modules not found)
# ---------------------------

def load_data(csv_path: str):
    if _load_data is not None:
        return _load_data(csv_path)
    try:
        print(f"[INFO] Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"[INFO] Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return None


def engineer_features(df: pd.DataFrame):
    if _engineer_features is not None:
        return _engineer_features(df)
    # Minimal fallback: split by TARGET_COLUMN
    if TARGET_COLUMN not in df.columns:
        raise KeyError(
            f"TARGET_COLUMN '{TARGET_COLUMN}' not found. Update TARGET_COLUMN or preprocess your data.")
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])
    print(f"[INFO] Features and labels prepared. X: {X.shape}, y: {y.shape}")
    return X, y

# ---------------------------
# Visualization
# ---------------------------

def plot_label_distribution(y: pd.Series, output_dir: str):
    if y is None:
        print('[WARN] No labels (y) to plot.')
        return
    print('\n[INFO] Generating label distribution plot...')
    plt.figure(figsize=(8, 6))
    sns.countplot(x=y)
    plt.title('Distribution of Drowsiness Labels (after windowing)')
    plt.xlabel('Label')
    plt.ylabel('Count')
    savefig_safe(os.path.join(output_dir, 'rf_label_distribution.png'))


def plot_key_feature_distributions(X: pd.DataFrame, output_dir: str):
    if X is None:
        print('[WARN] No features (X) to plot.')
        return
    print('\n[INFO] Generating key feature distribution plots...')
    key_features = [
        'HRV_RMSSD', 'HRV_SDNN', 'HRV_HF', 'HRV_LFHF', 'HRV_DFA_alpha1', 'HRV_SampEn'
    ]
    features_to_plot = [f for f in key_features if f in X.columns]
    if not features_to_plot:
        print('[WARN] None of the predefined key features found in X; skipping.')
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
    savefig_safe(os.path.join(output_dir, 'rf_key_feature_distributions.png'))

# ---------------------------
# Train & Evaluate
# ---------------------------

def train_and_evaluate(X: pd.DataFrame, y: pd.Series, output_dir: str):
    if X is None or y is None:
        print('[ERROR] No features (X) or labels (y).')
        return

    print('\n[INFO] Starting model training and evaluation...')
    unique_classes = np.unique(y)
    print(f"- Found {len(unique_classes)} unique classes: {unique_classes}")
    if len(unique_classes) < 2:
        print('[ERROR] Need at least 2 classes to train a classifier.')
        return

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
    except ValueError:
        print('[WARN] Stratified split failed (likely small/imbalanced). Using non-stratified split.')
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    print('[INFO] Training Random Forest...')
    model.fit(X_train, y_train)
    print('[INFO] Model training complete.')

    y_pred = model.predict(X_test)
    print("\n--- 1. Classification Report ---")
    report_text = classification_report(y_test, y_pred, target_names=[str(c) for c in unique_classes], zero_division=0)
    print(report_text)

    _ensure_dir(output_dir)
    with open(os.path.join(output_dir, 'rf_classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report_text)

    # Confusion matrix heatmap
    print('\n[INFO] Generating confusion matrix heatmap...')
    cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
    cm_df = pd.DataFrame(cm, index=[f'True {c}' for c in unique_classes], columns=[f'Pred {c}' for c in unique_classes])

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Greens')
    plt.title('Random Forest Confusion Matrix Heatmap')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    savefig_safe(os.path.join(output_dir, 'rf_confusion_matrix_heatmap.png'))

    # Feature importances
    print('\n--- 3. Top 10 Most Important Features ---')
    importances = model.feature_importances_
    feature_names = X.columns
    feature_importance_df = (
        pd.DataFrame({'feature': feature_names, 'importance': importances})
          .sort_values(by='importance', ascending=False)
    )
    print(feature_importance_df.head(10))

    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20))
    plt.title('Top 20 Random Forest Feature Importances')
    plt.tight_layout()
    savefig_safe(os.path.join(output_dir, 'rf_feature_importances.png'))

    # Save model (optional)
    try:
        import joblib
        model_path = os.path.join(output_dir, 'rf_model.joblib')
        joblib.dump(model, model_path)
        print(f"[INFO] Saved trained model to '{model_path}'")
    except Exception as e:
        print(f"[WARN] Could not save model: {e}")

# ---------------------------
# Main
# ---------------------------
if __name__ == '__main__':
    _ensure_dir(OUTPUT_DIR)

    print('=' * 8)
    print('Drowsiness Prediction Pipeline: START')
    print('=' * 8)

    df = load_data(CSV_FILE_PATH)
    if df is not None:
        try:
            X, y = engineer_features(df)
        except Exception as e:
            print(f"[ERROR] Feature engineering failed: {e}")
            print('[FAILURE] Feature engineering failed. Cannot train model.')
            print('\n' + '=' * 8)
            print('Drowsiness Prediction Pipeline: END')
            print('=' * 8)
            sys.exit(1)

        if X is not None and y is not None:
            # Visualizations
            plot_label_distribution(y, OUTPUT_DIR)
            plot_key_feature_distributions(X, OUTPUT_DIR)
            # Training
            train_and_evaluate(X, y, OUTPUT_DIR)
        else:
            print('[FAILURE] Feature engineering returned None.')
    else:
        print('[FAILURE] Data loading failed. Cannot continue pipeline.')

    print('\n' + '=' * 8)
    print('Drowsiness Prediction Pipeline: END')
    print('=' * 8)
