"""
main.py

---------------------------------
Combined execution of:
1. load_data.py
2. feature_engineering.py
3. PCA_datasets.py

Fixes:
- Resolves MemoryError in seaborn.histplot by enforcing explicit bin counts.
- Excludes 'Timestamp' columns from numerical distribution plots.
- Retains all outputs and plots.
---------------------------------
"""

import pandas as pd
import numpy as np
import ast
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ================= Configuration =================

BASE_DIR = r'src/drowsiness_detection_pkg/drowsiness_detection/heart_rate'
DATASET_DIR = os.path.join(BASE_DIR, 'Datasets')
RESULT_DIR = os.path.join(BASE_DIR, 'Results')

# Step 1 Paths
INPUT_RAW_METRICS = os.path.join(DATASET_DIR, 'session_metrics.csv')
OUTPUT_REFINED_DATA = os.path.join(DATASET_DIR, 'refined_data.csv')
DIR_REFINED_PLOTS = os.path.join(RESULT_DIR, 'Refined_Data')

# Step 2 Paths
OUTPUT_ENGINEERED_DATA = os.path.join(DATASET_DIR, 'engineered_features.csv')
DIR_ENGINEERED_PLOTS = os.path.join(RESULT_DIR, 'Engineered_Features')

# Step 3 Paths
OUTPUT_PCA_TRAIN = os.path.join(DATASET_DIR, 'PCA_training_dataset.csv')
OUTPUT_SELECTED_FEATURES = os.path.join(DATASET_DIR, 'combined_training_dataset.csv')
OUTPUT_LOADINGS = os.path.join(DATASET_DIR, 'pca_loadings_matrix.csv')
DIR_PCA_PLOTS = os.path.join(RESULT_DIR, 'PCA_Results')

# Constants
START_TIME = pd.Timestamp("2025-12-15 15:00:00")
WINDOW_DURATION_SEC = 60
LABEL_COLUMN = 'drowsiness_label'
FIXED_COLS_DATASET2 = ["Time_Period", "raw_hr", "Smooth_BPM"]

# ================= Data Helpers =================

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def parse_array_string(array_str):
    try:
        if isinstance(array_str, list): return array_str
        if isinstance(array_str, str): return ast.literal_eval(array_str)
        return []
    except (ValueError, SyntaxError):
        return []

def get_time_period(dt):
    hour = dt.hour
    if 6 <= hour < 13: return 1
    elif 13 <= hour < 18: return 2
    elif 18 <= hour < 22: return 3
    else: return 4

def get_unique_consecutive_values(hr_list):
    if not hr_list: return []
    series = pd.Series(hr_list)
    return series[series != series.shift()].tolist()

def calculate_smooth_bpm_per_window(hr_list, window_size=3):
    if not hr_list: return []
    smooth_values = []
    for i in range(0, len(hr_list), window_size):
        chunk = hr_list[i:i + window_size]
        avg = sum(chunk) / len(chunk)
        smooth_values.append(avg)
    return smooth_values

def calculate_hrv_numpy(bpm_values):
    bpm_safe = np.array(bpm_values, dtype=float)
    bpm_safe = bpm_safe[bpm_safe > 0]
    if len(bpm_safe) < 2: return None
    
    rr_intervals = 60000.0 / bpm_safe
    mean_nn = np.mean(rr_intervals)
    sdnn = np.std(rr_intervals, ddof=1)
    diff_rr = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(diff_rr ** 2))
    sdsd = np.std(diff_rr, ddof=1)
    nn50 = np.sum(np.abs(diff_rr) > 50)
    pnn50 = (nn50 / len(diff_rr)) * 100 if len(diff_rr) > 0 else 0
    cvnn = sdnn / mean_nn if mean_nn > 0 else 0
    min_nn = np.min(rr_intervals)
    max_nn = np.max(rr_intervals)
    
    return {
        'HRV_MeanNN': mean_nn, 'HRV_SDNN': sdnn, 'HRV_RMSSD': rmssd,
        'HRV_SDSD': sdsd, 'HRV_CVNN': cvnn, 'HRV_pNN50': pnn50,
        'HRV_MinNN': min_nn, 'HRV_MaxNN': max_nn
    }

def convert_duration_to_seconds(val):
    if pd.isna(val): return 0
    if isinstance(val, (int, float)): return val
    if isinstance(val, pd.Timedelta): return val.total_seconds()
    try:
        dt = pd.to_timedelta(str(val))
        return dt.total_seconds()
    except:
        return 0

# ================= PLOTTING FUNCTIONS (Fixed for Memory Safety) =================

def prepare_plot_df(df):
    plot_df = df.copy()
    
    # Label Maps
    if 'Time_Period' in plot_df.columns:
        tp_map = {1: "1 (Morning)", 2: "2 (Afternoon)", 3: "3 (Evening)", 4: "4 (Night)"}
        plot_df['Time_Period_Label'] = plot_df['Time_Period'].map(tp_map).fillna(plot_df['Time_Period'].astype(str))
    
    if 'drowsiness_label' in plot_df.columns:
        dl_map = {1: "1 (Low)", 2: "2 (Moderate)", 3: "3 (High)"}
        plot_df['Drowsiness_Label_Str'] = plot_df['drowsiness_label'].map(dl_map).fillna("Unknown")

    if 'Timestamp' in plot_df.columns and not pd.api.types.is_datetime64_any_dtype(plot_df['Timestamp']):
        plot_df['Timestamp'] = pd.to_datetime(plot_df['Timestamp'])
        
    return plot_df

def plot_numerical_distributions(df, numeric_cols, output_dir):
    """
    Plots distributions.
    FIX: explicitly sets bins=30 to avoid MemoryError with large timestamp/numeric ranges.
    """
    valid_cols = [c for c in numeric_cols if c not in ['Timestamp', 'window_id']]
    if len(valid_cols) == 0: return

    num_plots = len(valid_cols)
    num_cols_grid = 3
    num_rows_grid = math.ceil(num_plots / num_cols_grid)

    fig, axes = plt.subplots(nrows=num_rows_grid, ncols=num_cols_grid, figsize=(15, 4 * num_rows_grid))
    axes = axes.flatten() if num_plots > 1 else [axes]
    
    for i, col in enumerate(valid_cols):
        data = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(data) == 0: continue
        
        # Use explicit bins to prevent memory explosion
        sns.histplot(data, bins=30, kde=True, ax=axes[i])
        axes[i].set_title(f'Dist: {col}')
    
    for j in range(i + 1, len(axes)): fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'numerical_distributions.png'))
    plt.close()

def plot_numerical_boxplots(df, numeric_cols, output_dir):
    """Plots boxplots. Skips 'Timestamp' and 'window_id'."""
    valid_cols = [c for c in numeric_cols if c not in ['Timestamp', 'window_id']]
    if len(valid_cols) == 0: return

    num_plots = len(valid_cols)
    num_cols_grid = 3
    num_rows_grid = math.ceil(num_plots / num_cols_grid)

    fig, axes = plt.subplots(nrows=num_rows_grid, ncols=num_cols_grid, figsize=(15, 4 * num_rows_grid))
    axes = axes.flatten() if num_plots > 1 else [axes]
    
    for i, col in enumerate(valid_cols):
        data = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(data) == 0: continue
        
        sns.boxplot(y=data, ax=axes[i])
        axes[i].set_title(f'Box: {col}')
    
    for j in range(i + 1, len(axes)): fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'numerical_boxplots.png'))
    plt.close()

def plot_pairplot(df, numeric_cols, output_dir):
    """Plots pairplot. Limits feature count to avoid overcrowding."""
    valid_cols = [c for c in numeric_cols if c not in ['Timestamp', 'window_id']]
    
    # Cap at 10 features for pairplot to keep it readable and performant
    if len(valid_cols) > 1:
        cols_to_plot = valid_cols[:10]
        g = sns.pairplot(df[cols_to_plot].dropna(), corner=True)
        g.fig.suptitle('Pairplot of Features (Top 10)', y=1.02)
        plt.savefig(os.path.join(output_dir, 'pairplot.png'), bbox_inches='tight')
        plt.close()

def plot_common_graphs(df, output_dir):
    """Standard plots used in both Step 1 and Step 2"""
    _ensure_dir(output_dir)
    plot_df = prepare_plot_df(df)
    
    # Select numeric columns but exclude Timestamp from the automatic list
    numeric_cols = plot_df.select_dtypes(include=['number']).columns.tolist()
    if 'Timestamp' in numeric_cols: numeric_cols.remove('Timestamp')

    # 1. HR Over Time
    if 'Timestamp' in plot_df.columns and 'raw_hr' in plot_df.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(plot_df['Timestamp'], plot_df['raw_hr'], label='Raw Heart Rate', marker='o', color='red', alpha=0.5)
        if 'Smooth_BPM' in plot_df.columns:
            plt.plot(plot_df['Timestamp'], plot_df['Smooth_BPM'], label='Smooth BPM', linestyle='--', color='blue', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('Heart Rate (BPM)')
        plt.title('Heart Rate (Raw vs Smooth) over Drive Duration')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'plot_hr_over_time.png'))
        plt.close()

    # 2. Drowsiness Distribution
    if 'Drowsiness_Label_Str' in plot_df.columns:
        order = sorted(plot_df['Drowsiness_Label_Str'].unique())
        plt.figure(figsize=(8, 6))
        sns.countplot(data=plot_df, x='Drowsiness_Label_Str', order=order)
        plt.title('Distribution of Drowsiness Labels')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'plot_drowsiness_dist.png'))
        plt.close()

    # 3. HR by Label
    if 'raw_hr' in plot_df.columns and 'Drowsiness_Label_Str' in plot_df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=plot_df, x='Drowsiness_Label_Str', y='raw_hr', order=sorted(plot_df['Drowsiness_Label_Str'].unique()))
        plt.title('Heart Rate Distribution by Drowsiness Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'plot_hr_by_label.png'))
        plt.close()

    # 4. Correlation Heatmap
    if len(numeric_cols) > 1:
        plt.figure(figsize=(12, 10))
        corr = plot_df[numeric_cols].corr().fillna(0)
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
        plt.close()

    # 5. Explicitly Called Restored Plots (Safe versions)
    plot_numerical_distributions(plot_df, numeric_cols, output_dir)
    plot_numerical_boxplots(plot_df, numeric_cols, output_dir)
    plot_pairplot(plot_df, numeric_cols, output_dir)
    
    print(f"[INFO] Plots saved to {output_dir}")

def plot_pca_explained_variance(pca_model, output_dir):
    explained_variance = pca_model.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
    plt.axhline(y=0.95, color='r', linestyle=':', label='95% Variance')
    plt.title('PCA Explained Variance (Aggregated Data)')
    plt.xlabel('Principal Component')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_explained_variance.png'))
    plt.close()

def plot_pca_2d(X_pca, y, output_dir, variance_ratio):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='viridis', alpha=0.7)
    plt.title(f'PCA: PC1 ({variance_ratio[0]:.1%}) vs PC2 ({variance_ratio[1]:.1%})')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_2d_visualization.png'))
    plt.close()

def plot_pca_feature_importance(pca_model, feature_names, output_folder, top_n=10):
    os.makedirs(output_folder, exist_ok=True)
    loadings_abs = np.abs(pca_model.components_)
    n_components = loadings_abs.shape[0]
    
    for i in range(n_components):
        pc_name = f"PC{i+1}"
        vals = loadings_abs[i, :]
        s = pd.Series(vals, index=feature_names).sort_values(ascending=False)
        top = s.head(top_n)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x=top.values, y=top.index, hue=top.index, palette="rocket", legend=False)
        plt.title(f"Feature Importance ({pc_name}) - Absolute PCA Loading")
        plt.xlabel("Absolute loading magnitude")
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"pca_feature_importance_{pc_name}.png"))
        plt.close()

# ================= Execution Steps =================

def step_1_load_data():
    print("\n=== STEP 1: LOAD DATA ===")
    if not os.path.exists(INPUT_RAW_METRICS):
        print(f"[ERROR] Could not find {INPUT_RAW_METRICS}")
        sys.exit(1)

    df = pd.read_csv(INPUT_RAW_METRICS)
    required_cols = ['window_id', 'raw_hr', 'Indirajithu_drowsiness_level']
    df_subset = df[required_cols].copy()
    
    df_subset.rename(columns={'Indirajithu_drowsiness_level': 'drowsiness_label'}, inplace=True)
    drowsiness_map = {'Low': 1, 'Moderate': 2, 'High': 3}
    df_subset['drowsiness_label'] = df_subset['drowsiness_label'].map(drowsiness_map)
    
    df_subset['raw_hr'] = df_subset['raw_hr'].apply(parse_array_string)
    df_subset['raw_hr'] = df_subset['raw_hr'].apply(get_unique_consecutive_values)
    df_subset['smooth_bpm_list'] = df_subset['raw_hr'].apply(lambda x: calculate_smooth_bpm_per_window(x, window_size=3))
    
    new_rows = []
    grouped = df_subset.groupby('window_id', sort=False)
    current_window_start = START_TIME
    
    for window_id, group in grouped:
        row = group.iloc[0]
        raw_hrs = row['raw_hr']
        smooth_bpms = row['smooth_bpm_list']
        label = row['drowsiness_label']
        n_samples = len(raw_hrs)
        
        if n_samples > 0:
            deltas = np.linspace(0, WINDOW_DURATION_SEC, n_samples, endpoint=False)
            timestamps = [current_window_start + pd.Timedelta(seconds=d) for d in deltas]
            
            window_df = pd.DataFrame({
                'Timestamp': timestamps, 'window_id': window_id,
                'raw_hr': raw_hrs, 'drowsiness_label': label
            })
            
            smooth_col = []
            for i in range(n_samples):
                smooth_idx = i // 3
                val = smooth_bpms[smooth_idx] if smooth_idx < len(smooth_bpms) else np.nan
                smooth_col.append(val)
            
            window_df['Smooth_BPM'] = smooth_col
            new_rows.append(window_df)
        
        current_window_start += pd.Timedelta(seconds=WINDOW_DURATION_SEC)
    
    df_final = pd.concat(new_rows, ignore_index=True)
    df_final['Drive_Duration'] = df_final['Timestamp'] - START_TIME
    df_final['Time_Period'] = df_final['Timestamp'].apply(get_time_period)
    
    final_cols = ["Timestamp", "Time_Period", "Drive_Duration", "window_id", "raw_hr", "Smooth_BPM", "drowsiness_label"]
    df_final = df_final[final_cols]
    
    print(f"[INFO] Saving refined data to {OUTPUT_REFINED_DATA}...")
    _ensure_dir(os.path.dirname(OUTPUT_REFINED_DATA))
    df_final.to_csv(OUTPUT_REFINED_DATA, index=False)
    
    # Plot Step 1 Results
    plot_common_graphs(df_final, DIR_REFINED_PLOTS)
    
    return df_final

def step_2_feature_engineering(df):
    print("\n=== STEP 2: FEATURE ENGINEERING ===")
    
    hrv_results = []
    grouped = df.groupby('window_id')
    
    for window_id, group in grouped:
        bpm_vals = group['raw_hr'].values
        metrics = calculate_hrv_numpy(bpm_vals)
        if metrics:
            metrics['window_id'] = window_id
            hrv_results.append(metrics)
        else:
            hrv_results.append({'window_id': window_id})
            
    hrv_df = pd.DataFrame(hrv_results)
    df_merged = pd.merge(df, hrv_df, on='window_id', how='left')
    
    hrv_cols = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_SDSD', 'HRV_CVNN', 'HRV_pNN50', 'HRV_MinNN', 'HRV_MaxNN']
    for col in hrv_cols:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].fillna(0)
            
    print(f"[INFO] Saving engineered data to {OUTPUT_ENGINEERED_DATA}...")
    _ensure_dir(os.path.dirname(OUTPUT_ENGINEERED_DATA))
    df_merged.to_csv(OUTPUT_ENGINEERED_DATA, index=False)
    
    # Plot Step 2 Results
    plot_common_graphs(df_merged, DIR_ENGINEERED_PLOTS)
    
    return df_merged

def step_3_pca(df):
    print("\n=== STEP 3: PCA ANALYSIS ===")
    _ensure_dir(DIR_PCA_PLOTS)
    
    if 'Drive_Duration' in df.columns:
        df['Drive_Duration'] = df['Drive_Duration'].apply(convert_duration_to_seconds)

    # --- PCA Part 1 (Aggregated) ---
    print("[PART 1] PCA on Aggregated Data...")
    agg_dict = {col: 'first' for col in df.columns if col not in ['raw_hr', 'window_id']}
    agg_dict['raw_hr'] = 'mean'
    df_agg = df.groupby('window_id').agg(agg_dict).reset_index()
    df_agg.dropna(inplace=True)
    
    cols_drop = ['window_id', 'Timestamp', 'Time_Period_Label', 'Drowsiness_Label_Str', 'Time_Period', 'Smooth_BPM']
    y_agg = df_agg[LABEL_COLUMN]
    X_agg = df_agg.drop(columns=[c for c in cols_drop + [LABEL_COLUMN] if c in df_agg.columns])
    X_agg = X_agg.apply(pd.to_numeric, errors='coerce').dropna(axis=1)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_agg)
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Generate PCA Plots
    plot_pca_explained_variance(pca, DIR_PCA_PLOTS)
    plot_pca_2d(X_pca, y_agg, DIR_PCA_PLOTS, pca.explained_variance_ratio_)
    plot_pca_feature_importance(pca, list(X_agg.columns), DIR_PCA_PLOTS)
    
    # Save PCA Dataset
    num_components = min(4, X_pca.shape[1])
    df_pca_train = pd.DataFrame(X_pca[:, :num_components], columns=[f'PC{i+1}' for i in range(num_components)])
    df_pca_train['label'] = y_agg.values
    df_pca_train.to_csv(OUTPUT_PCA_TRAIN, index=False)
    
    # Save Loadings
    loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(len(X_agg.columns))], index=X_agg.columns)
    loadings.to_csv(OUTPUT_LOADINGS)

    # --- PCA Part 2 (Feature Selection) ---
    print("[PART 2] Feature Selection...")
    df_expl = df.copy().dropna()
    cols_drop_2 = ['window_id', 'Timestamp', 'Time_Period_Label', 'Drowsiness_Label_Str', 'raw_hr', 'Drive_Duration', 'Time_Period', 'Smooth_BPM']
    X_expl = df_expl.drop(columns=[c for c in cols_drop_2 + [LABEL_COLUMN] if c in df_expl.columns])
    X_expl = X_expl.apply(pd.to_numeric, errors='coerce').dropna(axis=1)
    
    pca2 = PCA()
    pca2.fit(StandardScaler().fit_transform(X_expl))
    
    # Select Top 5 Features based on PC1
    top_indices = np.argsort(np.abs(pca2.components_[0]))[::-1][:5]
    top_features = [X_expl.columns[i] for i in top_indices]
    print(f"[INFO] Selected Features: {top_features}")
    
    final_cols = list(set(FIXED_COLS_DATASET2 + top_features + [LABEL_COLUMN, 'Drive_Duration']))
    df_selected = df[ [c for c in final_cols if c in df.columns] ].copy().dropna()
    df_selected.to_csv(OUTPUT_SELECTED_FEATURES, index=False)

if __name__ == "__main__":
    _ensure_dir(DATASET_DIR)
    _ensure_dir(RESULT_DIR)
    
    refined_df = step_1_load_data()
    engineered_df = step_2_feature_engineering(refined_df)
    step_3_pca(engineered_df)
    
    print("\n[COMPLETE] Pipeline finished.")
