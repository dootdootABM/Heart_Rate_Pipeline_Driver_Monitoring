from __future__ import annotations

import sys
import warnings
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

# --- load_data import (same as your project layout) ---
try:
    from load_data import load_data
except ImportError:
    print("[ERROR] Could not import 'load_data'. Make sure 'load_data.py' is in the same directory.")
    sys.exit(1)

# --- Third-party dependency (NeuroKit2) ---
try:
    import neurokit2 as nk
except Exception:
    print("[ERROR] Install neurokit2: pip install neurokit2")
    raise


# ===================== CONFIGURATION (adjust if needed) =====================
SAMPLING_RATE: int = 25                # Hz
PPG_COLUMN: str = "ppgGreen"           # e.g., 'ppgIR' or 'ppgGreen'
WINDOW_SECONDS: int = 1200               # analysis window length (50% overlap is used)
LABEL_COLUMN: Optional[str] = "drowsiness"      # optional; majority vote per window
TIMESTAMP_COLUMN: Optional[str] = "timestamp"   # set to None if no timestamps in your data

# Peak detection / HRV
MIN_PEAKS_REQUIRED: int = 5
PPG_METHOD: str = "elgendi"            # 'elgendi' | 'bishop' | 'charlton'
INVERT_PPG: bool = False

# HRV feature groups ('auto' = time+freq for <120s; adds nonlinear for >=120s)
HRV_FEATURES: str = "auto"
# ==========================================================================


def _resolve_hrv_features(req: Optional[str], window_sec: int) -> List[str]:
    """Select NeuroKit2 HRV feature groups consistently."""
    r = (req or "auto").strip().lower()
    if r == "auto":
        return ["time", "frequency"] if window_sec < 120 else ["time", "frequency", "nonlinear"]
    tokens = [t.strip() for t in r.replace(";", ",").replace("\n", ",").split(",") if t.strip()]
    allowed = {"time", "frequency", "nonlinear"}
    out = [t for t in tokens if t in allowed]
    return out or ["time", "frequency"]


def _to_peak_indices(peaks_obj: Any, window_len: int) -> np.ndarray:
    """NeuroKit2 may return a boolean mask or integer indices; normalize to integer indices."""
    if peaks_obj is None:
        return np.array([], dtype=int)
    arr = np.asarray(peaks_obj)
    # boolean mask
    if arr.ndim == 1 and arr.size == int(window_len):
        if arr.dtype == bool or np.all(np.isin(np.unique(arr), [0, 1])):
            return np.flatnonzero(arr).astype(int, copy=False)
    # integer indices
    if arr.ndim == 1 and np.issubdtype(arr.dtype, np.integer):
        keep = (arr >= 0) & (arr < int(window_len))
        return arr[keep].astype(int, copy=False)
    # fallback
    return np.flatnonzero(arr).astype(int, copy=False)


def engineer_features(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray]]:
    """
    Applies sliding windows, extracts HRV features from PPG, and adds:
    - HR_BPM_* stats
    - time-of-day fields (if TIMESTAMP_COLUMN exists)

    Returns
    -------
    X : pd.DataFrame or None
    y : np.ndarray or None
    """
    # ---- Basic validation (mirrors your original) ----
    if df is None or df.empty:
        print("[ERROR] Input DataFrame is empty or None. Cannot engineer features.")
        return None, None

    if PPG_COLUMN not in df.columns:
        print(f"[ERROR] Specified PPG column '{PPG_COLUMN}' not found.")
        print(f"Available columns: {df.columns.tolist()}")
        return None, None

    # LABEL_COLUMN is optional; proceed without if absent.
    has_label = LABEL_COLUMN is not None and LABEL_COLUMN in df.columns

    # Windowing with 50% overlap
    window_size = int(SAMPLING_RATE * WINDOW_SECONDS)
    step_size = window_size // 2

    if len(df) < window_size:
        print(f"[ERROR] Dataset length ({len(df)}) is shorter than window size ({window_size}). Cannot process.")
        return None, None

    print("\n[INFO] Starting feature engineering ... ")
    print(f"- Sampling Rate: {SAMPLING_RATE} Hz")
    print(f"- PPG Column: '{PPG_COLUMN}'")
    print(f"- Window Duration: {WINDOW_SECONDS} seconds")
    print(f"- Window Size: {window_size} samples")
    print(f"- Step Size: {step_size} samples")

    work_df = df.reset_index(drop=True).copy()

    ppg = work_df[PPG_COLUMN].to_numpy(dtype=float, copy=False)
    if INVERT_PPG:
        ppg = -ppg

    labels_ser = work_df[LABEL_COLUMN] if has_label else None
    ts_ser = work_df[TIMESTAMP_COLUMN] if (TIMESTAMP_COLUMN and TIMESTAMP_COLUMN in work_df.columns) else None

    groups = _resolve_hrv_features(HRV_FEATURES, WINDOW_SECONDS)

    features_list: List[pd.Series] = []
    labels_list: List[Any] = []

    processed_windows = 0
    skipped_windows = 0
    max_errors_to_show = 5

    # ---- Sliding Window Loop ----
    for start in range(0, len(work_df) - window_size + 1, step_size):
        end = start + window_size
        win_raw = ppg[start:end]

        # Majority label for the window (if present)
        label_value = None
        if labels_ser is not None:
            try:
                label_value = labels_ser.iloc[start:end].mode(dropna=True).iloc[0]
            except Exception:
                label_value = np.nan

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Primary: ppg_process -> hrv(info)
                signals, info = nk.ppg_process(win_raw, sampling_rate=SAMPLING_RATE, method=PPG_METHOD)
                peaks_mask = info.get("PPG_Peaks", None)
                peak_idx = _to_peak_indices(peaks_mask, len(win_raw))
                if len(peak_idx) < MIN_PEAKS_REQUIRED:
                    raise ValueError(f"Too few peaks after ppg_process ({len(peak_idx)} < {MIN_PEAKS_REQUIRED}).")

                hrv_df = nk.hrv(info, sampling_rate=SAMPLING_RATE, show=False, features=groups)
                if hrv_df is None or hrv_df.empty:
                    raise ValueError("HRV calculation returned no results.")

                row = hrv_df.iloc[0].copy()

        except Exception:
            # Fallback: clean + findpeaks + hrv
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cleaned = nk.ppg_clean(win_raw, sampling_rate=SAMPLING_RATE)
                    info2 = nk.ppg_findpeaks(cleaned, sampling_rate=SAMPLING_RATE, method=PPG_METHOD)
                    peaks_mask = info2.get("PPG_Peaks", None)
                    peak_idx = _to_peak_indices(peaks_mask, len(win_raw))
                    if len(peak_idx) < MIN_PEAKS_REQUIRED:
                        raise ValueError(f"Too few peaks after findpeaks ({len(peak_idx)} < {MIN_PEAKS_REQUIRED}).")

                    hrv_df = nk.hrv(info2, sampling_rate=SAMPLING_RATE, show=False, features=groups)
                    if hrv_df is None or hrv_df.empty:
                        raise ValueError("HRV calculation returned no results (fallback).")

                    row = hrv_df.iloc[0].copy()
            except Exception as e2:
                skipped_windows += 1
                if skipped_windows <= max_errors_to_show:
                    msg = str(e2)
                    if len(msg) > 150:
                        msg = msg[:150] + " ..."
                    print(f"[WARN] Skipping window at index {start}. Reason: {type(e2).__name__}: {msg}")
                if skipped_windows == max_errors_to_show:
                    print("[WARN] ... (suppressing further error messages for skipped windows)")
                continue

        # ---- Heart-Rate (BPM) stats per window ----
        HR_BPM_mean = HR_BPM_median = HR_BPM_min = HR_BPM_max = np.nan
        if len(peak_idx) >= 2:
            try:
                hr_trace = nk.signal_rate(
                    peak_idx.astype(np.int64), sampling_rate=SAMPLING_RATE, desired_length=len(win_raw)
                )
                HR_BPM_mean = float(np.nanmean(hr_trace))
                HR_BPM_median = float(np.nanmedian(hr_trace))
                HR_BPM_min = float(np.nanmin(hr_trace))
                HR_BPM_max = float(np.nanmax(hr_trace))
            except Exception:
                # Fallback: from inter-beat intervals (no resampling)
                ibi_s = np.diff(peak_idx.astype(np.int64)) / float(SAMPLING_RATE)
                if ibi_s.size > 0:
                    bpm_inst = 60.0 / ibi_s
                    HR_BPM_mean = float(np.nanmean(bpm_inst))
                    HR_BPM_median = float(np.nanmedian(bpm_inst))
                    HR_BPM_min = float(np.nanmin(bpm_inst))
                    HR_BPM_max = float(np.nanmax(bpm_inst))

        # ---- Window indices, label, time-of-day ----
        row["window_start_idx"] = int(start)
        row["window_end_idx"] = int(end)
        if labels_ser is not None:
            row[LABEL_COLUMN] = label_value

        if ts_ser is not None:
            try:
                t_start = pd.to_datetime(ts_ser.iloc[start])
                t_end = pd.to_datetime(ts_ser.iloc[end - 1])
                t_mid = t_start + (t_end - t_start) / 2
                row["window_start_time"] = t_start
                row["window_end_time"] = t_end
                row["window_mid_time"] = t_mid
                row["time_of_day"] = t_mid.strftime("%H:%M:%S")
                row["hour"] = int(t_mid.hour)
                row["minute"] = int(t_mid.minute)
                row["weekday"] = int(t_mid.weekday())  # 0=Mon
                row["window_duration_s"] = float((t_end - t_start).total_seconds())
            except Exception:
                row["window_mid_time"] = np.nan
                row["time_of_day"] = np.nan

        # HR stats
        row["HR_BPM_mean"] = HR_BPM_mean
        row["HR_BPM_median"] = HR_BPM_median
        row["HR_BPM_min"] = HR_BPM_min
        row["HR_BPM_max"] = HR_BPM_max

        features_list.append(row)
        labels_list.append(label_value)
        processed_windows += 1

    # ---- Post-processing ----
    print("\n[INFO] Feature engineering finished.")
    print(f"- Successfully processed {processed_windows} windows.")
    print(f"- Skipped {skipped_windows} windows due to errors.")

    if not features_list:
        print("[ERROR] No features were successfully extracted from any window.")
        return None, None

    try:
        X = pd.DataFrame(features_list)
        y = np.array(labels_list) if labels_ser is not None else None

        # Drop all-NaN columns
        cols_before = X.shape[1]
        X = X.dropna(axis=1, how="all")
        cols_after = X.shape[1]
        if cols_before > cols_after:
            print(f"[INFO] Dropped {cols_before - cols_after} all-NaN columns.")

        # Replace inf with NaN and fill numeric NaNs with means
        if np.isinf(X.values).any():
            print("[WARN] Infinite values detected. Replacing with NaN.")
            X.replace([np.inf, -np.inf], np.nan, inplace=True)

        if X.isnull().values.any():
            print("[INFO] Filling remaining NaNs with column means.")
            X = X.fillna(X.mean(numeric_only=True))

        print(f"[INFO] Final feature set shape: {X.shape}")
        if y is not None:
            print(f"[INFO] Labels array shape: {y.shape}")
        return X, y

    except Exception as e:
        print(f"[ERROR] Failed to finalize feature DataFrame: {e}")
        return None, None


def create_combined_dataset(features_df: pd.DataFrame, labels_array: Optional[np.ndarray]) -> Optional[pd.DataFrame]:
    """
    Combine X and y into a single DataFrame (parity with original helper).
    Adds labels as 'drowsiness_label' if provided; otherwise leaves rows as-is.
    """
    if features_df is None:
        print("[ERROR] Cannot combine dataset: features is None.")
        return None

    combined = features_df.copy()
    if labels_array is not None:
        if len(features_df) != len(labels_array):
            print(f"[ERROR] Length mismatch: features={len(features_df)}, labels={len(labels_array)}")
            return None
        combined["drowsiness_label"] = labels_array

    print(f"[INFO] Combined DataFrame shape: {combined.shape}")
    return combined


# =============================== MAIN (hard paths) ===============================
if __name__ == "__main__":
    print(f"NeuroKit2 version: {getattr(nk, '_version_', getattr(nk, '__version__', 'N/A'))}")

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # TODO: UPDATE THESE TWO PATHS FOR YOUR MACHINE
    input_csv_path  = r"C:\Users\devan\Dropbox\Aktif\Germany\TH Ingolstadt\2. Semester\PRJ Project\My Project\03 Team 2\02 VScode\drowsiness_dataset.csv"
    output_csv_path = r"C:\Users\devan\Dropbox\Aktif\Germany\TH Ingolstadt\2. Semester\PRJ Project\My Project\03 Team 2\02 VScode\artifacts 2\01_features\engineered_features_with_hrv_hr_time.csv"
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # 1) Load data
    raw_df = load_data(input_csv_path)
    if raw_df is None or raw_df.empty:
        print("[ERROR] Could not load input data or it is empty.")
        sys.exit(1)

    # 2) Engineer features
    X, y = engineer_features(raw_df)
    if X is None or X.empty:
        print("[ERROR] Feature engineering failed.")
        sys.exit(1)

    # 3) Combine (optional) and save to CSV at the hard path
    combined = create_combined_dataset(X, y)
    to_save = combined if combined is not None else X
    try:
        from pathlib import Path
        Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
        to_save.to_csv(output_csv_path, index=False)
        print(f"\n[SUCCESS] Saved engineered features to:\n{output_csv_path}")
    except Exception as e:
        print(f"\n[ERROR] Failed to save final dataset to CSV: {e}")