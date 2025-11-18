import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import joblib


def make_splits(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """
    Reproduce the same split logic as URLPhishDataLoader:
    - test_size = 0.15
    - val_size = 0.12 of train_val (but we only care about test here)
    Returns X_test, y_test, idx_test (indices into original df).
    """
    y = df["label"].values
    indices = np.arange(len(df))

    # First split: train+val vs test
    X_train_val, X_test, y_train_val, y_test, idx_train_val, idx_test = train_test_split(
        df.drop(columns=["label"]),
        y,
        indices,
        test_size=0.15,
        stratify=y,
        random_state=42,
    )

    return X_test, y_test, idx_test


def load_model(models_dir: Path, preferred: str = "xgboost") -> Tuple[str, object]:
    """
    Load preferred model if available, else fall back to random_forest, else log_reg.
    """
    candidates = [preferred, "random_forest", "log_reg"]
    for name in candidates:
        path = models_dir / f"{name}.joblib"
        if path.exists():
            print(f"[INFO] Using model '{name}' from {path}")
            model = joblib.load(path)
            return name, model

    raise FileNotFoundError(
        f"No suitable model found in {models_dir}. "
        f"Expected one of: {', '.join(candidates)}"
    )


def compute_subset_stats(
    df_subset: pd.DataFrame, numeric_cols, name: str
) -> dict:
    """
    Compute simple stats for a subset of URLs: count, avg length, avg entropy, etc.
    Uses the actual column names from the dataset.
    """
    row = {"subset": name, "count": len(df_subset)}

    # Only compute if the column exists
    if "url_len" in numeric_cols:
        row["avg_url_len"] = df_subset["url_len"].mean()
    if "entropy" in numeric_cols:
        row["avg_entropy"] = df_subset["entropy"].mean()
    if "digit_cnt" in numeric_cols:
        row["avg_digit_cnt"] = df_subset["digit_cnt"].mean()
    if "special_cnt" in numeric_cols:
        row["avg_special_cnt"] = df_subset["special_cnt"].mean()
    if "subdom_cnt" in numeric_cols:
        row["avg_subdom_cnt"] = df_subset["subdom_cnt"].mean()
    if "slash_cnt" in numeric_cols:
        row["avg_slash_cnt"] = df_subset["slash_cnt"].mean()

    return row



def main():
    parser = argparse.ArgumentParser(
        description="Error analysis: false positives/negatives with URL samples and stats."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/url_phish.csv",
        help="Path to the CSV dataset (must include 'url' and 'label' columns).",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory containing trained models.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory where error analysis results will be saved.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=30,
        help="Max number of example URLs to save for each error type.",
    )
    args = parser.parse_args()

    csv_path = Path(args.data_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path.resolve()}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(csv_path)
    if "label" not in df_raw.columns:
        raise ValueError("Dataset must contain a 'label' column.")
    if "url" not in df_raw.columns:
        print("[WARN] 'url' column not found; you won't see raw URLs in samples.")

    # Create test split and keep original indices
    X_test_full, y_test, idx_test = make_splits(df_raw)

    # For prediction, drop non-numeric feature columns that models didn't use
    # (we know models were trained on df.drop(['url', 'dom', 'tld'], 'label'))
    X_test_for_model = X_test_full.drop(
        columns=["url", "dom", "tld"], errors="ignore"
    )

    numeric_cols = X_test_for_model.select_dtypes(include=[np.number]).columns

    model_name, model = load_model(Path(args.models_dir))

    # Predictions
    y_pred = model.predict(X_test_for_model.values)

    # Confusion matrix / sanity check
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("Confusion matrix on TEST set (rows=true, cols=pred):")
    print(cm)
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # Identify error types
    y_test_series = pd.Series(y_test, index=idx_test, name="true_label")
    y_pred_series = pd.Series(y_pred, index=idx_test, name="pred_label")

    false_pos_idx = y_test_series[(y_test_series == 0) & (y_pred_series == 1)].index
    false_neg_idx = y_test_series[(y_test_series == 1) & (y_pred_series == 0)].index
    true_pos_idx = y_test_series[(y_test_series == 1) & (y_pred_series == 1)].index
    true_neg_idx = y_test_series[(y_test_series == 0) & (y_pred_series == 0)].index

    print(
        f"[INFO] False positives: {len(false_pos_idx)}, "
        f"False negatives: {len(false_neg_idx)}"
    )

    # Build DataFrames for each subset, including URL and numeric features
    cols_to_keep = ["url", "label"] + list(numeric_cols)
    cols_to_keep = [c for c in cols_to_keep if c in df_raw.columns]

    df_fp = df_raw.loc[false_pos_idx, cols_to_keep].copy()
    df_fn = df_raw.loc[false_neg_idx, cols_to_keep].copy()
    df_tp = df_raw.loc[true_pos_idx, cols_to_keep].copy()
    df_tn = df_raw.loc[true_neg_idx, cols_to_keep].copy()

    # Save small samples of URLs for qualitative analysis
    fp_sample = df_fp.head(args.max_samples)
    fn_sample = df_fn.head(args.max_samples)

    fp_path = results_dir / "error_false_positives_sample.csv"
    fn_path = results_dir / "error_false_negatives_sample.csv"
    fp_sample.to_csv(fp_path, index=False)
    fn_sample.to_csv(fn_path, index=False)
    print(f"[INFO] Saved false positive samples to {fp_path}")
    print(f"[INFO] Saved false negative samples to {fn_path}")

    # Compute simple stats table
    stats_rows = []
    stats_rows.append(compute_subset_stats(df_tn, numeric_cols, "true_negatives"))
    stats_rows.append(compute_subset_stats(df_tp, numeric_cols, "true_positives"))
    stats_rows.append(compute_subset_stats(df_fp, numeric_cols, "false_positives"))
    stats_rows.append(compute_subset_stats(df_fn, numeric_cols, "false_negatives"))

    df_stats = pd.DataFrame(stats_rows)
    stats_path = results_dir / "error_analysis_summary.csv"
    df_stats.to_csv(stats_path, index=False)
    print(f"[INFO] Saved error analysis summary to {stats_path}")
    print(df_stats)


if __name__ == "__main__":
    main()
