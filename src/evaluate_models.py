import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
import joblib

from data_loader import URLPhishDataLoader


def load_trained_models(models_dir: Path) -> Dict[str, object]:
    """
    Load previously trained models from disk.
    Expects the same filenames as in train_models.py.
    """
    models = {}
    for name in ["log_reg", "random_forest", "xgboost"]:
        model_path = models_dir / f"{name}.joblib"
        if not model_path.exists():
            print(f"[WARN] Model file not found: {model_path} (skipping)")
            continue
        models[name] = joblib.load(model_path)
        print(f"[INFO] Loaded model '{name}' from {model_path}")
    return models


def evaluate_on_test_set(
    models: Dict[str, object],
    X_test,
    y_test,
) -> pd.DataFrame:
    """
    Evaluate each model on the test set and return a DataFrame with metrics.
    Also prints classification reports and confusion matrices.
    """
    rows = []

    for name, model in models.items():
        print(f"\n=== Evaluating model on TEST set: {name} ===")
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", zero_division=0
        )

        print(f"[{name}] Test Accuracy:  {acc:.4f}")
        print(f"[{name}] Test Precision: {precision:.4f}")
        print(f"[{name}] Test Recall:    {recall:.4f}")
        print(f"[{name}] Test F1-score:  {f1:.4f}")
        print("\nClassification report:")
        print(classification_report(y_test, y_pred, digits=4))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        print("Confusion matrix (rows=true, cols=pred):")
        print(cm)
        print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

        rows.append(
            {
                "model": name,
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models on test set.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/url_phish.csv",
        help="Path to the CSV dataset (same used in training).",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory containing trained model .joblib files.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory where evaluation metrics will be saved.",
    )

    args = parser.parse_args()

    csv_path = Path(args.data_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path.resolve()}")

    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir.resolve()}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load the same split as in training (same random_state inside loader)
    loader = URLPhishDataLoader(str(csv_path), scale_features=False)
    X_train, X_val, X_test, y_train, y_val, y_test = loader.load_data()

    print(f"[INFO] Test shape: {X_test.shape}, Test positives: {int(y_test.sum())}")

    models = load_trained_models(models_dir)
    if not models:
        raise RuntimeError("No models loaded. Make sure you trained them first.")

    df_metrics = evaluate_on_test_set(models, X_test, y_test)

    # Save metrics to CSV for later use in the paper
    metrics_path = results_dir / "test_metrics.csv"
    df_metrics.to_csv(metrics_path, index=False)
    print(f"\n[INFO] Saved test metrics to {metrics_path}")


if __name__ == "__main__":
    main()
