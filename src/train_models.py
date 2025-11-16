import argparse
import os
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import joblib

from .data_loader import URLPhishDataLoader


def build_models(random_state: int = 42) -> Dict[str, object]:
    """
    Construct the set of ML models to be evaluated.
    """
    models = {
        "log_reg": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            n_jobs=-1
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=random_state
        ),
        "xgboost": XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=random_state,
            tree_method="hist"  # fast and scalable
        ),
    }
    return models


def evaluate_on_validation(
    models: Dict[str, object],
    X_train,
    y_train,
    X_val,
    y_val
):
    """
    Train each model on (X_train, y_train) and evaluate on validation set.
    Returns a simple metrics dictionary for comparison.
    """
    metrics_summary = {}

    for name, model in models.items():
        print(f"\n=== Training model: {name} ===")
        model.fit(X_train, y_train)

        y_val_pred = model.predict(X_val)

        acc = accuracy_score(y_val, y_val_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, y_val_pred, average="binary", zero_division=0
        )

        print(f"[{name}] Validation Accuracy:  {acc:.4f}")
        print(f"[{name}] Validation Precision: {precision:.4f}")
        print(f"[{name}] Validation Recall:    {recall:.4f}")
        print(f"[{name}] Validation F1-score:  {f1:.4f}")
        print("\nClassification report:")
        print(classification_report(y_val, y_val_pred, digits=4))

        metrics_summary[name] = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return metrics_summary


def save_models(models: Dict[str, object], output_dir: Path):
    """
    Persist trained models to disk using joblib.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        path = output_dir / f"{name}.joblib"
        joblib.dump(model, path)
        print(f"Saved model '{name}' to {path}")


def main():
    parser = argparse.ArgumentParser(description="Train ML models for phishing URL detection.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/url_phish.csv",
        help="Path to the CSV dataset (relative or absolute)."
    )
    parser.add_argument(
        "--scale-features",
        action="store_true",
        help="If set, apply StandardScaler to features."
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory where trained models will be saved."
    )
    args = parser.parse_args()

    csv_path = Path(args.data_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path.resolve()}")

    print(f"Loading data from: {csv_path}")
    loader = URLPhishDataLoader(str(csv_path), scale_features=args.scale_features)
    X_train, X_val, X_test, y_train, y_val, y_test = loader.load_data()

    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

    models = build_models()
    metrics_summary = evaluate_on_validation(models, X_train, y_train, X_val, y_val)

    print("\n=== Validation metrics summary ===")
    for name, metrics in metrics_summary.items():
        print(
            f"{name}: "
            f"Acc={metrics['accuracy']:.4f}, "
            f"Prec={metrics['precision']:.4f}, "
            f"Rec={metrics['recall']:.4f}, "
            f"F1={metrics['f1']:.4f}"
        )

    # Save trained models
    models_dir = Path(args.models_dir)
    save_models(models, models_dir)


if __name__ == "__main__":
    main()
