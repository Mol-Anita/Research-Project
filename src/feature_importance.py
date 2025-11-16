import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.inspection import permutation_importance

from data_loader import URLPhishDataLoader

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)


def load_models(models_dir: Path) -> Dict[str, object]:
    models = {}
    for name in ["log_reg", "random_forest", "xgboost"]:
        model_path = models_dir / f"{name}.joblib"
        if not model_path.exists():
            print(f"[WARN] Model file not found: {model_path} (skipping)")
            continue
        models[name] = joblib.load(model_path)
        print(f"[INFO] Loaded model '{name}' from {model_path}")
    return models


def _global_importance_for_model(model, feature_names: List[str]) -> pd.DataFrame:
    """
    Compute a simple "global importance" score for a model and return
    a DataFrame with columns: feature, importance.
    - For tree models: use feature_importances_
    - For linear models: use abs(coefficients)
    """
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        if coef.ndim > 1:
            # multiclass case: aggregate over classes
            importances = np.mean(np.abs(coef), axis=0)
        else:
            importances = np.abs(coef)
    else:
        raise ValueError("Model does not expose feature_importances_ or coef_.")

    if importances.shape[0] != len(feature_names):
        raise ValueError(
            f"Number of importances ({importances.shape[0]}) != "
            f"number of features ({len(feature_names)})."
        )

    df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)
    return df


def plot_top_features(df_imp: pd.DataFrame, model_name: str, top_k: int, out_dir: Path):
    df_top = df_imp.head(top_k).sort_values("importance", ascending=True)

    plt.figure(figsize=(8, max(4, top_k * 0.4)))
    sns.barplot(x="importance", y="feature", data=df_top)
    plt.title(f"Top {top_k} features – {model_name}")
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"feature_importance_{model_name}_top{top_k}.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved feature importance plot for {model_name} to {out_path}")


def compute_permutation_importance(
    model,
    X_val,
    y_val,
    feature_names: List[str],
    n_repeats: int = 10,
    random_state: int = 42,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Compute permutation importance on the validation set.
    """
    print("[INFO] Computing permutation importance...")
    result = permutation_importance(
        model,
        X_val,
        y_val,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    importances = result.importances_mean
    df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)
    return df


def main():
    parser = argparse.ArgumentParser(description="Feature importance analysis for phishing URL models.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/url_phish.csv",
        help="Path to the CSV dataset.",
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
        help="Directory where CSV results will be stored.",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="figures",
        help="Directory where plots will be saved.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="How many top features to show/plot.",
    )

    args = parser.parse_args()

    csv_path = Path(args.data_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path.resolve()}")

    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir.resolve()}")

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load same split as training
    loader = URLPhishDataLoader(str(csv_path), scale_features=False)
    X_train, X_val, X_test, y_train, y_val, y_test = loader.load_data()

    # Feature names are taken from the training DataFrame
    feature_names = list(X_train.columns)
    print(f"[INFO] Number of features: {len(feature_names)}")

    models = load_models(models_dir)
    if not models:
        raise RuntimeError("No models loaded. Make sure you trained them first.")

    # Global feature importance for each model
    all_importances = []

    for name, model in models.items():
        try:
            df_imp = _global_importance_for_model(model, feature_names)
        except ValueError as e:
            print(f"[WARN] Could not compute global importance for {name}: {e}")
            continue

        df_imp["model"] = name
        all_importances.append(df_imp)

        # Save per-model importance CSV
        model_imp_path = results_dir / f"feature_importance_{name}.csv"
        df_imp.to_csv(model_imp_path, index=False)
        print(f"[INFO] Saved global feature importance for {name} to {model_imp_path}")

        # Plot top-k
        plot_top_features(df_imp, name, args.top_k, figures_dir)

    if all_importances:
        df_all = pd.concat(all_importances, ignore_index=True)
        all_path = results_dir / "feature_importance_all_models.csv"
        df_all.to_csv(all_path, index=False)
        print(f"[INFO] Saved combined feature importance to {all_path}")

    # Permutation importance for one of the tree models (prefer xgboost, else random_forest)
    perm_model_name = None
    perm_model = None

    if "xgboost" in models:
        perm_model_name = "xgboost"
        perm_model = models["xgboost"]
    elif "random_forest" in models:
        perm_model_name = "random_forest"
        perm_model = models["random_forest"]

    if perm_model is not None:
        df_perm = compute_permutation_importance(
            perm_model,
            X_val,
            y_val,
            feature_names,
            n_repeats=10,
            random_state=42,
            n_jobs=-1,
        )
        perm_path = results_dir / f"permutation_importance_{perm_model_name}.csv"
        df_perm.to_csv(perm_path, index=False)
        print(f"[INFO] Saved permutation importance for {perm_model_name} to {perm_path}")

        # Plot permutation importance top-k
        plot_top_features(df_perm, f"{perm_model_name}_perm", args.top_k, figures_dir)
    else:
        print("[WARN] No tree model found for permutation importance (xgboost or random_forest).")


if __name__ == "__main__":
    main()
