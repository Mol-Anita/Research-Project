import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .data_loader import URLPhishDataLoader
from .train_models import build_models, evaluate_on_validation


def build_feature_groups(feature_names: List[str]) -> Dict[str, List[str]]:
    lexical_candidates = [
        "url_len",
        "dom_len",
        "tld_len",
        "letter_cnt",
        "digit_cnt",
        "special_cnt",
        "letter_ratio",
        "digit_ratio",
        "spec_ratio",
        "entropy",
        "path_len",
        "query_len",
    ]

    # Structural features (URL structure / separators)
    structural_candidates = [
        "subdom_cnt",
        "dot_cnt",
        "slash_cnt",
        "dash_cnt",
        "under_cnt",
        "eq_cnt",
        "qm_cnt",
        "amp_cnt",
    ]

    # Protocol / domain-like flags
    protocol_candidates = [
        "is_ip",
        "is_https",
    ]

    # Keep only the features that actually exist in the dataset
    lexical = [f for f in lexical_candidates if f in feature_names]
    structural = [f for f in structural_candidates if f in feature_names]
    protocol = [f for f in protocol_candidates if f in feature_names]

    already_used = set(lexical + structural + protocol)
    other = [f for f in feature_names if f not in already_used]

    groups = {
        "lexical": lexical,
        "structural": structural,
        "protocol": protocol,
        "other": other,
    }

    print("\n[INFO] Feature groups:")
    for name, feats in groups.items():
        print(f"  {name}: {len(feats)} features -> {feats}")

    return groups


def main():
    parser = argparse.ArgumentParser(
        description="Feature group ablation: lexical vs structural vs combined."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/url_phish.csv",
        help="Path to the CSV dataset.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory where ablation results will be saved.",
    )
    args = parser.parse_args()

    csv_path = Path(args.data_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path.resolve()}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    loader = URLPhishDataLoader(str(csv_path), scale_features=False)
    X_train, X_val, X_test, y_train, y_val, y_test = loader.load_data()

    feature_names = list(X_train.columns)
    groups = build_feature_groups(feature_names)

    # Define configurations: which groups we include in each experiment
    configs = {
        "lexical_only": groups["lexical"],
        "structural_only": groups["structural"],
        "protocol_only": groups["protocol"],
        "lexical_structural": groups["lexical"] + groups["structural"],
        "all_features": feature_names,
    }

    all_rows = []

    for cfg_name, selected_feats in configs.items():
        if not selected_feats:
            print(f"\n[WARN] Config '{cfg_name}' has no features, skipping.")
            continue

        print(f"\n=== Running ablation config: {cfg_name} ===")
        print(f"[INFO] Using {len(selected_feats)} features.")

        X_train_cfg = X_train[selected_feats]
        X_val_cfg = X_val[selected_feats]

        models = build_models()
        metrics_summary = evaluate_on_validation(
            models, X_train_cfg, y_train, X_val_cfg, y_val
        )

        for model_name, metrics in metrics_summary.items():
            row = {
                "config": cfg_name,
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "n_features": len(selected_feats),
            }
            all_rows.append(row)

    if not all_rows:
        print("[ERROR] No ablation results produced.")
        return

    df_results = pd.DataFrame(all_rows)
    out_path = results_dir / "ablation_results.csv"
    df_results.to_csv(out_path, index=False)
    print(f"\n[INFO] Saved ablation results to {out_path}")
    print(df_results)


if __name__ == "__main__":
    main()
