"""Evaluate frozen HaluEval-trained models on RAGTruth's official test partition.

No RAGTruth labels are used for training or threshold tuning. This is the
primary external-transfer experiment; do not conflate it with RAGTruth
in-domain training and testing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support, roc_auc_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ragtruth-features", type=Path, required=True)
    parser.add_argument("--halueval-model-dir", type=Path, required=True)
    parser.add_argument("--halueval-results", type=Path, required=True)
    parser.add_argument("--method", default="direct_length_excluded_gradient_boosting")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ragtruth = pd.read_csv(args.ragtruth_features)
    test = ragtruth[ragtruth.official_split == "test"].copy()
    if test.empty or test.label.nunique() < 2:
        raise ValueError("RAGTruth official test must contain both classes")
    halueval_results = pd.read_csv(args.halueval_results)
    thresholds = halueval_results[halueval_results.method == args.method].set_index("seed").threshold.to_dict()
    output: list[dict[str, float | int | str]] = []
    for seed, threshold in sorted(thresholds.items()):
        model_path = args.halueval_model_dir / f"seed_{seed}_{args.method}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing frozen HaluEval model: {model_path}")
        model = joblib.load(model_path)
        columns = list(model.feature_names_in_)
        missing = sorted(set(columns) - set(test.columns))
        if missing:
            raise ValueError(f"RAGTruth features lack model columns: {missing}")
        score = model.predict_proba(test[columns])[:, 1]
        prediction = (score >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(test.label, prediction, average="binary", zero_division=0)
        output.append({
            "halueval_seed": int(seed), "frozen_halueval_threshold": threshold, "n": len(test),
            "accuracy": accuracy_score(test.label, prediction), "precision": precision, "recall": recall,
            "f1": f1, "roc_auc": roc_auc_score(test.label, score), "pr_auc": average_precision_score(test.label, score),
        })
    results = pd.DataFrame(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    summary = results.drop(columns=["halueval_seed", "frozen_halueval_threshold"]).agg(["mean", "std"])
    summary.to_csv(args.output.with_name(args.output.stem + "_summary.csv"))


if __name__ == "__main__":
    main()
