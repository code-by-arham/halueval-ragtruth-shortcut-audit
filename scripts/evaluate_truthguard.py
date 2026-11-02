"""Evaluate artifact baselines and fusion models on fixed group-safe splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DIRECT_LENGTH = ["answer_len", "evidence_len", "question_len", "answer_evidence_len_ratio"]
DIRECT_LENGTH_EXCLUDED = [
    "lexical_overlap", "lexical_mismatch", "jaccard", "semantic_answer_evidence_sim",
    "semantic_answer_question_sim", "semantic_mismatch", "nli_entailment", "nli_neutral",
    "nli_contradiction", "nli_unsupported", "uncertainty",
]
# (feature column, whether a high raw feature value indicates factuality rather
# than hallucination). Every baseline is converted to a hallucination score
# before validation threshold selection. This avoids unfairly reporting an
# avoidable 0.5 accuracy for a correctly informative signal in the opposite
# direction.
BASELINES = {
    "artifact_answer_length": ("answer_len", False),
    "artifact_answer_evidence_ratio": ("answer_evidence_len_ratio", False),
    "lexical_overlap_inverse": ("lexical_overlap", True),
    "lexical_mismatch": ("lexical_mismatch", False),
    "jaccard": ("jaccard", False),
    "semantic_answer_evidence": ("semantic_answer_evidence_sim", False),
    "semantic_mismatch_inverse": ("semantic_mismatch", True),
    "semantic_answer_question": ("semantic_answer_question_sim", False),
    "nli_entailment_inverse": ("nli_entailment", True),
    "nli_neutral_inverse": ("nli_neutral", True),
    "nli_contradiction": ("nli_contradiction", False),
    "nli_unsupported": ("nli_unsupported", False),
}


def threshold_for_f1(y: np.ndarray, score: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0.0], score, [1.0])))
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in candidates:
        prediction = (score >= threshold).astype(int)
        f1 = precision_recall_fscore_support(y, prediction, average="binary", zero_division=0)[2]
        if f1 > best_f1:
            best_threshold, best_f1 = float(threshold), float(f1)
    return best_threshold


def score_result(method: str, seed: int, y: np.ndarray, score: np.ndarray, threshold: float) -> tuple[dict[str, float | int | str], np.ndarray]:
    prediction = (score >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, prediction, average="binary", zero_division=0)
    return {
        "seed": seed, "method": method, "threshold": threshold,
        "accuracy": accuracy_score(y, prediction), "precision": precision, "recall": recall, "f1": f1,
        "roc_auc": roc_auc_score(y, score), "pr_auc": average_precision_score(y, score),
    }, prediction


def model_definitions(seed: int) -> dict[str, tuple[list[str], object]]:
    length_aware = DIRECT_LENGTH_EXCLUDED + DIRECT_LENGTH
    return {
        "direct_length_excluded_logistic_regression": (
            DIRECT_LENGTH_EXCLUDED,
            Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))]),
        ),
        "direct_length_excluded_random_forest": (
            DIRECT_LENGTH_EXCLUDED,
            RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=4, class_weight="balanced", random_state=seed, n_jobs=-1),
        ),
        "direct_length_excluded_gradient_boosting": (
            DIRECT_LENGTH_EXCLUDED,
            GradientBoostingClassifier(n_estimators=120, learning_rate=0.05, max_depth=2, random_state=seed),
        ),
        "length_aware_gradient_boosting": (
            length_aware,
            GradientBoostingClassifier(n_estimators=120, learning_rate=0.05, max_depth=2, random_state=seed),
        ),
    }


def evaluate_split(features: pd.DataFrame, split_file: Path, output_dir: Path, save_models: bool) -> list[dict[str, float | int | str]]:
    seed = int(split_file.stem.split("_")[3])
    split = pd.read_csv(split_file, usecols=["question_id", "variant", "split"])
    joined = features.merge(split, on=["question_id", "variant"], how="inner", validate="one_to_one")
    if len(joined) != len(features):
        raise ValueError(f"Feature/split mismatch in {split_file}")
    parts = {name: joined[joined["split"] == name].copy() for name in ("train", "validation", "test")}
    if any(part.empty for part in parts.values()):
        raise ValueError(f"Missing split partition in {split_file}")

    results: list[dict[str, float | int | str]] = []
    prediction_output = parts["test"][["question_id", "variant", "label"]].copy()
    y_val, y_test = parts["validation"].label.to_numpy(), parts["test"].label.to_numpy()

    for method, (column, invert) in BASELINES.items():
        validation_score = parts["validation"][column].to_numpy()
        test_score = parts["test"][column].to_numpy()
        if invert:
            validation_score = 1.0 - validation_score
            test_score = 1.0 - test_score
        threshold = threshold_for_f1(y_val, validation_score)
        result, prediction = score_result(method, seed, y_test, test_score, threshold)
        results.append(result)
        prediction_output[f"{method}_score"] = test_score
        prediction_output[f"{method}_prediction"] = prediction

    for method, (columns, model) in model_definitions(seed).items():
        model.fit(parts["train"][columns], parts["train"].label)
        validation_score = model.predict_proba(parts["validation"][columns])[:, 1]
        test_score = model.predict_proba(parts["test"][columns])[:, 1]
        threshold = threshold_for_f1(y_val, validation_score)
        result, prediction = score_result(method, seed, y_test, test_score, threshold)
        results.append(result)
        prediction_output[f"{method}_score"] = test_score
        prediction_output[f"{method}_prediction"] = prediction
        if save_models:
            joblib.dump(model, output_dir / f"seed_{seed}_{method}.joblib")

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_output.to_csv(output_dir / f"seed_{seed}_test_predictions.csv", index=False)
    return results


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    measures = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    aggregate = results.groupby("method")[measures].agg(["mean", "std", "count"])
    summary = pd.DataFrame(index=aggregate.index)
    for measure in measures:
        summary[f"{measure}_mean"] = aggregate[(measure, "mean")]
        summary[f"{measure}_std"] = aggregate[(measure, "std")].fillna(0.0)
        summary[f"{measure}_ci95_halfwidth"] = 1.96 * summary[f"{measure}_std"] / np.sqrt(aggregate[(measure, "count")])
    return summary.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--split-pattern", default="halueval_qa_seed_*_splits.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-models", action="store_true")
    args = parser.parse_args()
    features = pd.read_csv(args.features)
    required = {"question_id", "variant", "label", *DIRECT_LENGTH, *DIRECT_LENGTH_EXCLUDED}
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"Feature file misses required columns: {missing}")

    split_files = sorted(args.splits_dir.glob(args.split_pattern))
    if not split_files:
        raise FileNotFoundError(f"No split manifests in {args.splits_dir}")
    raw_results = [row for split_file in split_files for row in evaluate_split(features, split_file, args.output_dir, args.save_models)]
    results = pd.DataFrame(raw_results).sort_values(["method", "seed"])
    results.to_csv(args.output_dir / "all_seed_results.csv", index=False)
    summarize(results).to_csv(args.output_dir / "summary_results.csv", index=False)


if __name__ == "__main__":
    main()
