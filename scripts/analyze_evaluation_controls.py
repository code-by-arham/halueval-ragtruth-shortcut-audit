"""Run post-evaluation length-control diagnostics across all saved seed predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def binary_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float | int | None]:
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {"n": int(len(y)), "accuracy": None, "f1": None}
    tp = int(np.sum((y == 1) & (prediction == 1)))
    tn = int(np.sum((y == 0) & (prediction == 0)))
    fp = int(np.sum((y == 0) & (prediction == 1)))
    fn = int(np.sum((y == 1) & (prediction == 0)))
    return {
        "n": int(len(y)), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / len(y),
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def length_match(frame: pd.DataFrame, width: int, seed: int) -> pd.DataFrame:
    work = frame.copy()
    work["_bin"] = (work.answer_len // width) * width
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for _, group in work.groupby("_bin", sort=True):
        factual = group[group.label == 0]
        hallucinated = group[group.label == 1]
        count = min(len(factual), len(hallucinated))
        if count:
            chosen.extend(rng.choice(factual.index.to_numpy(), count, replace=False))
            chosen.extend(rng.choice(hallucinated.index.to_numpy(), count, replace=False))
    return work.loc[chosen].drop(columns="_bin")


def aggregate(entries: list[dict[str, object]]) -> dict[str, float | int]:
    metrics = pd.DataFrame(entries)
    result: dict[str, float | int] = {"seeds": int(len(metrics))}
    for column in ("n", "accuracy", "f1", "score_answer_length_correlation"):
        if column in metrics:
            result[f"{column}_mean"] = float(metrics[column].mean())
            result[f"{column}_std"] = float(metrics[column].std(ddof=1)) if len(metrics) > 1 else 0.0
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--match-bin-width", type=int, default=5)
    args = parser.parse_args()

    features = pd.read_csv(args.features, usecols=["question_id", "variant", "label", "answer_len"])
    score_column = f"{args.method}_score"
    prediction_column = f"{args.method}_prediction"
    all_rows: list[dict[str, object]] = []
    for prediction_file in sorted(args.predictions_dir.glob("seed_*_test_predictions.csv")):
        seed = int(prediction_file.stem.split("_")[1])
        predictions = pd.read_csv(prediction_file)
        required = {"question_id", "variant", "label", score_column, prediction_column}
        missing = required - set(predictions.columns)
        if missing:
            raise ValueError(f"{prediction_file} lacks columns: {sorted(missing)}")
        merged = predictions.merge(features, on=["question_id", "variant", "label"], validate="one_to_one")
        score = merged[score_column].to_numpy()
        pred = merged[prediction_column].to_numpy(dtype=int)
        base = binary_metrics(merged.label.to_numpy(dtype=int), pred)
        base.update({"seed": seed, "setting": "original", "score_answer_length_correlation": float(pd.Series(score).corr(merged.answer_len))})
        all_rows.append(base)

        matched = length_match(merged, args.match_bin_width, seed)
        matched_metrics = binary_metrics(matched.label.to_numpy(dtype=int), matched[prediction_column].to_numpy(dtype=int))
        matched_metrics.update({"seed": seed, "setting": "length_matched", "score_answer_length_correlation": float(pd.Series(matched[score_column].to_numpy()).corr(matched.answer_len))})
        all_rows.append(matched_metrics)

        strata = pd.cut(merged.answer_len, bins=[0, 10, 20, 40, 60, np.inf], right=False, labels=["1-9", "10-19", "20-39", "40-59", "60+"])
        for name, group in merged.assign(_stratum=strata).groupby("_stratum", observed=False):
            item = binary_metrics(group.label.to_numpy(dtype=int), group[prediction_column].to_numpy(dtype=int))
            item.update({"seed": seed, "setting": f"stratum_{name}", "score_answer_length_correlation": float(pd.Series(group[score_column].to_numpy()).corr(group.answer_len)) if len(group) > 1 else None})
            all_rows.append(item)

    detailed = pd.DataFrame(all_rows)
    report = {
        "method": args.method,
        "scope": "Post-evaluation diagnostic with thresholds and predictions fixed from the main evaluator.",
        "length_matching": {"bin_width_tokens": args.match_bin_width, "rule": "Equal class sampling within fixed-width answer-length bins."},
        "aggregate_by_setting": {setting: aggregate(rows.to_dict(orient="records")) for setting, rows in detailed.groupby("setting")},
        "per_seed": detailed.to_dict(orient="records"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    detailed.to_csv(args.output.with_suffix(".csv"), index=False)


if __name__ == "__main__":
    main()
