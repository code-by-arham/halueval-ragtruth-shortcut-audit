"""Create group-safe RAGTruth train/validation splits while retaining official test."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    args = parser.parse_args()
    records = pd.read_csv(args.records, usecols=["question_id", "variant", "official_split", "label"])
    train = records[records.official_split == "train"].copy()
    test = records[records.official_split == "test"].copy()
    if train.empty or test.empty:
        raise ValueError("Expected both official train and official test records")
    groups = train.question_id.drop_duplicates().to_numpy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(groups)
        validation_groups = set(shuffled[: max(1, round(0.2 * len(shuffled)))])
        split = records[["question_id", "variant"]].copy()
        split["split"] = np.where(
            records.official_split.eq("test"), "test",
            np.where(records.question_id.isin(validation_groups), "validation", "train"),
        )
        output = split.join(records[["label"]])
        output.to_csv(args.output_dir / f"ragtruth_qa_seed_{seed}_splits.csv", index=False)
        print(f"seed={seed}", output.groupby("split").size().to_dict(), output.groupby(["split", "label"]).size().to_dict())


if __name__ == "__main__":
    main()
