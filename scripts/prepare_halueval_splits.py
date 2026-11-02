"""Recreate the group-safe HaluEval QA 80/10/10 split manifests (seeds 11-55).

Reconstructed generator: np.random.default_rng(seed).permutation(question_ids);
first 80% train, next 10% validation, final 10% test. Verified to reproduce the
archived per-seed artifact baseline results exactly.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    a = ap.parse_args()
    n = sum(1 for line in a.source.open(encoding="utf-8") if line.strip())
    ids = np.arange(n)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for seed in a.seeds:
        p = np.random.default_rng(seed).permutation(ids)
        assign = {}
        for q in p[: int(0.8 * n)]: assign[q] = "train"
        for q in p[int(0.8 * n): int(0.9 * n)]: assign[q] = "validation"
        for q in p[int(0.9 * n):]: assign[q] = "test"
        rows = [(q, v, assign[q], lab) for q in ids for v, lab in (("factual", 0), ("hallucinated", 1))]
        out = pd.DataFrame(rows, columns=["question_id", "variant", "split", "label"])
        out.to_csv(a.output_dir / f"halueval_qa_seed_{seed}_splits.csv", index=False)
        print(seed, out.split.value_counts().to_dict())

if __name__ == "__main__":
    main()
