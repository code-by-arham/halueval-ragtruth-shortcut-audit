"""Convert HaluBench (Patronus AI) external subsets into standardized response-level records.

Kept subsets: DROP, FinanceBench, covidQA, pubmedQA (HaluEval and RAGTruth rows are
excluded because those sources are already used directly). Label 1 = FAIL
(hallucinated answer), 0 = PASS. Group id = subset + passage hash, so every
question that shares a passage stays in one split. Five group-safe 80/10/10
splits use the same generator as the HaluEval splits.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd

KEEP = ["DROP", "FinanceBench", "covidQA", "pubmedQA"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--splits-dir", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    a = ap.parse_args()
    h = pd.read_parquet(a.parquet)
    h = h[h.source_ds.isin(KEEP)].reset_index(drop=True)
    gid = [f"{s}_{hashlib.sha1(p.encode('utf-8')).hexdigest()[:12]}" for s, p in zip(h.source_ds, h.passage)]
    rec = pd.DataFrame({"question_id": gid, "variant": h["id"], "source_ds": h.source_ds,
                        "label": (h.label == "FAIL").astype(int), "question": h.question,
                        "evidence": h.passage, "answer": h.answer.astype(str)})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    rec.to_csv(a.output, index=False)
    groups = np.array(sorted(rec.question_id.unique()))
    a.splits_dir.mkdir(parents=True, exist_ok=True)
    for seed in a.seeds:
        p = np.random.default_rng(seed).permutation(groups); n = len(p)
        assign = {**{g: "train" for g in p[:int(.8 * n)]}, **{g: "validation" for g in p[int(.8 * n):int(.9 * n)]},
                  **{g: "test" for g in p[int(.9 * n):]}}
        out = rec[["question_id", "variant", "label"]].copy(); out["split"] = out.question_id.map(assign)
        out.to_csv(a.splits_dir / f"halubench_qa_seed_{seed}_splits.csv", index=False)
        print(seed, out.groupby(["split", "label"]).size().to_dict())
    manifest = {"source": "PatronusAI/HaluBench data/test-00000-of-00001.parquet",
                "sha256": hashlib.sha256(a.parquet.read_bytes()).hexdigest(), "licence": "CC BY-NC 2.0",
                "subsets": KEEP, "records": int(len(rec)), "groups": int(len(groups)),
                "label_counts": {str(k): int(v) for k, v in rec.label.value_counts().items()},
                "per_subset": rec.groupby("source_ds").agg(n=("label", "size"), hallucinated=("label", "sum"),
                                                            groups=("question_id", "nunique")).to_dict(orient="index")}
    a.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, default=int))
    print(json.dumps(manifest, indent=1, default=int))


if __name__ == "__main__":
    main()
