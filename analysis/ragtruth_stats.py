"""Descriptive statistics for the RAGTruth QA response-level conversion (quality filter, span types, implicit_true)."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import json, collections
import pandas as pd
R = [json.loads(l) for l in open(ROOT / "data/RAGTruth/response.jsonl")]
S = {str(json.loads(l)["source_id"]): json.loads(l) for l in open(ROOT / "data/RAGTruth/source_info.jsonl")}
print("all responses", len(R), "task types", collections.Counter(S[str(r["source_id"])]["task_type"] for r in R))
qa = [r for r in R if S[str(r["source_id"])]["task_type"] == "QA"]
print("QA responses", len(qa), "QA sources", len({r['source_id'] for r in qa}))
print("quality", collections.Counter(r.get("quality") for r in qa))
print("quality x split", collections.Counter((r.get("quality"), r["split"]) for r in qa))
good = [r for r in qa if r.get("quality") == "good"]
print("good", len(good), "split", collections.Counter(r["split"] for r in good))
lab = [int(len(r["labels"])>0) for r in good]; print("labels", collections.Counter(lab))
# hallucination rate by quality
for q in set(r.get("quality") for r in qa):
    sub=[r for r in qa if r.get("quality")==q]; print(q, len(sub), "halluc rate", sum(len(r["labels"])>0 for r in sub)/len(sub))
spans = [s for r in good for s in r["labels"]]
print("total spans", len(spans), collections.Counter(s["label_type"] for s in spans))
print("implicit_true spans", sum(bool(s.get("implicit_true")) for s in spans))
hall = [r for r in good if r["labels"]]
only_impl = sum(all(s.get("implicit_true") for s in r["labels"]) for r in hall)
print("hallucinated responses", len(hall), "label solely from implicit_true", only_impl)
print("due_to_null field?", collections.Counter(k for s in spans for k in s.keys()))
print("keys response", list(good[0].keys()))
# test set breakdown
test=[r for r in good if r["split"]=="test"]
print("test", len(test), collections.Counter(int(len(r['labels'])>0) for r in test))
print("test only implicit", sum(1 for r in test if r['labels'] and all(s.get('implicit_true') for s in r['labels'])))
print("model x label (good)", pd.crosstab([r['model'] for r in good],[int(len(r['labels'])>0) for r in good]))
