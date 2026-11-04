"""Checkpointed feature extraction for a standardized records CSV (e.g. HaluBench).

Uses the exact functions of extract_halueval_features.py (tokeniser, lexical,
MiniLM semantic, BART-MNLI with the same 16-pair call pattern, 0.35 rule,
uncertainty markers). NLI output is saved every CHUNK pairs so a time-limited
run can continue where it stopped. MAX_CHUNKS limits the chunks done per call.
"""
import argparse, importlib.metadata, json, os, platform, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_halueval_features as X  # noqa: E402
from transformers import pipeline  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--records", type=Path, required=True)
ap.add_argument("--output", type=Path, required=True)
ap.add_argument("--chunk", type=int, default=250)
a = ap.parse_args()
torch.set_num_threads(int(os.environ.get("NUM_THREADS", max(1, os.cpu_count() or 1))))
MAX_CHUNKS = int(os.environ.get("MAX_CHUNKS", "100000"))
import time
DEADLINE = time.time() + 60 * float(os.environ.get("BUDGET_MIN", "1e9"))
ck = a.output.parent / (a.output.stem + "_nli_chunks"); ck.mkdir(parents=True, exist_ok=True)
base = a.output.parent / (a.output.stem + "_base_semantic.csv")

if base.exists():
    frame = pd.read_csv(base, keep_default_na=False)
    frame["label"] = frame.label.astype(int)
else:
    frame = X.load_standardized_records(a.records)
    frame = X.add_basic_features(frame)
    frame = X.add_semantic_features(frame, "sentence-transformers/all-MiniLM-L6-v2", "cpu", 64)
    frame.to_csv(base, index=False)
    print("base+semantic saved", flush=True)

clf = pipeline("text-classification", model="facebook/bart-large-mnli", device=-1, top_k=None, truncation=True, max_length=512)
pairs = [{"text": e, "text_pair": s} for e, s in zip(frame.evidence.astype(str), frame.answer.astype(str))]
done_now = 0
for c0 in range(0, len(pairs), a.chunk):
    f = ck / f"chunk_{c0:06d}.json"
    if f.exists():
        continue
    if done_now >= MAX_CHUNKS or time.time() > DEADLINE:
        print("STOP_AFTER_MAX", flush=True); sys.exit(0)
    rows = []
    end = min(c0 + a.chunk, len(pairs))
    for s in range(c0, end, 16):
        rows.extend(X.normalize_nli(lab) for lab in clf(pairs[s:min(s + 16, end)]))
    f.write_text(json.dumps(rows)); done_now += 1
    print("chunk", c0, "done", datetime.now().strftime("%H:%M:%S"), flush=True)

rows = []
for c0 in range(0, len(pairs), a.chunk):
    rows.extend(json.loads((ck / f"chunk_{c0:06d}.json").read_text()))
nli = pd.DataFrame(rows)
frame["nli_entailment"] = nli["entailment"].to_numpy()
frame["nli_neutral"] = nli["neutral"].to_numpy()
frame["nli_contradiction"] = nli["contradiction"].to_numpy()
frame["nli_unsupported"] = np.clip(frame.nli_contradiction + 0.35 * frame.nli_neutral, 0, 1)
frame.to_csv(a.output, index=False)
a.output.with_suffix(".manifest.json").write_text(json.dumps({
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "records": str(a.records), "examples": int(len(frame)),
    "models": {"embedding": "sentence-transformers/all-MiniLM-L6-v2", "nli": "facebook/bart-large-mnli"},
    "software": {n: importlib.metadata.version(n) for n in ("numpy", "pandas", "torch", "transformers", "sentence-transformers")},
    "python": sys.version, "platform": platform.platform()}, indent=2))
print("ALL_DONE", flush=True)
