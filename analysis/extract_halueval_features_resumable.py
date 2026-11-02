"""Resumable wrapper around scripts/extract_halueval_features.py.

Uses the *same* functions (tokenizer, lexical, semantic, NLI normalisation,
0.35 unsupportedness rule) as the original script, but checkpoints NLI output
in 1,000-pair chunks so a time-limited run can resume.
"""
import json, sys, importlib.metadata, platform
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import extract_halueval_features as X  # noqa: E402
from transformers import pipeline  # noqa: E402

torch.set_num_threads(2)
W = Path(__file__).resolve().parents[1]
CK = W / "derived/features/nli_chunks"; CK.mkdir(parents=True, exist_ok=True)
BASE = W / "derived/features/halueval_base_semantic.csv"
OUT = W / "derived/features/halueval_full_features.csv"
SRC = W / "data/halueval_qa_main.json"
CHUNK = 1000
import os
MAX_CHUNKS = int(os.environ.get('MAX_CHUNKS', '999'))
done_now = 0

if BASE.exists():
    frame = pd.read_csv(BASE)
else:
    frame = X.load_source(SRC, None, 11)
    frame = X.add_basic_features(frame)
    frame = X.add_semantic_features(frame, "sentence-transformers/all-MiniLM-L6-v2", "cpu", 128)
    frame.to_csv(BASE, index=False)
    print("base+semantic saved", flush=True)

clf = pipeline("text-classification", model="facebook/bart-large-mnli", device=-1, top_k=None, truncation=True, max_length=512)
pairs = [{"text": e, "text_pair": a} for e, a in zip(frame.evidence, frame.answer)]
for c0 in range(0, len(pairs), CHUNK):
    f = CK / f"chunk_{c0:05d}.json"
    if f.exists():
        continue
    if done_now >= MAX_CHUNKS:
        print('STOP_AFTER_MAX', flush=True); sys.exit(0)
    rows = []
    for s in range(c0, min(c0 + CHUNK, len(pairs)), 16):  # same 16-pair call pattern as original
        rows.extend(X.normalize_nli(lab) for lab in clf(pairs[s:min(s + 16, c0 + CHUNK)]))
    f.write_text(json.dumps(rows))
    done_now += 1
    print("chunk", c0, "done", datetime.now().strftime("%H:%M:%S"), flush=True)

rows = []
for c0 in range(0, len(pairs), CHUNK):
    rows.extend(json.loads((CK / f"chunk_{c0:05d}.json").read_text()))
nli = pd.DataFrame(rows)
frame["nli_entailment"] = nli["entailment"].to_numpy()
frame["nli_neutral"] = nli["neutral"].to_numpy()
frame["nli_contradiction"] = nli["contradiction"].to_numpy()
frame["nli_unsupported"] = np.clip(frame.nli_contradiction + 0.35 * frame.nli_neutral, 0, 1)
frame.to_csv(OUT, index=False)
manifest = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "source": {"path": str(SRC), "sha256": X.file_sha256(SRC)},
    "note": "Regenerated on CPU with checkpointed NLI for the major revision; same functions as extract_halueval_features.py",
    "examples": int(len(frame)), "question_groups": int(frame.question_id.nunique()),
    "software": {n: importlib.metadata.version(n) for n in ("numpy", "pandas", "torch", "transformers", "sentence-transformers", "scikit-learn")},
    "python": sys.version, "platform": platform.platform(),
}
OUT.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
print("ALL_DONE", flush=True)
