"""Score (evidence, response) pairs with MiniCheck (Tang et al., 2024), checkpointed and batched.

Follows the MiniCheck authors' recommendation for multi-sentence responses:
the response is split into sentences (NLTK), every sentence is checked against
the document (document split into ~400-token chunks exactly as in the official
Inferencer, max support over chunks), and the response-level support is the
minimum over its sentences. Hallucination score = 1 - min sentence support.

The official Inferencer scores one example at a time. Here the identical
(chunk, claim) inputs are flattened, sorted by length and batched, which gives
the same probabilities (verified against the official per-example path,
see --verify) at a fraction of the CPU time.

Claim modes
  answer : the answer text alone (same hypothesis as the NLI feature)
  qa     : "<question> <answer>", so short QA answers become a full statement
"""
import argparse, json, os, sys
from datetime import datetime
from pathlib import Path

import numpy as np, pandas as pd, torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party"))
from minicheck.inference import Inferencer, sent_tokenize_with_newlines  # noqa: E402
from nltk.tokenize import sent_tokenize  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--input", type=Path, required=True)
ap.add_argument("--id-cols", nargs="+", required=True)
ap.add_argument("--output", type=Path, required=True)
ap.add_argument("--claim-mode", choices=["answer", "qa"], default="answer")
ap.add_argument("--split-sentences", action="store_true")
ap.add_argument("--filter", default=None)
ap.add_argument("--chunk", type=int, default=1000, help="examples per checkpoint")
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--verify", type=int, default=0, help="also run the official per-example path on N rows and compare")
a = ap.parse_args()
torch.set_num_threads(int(os.environ.get("NUM_THREADS", max(1, os.cpu_count() or 1))))
MAX_CHUNKS = int(os.environ.get("MAX_CHUNKS", "100000"))
import time
DEADLINE = float(os.environ["DEADLINE_EPOCH"]) if os.environ.get("DEADLINE_EPOCH") else time.time() + 60 * float(os.environ.get("BUDGET_MIN", "1e9"))

df = pd.read_csv(a.input, keep_default_na=False)
if a.filter:
    df = df.query(a.filter).reset_index(drop=True)
docs = df.evidence.astype(str).tolist()
claims = (df.question.astype(str) + " " + df.answer.astype(str)).tolist() if a.claim_mode == "qa" else df.answer.astype(str).tolist()

inf = Inferencer(model_name="roberta-large", max_model_len=None, batch_size=16, cache_dir=None)
inf.chunk_size = 400  # MiniCheck default for RoBERTa/DeBERTa
tok, model = inf.tokenizer, inf.model


def doc_chunks(doc):
    """Identical to Inferencer.inference_per_example's chunking for RoBERTa."""
    sents = sent_tokenize_with_newlines(doc) or [""]
    out, cur, cnt = [], [], 0
    for s in sents:
        n = len(tok(s, padding=False, add_special_tokens=False, max_length=inf.max_model_len, truncation=True)["input_ids"])
        if cnt + n > inf.chunk_size:
            out.append(" ".join(cur)); cur, cnt = [s], n
        else:
            cur.append(s); cnt += n
    if cur:
        out.append(" ".join(cur))
    out = [c.replace(" \n ", "\n").strip() for c in out]
    return [c for c in out if c != ""]


def claim_sents(c):
    if not a.split_sentences:
        return [c]
    s = [x for x in sent_tokenize(c) if x.strip()]
    return s or [c if c.strip() else "."]


def score_rows(idx):
    pairs = []  # (row, sentence index, text)
    meta = {}
    for i in idx:
        ch = doc_chunks(docs[i]) or [""]
        ss = claim_sents(claims[i])
        meta[i] = len(ss)
        for j, s in enumerate(ss):
            for c in ch:
                pairs.append((i, j, tok.eos_token.join([c, s])))
    lens = np.array([len(p[2]) for p in pairs]); order = np.argsort(lens)
    probs = np.empty(len(pairs))
    with torch.no_grad():
        for b in range(0, len(order), a.batch):
            sel = order[b:b + a.batch]
            enc = tok([pairs[k][2] for k in sel], max_length=inf.max_model_len, truncation=True, padding=True, return_tensors="pt")
            probs[sel] = F.softmax(model(**enc).logits, dim=1)[:, 1].numpy()
    best = {}
    for (i, j, _), p in zip(pairs, probs):
        best[(i, j)] = max(best.get((i, j), -1.0), p)
    rows = []
    for i in idx:
        sp = [best[(i, j)] for j in range(meta[i])]
        rows.append({"n_sentences": meta[i], "min_support": float(min(sp)), "mean_support": float(np.mean(sp))})
    return rows


if a.verify:
    vi = list(range(min(a.verify, len(df))))
    fast = [r["min_support"] for r in score_rows(vi)]
    slow = [min(inf.inference_per_example(docs[i], s)["max_support_prob"] for s in claim_sents(claims[i])) for i in vi]
    print("VERIFY max |batched - official| =", float(np.max(np.abs(np.array(fast) - np.array(slow)))), flush=True)

ck = a.output.parent / (a.output.stem + "_chunks"); ck.mkdir(parents=True, exist_ok=True)
done_now = 0
for c0 in range(0, len(df), a.chunk):
    f = ck / f"chunk_{c0:06d}.json"
    if f.exists():
        continue
    if done_now >= MAX_CHUNKS or time.time() > DEADLINE:
        print("STOP_AFTER_MAX", flush=True); sys.exit(0)
    part = f.with_suffix(".partial.json")
    rows_done = json.loads(part.read_text()) if part.exists() else []
    end = min(c0 + a.chunk, len(df))
    for s0 in range(c0 + len(rows_done), end, 25):  # save progress every 25 rows inside a chunk
        if time.time() > DEADLINE:
            print("STOP_AFTER_MAX", flush=True); sys.exit(0)
        rows_done.extend(score_rows(list(range(s0, min(s0 + 25, end)))))
        part.write_text(json.dumps(rows_done))
    f.write_text(json.dumps(rows_done)); part.unlink(missing_ok=True)
    done_now += 1
    print("chunk", c0, "done", datetime.now().strftime("%H:%M:%S"), flush=True)

rows = []
for c0 in range(0, len(df), a.chunk):
    rows.extend(json.loads((ck / f"chunk_{c0:06d}.json").read_text()))
out = df[a.id_cols].copy()
res = pd.DataFrame(rows)
out["minicheck_n_sentences"] = res.n_sentences.to_numpy()
out["minicheck_min_support"] = res.min_support.to_numpy()
out["minicheck_mean_support"] = res.mean_support.to_numpy()
out["minicheck_halluc_score"] = 1.0 - out.minicheck_min_support
out.to_csv(a.output, index=False)
print("ALL_DONE", flush=True)
