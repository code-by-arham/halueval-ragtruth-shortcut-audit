import json, re, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
def frac(ev, an):
    n = np.array([len(tk(e, a, truncation=False)["input_ids"]) for e, a in zip(ev, an)])
    return n
rt = pd.read_csv(ROOT / "derived/ragtruth/ragtruth_qa_features.csv")
n = frac(rt.evidence.tolist(), rt.answer.tolist())
out = {"ragtruth_pair_tokens_median": float(np.median(n)), "ragtruth_pair_tokens_mean": float(n.mean()),
       "ragtruth_frac_over_512": float((n > 512).mean()), "ragtruth_test_frac_over_512": float((n[rt.official_split.eq("test").to_numpy()] > 512).mean())}
rows = [json.loads(l) for l in open(ROOT / "data/halueval_qa_main.json")]
ev = [r["knowledge"] for r in rows for _ in (0, 1)]; an = [x for r in rows for x in (r["right_answer"], r["hallucinated_answer"])]
h = frac(ev, an)
out.update({"halueval_pair_tokens_median": float(np.median(h)), "halueval_pair_tokens_mean": float(h.mean()), "halueval_frac_over_512": float((h > 512).mean())})
print(json.dumps(out, indent=1)); json.dump(out, open(ROOT / "results/reviewer_analyses/nli_truncation.json", "w"), indent=1)
