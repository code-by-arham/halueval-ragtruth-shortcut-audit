"""Second-round revision analyses (R1.5): HaluBench external subsets and MiniCheck.

Protocol is unchanged: HaluEval group-safe splits (seeds 11-55), validation-only
thresholds, fixed-prediction length matching, frozen transfer with HaluEval
models and HaluEval validation thresholds.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_truthguard import DIRECT_LENGTH_EXCLUDED as FEATS, threshold_for_f1  # noqa: E402

OUT = ROOT / "results" / "revision2"; OUT.mkdir(parents=True, exist_ok=True)
EV = ROOT / "results/halueval_full"
MC = ROOT / "derived/minicheck"
SEEDS = [11, 22, 33, 44, 55]
WIDTHS = [1, 2, 5, 10, 20]
SUBSETS = ["DROP", "FinanceBench", "covidQA", "pubmedQA"]
V: dict[str, object] = {}

H = pd.read_csv(ROOT / "derived/features/halueval_full_features.csv")
HB = pd.read_csv(ROOT / "halubench/features/halubench_features.csv", keep_default_na=False)
RG = pd.read_csv(ROOT / "derived/ragtruth/ragtruth_qa_features.csv")
RT = RG[RG.official_split == "test"].reset_index(drop=True)
SPL = {s: pd.read_csv(ROOT / f"derived/split_manifests/halueval_qa_seed_{s}_splits.csv") for s in SEEDS}
ARCH = pd.read_csv(EV / "all_seed_results.csv")
THR = {(m, s): t for m, s, t in zip(ARCH.method, ARCH.seed, ARCH.threshold)}


def parts(seed, frame=H):
    j = frame.merge(SPL[seed][["question_id", "variant", "split"]], on=["question_id", "variant"], validate="one_to_one")
    return {k: j[j.split == k].reset_index(drop=True) for k in ("train", "validation", "test")}


def metr(y, score, thr):
    y = np.asarray(y); score = np.asarray(score, dtype=float); pred = (score >= thr).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {"n": int(len(y)), "accuracy": accuracy_score(y, pred), "precision": p, "recall": r, "f1": f,
            "roc_auc": roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan,
            "pr_auc": average_precision_score(y, score) if len(np.unique(y)) == 2 else np.nan,
            "pred_positive_frac": float(pred.mean())}


def length_match(frame, width, seed):
    work = frame.copy(); work["_bin"] = (work.answer_len // width) * width
    rng = np.random.default_rng(seed); chosen = []
    for _, g in work.groupby("_bin", sort=True):
        f0, f1 = g[g.label == 0], g[g.label == 1]; c = min(len(f0), len(f1))
        if c:
            chosen.extend(rng.choice(f0.index.to_numpy(), c, replace=False)); chosen.extend(rng.choice(f1.index.to_numpy(), c, replace=False))
    return work.loc[chosen]


def matched(test, score, thr, seed):
    t = test.assign(_p=(np.asarray(score) >= thr).astype(int))
    out = {}
    for w in WIDTHS:
        m = length_match(t, w, seed)
        out[f"matched_f1_w{w}"] = precision_recall_fscore_support(m.label, m._p, average="binary", zero_division=0)[2]
        out[f"matched_n_w{w}"] = len(m)
    return out


def summ(df, keys):
    num = df.select_dtypes("number").drop(columns=[c for c in ("seed",) if c in df], errors="ignore")
    g = pd.concat([df[keys], num], axis=1).groupby(keys)
    m, s = g.mean(), g.std()
    m.columns = [f"{c}_mean" for c in m.columns]; s.columns = [f"{c}_std" for c in s.columns]
    return pd.concat([m, s], axis=1).reset_index()


# ------------------------------------------------------------------ 1. HaluBench descriptives
from transformers import AutoTokenizer  # noqa: E402
tk = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
HB["pair_tokens"] = [len(tk(e, a, truncation=False)["input_ids"]) for e, a in zip(HB.evidence, HB.answer)]
desc = []
for s in SUBSETS + ["pooled"]:
    d = HB if s == "pooled" else HB[HB.source_ds == s]
    desc.append({"subset": s, "n": len(d), "hallucinated": int(d.label.sum()), "groups": d.question_id.nunique(),
                 "answer_len_median_factual": d[d.label == 0].answer_len.median(),
                 "answer_len_median_halluc": d[d.label == 1].answer_len.median(),
                 "answer_len_mean_factual": d[d.label == 0].answer_len.mean(),
                 "answer_len_mean_halluc": d[d.label == 1].answer_len.mean(),
                 "length_auc": roc_auc_score(d.label, d.answer_len),
                 "length_label_pointbiserial": float(np.corrcoef(d.answer_len, d.label)[0, 1]),
                 "evidence_len_median": d.evidence_len.median(),
                 "nli_pairs_over_512_frac": float((d.pair_tokens > 512).mean())})
pd.DataFrame(desc).to_csv(OUT / "halubench_descriptives.csv", index=False)

# ------------------------------------------------------------------ 2. frozen HaluEval -> HaluBench
MODELS = ["direct_length_excluded_gradient_boosting", "direct_length_excluded_random_forest",
          "direct_length_excluded_logistic_regression", "length_aware_gradient_boosting"]
BASE = {"artifact_answer_length": "answer_len", "lexical_mismatch": "lexical_mismatch"}
rows = []
for s in SEEDS:
    for m in MODELS:
        mdl = joblib.load(EV / f"seed_{s}_{m}.joblib"); cols = list(mdl.feature_names_in_)
        sc = mdl.predict_proba(HB[cols])[:, 1]
        for sub in SUBSETS + ["pooled"]:
            ix = HB.index if sub == "pooled" else HB.index[HB.source_ds == sub]
            r = metr(HB.label.loc[ix], sc[ix], THR[(m, s)]); r.update({"seed": s, "method": m, "subset": sub,
                     "score_length_corr": float(np.corrcoef(sc[ix], HB.answer_len.loc[ix])[0, 1])})
            rows.append(r)
    for m, col in BASE.items():
        sc = HB[col].to_numpy(dtype=float)
        for sub in SUBSETS + ["pooled"]:
            ix = HB.index if sub == "pooled" else HB.index[HB.source_ds == sub]
            r = metr(HB.label.loc[ix], sc[ix], THR[(m, s)]); r.update({"seed": s, "method": m, "subset": sub})
            rows.append(r)
fz = pd.DataFrame(rows); fz.to_csv(OUT / "halubench_frozen_per_seed.csv", index=False)
summ(fz, ["method", "subset"]).to_csv(OUT / "halubench_frozen_summary.csv", index=False)

# ------------------------------------------------------------------ 3. HaluBench in-domain (from evaluate_truthguard outputs)
IDD = ROOT / "results/halubench/in_domain"
if (IDD / "all_seed_results.csv").exists():
    rows = []
    src = HB[["question_id", "variant", "source_ds", "answer_len"]]
    for s in SEEDS:
        p = pd.read_csv(IDD / f"seed_{s}_test_predictions.csv", keep_default_na=False).merge(src, on=["question_id", "variant"])
        for m in ["direct_length_excluded_gradient_boosting", "artifact_answer_length", "length_aware_gradient_boosting"]:
            for sub in SUBSETS + ["pooled"]:
                d = p if sub == "pooled" else p[p.source_ds == sub]
                if d.label.nunique() < 2:
                    continue
                y, sc, pr = d.label, d[f"{m}_score"], d[f"{m}_prediction"]
                pp, rr, ff, _ = precision_recall_fscore_support(y, pr, average="binary", zero_division=0)
                rows.append({"seed": s, "method": m, "subset": sub, "n": len(d), "f1": ff, "accuracy": accuracy_score(y, pr),
                             "roc_auc": roc_auc_score(y, sc)})
    ind = pd.DataFrame(rows); ind.to_csv(OUT / "halubench_in_domain_per_subset_per_seed.csv", index=False)
    summ(ind, ["method", "subset"]).to_csv(OUT / "halubench_in_domain_per_subset_summary.csv", index=False)

# ------------------------------------------------------------------ 4. MiniCheck
def mc_load(name, keys):
    f = MC / name
    if f.exists():
        return pd.read_csv(f, keep_default_na=False)
    if name == "halubench_minicheck_answer.csv":
        # DROP, FinanceBench, PubMedQA in full + CovidQA stratified subsample of 200 (cost of 2,600-word passages)
        parts_ = [MC / "halubench_minicheck_answer_noncovid.csv", MC / "halubench_covidqa200_minicheck_answer.csv"]
        if all(p.exists() for p in parts_):
            return pd.concat([pd.read_csv(p, keep_default_na=False) for p in parts_], ignore_index=True)
    if name == "halubench_minicheck_qa.csv":
        # question+answer claims were run on DROP, FinanceBench and PubMedQA only (span answers); not on CovidQA
        p = MC / "halubench_minicheck_qa_noncovid.csv"
        if p.exists():
            return pd.read_csv(p, keep_default_na=False)
    return None


for mode in ("answer", "qa"):
    he = mc_load(f"halueval_minicheck_{mode}.csv", None)
    if he is None:
        continue
    Hm = H.merge(he[["question_id", "variant", "minicheck_halluc_score"]], on=["question_id", "variant"], validate="one_to_one")
    rows, rows12 = [], []
    for s in SEEDS:
        p = parts(s, Hm)
        va, te = p["validation"], p["test"]
        thr = threshold_for_f1(va.label.to_numpy(), va.minicheck_halluc_score.to_numpy())
        for tname, t in (("default_0.5", 0.5), ("validation_tuned", thr)):
            r = metr(te.label, te.minicheck_halluc_score, t)
            r.update({"seed": s, "threshold_rule": tname, "threshold": t,
                      "score_length_corr": float(np.corrcoef(te.minicheck_halluc_score, te.answer_len)[0, 1])})
            r.update(matched(te, te.minicheck_halluc_score, t, s)); rows.append(r)
        # meta classifier + MiniCheck as a 12th feature
        cols = FEATS + ["minicheck_halluc_score"]
        g = GradientBoostingClassifier(n_estimators=120, learning_rate=0.05, max_depth=2, random_state=s).fit(p["train"][cols], p["train"].label)
        tv = threshold_for_f1(va.label.to_numpy(), g.predict_proba(va[cols])[:, 1]); st = g.predict_proba(te[cols])[:, 1]
        r = metr(te.label, st, tv); r.update({"seed": s, "threshold": tv, "score_length_corr": float(np.corrcoef(st, te.answer_len)[0, 1])})
        r.update(matched(te, st, tv, s))
        rt = mc_load("ragtruth_test_minicheck.csv", None)
        if rt is not None and mode == "answer":
            R2 = RT.merge(rt[["response_id", "minicheck_halluc_score"]], on="response_id")
            sr = g.predict_proba(R2[cols])[:, 1]; rr = metr(R2.label, sr, tv)
            r.update({f"ragtruth_frozen_{k}": v for k, v in rr.items()})
        hb = mc_load(f"halubench_minicheck_{mode}.csv", None)
        if hb is not None:
            B2 = HB.merge(hb[["variant", "minicheck_halluc_score"]], on="variant")
            sb = g.predict_proba(B2[cols])[:, 1]; rb = metr(B2.label, sb, tv)
            r.update({f"halubench_frozen_{k}": v for k, v in rb.items()})
        rows12.append(r)
    d = pd.DataFrame(rows); d.to_csv(OUT / f"minicheck_halueval_{mode}_per_seed.csv", index=False)
    summ(d, ["threshold_rule"]).to_csv(OUT / f"minicheck_halueval_{mode}_summary.csv", index=False)
    d12 = pd.DataFrame(rows12); d12.to_csv(OUT / f"meta_gb_plus_minicheck_{mode}_per_seed.csv", index=False)
    d12.drop(columns="seed").mean(numeric_only=True).to_frame("mean").join(d12.drop(columns="seed").std(numeric_only=True).to_frame("std")).to_csv(OUT / f"meta_gb_plus_minicheck_{mode}_summary.csv")
    V[f"minicheck_halueval_{mode}_val_thresholds"] = {}
    for s in SEEDS:
        p = parts(s, Hm); V[f"minicheck_halueval_{mode}_val_thresholds"][s] = threshold_for_f1(p["validation"].label.to_numpy(), p["validation"].minicheck_halluc_score.to_numpy())

# MiniCheck on RAGTruth test and HaluBench (zero-shot; default 0.5 and frozen HaluEval-validation threshold)
ext = []
rt = mc_load("ragtruth_test_minicheck.csv", None)
if rt is not None:
    R2 = RT.merge(rt[["response_id", "minicheck_halluc_score", "minicheck_n_sentences"]], on="response_id")
    V["ragtruth_test_minicheck_sentences_median"] = float(R2.minicheck_n_sentences.median())
    thrs = V.get("minicheck_halueval_answer_val_thresholds", {})
    r = metr(R2.label, R2.minicheck_halluc_score, 0.5); r.update({"dataset": "RAGTruth test", "subset": "all", "claim_mode": "answer (sentence-split)", "threshold_rule": "default_0.5",
              "score_length_corr": float(np.corrcoef(R2.minicheck_halluc_score, R2.answer_len)[0, 1])}); ext.append(r)
    for s, t in thrs.items():
        r = metr(R2.label, R2.minicheck_halluc_score, t); r.update({"dataset": "RAGTruth test", "subset": "all", "claim_mode": "answer (sentence-split)", "threshold_rule": f"frozen_halueval_val_seed{s}"}); ext.append(r)
for mode in ("answer", "qa"):
    hb = mc_load(f"halubench_minicheck_{mode}.csv", None)
    if hb is None:
        continue
    B2 = HB.merge(hb[["variant", "minicheck_halluc_score"]], on="variant")
    # frozen direct-length-excluded GB on exactly the same rows, for a like-for-like comparison
    gb_sc = np.mean([joblib.load(EV / f"seed_{s}_direct_length_excluded_gradient_boosting.joblib").predict_proba(B2[FEATS])[:, 1] for s in SEEDS], axis=0)
    thrs = V.get(f"minicheck_halueval_{mode}_val_thresholds", {})
    for sub in SUBSETS + ["pooled"]:
        d = B2 if sub == "pooled" else B2[B2.source_ds == sub]
        if d.empty:
            continue
        if sub == "pooled" and "covidQA" not in set(B2.source_ds):
            sub = "pooled_excl_covidQA"
        V.setdefault(f"halubench_same_rows_frozen_gb_seedavg_auc_{mode}", {})[sub] = {
            "n": int(len(d)), "frozen_gb_auc": float(roc_auc_score(d.label, gb_sc[d.index])),
            "minicheck_auc": float(roc_auc_score(d.label, d.minicheck_halluc_score))}
        r = metr(d.label, d.minicheck_halluc_score, 0.5); r.update({"dataset": "HaluBench", "subset": sub, "claim_mode": mode, "threshold_rule": "default_0.5",
                 "score_length_corr": float(np.corrcoef(d.minicheck_halluc_score, d.answer_len)[0, 1])}); ext.append(r)
        for s, t in thrs.items():
            r = metr(d.label, d.minicheck_halluc_score, t); r.update({"dataset": "HaluBench", "subset": sub, "claim_mode": mode, "threshold_rule": f"frozen_halueval_val_seed{s}"}); ext.append(r)
# answer vs question+answer claims on the identical 3,000 DROP/FinanceBench/PubMedQA rows
ha, hq = mc_load("halubench_minicheck_answer_noncovid.csv", None), mc_load("halubench_minicheck_qa.csv", None)
if ha is not None and hq is not None:
    P = HB.merge(ha[["variant", "minicheck_halluc_score"]], on="variant").merge(
        hq[["variant", "minicheck_halluc_score"]], on="variant", suffixes=("_answer", "_qa"))
    V["halubench_answer_vs_qa_claims_auc"] = {
        sub: {"n": int(len(d)), "answer_auc": float(roc_auc_score(d.label, d.minicheck_halluc_score_answer)),
              "qa_auc": float(roc_auc_score(d.label, d.minicheck_halluc_score_qa))}
        for sub, d in [("pooled_excl_covidQA", P)] + [(s, P[P.source_ds == s]) for s in ["DROP", "FinanceBench", "pubmedQA"]]}
if ext:
    e = pd.DataFrame(ext); e.to_csv(OUT / "minicheck_external_all_rows.csv", index=False)
    e["rule"] = np.where(e.threshold_rule.str.startswith("frozen"), "frozen_halueval_val", e.threshold_rule)
    e.groupby(["dataset", "subset", "claim_mode", "rule"]).mean(numeric_only=True).reset_index().to_csv(OUT / "minicheck_external_summary.csv", index=False)


def default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


(OUT / "revision2_values.json").write_text(json.dumps(V, indent=1, default=default))
print("DONE")
