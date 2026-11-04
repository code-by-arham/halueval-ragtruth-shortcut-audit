"""Independent re-computation of the headline second-round numbers.

Written separately from new_dataset_analyses.py (no shared helper code): reads the
raw score/feature CSVs, recomputes each value with plain sklearn calls and compares
it with results/revision2/revision2_values.json and the summary CSVs.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

R = Path(__file__).resolve().parents[1]
OUT = R / "results/revision2"
V = json.loads((OUT / "revision2_values.json").read_text())
MC = R / "derived/minicheck"
EV = R / "results/halueval_full"
SEEDS = [11, 22, 33, 44, 55]
FEATS = ["lexical_overlap", "lexical_mismatch", "jaccard", "semantic_answer_evidence_sim", "semantic_answer_question_sim",
         "semantic_mismatch", "nli_entailment", "nli_neutral", "nli_contradiction", "nli_unsupported", "uncertainty"]
checks = []


def chk(name, mine, reported, tol=5e-4):
    ok = abs(float(mine) - float(reported)) <= tol
    checks.append({"check": name, "recomputed": round(float(mine), 4), "reported": round(float(reported), 4), "ok": ok})


# HaluBench composition and length
HB = pd.read_csv(R / "halubench/features/halubench_features.csv", keep_default_na=False)
desc = pd.read_csv(OUT / "halubench_descriptives.csv").set_index("subset")
for s, d in list(HB.groupby("source_ds")) + [("pooled", HB)]:
    chk(f"HaluBench {s} n", len(d), desc.loc[s, "n"], 0)
    chk(f"HaluBench {s} hallucinated", d.label.sum(), desc.loc[s, "hallucinated"], 0)
    chk(f"HaluBench {s} length AUC", roc_auc_score(d.label, d.answer_len), desc.loc[s, "length_auc"])

# Frozen HaluEval GB on HaluBench (seed-mean AUC, pooled and per subset)
fz = pd.read_csv(OUT / "halubench_frozen_summary.csv")
fz = fz[fz.method == "direct_length_excluded_gradient_boosting"].set_index("subset")
sc = {s: joblib.load(EV / f"seed_{s}_direct_length_excluded_gradient_boosting.joblib").predict_proba(HB[FEATS])[:, 1] for s in SEEDS}
for sub in ["DROP", "FinanceBench", "covidQA", "pubmedQA", "pooled"]:
    m = np.ones(len(HB), bool) if sub == "pooled" else (HB.source_ds == sub).to_numpy()
    chk(f"Frozen GB HaluBench {sub} AUC", np.mean([roc_auc_score(HB.label[m], sc[s][m]) for s in SEEDS]), fz.loc[sub, "roc_auc_mean"])

# MiniCheck on HaluEval test splits (answer claims)
H = pd.read_csv(R / "derived/features/halueval_full_features.csv")
mc = pd.read_csv(MC / "halueval_minicheck_answer.csv")
Hm = H[["question_id", "variant", "label", "answer_len"]].merge(mc[["question_id", "variant", "minicheck_halluc_score"]], on=["question_id", "variant"])
summ = pd.read_csv(OUT / "minicheck_halueval_answer_summary.csv").set_index("threshold_rule")
aucs, f1s, corrs = [], [], []
for s in SEEDS:
    sp = pd.read_csv(R / f"derived/split_manifests/halueval_qa_seed_{s}_splits.csv")
    te = Hm.merge(sp[sp.split == "test"][["question_id", "variant"]], on=["question_id", "variant"])
    aucs.append(roc_auc_score(te.label, te.minicheck_halluc_score))
    f1s.append(f1_score(te.label, (te.minicheck_halluc_score >= 0.5).astype(int)))
    corrs.append(np.corrcoef(te.minicheck_halluc_score, te.answer_len)[0, 1])
chk("MiniCheck HaluEval test AUC", np.mean(aucs), summ.loc["default_0.5", "roc_auc_mean"])
chk("MiniCheck HaluEval test F1@0.5", np.mean(f1s), summ.loc["default_0.5", "f1_mean"])
chk("MiniCheck HaluEval score-length r", np.mean(corrs), summ.loc["default_0.5", "score_length_corr_mean"])

# MiniCheck on RAGTruth official test
RG = pd.read_csv(R / "derived/ragtruth/ragtruth_qa_features.csv")
rt = RG[RG.official_split == "test"][["response_id", "label"]].merge(pd.read_csv(MC / "ragtruth_test_minicheck.csv")[["response_id", "minicheck_halluc_score"]], on="response_id")
ext = pd.read_csv(OUT / "minicheck_external_summary.csv")
row = ext[(ext.dataset == "RAGTruth test") & (ext.rule == "default_0.5")].iloc[0]
chk("MiniCheck RAGTruth test n", len(rt), row.n, 0)
chk("MiniCheck RAGTruth test AUC", roc_auc_score(rt.label, rt.minicheck_halluc_score), row.roc_auc)
chk("MiniCheck RAGTruth test F1@0.5", f1_score(rt.label, (rt.minicheck_halluc_score >= 0.5).astype(int)), row.f1)

# MiniCheck on HaluBench (answer claims), same rows as frozen GB
hb = pd.concat([pd.read_csv(MC / f, keep_default_na=False) for f in ("halubench_minicheck_answer_noncovid.csv", "halubench_covidqa200_minicheck_answer.csv")])
B = HB.merge(hb[["variant", "minicheck_halluc_score"]], on="variant")
gb = np.mean([joblib.load(EV / f"seed_{s}_direct_length_excluded_gradient_boosting.joblib").predict_proba(B[FEATS])[:, 1] for s in SEEDS], axis=0)
rep = V["halubench_same_rows_frozen_gb_seedavg_auc_answer"]
for sub in ["DROP", "FinanceBench", "covidQA", "pubmedQA", "pooled"]:
    m = np.ones(len(B), bool) if sub == "pooled" else (B.source_ds == sub).to_numpy()
    chk(f"Same-rows {sub} n", m.sum(), rep[sub]["n"], 0)
    chk(f"Same-rows {sub} MiniCheck AUC", roc_auc_score(B.label[m], B.minicheck_halluc_score[m]), rep[sub]["minicheck_auc"])
    chk(f"Same-rows {sub} frozen GB AUC", roc_auc_score(B.label[m], gb[m]), rep[sub]["frozen_gb_auc"])

if "halubench_answer_vs_qa_claims_auc" in V:
    hq = pd.read_csv(MC / "halubench_minicheck_qa_noncovid.csv", keep_default_na=False)
    Q = HB.merge(hq[["variant", "minicheck_halluc_score"]], on="variant")
    for sub, d in [("pooled_excl_covidQA", Q)] + [(s, Q[Q.source_ds == s]) for s in ["DROP", "FinanceBench", "pubmedQA"]]:
        chk(f"HaluBench q+a claims {sub} AUC", roc_auc_score(d.label, d.minicheck_halluc_score), V["halubench_answer_vs_qa_claims_auc"][sub]["qa_auc"])

c = pd.DataFrame(checks)
c.to_csv(OUT / "verification_revision2.csv", index=False)
print(c.to_string(index=False))
print(f"\n{c.ok.sum()}/{len(c)} checks agree")
