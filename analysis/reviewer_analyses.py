"""Reviewer-driven analyses for the Engineering Reports major revision.

Runs on the regenerated HaluEval feature matrix + the archived RAGTruth feature
matrix. Nothing here changes the original protocol; every analysis reuses the
same group-safe splits, validation-only thresholds and fixed-prediction
length matching as the paper.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (accuracy_score, average_precision_score, precision_recall_fscore_support,
                             r2_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_truthguard import DIRECT_LENGTH_EXCLUDED as FEATS, threshold_for_f1  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "reviewer_analyses"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [11, 22, 33, 44, 55]
WIDTHS = [1, 2, 5, 10, 20]
MARKERS = ("i think", "probably", "maybe", "it is likely", "it seems", "as an ai",
           "i cannot verify", "not sure", "unknown")

H = pd.read_csv(ROOT / "derived/features/halueval_full_features.csv")
R = pd.read_csv(ROOT / "derived/ragtruth/ragtruth_qa_features.csv")
RT = R[R.official_split == "test"].reset_index(drop=True)
RTR = R[R.official_split == "train"].reset_index(drop=True)
SPLITS = {s: pd.read_csv(ROOT / f"derived/split_manifests/halueval_qa_seed_{s}_splits.csv") for s in SEEDS}
values: dict[str, object] = {}


def parts(seed: int, frame: pd.DataFrame = H) -> dict[str, pd.DataFrame]:
    j = frame.merge(SPLITS[seed][["question_id", "variant", "split"]], on=["question_id", "variant"], validate="one_to_one")
    return {k: j[j.split == k].reset_index(drop=True) for k in ("train", "validation", "test")}


def metrics(y, score, thr) -> dict[str, float]:
    pred = (np.asarray(score) >= thr).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {"accuracy": accuracy_score(y, pred), "precision": p, "recall": r, "f1": f,
            "roc_auc": roc_auc_score(y, score), "pr_auc": average_precision_score(y, score)}


def length_match(frame: pd.DataFrame, width: int, seed: int) -> pd.DataFrame:
    """Identical to scripts/analyze_evaluation_controls.length_match."""
    work = frame.copy()
    work["_bin"] = (work.answer_len // width) * width
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for _, g in work.groupby("_bin", sort=True):
        f0, f1 = g[g.label == 0], g[g.label == 1]
        c = min(len(f0), len(f1))
        if c:
            chosen.extend(rng.choice(f0.index.to_numpy(), c, replace=False))
            chosen.extend(rng.choice(f1.index.to_numpy(), c, replace=False))
    return work.loc[chosen].drop(columns="_bin")


def matched_f1(test: pd.DataFrame, score, thr, seed) -> dict[str, float]:
    t = test.assign(_s=score, _p=(np.asarray(score) >= thr).astype(int))
    out = {}
    for w in WIDTHS:
        m = length_match(t, w, seed)
        _, _, f, _ = precision_recall_fscore_support(m.label, m._p, average="binary", zero_division=0)
        out[f"matched_f1_w{w}"] = f
        out[f"matched_n_w{w}"] = len(m)
    return out


def gb(seed):
    return GradientBoostingClassifier(n_estimators=120, learning_rate=0.05, max_depth=2, random_state=seed)


def run_condition(name, cols, transform=None, train_filter=None, rag_transform=None) -> pd.DataFrame:
    """Train GB per seed; report conventional, residual-length and matched metrics + frozen RAGTruth transfer."""
    rows = []
    for s in SEEDS:
        p = parts(s)
        tr, va, te = p["train"], p["validation"], p["test"]
        state = None
        if transform is not None:
            state = transform("fit", tr)
            tr, va, te = (transform("apply", d, state) for d in (tr, va, te))
        if train_filter is not None:
            tr = train_filter(tr, s)
        m = gb(s).fit(tr[cols], tr.label)
        sv, st = m.predict_proba(va[cols])[:, 1], m.predict_proba(te[cols])[:, 1]
        thr = threshold_for_f1(va.label.to_numpy(), sv)
        row = {"condition": name, "seed": s, "threshold": thr, "train_n": len(tr), **metrics(te.label, st, thr),
               "score_length_corr": float(np.corrcoef(st, te.answer_len)[0, 1])}
        row.update(matched_f1(te, st, thr, s))
        rt = RT if transform is None else transform("apply", RT, state)
        sr = m.predict_proba(rt[cols])[:, 1]
        rm = metrics(rt.label, sr, thr)
        row.update({f"ragtruth_frozen_{k}": v for k, v in rm.items()})
        row["ragtruth_frozen_pred_positive_frac"] = float((sr >= thr).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, key="condition") -> pd.DataFrame:
    num = df.drop(columns=["seed"]).groupby(key).agg(["mean", "std"])
    num.columns = [f"{a}_{b}" for a, b in num.columns]
    return num.reset_index()


# ---------------------------------------------------------------- 1. length stats
def q(x):
    x = pd.Series(x)
    return {"mean": x.mean(), "median": x.median(), "p25": x.quantile(.25), "p75": x.quantile(.75),
            "p95": x.quantile(.95), "max": x.max(), "frac_ge_40": (x >= 40).mean(), "frac_le_9": (x <= 9).mean()}


values["length_stats"] = {
    "halueval_factual": q(H[H.label == 0].answer_len), "halueval_hallucinated": q(H[H.label == 1].answer_len),
    "halueval_all": q(H.answer_len),
    "ragtruth_all": q(R.answer_len), "ragtruth_factual": q(R[R.label == 0].answer_len),
    "ragtruth_hallucinated": q(R[R.label == 1].answer_len), "ragtruth_test": q(RT.answer_len),
    "halueval_evidence_len": q(H.evidence_len), "ragtruth_evidence_len": q(R.evidence_len),
    "halueval_len_label_pointbiserial": float(np.corrcoef(H.answer_len, H.label)[0, 1]),
    "ragtruth_len_label_pointbiserial": float(np.corrcoef(R.answer_len, R.label)[0, 1]),
    "halueval_len_auc": roc_auc_score(H.label, H.answer_len), "ragtruth_len_auc": roc_auc_score(R.label, R.answer_len),
    "ragtruth_frac_longer_than_halueval_max": float((R.answer_len > H.answer_len.max()).mean()),
    "ragtruth_frac_longer_than_halueval_p99": float((R.answer_len > H.answer_len.quantile(.99)).mean()),
    "halueval_pairs_where_halluc_longer": float(
        (H[H.label == 1].set_index("question_id").answer_len > H[H.label == 0].set_index("question_id").answer_len).mean()),
    "halueval_pairs_equal_length": float(
        (H[H.label == 1].set_index("question_id").answer_len == H[H.label == 0].set_index("question_id").answer_len).mean()),
}

# ---------------------------------------------------------------- 2. feature-length association (R3.4 / R1.2)
def assoc(frame):
    out = {}
    for f in FEATS:
        out[f] = {"pearson": float(np.corrcoef(frame[f], frame.answer_len)[0, 1]) if frame[f].std() > 0 else None,
                  "spearman": float(spearmanr(frame[f], frame.answer_len).statistic) if frame[f].std() > 0 else None}
        for lab in (0, 1):
            sub = frame[frame.label == lab]
            out[f][f"spearman_label{lab}"] = float(spearmanr(sub[f], sub.answer_len).statistic) if sub[f].std() > 0 else None
    return out


values["feature_length_assoc_halueval"] = assoc(H)
values["feature_length_assoc_ragtruth"] = assoc(R)

# How much answer length is recoverable from the 11 non-length features?
r2 = []
for s in SEEDS:
    p = parts(s)
    reg = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=s)
    reg.fit(p["train"][FEATS], p["train"].answer_len)
    pr = reg.predict(p["test"][FEATS])
    r2.append({"seed": s, "r2": r2_score(p["test"].answer_len, pr),
               "spearman": spearmanr(pr, p["test"].answer_len).statistic,
               "r2_log": None})
    regl = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=s)
    regl.fit(p["train"][FEATS], np.log1p(p["train"].answer_len))
    r2[-1]["r2_log"] = r2_score(np.log1p(p["test"].answer_len), regl.predict(p["test"][FEATS]))
r2 = pd.DataFrame(r2)
values["length_recoverable_from_11_features_halueval"] = {
    "r2_mean": r2.r2.mean(), "r2_std": r2.r2.std(), "r2_log_mean": r2.r2_log.mean(), "spearman_mean": r2.spearman.mean()}
# per-group recoverability
grp = {"lexical": ["lexical_overlap", "lexical_mismatch", "jaccard"],
       "semantic": ["semantic_answer_evidence_sim", "semantic_answer_question_sim", "semantic_mismatch"],
       "nli": ["nli_entailment", "nli_neutral", "nli_contradiction", "nli_unsupported"],
       "uncertainty": ["uncertainty"]}
values["length_recoverable_by_group_r2_log"] = {}
for g, cols in grp.items():
    vals = []
    for s in SEEDS:
        p = parts(s)
        reg = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=s)
        reg.fit(p["train"][cols], np.log1p(p["train"].answer_len))
        vals.append(r2_score(np.log1p(p["test"].answer_len), reg.predict(p["test"][cols])))
    values["length_recoverable_by_group_r2_log"][g] = float(np.mean(vals))

# ---------------------------------------------------------------- 3. lexical / semantic redundancy (R2.7)
values["lexical_semantic_intercorrelation_halueval"] = H[
    ["lexical_overlap", "lexical_mismatch", "jaccard", "semantic_answer_evidence_sim", "semantic_mismatch",
     "semantic_answer_question_sim"]].corr(method="spearman").round(4).to_dict()

# ---------------------------------------------------------------- 4. uncertainty marker prevalence (R2.8 / R3.1)
def marker_stats(frame):
    low = frame.answer.str.lower()
    d = {m: int(low.str.contains(m, regex=False).sum()) for m in MARKERS}
    d["any_marker_frac"] = float((frame.uncertainty > 0).mean())
    d["any_marker_frac_factual"] = float((frame[frame.label == 0].uncertainty > 0).mean())
    d["any_marker_frac_hallucinated"] = float((frame[frame.label == 1].uncertainty > 0).mean())
    d["value_counts"] = {str(k): int(v) for k, v in frame.uncertainty.value_counts().sort_index().items()}
    return d


values["uncertainty_markers_halueval"] = marker_stats(H)
values["uncertainty_markers_ragtruth"] = marker_stats(R)

# ---------------------------------------------------------------- 5. feature importance + group ablation (R2.6, R3.4)
imp = []
for s in SEEDS:
    p = parts(s)
    m = gb(s).fit(p["train"][FEATS], p["train"].label)
    imp.append(pd.Series(m.feature_importances_, index=FEATS))
values["gb_impurity_importance_mean"] = pd.concat(imp, axis=1).mean(axis=1).sort_values(ascending=False).round(4).to_dict()

ablations = {
    "all_11": FEATS,
    "without_uncertainty": [f for f in FEATS if f != "uncertainty"],
    "without_nli_unsupported": [f for f in FEATS if f != "nli_unsupported"],
    "lexical_only": grp["lexical"],
    "semantic_only": grp["semantic"],
    "nli_only": grp["nli"],
    "without_lexical": [f for f in FEATS if f not in grp["lexical"]],
    "semantic_plus_nli": grp["semantic"] + grp["nli"],
}
abl = pd.concat([run_condition(k, v) for k, v in ablations.items()])
abl.to_csv(OUT / "ablation_per_seed.csv", index=False)
summarise(abl).to_csv(OUT / "ablation_summary.csv", index=False)

# ---------------------------------------------------------------- 6. Eq. (3) coefficient sensitivity (R2.5)
lam_rows = []
for lam in (0.0, 0.25, 0.35, 0.5, 0.75, 1.0):
    def tf(mode, d, st=None, lam=lam):
        if mode == "fit":
            return None
        d = d.copy()
        d["nli_unsupported"] = np.clip(d.nli_contradiction + lam * d.nli_neutral, 0, 1)
        return d
    df = run_condition(f"lambda_{lam}", FEATS, transform=tf)
    lam_rows.append(df)
    # single-signal baseline with this lambda
lam = pd.concat(lam_rows)
lam.to_csv(OUT / "lambda_sensitivity_per_seed.csv", index=False)
summarise(lam).to_csv(OUT / "lambda_sensitivity_summary.csv", index=False)
single = []
for lamv in (0.0, 0.25, 0.35, 0.5, 0.75, 1.0):
    for s in SEEDS:
        p = parts(s)
        sv = np.clip(p["validation"].nli_contradiction + lamv * p["validation"].nli_neutral, 0, 1)
        st = np.clip(p["test"].nli_contradiction + lamv * p["test"].nli_neutral, 0, 1)
        thr = threshold_for_f1(p["validation"].label.to_numpy(), sv.to_numpy())
        single.append({"lambda": lamv, "seed": s, **metrics(p["test"].label, st, thr)})
single = pd.DataFrame(single)
single.groupby("lambda")[["f1", "roc_auc"]].agg(["mean", "std"]).to_csv(OUT / "lambda_single_signal_summary.csv")

# ---------------------------------------------------------------- 7. mitigation attempts (R1.2 / R1.1)
def residualise(mode, d, st=None):
    """Remove the length-explained component of every feature (fit on HaluEval train only)."""
    if mode == "fit":
        models = {}
        x = np.log1p(d[["answer_len"]].to_numpy())
        for f in FEATS:
            r = GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=0.1, random_state=0)
            r.fit(x, d[f])
            models[f] = r
        return models
    d = d.copy()
    x = np.log1p(d[["answer_len"]].to_numpy())
    for f in FEATS:
        d[f] = d[f] - st[f].predict(x)
    return d


def balanced_train(tr, seed):
    return length_match(tr, 1, seed).reset_index(drop=True)


def reweight_train(tr, seed):
    # Inverse-frequency resampling within 1-token bins so each bin is label-balanced
    # (resample with replacement to keep size); equivalent to importance weighting.
    work = tr.copy()
    work["_b"] = work.answer_len
    rng = np.random.default_rng(seed)
    idx = []
    for _, g in work.groupby("_b"):
        if g.label.nunique() < 2:
            continue
        n = len(g)
        for lab in (0, 1):
            sub = g[g.label == lab].index.to_numpy()
            idx.extend(rng.choice(sub, n // 2, replace=True))
    return work.loc[idx].drop(columns="_b").reset_index(drop=True)


mit = pd.concat([
    run_condition("baseline_direct_length_excluded", FEATS),
    run_condition("length_residualised_features", FEATS, transform=residualise),
    run_condition("length_balanced_training_1tok", FEATS, train_filter=balanced_train),
    run_condition("length_reweighted_training", FEATS, train_filter=reweight_train),
    run_condition("residualised_plus_balanced", FEATS, transform=residualise, train_filter=balanced_train),
])
mit.to_csv(OUT / "mitigation_per_seed.csv", index=False)
summarise(mit).to_csv(OUT / "mitigation_summary.csv", index=False)

# ---------------------------------------------------------------- 8. transfer diagnostics (R1.3 / R3.5)
# 8a. domain classifier: can the 11 features tell HaluEval from RAGTruth?
dom = pd.concat([H[FEATS].assign(d=0), R[FEATS].assign(d=1)], ignore_index=True)
skf = StratifiedKFold(5, shuffle=True, random_state=11)
pd_ = cross_val_predict(GradientBoostingClassifier(n_estimators=120, max_depth=2, learning_rate=0.05, random_state=11),
                        dom[FEATS], dom.d, cv=skf, method="predict_proba")[:, 1]
values["domain_classifier_auc_halueval_vs_ragtruth_11feat"] = roc_auc_score(dom.d, pd_)

# 8b. per-feature shift and AUC orientation
shift = {}
for f in FEATS + ["answer_len"]:
    lo, hi = H[f].quantile(.01), H[f].quantile(.99)
    shift[f] = {
        "halueval_factual_median": float(H[H.label == 0][f].median()),
        "halueval_halluc_median": float(H[H.label == 1][f].median()),
        "ragtruth_factual_median": float(R[R.label == 0][f].median()),
        "ragtruth_halluc_median": float(R[R.label == 1][f].median()),
        "ks_stat_halueval_vs_ragtruth": float(ks_2samp(H[f], R[f]).statistic),
        "ragtruth_frac_outside_halueval_p1_p99": float(((R[f] < lo) | (R[f] > hi)).mean()),
        "auc_halueval": float(roc_auc_score(H.label, H[f])) if H[f].nunique() > 1 else None,
        "auc_ragtruth_test": float(roc_auc_score(RT.label, RT[f])) if RT[f].nunique() > 1 else None,
    }
values["feature_shift_halueval_vs_ragtruth"] = shift
values["ragtruth_factual_closer_to_halueval_halluc"] = {
    f: abs(v["ragtruth_factual_median"] - v["halueval_halluc_median"]) < abs(v["ragtruth_factual_median"] - v["halueval_factual_median"])
    for f, v in shift.items()}

# 8c. frozen baselines (HaluEval-validation thresholds applied to RAGTruth test)
fro = []
for s in SEEDS:
    p = parts(s)
    for col in ("answer_len", "lexical_mismatch"):
        thr = threshold_for_f1(p["validation"].label.to_numpy(), p["validation"][col].to_numpy())
        r = metrics(RT.label, RT[col].to_numpy(), thr)
        fro.append({"baseline": col, "seed": s, "threshold": thr, **r,
                    "pred_positive_frac": float((RT[col] >= thr).mean())})
fro = pd.DataFrame(fro)
fro.to_csv(OUT / "frozen_single_baselines_ragtruth.csv", index=False)
values["frozen_single_baselines_ragtruth_mean"] = fro.drop(columns="seed").groupby("baseline").mean().to_dict(orient="index")

# 8d. frozen main GB: score distribution + subgroup AUCs (length tercile, NLI not truncated, model family)
from transformers import AutoTokenizer  # noqa: E402
tk = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
RT["pair_tokens"] = [len(tk(e, a, truncation=False)["input_ids"]) for e, a in zip(RT.evidence, RT.answer)]
RT["len_tercile"] = pd.qcut(RT.answer_len, 3, labels=["short", "medium", "long"])
sub_rows, score_stats = [], []
for s in SEEDS:
    p = parts(s)
    m = gb(s).fit(p["train"][FEATS], p["train"].label)
    thr = threshold_for_f1(p["validation"].label.to_numpy(), m.predict_proba(p["validation"][FEATS])[:, 1])
    sc = m.predict_proba(RT[FEATS])[:, 1]
    hv = m.predict_proba(p["test"][FEATS])[:, 1]
    score_stats.append({"seed": s, "threshold": thr, "ragtruth_score_mean": sc.mean(), "ragtruth_score_min": sc.min(),
                        "ragtruth_score_median": np.median(sc),
                        "ragtruth_frac_above_thr": (sc >= thr).mean(),
                        "halueval_test_score_mean_factual": hv[p["test"].label == 0].mean(),
                        "halueval_test_score_mean_halluc": hv[p["test"].label == 1].mean(),
                        "ragtruth_score_mean_factual": sc[RT.label == 0].mean(),
                        "ragtruth_score_mean_halluc": sc[RT.label == 1].mean()})
    groups = {"all": RT.index, "nli_not_truncated(<=512)": RT.index[RT.pair_tokens <= 512],
              "nli_truncated(>512)": RT.index[RT.pair_tokens > 512]}
    for t in ("short", "medium", "long"):
        groups[f"len_{t}"] = RT.index[RT.len_tercile == t]
    for fam, pat in (("gpt", "gpt"), ("llama", "llama"), ("mistral", "mistral")):
        groups[f"model_{fam}"] = RT.index[RT.generator_model.str.contains(pat)]
    for g, ix in groups.items():
        y = RT.label.loc[ix]
        if y.nunique() == 2:
            sub_rows.append({"seed": s, "subset": g, "n": len(ix), "n_halluc": int(y.sum()),
                             "auc": roc_auc_score(y, sc[ix]), "f1": precision_recall_fscore_support(
                                 y, (sc[ix] >= thr).astype(int), average="binary", zero_division=0)[2]})
pd.DataFrame(score_stats).to_csv(OUT / "frozen_gb_score_stats.csv", index=False)
values["frozen_gb_score_stats_mean"] = pd.DataFrame(score_stats).drop(columns="seed").mean().to_dict()
sub = pd.DataFrame(sub_rows)
sub.to_csv(OUT / "frozen_gb_subgroups_per_seed.csv", index=False)
sub.drop(columns="seed").groupby("subset").mean().to_csv(OUT / "frozen_gb_subgroups_summary.csv")
values["ragtruth_test_len_tercile_edges"] = RT.groupby("len_tercile", observed=True).answer_len.agg(["min", "max"]).to_dict()
values["ragtruth_test_pair_tokens_over_512_frac"] = float((RT.pair_tokens > 512).mean())


def default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


(OUT / "reviewer_values.json").write_text(json.dumps(values, indent=1, default=default), encoding="utf-8")
print("DONE")
