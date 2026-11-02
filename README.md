# Benchmark-Specific Shortcuts in Hallucination Detection: HaluEval QA and RAGTruth

Code, data splits, feature matrices, per-seed predictions and all result files for the paper

> **Uncovering Benchmark-Specific Shortcuts in Hallucination Detection: A Controlled Audit of HaluEval QA and RAGTruth**
> Syed Muhammad Waqas, Muhammad Arham, Ziyan Umer, Syed Jafar Ali Kazmi, Mazliham Mohd Su'ud, Muhammad Mansoor Alam.
> *Under review (Engineering Reports, Wiley).*

The study asks whether a high hallucination-detection score on HaluEval QA reflects evidence grounding or a benchmark artifact. It uses the full HaluEval QA source (10,000 question groups, 20,000 answers), five group-safe 80/10/10 splits, validation-only thresholds, a Meta Classifier that withholds all direct length variables, fixed-prediction answer-length matching, and a frozen transfer test on the official RAGTruth QA test set.

## Headline results

| Setting | F1 (%) | ROC AUC (%) |
|---|---|---|
| Answer length only (HaluEval, validation-tuned threshold) | 94.98 ± 0.34 | 97.45 ± 0.25 |
| Direct-length-excluded Gradient Boosting (proposed, HaluEval) | 97.81 ± 0.31 | 99.51 ± 0.10 |
| Same model, strict 1-token length matching | 88.88 | n/a |
| Same model, frozen transfer to RAGTruth QA test | 30.95 | 42.82 |
| RAGTruth in-domain direct-length-excluded GB | 48.15 ± 1.15 | 79.89 ± 0.18 |

Values are five-seed means with descriptive 95% half-widths. The mean score–answer-length Pearson correlation of the proposed model is 0.7187.

## Repository layout

```
scripts/                     core pipeline (used for every number in the paper)
  prepare_halueval_splits.py   group-safe 80/10/10 splits, seeds 11-55
  extract_halueval_features.py lexical, semantic (MiniLM), NLI (BART-MNLI), uncertainty features
  evaluate_truthguard.py       artifact baselines + LR / RF / GB, validation-only thresholds
  analyze_evaluation_controls.py  fixed-prediction length matching and length strata
  prepare_ragtruth_qa.py       span annotations -> response-level labels (quality == "good")
  prepare_ragtruth_splits.py   group-safe validation inside the official RAGTruth train split
  evaluate_halueval_to_ragtruth_transfer.py  frozen external transfer
analysis/                    additional analyses requested during peer review
configs/  data/SHA256SUMS    configuration and checksums of the raw upstream files
derived/                     split manifests and released feature matrices
results/                     all aggregate and per-seed outputs (incl. test predictions)
run_all.sh                   one-command reproduction
```

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash run_all.sh            # reuses released features; CPU, a few minutes
bash run_all.sh --extract  # recomputes all features from raw text (GPU recommended)
```

`run_all.sh` downloads the raw HaluEval QA and RAGTruth files and checks them against `data/SHA256SUMS`. It then rebuilds the splits, re-runs every evaluation and writes the results to `results/`.

**Reproduction status.** The released HaluEval feature matrix was regenerated on CPU. `derived/features/halueval_full_features.original_gpu_run.manifest.json` records the original Colab GPU run. Re-running the pipeline on the regenerated matrix gives the following:
- Tables 5, 7 and 8 match the submitted manuscript to two decimals. The one exception is direct-length-excluded Random Forest, with F1 97.86 against 97.87 in the manuscript.
- Per-seed frozen-transfer thresholds and RAGTruth in-domain per-seed results match the manuscript exactly.
- Non-NLI features differ from the GPU run by at most 5×10⁻⁷.

## Key implementation details

- **Evidence**: the HaluEval `knowledge` field, never concatenated with the question.
- **Uncertainty feature**: the answer is lowercased and checked for 9 markers by substring: `i think`, `probably`, `maybe`, `it is likely`, `it seems`, `as an ai`, `i cannot verify`, `not sure`, `unknown`. The score is U(a) = min(1, ½·#markers present).
- **NLI unsupportedness**: S_unsup = min(1, P_con + 0.35·P_neu), with premise = evidence, hypothesis = answer, `facebook/bart-large-mnli`, max 512 tokens.
- **RAGTruth conversion**: QA responses with RAGTruth's own annotator `quality == "good"` are kept (5,767 of 5,934). Label = 1 if the response has ≥ 1 official hallucination span of any type, otherwise 0. The official train/test partition is kept (4,892 / 875).

## Revision analyses (`results/reviewer_analyses/`)

| File | Content |
|---|---|
| `reviewer_values.json` | length statistics, feature–length associations, recoverability of answer length from the 11 non-length features, uncertainty-marker prevalence, GB importances, domain-classifier AUC, per-feature HaluEval→RAGTruth shift |
| `ablation_*.csv` | feature-group ablations (conventional, length-matched, frozen transfer) |
| `lambda_*.csv` | sensitivity of the Eq. (3) coefficient (0 to 1) |
| `mitigation_*.csv` | length-residualised features, length-balanced and length-reweighted training |
| `frozen_*.csv` | frozen-transfer score saturation, single-signal frozen baselines, subgroup AUCs (length tercile, NLI truncation, generator family) |
| `nli_truncation.json` | share of evidence+answer pairs above the 512-token NLI limit |

## Data and licences

Code is released under the MIT licence. The derived feature matrices contain question, evidence and answer text from:
- [HaluEval](https://github.com/RUCAIBox/HaluEval) (MIT, © 2020 RUCAIBox), Li et al., EMNLP 2023
- [RAGTruth](https://github.com/ParticleMedia/RAGTruth) (MIT, © 2023 Particle Media), Niu et al., ACL 2024

Please cite the original datasets when using these files.
