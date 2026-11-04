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
  prepare_halubench.py         HaluBench subsets -> records + group-safe splits (second round)
  make_covidqa_subsample.py    fixed 200-row CovidQA subsample for MiniCheck (100 per label, seed 11)
  extract_records_resumable.py same feature extractor, resumable on CPU
  run_minicheck.py / run_minicheck_all.sh  MiniCheck-RoBERTa-Large scores (resumable)
analysis/                    additional analyses requested during peer review
configs/  data/SHA256SUMS    configuration and checksums of the raw upstream files
derived/                     split manifests, released feature matrices, MiniCheck scores
halubench/                   HaluBench records, splits and feature matrix
third_party/minicheck/       vendored MiniCheck inference code (Apache 2.0, unmodified)
results/                     all aggregate and per-seed outputs (incl. test predictions)
run_all.sh                   one-command reproduction
```

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash run_all.sh            # reuses released features; CPU, a few minutes
bash run_all.sh --extract  # recomputes all features from raw text (GPU recommended)
bash run_all.sh --minicheck  # also recomputes all MiniCheck scores (~10 h on 2 CPU threads)
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

## Second-round analyses: HaluBench and MiniCheck (`results/halubench/`, `results/revision2/`)

**HaluBench.** The four source subsets of [HaluBench](https://huggingface.co/datasets/PatronusAI/HaluBench) are used: DROP, FinanceBench, CovidQA and PubMedQA, with 1,000 rows each (500 PASS, 500 FAIL). `scripts/prepare_halubench.py` builds the records and five group-safe 80/10/10 splits, using passage hash as the group (2,333 groups). The median answer length is identical across labels in every subset, and pooled length-only AUC is 0.489, so the HaluEval length shortcut is absent by construction.

| HaluBench AUC | DROP | FinanceBench | CovidQA | PubMedQA | Pooled |
|---|---|---|---|---|---|
| Frozen HaluEval direct-length-excluded GB (5-seed mean) | 0.566 | 0.515 | 0.729 | 0.633 | 0.570 |
| In-domain direct-length-excluded GB (HaluBench splits) | 0.593 | 0.498 | 0.808 | 0.796 | 0.686 ± 0.041 |

**MiniCheck.** [MiniCheck-RoBERTa-Large](https://github.com/Liyan06/MiniCheck) (Tang et al., EMNLP 2024) is used zero-shot as an atomic-claim fact checker. The inference code is vendored unmodified in `third_party/minicheck/`.
- Each document is split into 400-token chunks. A claim's support is the maximum over chunks, and a response's support is the minimum over its sentences. The hallucination score is 1 − support.
- On RAGTruth, responses are split into sentences.

| MiniCheck | ROC AUC | F1 @0.5 |
|---|---|---|
| HaluEval QA test (5 seeds) | 0.895 | 0.825 (1-token matched: 0.848) |
| RAGTruth QA official test (n = 875) | 0.853 | 0.412 |
| HaluBench pooled (n = 3,200; same rows as frozen GB = 0.543) | 0.610 | n/a |

**Claim format.** DROP and FinanceBench answers are often very short (for example "2"). Rewriting each claim as question + answer changes little: on the same 3,000 non-CovidQA rows, AUC goes from 0.593 to 0.588 pooled, 0.516 to 0.541 on DROP, 0.562 to 0.536 on FinanceBench, and 0.706 to 0.710 on PubMedQA.

**CovidQA subsample.** CovidQA passages average about 2,600 words, roughly 9 MiniCheck chunks per claim. MiniCheck is therefore run on a fixed stratified random subsample of 200 CovidQA rows (100 per label, `numpy.random.default_rng(11)`), produced by `scripts/make_covidqa_subsample.py`. The selection does not depend on any model score. Frozen GB AUC is 0.728 on these 200 rows and 0.729 on all 1,000.

`analysis/new_dataset_analyses.py` writes every table to `results/revision2/`. `analysis/verify_revision2.py` recomputes the headline values independently and writes `verification_revision2.csv`.

## Data and licences

Code is released under the MIT licence. The derived feature matrices contain question, evidence and answer text from:
- [HaluEval](https://github.com/RUCAIBox/HaluEval) (MIT, © 2020 RUCAIBox), Li et al., EMNLP 2023
- [RAGTruth](https://github.com/ParticleMedia/RAGTruth) (MIT, © 2023 Particle Media), Niu et al., ACL 2024

The second-round files additionally contain or use:
- [HaluBench](https://huggingface.co/datasets/PatronusAI/HaluBench) (Patronus AI), released under **CC BY-NC 2.0**. The files under `halubench/` and the HaluBench score files carry this non-commercial licence.
- [MiniCheck](https://github.com/Liyan06/MiniCheck) (Tang, Laban and Durrett, EMNLP 2024). The code at commit b58b9fa is Apache 2.0 (see `third_party/minicheck/LICENSE`), and the `lytang/MiniCheck-RoBERTa-Large` weights are MIT.

Please cite the original datasets when using these files.
