#!/usr/bin/env bash
# End-to-end reproduction of every number in the paper and in the revision analyses.
#
#   bash run_all.sh              # reuse the released feature matrices and MiniCheck scores (minutes, CPU)
#   bash run_all.sh --extract    # also recompute all features from raw text (GPU recommended;
#                                # ~2.2 h for HaluEval NLI on a 2-core CPU)
#   bash run_all.sh --minicheck  # also recompute all MiniCheck scores (~10 h on 2 CPU threads)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
EXTRACT=""; MINICHECK=""
for arg in "$@"; do
  case "$arg" in --extract) EXTRACT=--extract ;; --minicheck) MINICHECK=--minicheck ;; esac
done
mkdir -p results/halueval_full results/ragtruth/in_domain results/reviewer_analyses derived/ragtruth derived/features

echo "== 0. Raw data (MIT-licensed upstream sources) =="
mkdir -p data/RAGTruth
[ -f data/halueval_qa_main.json ] || curl -sSL -o data/halueval_qa_main.json \
  https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json
for f in response.jsonl source_info.jsonl; do
  [ -f "data/RAGTruth/$f" ] || curl -sSL -o "data/RAGTruth/$f" \
    "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/$f"
done
(cd data && sha256sum -c SHA256SUMS)

echo "== 1. Group-safe splits =="
$PY scripts/prepare_halueval_splits.py --source data/halueval_qa_main.json --output-dir derived/split_manifests
$PY scripts/prepare_ragtruth_qa.py --responses data/RAGTruth/response.jsonl --sources data/RAGTruth/source_info.jsonl \
    --output derived/ragtruth/ragtruth_qa_response_level.csv
$PY scripts/prepare_ragtruth_splits.py --records derived/ragtruth/ragtruth_qa_response_level.csv --output-dir derived/ragtruth/splits

if [ "$EXTRACT" = "--extract" ]; then
  echo "== 2. Feature extraction (MiniLM + BART-MNLI) =="
  DEV=$($PY -c "import torch;print('cuda' if torch.cuda.is_available() else 'cpu')")
  $PY scripts/extract_halueval_features.py --source data/halueval_qa_main.json \
      --output derived/features/halueval_full_features.csv --device "$DEV"
  $PY scripts/extract_halueval_features.py --records derived/ragtruth/ragtruth_qa_response_level.csv \
      --output derived/ragtruth/ragtruth_qa_features.csv --device "$DEV"
fi

echo "== 3. HaluEval conventional evaluation (Table 5) =="
EV=results/halueval_full
$PY scripts/evaluate_truthguard.py --features derived/features/halueval_full_features.csv \
    --splits-dir derived/split_manifests --output-dir "$EV" --save-models

echo "== 4. Fixed-prediction length matching (Tables 6-7) =="
for W in 1 2 5 10 20; do
  $PY scripts/analyze_evaluation_controls.py --features derived/features/halueval_full_features.csv \
      --predictions-dir "$EV" --method direct_length_excluded_gradient_boosting \
      --match-bin-width "$W" --output "$EV/length_controls_w${W}.json"
done

echo "== 5. Frozen HaluEval -> RAGTruth transfer and RAGTruth in-domain (Table 8) =="
$PY scripts/evaluate_halueval_to_ragtruth_transfer.py --ragtruth-features derived/ragtruth/ragtruth_qa_features.csv \
    --halueval-model-dir "$EV" --halueval-results "$EV/all_seed_results.csv" \
    --output results/ragtruth/halueval_to_ragtruth_transfer.csv
$PY scripts/evaluate_truthguard.py --features derived/ragtruth/ragtruth_qa_features.csv \
    --splits-dir derived/ragtruth/splits --split-pattern "ragtruth_qa_seed_*_splits.csv" \
    --output-dir results/ragtruth/in_domain

echo "== 6. Revision analyses =="
$PY analysis/ragtruth_stats.py
$PY analysis/nli_truncation.py
$PY analysis/reviewer_analyses.py

echo "== 7. HaluBench external subsets: DROP, FinanceBench, CovidQA, PubMedQA (CC BY-NC 2.0) =="
mkdir -p halubench/data halubench/features results/halubench/in_domain
[ -f halubench/data/halubench.parquet ] || curl -sSL -o halubench/data/halubench.parquet \
  https://huggingface.co/datasets/PatronusAI/HaluBench/resolve/main/data/test-00000-of-00001.parquet
echo "c7e9cf966085ffae88d2947744418a05a26ea94380c35c238b9fc12ecb874cdc  halubench/data/halubench.parquet" | sha256sum -c -
$PY scripts/prepare_halubench.py --parquet halubench/data/halubench.parquet \
    --output halubench/data/halubench_records.csv --splits-dir halubench/splits
$PY scripts/make_covidqa_subsample.py
if [ "$EXTRACT" = "--extract" ]; then
  $PY scripts/extract_records_resumable.py --records halubench/data/halubench_records.csv \
      --output halubench/features/halubench_features.csv
fi
$PY scripts/evaluate_truthguard.py --features halubench/features/halubench_features.csv \
    --splits-dir halubench/splits --split-pattern "halubench_qa_seed_*_splits.csv" \
    --output-dir results/halubench/in_domain

echo "== 8. MiniCheck-RoBERTa-Large (atomic-claim fact checker) on HaluEval, RAGTruth and HaluBench =="
if [ "$MINICHECK" = "--minicheck" ]; then
  bash scripts/run_minicheck_all.sh   # resumable; rerun to continue after an interruption
fi
$PY analysis/new_dataset_analyses.py
$PY analysis/verify_revision2.py
echo "All done."
