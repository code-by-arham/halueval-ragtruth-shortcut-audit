#!/usr/bin/env bash
# End-to-end reproduction of every number in the paper and in the revision analyses.
#
#   bash run_all.sh            # reuse the released feature matrices (minutes, CPU)
#   bash run_all.sh --extract  # also recompute all features from raw text (GPU recommended;
#                              # ~2.2 h for HaluEval NLI on a 2-core CPU)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
EXTRACT=${1:-}
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
echo "All done."
