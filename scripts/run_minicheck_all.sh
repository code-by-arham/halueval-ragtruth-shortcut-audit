#!/usr/bin/env bash
# MiniCheck-RoBERTa-Large scores for all three benchmarks (resumable; rerun to continue).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
OUT=derived/minicheck; mkdir -p "$OUT"
# one deadline for the whole batch (all datasets), not per dataset
export DEADLINE_EPOCH=${DEADLINE_EPOCH:-$(( $(date +%s) + ${BUDGET_MIN:-100000} * 60 ))}
run() { [ -f "$OUT/$1" ] && { echo "skip $1"; return; }; shift_name=$1; shift; $PY scripts/run_minicheck.py --output "$OUT/$shift_name" "$@"; [ -f "$OUT/$shift_name" ] || { echo "BUDGET_STOP before finishing $shift_name"; exit 0; }; }
# 1. RAGTruth official test: multi-sentence responses, sentence-level checking (MiniCheck recommendation)
run ragtruth_test_minicheck.csv --input derived/ragtruth/ragtruth_qa_features.csv --filter "official_split=='test'" \
    --id-cols response_id label --split-sentences --chunk 125
# 2. HaluEval QA (all 20,000 answers), answer as claim
run halueval_minicheck_answer.csv --input derived/features/halueval_full_features.csv --id-cols question_id variant label \
    --claim-mode answer --chunk 1000
# 3. HaluBench DROP, FinanceBench, PubMedQA: all 3,000 rows (rows 0-2999 of the records file)
run halubench_minicheck_answer_noncovid.csv --input halubench/data/halubench_records.csv --filter "source_ds!='covidQA'" \
    --id-cols question_id variant source_ds label --claim-mode answer --chunk 250
# 4. HaluBench CovidQA: stratified random subsample of 200 (100 per label, seed 11). Each CovidQA passage
#    is ~2,600 words (~9 MiniCheck chunks per claim), which makes the full 1,000 rows ~5 h on 2 CPU threads.
run halubench_covidqa200_minicheck_answer.csv --input halubench/data/halubench_covidqa_subsample200.csv \
    --id-cols question_id variant source_ds label --claim-mode answer --chunk 25
# 5. Question+answer claims for HaluBench DROP, FinanceBench, PubMedQA (short span answers such as "2" or "Rams")
run halubench_minicheck_qa_noncovid.csv --input halubench/data/halubench_records.csv --filter "source_ds!='covidQA'" \
    --id-cols question_id variant source_ds label --claim-mode qa --chunk 250
# 6. Question+answer claims for HaluEval QA (very short answers)
run halueval_minicheck_qa.csv --input derived/features/halueval_full_features.csv --id-cols question_id variant label \
    --claim-mode qa --chunk 1000
echo MINICHECK_ALL_DONE
