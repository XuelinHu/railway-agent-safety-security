#!/usr/bin/env bash
# Continue the full validation stage after CoNLL04 using serial training and
# two isolated inference shards per validation split.
set -euo pipefail
export PATH="/home/xuelin/miniconda3/bin:$PATH"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL="/ds2/xuelin/cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
DATA_ROOT="data/processed/public_benchmarks_full"
RUN_ROOT="outputs/public_full_stage1"

run_dataset() {
  local dataset="$1" base="$DATA_ROOT/$1"
  for method in baseline kg; do
    local instruction=()
    [[ "$method" == "kg" ]] && instruction=(--use-job-instruction)
    if [[ ! -f "$RUN_ROOT/${dataset}_${method}/training_metrics.json" ]]; then
      python3 scripts/train_qlora.py --model-path "$MODEL" --gold "$base/train_gold.jsonl" --index "$base/train_index.jsonl" \
        --jobs "$base/train_${method}_jobs.jsonl" --output "$RUN_ROOT/${dataset}_${method}" --epochs 1 --batch-size 1 \
        --gradient-accumulation 4 --max-length 4096 --compact-target "${instruction[@]}"
    else
      echo "resume: training already complete for ${dataset}_${method}"
    fi
    bash scripts/run_qlora_inference_sharded.sh --workers 2 --model-path "$MODEL" --adapter "$RUN_ROOT/${dataset}_${method}" \
      --jobs "$base/validation_${method}_jobs.jsonl" --output "$RUN_ROOT/${dataset}_${method}_validation.jsonl" \
      --log "$RUN_ROOT/${dataset}_${method}_validation.log" --compact-target --max-input-tokens 4096 \
      --max-new-tokens 512 --max-seconds-per-job 45 "${instruction[@]}"
    python3 scripts/evaluate_annotations.py --gold "$base/validation_gold.jsonl" --gold-index "$base/validation_index.jsonl" \
      --predictions "$RUN_ROOT/${dataset}_${method}_validation.jsonl" --include-missing-as-empty \
      --output "$RUN_ROOT/${dataset}_${method}_validation_raw_metrics.json"
    python3 scripts/expand_compact_predictions.py --jobs "$base/validation_${method}_jobs.jsonl" \
      --predictions "$RUN_ROOT/${dataset}_${method}_validation.jsonl" --output "$RUN_ROOT/${dataset}_${method}_validation_expanded.jsonl" \
      --errors "$RUN_ROOT/${dataset}_${method}_validation_expand_errors.jsonl"
    python3 scripts/verify_relations.py --annotations "$RUN_ROOT/${dataset}_${method}_validation_expanded.jsonl" \
      --ontology "$base/ontology.yaml" --output "$RUN_ROOT/${dataset}_${method}_validation_verified.jsonl" \
      --audit "$RUN_ROOT/${dataset}_${method}_validation_audit.jsonl"
    python3 scripts/evaluate_annotations.py --gold "$base/validation_gold.jsonl" --gold-index "$base/validation_index.jsonl" \
      --predictions "$RUN_ROOT/${dataset}_${method}_validation_verified.jsonl" --include-missing-as-empty \
      --output "$RUN_ROOT/${dataset}_${method}_validation_metrics.json"
  done
}

for dataset in scierc ade; do run_dataset "$dataset"; done
printf '{"status":"validation_complete","finished_at":"%s","inference_workers":2}\n' "$(date --iso-8601=seconds)" > "$RUN_ROOT/status.json"
