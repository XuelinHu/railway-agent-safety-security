#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/xuelin/miniconda3/bin:$PATH"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
MODEL="/ds2/xuelin/cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
DATA_ROOT="data/processed/public_benchmarks_full"; RUN_ROOT="outputs/public_full_stage1"

eval_one() {
  local dataset="$1" method="$2"; local base="$DATA_ROOT/$dataset"; local instruction=()
  [[ "$method" == "kg" ]] && instruction=(--use-job-instruction)
  bash scripts/run_qlora_inference_sharded.sh --workers 2 --model-path "$MODEL" --adapter "$RUN_ROOT/${dataset}_${method}" \
    --jobs "$base/validation_${method}_jobs.jsonl" --output "$RUN_ROOT/${dataset}_${method}_validation.jsonl" \
    --log "$RUN_ROOT/${dataset}_${method}_validation.log" --compact-target --max-input-tokens 4096 --max-new-tokens 512 --max-seconds-per-job 45 "${instruction[@]}"
  python3 scripts/evaluate_annotations.py --gold "$base/validation_gold.jsonl" --gold-index "$base/validation_index.jsonl" --predictions "$RUN_ROOT/${dataset}_${method}_validation.jsonl" --include-missing-as-empty --output "$RUN_ROOT/${dataset}_${method}_validation_raw_metrics.json"
  python3 scripts/expand_compact_predictions.py --jobs "$base/validation_${method}_jobs.jsonl" --predictions "$RUN_ROOT/${dataset}_${method}_validation.jsonl" --output "$RUN_ROOT/${dataset}_${method}_validation_expanded.jsonl" --errors "$RUN_ROOT/${dataset}_${method}_validation_expand_errors.jsonl"
  python3 scripts/verify_relations.py --annotations "$RUN_ROOT/${dataset}_${method}_validation_expanded.jsonl" --ontology "$base/ontology.yaml" --output "$RUN_ROOT/${dataset}_${method}_validation_verified.jsonl" --audit "$RUN_ROOT/${dataset}_${method}_validation_audit.jsonl"
  python3 scripts/evaluate_annotations.py --gold "$base/validation_gold.jsonl" --gold-index "$base/validation_index.jsonl" --predictions "$RUN_ROOT/${dataset}_${method}_validation_verified.jsonl" --include-missing-as-empty --output "$RUN_ROOT/${dataset}_${method}_validation_metrics.json"
}

eval_one scierc kg
for dataset in ade; do
  base="$DATA_ROOT/$dataset"
  for method in baseline kg; do
    instruction=(); [[ "$method" == "kg" ]] && instruction=(--use-job-instruction)
    if [[ ! -f "$RUN_ROOT/${dataset}_${method}/training_metrics.json" ]]; then
      python3 scripts/train_qlora.py --model-path "$MODEL" --gold "$base/train_gold.jsonl" --index "$base/train_index.jsonl" --jobs "$base/train_${method}_jobs.jsonl" --output "$RUN_ROOT/${dataset}_${method}" --epochs 1 --batch-size 1 --gradient-accumulation 4 --max-length 4096 --compact-target "${instruction[@]}"
    fi
    eval_one "$dataset" "$method"
  done
done
printf '{"status":"validation_complete","finished_at":"%s"}\n' "$(date --iso-8601=seconds)" > "$RUN_ROOT/status.json"
