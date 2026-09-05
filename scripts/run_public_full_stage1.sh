#!/usr/bin/env bash
set -euo pipefail

export PATH="/home/xuelin/miniconda3/bin:$PATH"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL="/ds2/xuelin/cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
DATA_ROOT="data/processed/public_benchmarks_full"
RUN_ROOT="outputs/public_full_stage1"
mkdir -p "$RUN_ROOT"

run_dataset() {
  local dataset="$1"
  local base="$DATA_ROOT/$dataset"
  python3 scripts/import_spert_benchmarks.py --dataset "$dataset" --output-root "$DATA_ROOT" --seed public-full-42
  for split in train validation test; do
    for mode in baseline kg_constrained; do
      local suffix="baseline"
      [[ "$mode" == "kg_constrained" ]] && suffix="kg"
      python3 scripts/build_experiment_jobs.py --jobs "$base/jobs.jsonl" --manifest "$base/split_manifest.jsonl" \
        --mentions "$base/knowledge_graph/mentions.jsonl" --ontology "$base/ontology.yaml" --split "$split" --mode "$mode" \
        --output "$base/${split}_${suffix}_jobs.jsonl"
    done
  done

  for method in baseline kg; do
    local instruction=()
    [[ "$method" == "kg" ]] && instruction=(--use-job-instruction)
    python3 scripts/train_qlora.py --model-path "$MODEL" --gold "$base/train_gold.jsonl" --index "$base/train_index.jsonl" \
      --jobs "$base/train_${method}_jobs.jsonl" --output "$RUN_ROOT/${dataset}_${method}" --epochs 1 --batch-size 1 \
      --gradient-accumulation 4 --max-length 4096 --compact-target "${instruction[@]}"
    local raw="$RUN_ROOT/${dataset}_${method}_validation.jsonl"
    bash scripts/run_qlora_inference_sharded.sh --workers 2 --model-path "$MODEL" --adapter "$RUN_ROOT/${dataset}_${method}" \
      --jobs "$base/validation_${method}_jobs.jsonl" --output "$raw" --log "$RUN_ROOT/${dataset}_${method}_validation.log" \
      --compact-target --resume --max-input-tokens 4096 --max-new-tokens 512 --max-seconds-per-job 45 "${instruction[@]}"
    python3 scripts/evaluate_annotations.py --gold "$base/validation_gold.jsonl" --gold-index "$base/validation_index.jsonl" \
      --predictions "$raw" --include-missing-as-empty --output "$RUN_ROOT/${dataset}_${method}_validation_raw_metrics.json"
    python3 scripts/expand_compact_predictions.py --jobs "$base/validation_${method}_jobs.jsonl" --predictions "$raw" \
      --output "$RUN_ROOT/${dataset}_${method}_validation_expanded.jsonl" --errors "$RUN_ROOT/${dataset}_${method}_validation_expand_errors.jsonl"
    python3 scripts/verify_relations.py --annotations "$RUN_ROOT/${dataset}_${method}_validation_expanded.jsonl" --ontology "$base/ontology.yaml" \
      --output "$RUN_ROOT/${dataset}_${method}_validation_verified.jsonl" --audit "$RUN_ROOT/${dataset}_${method}_validation_audit.jsonl"
    python3 scripts/evaluate_annotations.py --gold "$base/validation_gold.jsonl" --gold-index "$base/validation_index.jsonl" \
      --predictions "$RUN_ROOT/${dataset}_${method}_validation_verified.jsonl" --include-missing-as-empty \
      --output "$RUN_ROOT/${dataset}_${method}_validation_metrics.json"
  done
}

started="$(date --iso-8601=seconds)"
for dataset in conll04 scierc ade; do run_dataset "$dataset"; done
printf '{"status":"validation_complete","started_at":"%s","finished_at":"%s"}\n' "$started" "$(date --iso-8601=seconds)" > "$RUN_ROOT/status.json"
