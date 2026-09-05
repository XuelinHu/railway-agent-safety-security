#!/usr/bin/env bash
# Resume all public validation datasets once the target GPU has no compute jobs.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PATH="/home/xuelin/miniconda3/bin:$PATH"
export HF_HOME="${HF_HOME:-/ds2/xuelin/cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

poll_seconds="${GPU_POLL_SECONDS:-1800}"
requested_workers="${GLINER_GLIREL_WORKERS:-2}"
workers="$requested_workers"
parallel_min_free_mib="${GLINER_GLIREL_PARALLEL_MIN_FREE_MIB:-12288}"
parallel_decision="pending_gpu_check"
output_root="${OUTPUT_ROOT:-outputs/public_horizontal_validation/gliner_glirel}"
entity_threshold="${ENTITY_THRESHOLD:-0.5}"
relation_threshold="${RELATION_THRESHOLD:-0.5}"
relation_label_mode="${RELATION_LABEL_MODE:-canonical}"
dtype="${MODEL_DTYPE:-float16}"
status_file="$output_root/status.json"
current_dataset=""

mkdir -p "$output_root"
if [[ ! "$requested_workers" =~ ^[12]$ ]]; then
  echo "GLINER_GLIREL_WORKERS must be 1 or 2 on the 24GB RTX 3090" >&2
  exit 2
fi
if [[ ! "$parallel_min_free_mib" =~ ^[1-9][0-9]*$ ]]; then
  echo "GLINER_GLIREL_PARALLEL_MIN_FREE_MIB must be a positive integer" >&2
  exit 2
fi

gpu_lock="${PUBLIC_GPU_LOCK_FILE:-outputs/.locks/public-validation-gpu.lock}"
mkdir -p "$(dirname "$gpu_lock")"
exec 8>"$gpu_lock"
while ! flock -n 8; do
  echo "$(date --iso-8601=seconds) waiting_for_public_validation_gpu_lock"
  sleep 30
done

write_status() {
  local state="$1" active_dataset="${2:-}"
  python3 - "$status_file" "$state" "$active_dataset" "$output_root" "$poll_seconds" \
    "$workers" "$requested_workers" "$parallel_min_free_mib" "$parallel_decision" \
    "$entity_threshold" "$relation_threshold" "$relation_label_mode" "$dtype" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

status_path = Path(sys.argv[1])
state = sys.argv[2]
active_dataset = sys.argv[3] or None
output_root = Path(sys.argv[4])
poll_seconds = int(sys.argv[5])
workers = int(sys.argv[6])
requested_workers = int(sys.argv[7])
parallel_min_free_mib = int(sys.argv[8])
parallel_decision = sys.argv[9]
entity_threshold = float(sys.argv[10])
relation_threshold = float(sys.argv[11])
relation_label_mode = sys.argv[12]
dtype = sys.argv[13]
datasets = {}
for name in ("conll04", "scierc", "ade"):
    base = Path("data/processed/public_benchmarks_full") / name
    jobs = base / "validation_baseline_jobs.jsonl"
    predictions = output_root / f"{name}_validation.jsonl"
    expected = sum(bool(line.strip()) for line in jobs.open(encoding="utf-8"))
    completed = (
        sum(bool(line.strip()) for line in predictions.open(encoding="utf-8"))
        if predictions.exists()
        else 0
    )
    datasets[name] = {
        "expected_jobs": expected,
        "prediction_rows": completed,
        "predictions_complete": completed == expected,
        "normalized_text_metrics": str(output_root / f"{name}_validation.normalized_text_metrics.json"),
        "normalized_text_metrics_ready": (output_root / f"{name}_validation.normalized_text_metrics.json").is_file(),
        "character_span_metrics": str(output_root / f"{name}_validation.character_span_metrics.json"),
        "character_span_metrics_ready": (output_root / f"{name}_validation.character_span_metrics.json").is_file(),
    }
result = {
    "status": state,
    "active_dataset": active_dataset,
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "poll_seconds": poll_seconds,
    "parallel_workers": workers,
    "requested_parallel_workers": requested_workers,
    "parallel_min_free_mib": parallel_min_free_mib,
    "parallel_decision": parallel_decision,
    "output_root": str(output_root),
    "parameters": {
        "entity_threshold": entity_threshold,
        "relation_threshold": relation_threshold,
        "relation_label_mode": relation_label_mode,
        "dtype": dtype,
    },
    "datasets": datasets,
}
temporary = status_path.with_name(f".{status_path.name}.tmp")
temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
temporary.replace(status_path)
PY
}

on_exit() {
  local code="$?"
  if (( code != 0 )); then
    write_status failed "$current_dataset"
  fi
}
trap on_exit EXIT

for dataset in conll04 scierc ade; do
  python3 scripts/run_gliner_glirel_validation.py \
    --dataset "$dataset" \
    --relation-label-mode "$relation_label_mode" \
    --preflight-only
done

write_status waiting_for_gpu
if ! gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)"; then
  echo "unable to query GPU compute processes" >&2
  exit 1
fi
while [[ -n "$gpu_pids" ]]; do
  echo "$(date --iso-8601=seconds) waiting_for_idle_gpu poll_seconds=$poll_seconds"
  sleep "$poll_seconds"
  if ! gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)"; then
    echo "unable to query GPU compute processes" >&2
    exit 1
  fi
done

if (( requested_workers == 2 )); then
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1)"
  if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= parallel_min_free_mib )); then
    workers=2
    parallel_decision="enabled_free_mib_${free_mib}"
  else
    workers=1
    parallel_decision="single_worker_free_mib_${free_mib:-unknown}"
  fi
else
  workers=1
  parallel_decision="single_worker_requested"
fi
echo "$(date --iso-8601=seconds) parallel_decision=$parallel_decision workers=$workers"
write_status ready_for_inference

for dataset in conll04 scierc ade; do
  current_dataset="$dataset"
  write_status running "$current_dataset"
  base="data/processed/public_benchmarks_full/$dataset"
  predictions="$output_root/${dataset}_validation.jsonl"
  if (( workers == 1 )); then
    python3 scripts/run_gliner_glirel_validation.py \
      --dataset "$dataset" \
      --jobs "$base/validation_baseline_jobs.jsonl" \
      --output "$predictions" \
      --entity-threshold "$entity_threshold" \
      --relation-threshold "$relation_threshold" \
      --relation-label-mode "$relation_label_mode" \
      --device cuda:0 \
      --dtype "$dtype" \
      --resume
  else
    total="$(awk 'NF {count++} END {print count + 0}' "$base/validation_baseline_jobs.jsonl")"
    chunk=$(( (total + workers - 1) / workers ))
    pids=()
    shards=()
    for (( worker=0; worker<workers; worker++ )); do
      offset=$(( worker * chunk ))
      (( offset >= total )) && break
      shard="$output_root/${dataset}_validation.part${worker}.jsonl"
      shards+=("$shard")
      python3 scripts/run_gliner_glirel_validation.py \
        --dataset "$dataset" \
        --jobs "$base/validation_baseline_jobs.jsonl" \
        --output "$shard" \
        --offset "$offset" \
        --limit "$chunk" \
        --entity-threshold "$entity_threshold" \
        --relation-threshold "$relation_threshold" \
        --relation-label-mode "$relation_label_mode" \
        --device cuda:0 \
        --dtype "$dtype" \
        --resume \
        > "$output_root/${dataset}_validation.part${worker}.log" 2>&1 &
      pids+=("$!")
    done
    worker_failed=0
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then
        worker_failed=1
      fi
    done
    if (( worker_failed != 0 )); then
      if rg -qi 'CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED' \
        "$output_root/${dataset}_validation.part"*.log; then
        workers=1
        parallel_decision="fallback_to_single_after_cuda_oom_${dataset}"
        echo "$(date --iso-8601=seconds) $parallel_decision"
        write_status running "$current_dataset"
        python3 scripts/run_gliner_glirel_validation.py \
          --dataset "$dataset" \
          --jobs "$base/validation_baseline_jobs.jsonl" \
          --output "$predictions" \
          --entity-threshold "$entity_threshold" \
          --relation-threshold "$relation_threshold" \
          --relation-label-mode "$relation_label_mode" \
          --device cuda:0 \
          --dtype "$dtype" \
          --resume
      else
        exit 1
      fi
    else
      python3 scripts/merge_gliner_glirel_shards.py \
        --jobs "$base/validation_baseline_jobs.jsonl" \
        --shards "${shards[@]}" \
        --output "$predictions"
    fi
  fi

  python3 scripts/evaluate_annotations.py \
    --gold "$base/validation_gold.jsonl" \
    --gold-index "$base/validation_index.jsonl" \
    --predictions "$predictions" \
    --jobs "$base/validation_baseline_jobs.jsonl" \
    --include-missing-as-empty \
    --output "$output_root/${dataset}_validation.normalized_text_metrics.json"

  python3 scripts/evaluate_public_validation_spans.py \
    --gold "$base/validation_gold.jsonl" \
    --gold-index "$base/validation_index.jsonl" \
    --predictions "$predictions" \
    --jobs "$base/validation_baseline_jobs.jsonl" \
    --output "$output_root/${dataset}_validation.character_span_metrics.json"
done

current_dataset=""
write_status complete
trap - EXIT
