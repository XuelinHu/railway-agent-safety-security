#!/usr/bin/env bash
# Retry only failed Qwen zero-shot validation jobs with a larger output budget.
set -euo pipefail

export PATH="/home/xuelin/miniconda3/bin:$PATH"
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

model="/ds2/xuelin/cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
data_root="data/processed/public_benchmarks_full"
run_root="outputs/public_horizontal_validation/qwen3_4b_zero_shot"
initial_status="$run_root/status.json"
retry_status="$run_root/retry_status.json"
mkdir -p "$run_root"

gpu_lock="${PUBLIC_GPU_LOCK_FILE:-outputs/.locks/public-validation-gpu.lock}"
mkdir -p "$(dirname "$gpu_lock")"
exec 8>"$gpu_lock"
while ! flock -n 8; do
  echo "$(date --iso-8601=seconds) waiting_for_public_validation_gpu_lock"
  sleep 30
done

if [[ ! -f "$initial_status" ]] || \
   [[ "$(jq -r '.status // ""' "$initial_status")" != "validation_complete" ]]; then
  echo "initial Qwen zero-shot validation is not complete" >&2
  exit 1
fi

write_status() {
  local state="$1"
  python3 - "$retry_status" "$state" "$run_root" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

target = Path(sys.argv[1])
state = sys.argv[2]
root = Path(sys.argv[3])
datasets = {}
total_terminal_failures = 0
for dataset in ("conll04", "scierc", "ade"):
    log_path = root / f"{dataset}_validation.log"
    terminal = {}
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("job_id"):
                    terminal[row["job_id"]] = row.get("status")
    remaining_failed_jobs = sum(value != "success" for value in terminal.values())
    total_terminal_failures += remaining_failed_jobs
    datasets[dataset] = {
        "terminal_jobs": len(terminal),
        "remaining_failed_jobs": remaining_failed_jobs,
    }
payload = {
    "status": state,
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "retry_policy": {
        "selection": "failed jobs only",
        "workers": 2,
        "max_new_tokens": 1024,
        "max_seconds_per_job": 90,
    },
    "terminal_failure_evaluation": "missing predictions are scored as empty annotations",
    "remaining_failed_jobs": total_terminal_failures,
    "datasets": datasets,
}
temporary = target.with_name(f".{target.name}.tmp")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
temporary.replace(target)
PY
}

on_exit() {
  local code="$?"
  if (( code != 0 )); then
    write_status failed
  fi
}
trap on_exit EXIT
write_status running

for dataset in conll04 scierc ade; do
  base="$data_root/$dataset"
  raw="$run_root/${dataset}_validation.jsonl"
  expanded="$run_root/${dataset}_validation_expanded.jsonl"
  verified="$run_root/${dataset}_validation_verified.jsonl"

  bash scripts/run_qlora_inference_sharded.sh \
    --workers 2 \
    --model-path "$model" \
    --jobs "$base/validation_baseline_jobs.jsonl" \
    --output "$raw" \
    --log "$run_root/${dataset}_validation.log" \
    --compact-target \
    --resume \
    --retry-failed \
    --max-input-tokens 4096 \
    --max-new-tokens 1024 \
    --max-seconds-per-job 90

  python3 scripts/evaluate_annotations.py \
    --gold "$base/validation_gold.jsonl" \
    --gold-index "$base/validation_index.jsonl" \
    --predictions "$raw" \
    --include-missing-as-empty \
    --output "$run_root/${dataset}_validation_raw_metrics.json"
  python3 scripts/expand_compact_predictions.py \
    --jobs "$base/validation_baseline_jobs.jsonl" \
    --predictions "$raw" \
    --output "$expanded" \
    --errors "$run_root/${dataset}_validation_expand_errors.jsonl"
  python3 scripts/verify_relations.py \
    --annotations "$expanded" \
    --ontology "$base/ontology.yaml" \
    --output "$verified" \
    --audit "$run_root/${dataset}_validation_audit.jsonl"
  python3 scripts/evaluate_annotations.py \
    --gold "$base/validation_gold.jsonl" \
    --gold-index "$base/validation_index.jsonl" \
    --predictions "$verified" \
    --include-missing-as-empty \
    --output "$run_root/${dataset}_validation_metrics.json"
  python3 scripts/evaluate_public_validation_spans.py \
    --gold "$base/validation_gold.jsonl" \
    --gold-index "$base/validation_index.jsonl" \
    --predictions "$verified" \
    --jobs "$base/validation_baseline_jobs.jsonl" \
    --output "$run_root/${dataset}_validation_character_span_metrics.json"
done

write_status finalizing
remaining_failed_jobs="$(jq -r '.remaining_failed_jobs // 0' "$retry_status")"
if (( remaining_failed_jobs > 0 )); then
  write_status complete_with_terminal_failures
else
  write_status complete
fi
trap - EXIT
