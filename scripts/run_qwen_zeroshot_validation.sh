#!/usr/bin/env bash
# Run a validation-only Qwen3-4B zero-shot control on all public benchmarks.
set -euo pipefail

export PATH="/home/xuelin/miniconda3/bin:$PATH"
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

model="/ds2/xuelin/cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
data_root="data/processed/public_benchmarks_full"
run_root="outputs/public_horizontal_validation/qwen3_4b_zero_shot"
mkdir -p "$run_root"

gpu_lock="${PUBLIC_GPU_LOCK_FILE:-outputs/.locks/public-validation-gpu.lock}"
mkdir -p "$(dirname "$gpu_lock")"
exec 8>"$gpu_lock"
while ! flock -n 8; do
  echo "$(date --iso-8601=seconds) waiting_for_public_validation_gpu_lock"
  sleep 30
done

for dataset in conll04 scierc ade; do
  base="$data_root/$dataset"
  raw="$run_root/${dataset}_validation.jsonl"
  expanded="$run_root/${dataset}_validation_expanded.jsonl"
  verified="$run_root/${dataset}_validation_verified.jsonl"

  if [[ ! -f "$run_root/${dataset}_validation_metrics.json" ]]; then
    bash scripts/run_qlora_inference_sharded.sh \
      --workers 2 \
      --model-path "$model" \
      --jobs "$base/validation_baseline_jobs.jsonl" \
      --output "$raw" \
      --log "$run_root/${dataset}_validation.log" \
      --compact-target \
      --resume \
      --max-input-tokens 4096 \
      --max-new-tokens 512 \
      --max-seconds-per-job 45

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
  fi
done

python3 - "$run_root/status.json" "$data_root" "$run_root" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

status_path = Path(sys.argv[1])
data_root = Path(sys.argv[2])
run_root = Path(sys.argv[3])
datasets = {}
total_failures = 0
for dataset in ("conll04", "scierc", "ade"):
    jobs_path = data_root / dataset / "validation_baseline_jobs.jsonl"
    prediction_path = run_root / f"{dataset}_validation.jsonl"
    log_path = run_root / f"{dataset}_validation.log"
    expected = [
        json.loads(line)["job_id"]
        for line in jobs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(expected) != len(set(expected)):
        raise SystemExit(f"{dataset}: duplicate validation job IDs")
    expected_ids = set(expected)
    predictions = [
        json.loads(line)["job_id"]
        for line in prediction_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(predictions) != len(set(predictions)):
        raise SystemExit(f"{dataset}: duplicate prediction IDs")
    terminal = {}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            terminal[row["job_id"]] = row.get("status")
    unknown = (set(predictions) | set(terminal)) - expected_ids
    missing_logs = expected_ids - set(terminal)
    invalid_statuses = {
        job_id: value for job_id, value in terminal.items()
        if value not in {"success", "failed"}
    }
    successful = {job_id for job_id, value in terminal.items() if value == "success"}
    if unknown or missing_logs or invalid_statuses or set(predictions) != successful:
        raise SystemExit(
            f"{dataset}: invalid terminal coverage "
            f"unknown={sorted(unknown)[:3]} missing_logs={sorted(missing_logs)[:3]} "
            f"invalid_statuses={list(invalid_statuses.items())[:3]} "
            f"prediction_mismatch={sorted(set(predictions) ^ successful)[:3]}"
        )
    failures = len(expected) - len(successful)
    total_failures += failures
    datasets[dataset] = {
        "expected_jobs": len(expected),
        "successful_predictions": len(successful),
        "terminal_failures": failures,
    }

payload = {
    "status": "validation_complete",
    "finished_at": datetime.now(timezone.utc).isoformat(),
    "terminal_failures": total_failures,
    "terminal_failure_evaluation": "missing predictions are scored as empty annotations",
    "datasets": datasets,
}
temporary = status_path.with_name(f".{status_path.name}.tmp")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
temporary.replace(status_path)
PY
