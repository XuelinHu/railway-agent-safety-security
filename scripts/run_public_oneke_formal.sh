#!/usr/bin/env bash
# Resumable OneKE validation runner for the three complete public datasets.
# It has no test mode.  Prediction generation finishes before validation gold
# is passed to the common evaluator.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

runtime="${PUBLIC_ONEKE_PYTHON:-/ds2/xuelin/envs/public-oneke-formal/bin/python}"
model="${PUBLIC_ONEKE_MODEL:-/ds2/xuelin/cache/huggingface/hub/models--zjunlp--OneKE/snapshots/696148c0581b29f530af738ddab500deaa8fe8f2}"
source_root="data/processed/public_benchmarks_full"
run_root="${PUBLIC_ONEKE_RUN_ROOT:-outputs/public_external_formal/oneke}"
canary="$run_root/gpu_canary.json"
runner_status="$run_root/status.json"
gpu_lock="${PUBLIC_GPU_LOCK_FILE:-outputs/.locks/public-validation-gpu.lock}"
poll_seconds="${PUBLIC_ONEKE_POLL_SECONDS:-30}"
current_dataset=""
current_dataset_status=""
gpu_lock_held=0
mkdir -p "$run_root" "$(dirname "$gpu_lock")"

write_status_file() {
  local path="$1" state="$2" stage="$3" detail="$4"
  python3 - "$path" "$state" "$stage" "$detail" "$current_dataset" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema_version": "public-oneke-formal-status-v1",
    "status": sys.argv[2],
    "stage": sys.argv[3],
    "detail": sys.argv[4],
    "dataset": sys.argv[5] or None,
    "split": "validation",
    "seed": 42,
    "formal_test_read": False,
    "gpu_required": sys.argv[2] == "running" and sys.argv[3] == "inference",
    "gpu_lock_scope": "inference_only",
    "terminal": sys.argv[2] in {"complete", "blocked_with_reason"},
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
temporary = path.with_name(f".{path.name}.tmp")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
temporary.replace(path)
PY
}

write_runner_status() {
  write_status_file "$runner_status" "$1" "$2" "$3"
}

release_gpu_lock() {
  if (( gpu_lock_held )); then
    flock -u 8 2>/dev/null || true
    gpu_lock_held=0
  fi
}

on_exit() {
  local code="$?"
  release_gpu_lock
  if (( code != 0 )); then
    write_runner_status blocked_with_reason failed "formal OneKE runner exited with code $code"
    if [[ -n "$current_dataset_status" ]]; then
      write_status_file "$current_dataset_status" blocked_with_reason failed "formal OneKE runner exited with code $code"
    fi
  fi
}
trap on_exit EXIT

if [[ ! -x "$runtime" ]]; then
  write_runner_status blocked_with_reason preflight "isolated OneKE runtime is missing"
  exit 2
fi
if [[ ! -f "$canary" ]] || [[ "$(jq -r '.status // ""' "$canary")" != passed ]]; then
  write_runner_status blocked_with_reason preflight "verified RTX 3090 canary is required"
  exit 2
fi

# Prepare model requests without reading either validation or test gold.
for dataset in conll04 scierc ade; do
  case "$dataset" in
    conll04) expected=231 ;;
    scierc) expected=275 ;;
    ade) expected=384 ;;
  esac
  dataset_root="$run_root/$dataset/seed42"
  mkdir -p "$dataset_root"
  requests="$dataset_root/validation_requests.jsonl"
  request_summary="$dataset_root/validation_request_summary.json"
  # Regenerate atomically on every resume so a stale or manually replaced
  # request file can never become the inference authority.
  python3 scripts/public_external_adapters/oneke.py prepare \
    --dataset "$dataset" \
    --jobs "$source_root/$dataset/validation_baseline_jobs.jsonl" \
    --expected-rows "$expected" \
    --output "$requests" \
    --summary "$request_summary"
  [[ "$(awk 'NF {count += 1} END {print count + 0}' "$requests")" -eq "$expected" ]]
done

exec 8>"$gpu_lock"

for dataset in conll04 scierc ade; do
  case "$dataset" in
    conll04) expected=231 ;;
    scierc) expected=275 ;;
    ade) expected=384 ;;
  esac
  current_dataset="$dataset"
  dataset_root="$run_root/$dataset/seed42"
  current_dataset_status="$dataset_root/status.json"
  requests="$dataset_root/validation_requests.jsonl"
  raw="$dataset_root/validation_raw_responses.jsonl"
  predictions="$dataset_root/validation_predictions.jsonl"
  audit="$dataset_root/conversion_audit.jsonl"
  conversion_summary="$dataset_root/conversion_summary.json"
  metrics="$dataset_root/validation_character_span_metrics.json"
  manifest="$dataset_root/run_manifest.json"

  if [[ -f "$current_dataset_status" ]] && \
     [[ "$(jq -r '.status // ""' "$current_dataset_status")" == complete ]] && \
     [[ -f "$manifest" ]] && [[ -f "$predictions" ]] && [[ -f "$metrics" ]]; then
    continue
  fi

  write_status_file "$current_dataset_status" waiting_for_gpu waiting_for_exclusive_gpu_lock \
    "waiting for the shared public GPU lock"
  write_runner_status waiting_for_gpu waiting_for_exclusive_gpu_lock "$dataset"
  while ! flock -n 8; do
    sleep "$poll_seconds"
    write_runner_status waiting_for_gpu waiting_for_exclusive_gpu_lock "$dataset"
  done
  gpu_lock_held=1

  write_status_file "$current_dataset_status" running inference "gold-inaccessible OneKE generation"
  write_runner_status running inference "$dataset"
  CUDA_VISIBLE_DEVICES=0 "$runtime" scripts/run_public_oneke_formal.py \
    --model "$model" \
    --max-input-tokens 3072 \
    --max-new-tokens 512 \
    validation \
    --requests "$requests" \
    --output "$raw" \
    --resume
  release_gpu_lock
  [[ "$(awk 'NF {count += 1} END {print count + 0}' "$raw")" -eq "$expected" ]]

  write_status_file "$current_dataset_status" running conversion "converting raw outputs without gold"
  python3 scripts/public_external_adapters/oneke.py convert \
    --requests "$requests" \
    --raw "$raw" \
    --output "$predictions" \
    --audit "$audit" \
    --summary "$conversion_summary"
  [[ "$(awk 'NF {count += 1} END {print count + 0}' "$predictions")" -eq "$expected" ]]

  # Validation gold is opened exactly here, after the prediction file is closed.
  write_status_file "$current_dataset_status" running evaluation "single canonical validation evaluation"
  python3 scripts/evaluate_public_validation_spans.py \
    --gold "$source_root/$dataset/validation_gold.jsonl" \
    --gold-index "$source_root/$dataset/validation_index.jsonl" \
    --predictions "$predictions" \
    --jobs "$source_root/$dataset/validation_baseline_jobs.jsonl" \
    --output "$metrics"

  python3 - \
    "$manifest" "$dataset" "$predictions" "$metrics" "$conversion_summary" "$canary" \
    "$source_root/$dataset/train_gold.jsonl" \
    "$source_root/$dataset/train_index.jsonl" \
    "$source_root/$dataset/train_baseline_jobs.jsonl" \
    "$source_root/$dataset/validation_gold.jsonl" \
    "$source_root/$dataset/validation_index.jsonl" \
    "$source_root/$dataset/validation_baseline_jobs.jsonl" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


manifest = Path(sys.argv[1])
dataset = sys.argv[2]
predictions = Path(sys.argv[3])
metrics = Path(sys.argv[4])
conversion = json.loads(Path(sys.argv[5]).read_text(encoding="utf-8"))
canary = Path(sys.argv[6])
input_names = (
    "train_gold",
    "train_index",
    "train_jobs",
    "validation_gold",
    "validation_index",
    "validation_jobs",
)
inputs = {name: sha256(path) for name, path in zip(input_names, sys.argv[7:])}
payload = {
    "schema_version": "public-external-formal-run-v1",
    "status": "complete",
    "baseline": "oneke",
    "dataset": dataset,
    "split": "validation",
    "seed": 42,
    "model_revision": "696148c0581b29f530af738ddab500deaa8fe8f2",
    "quantization": "bitsandbytes-nf4-double-quantization",
    "prompt_version": "oneke-upstream-jsonlike-batched-v3",
    "evaluator": "scripts/evaluate_public_validation_spans.py",
    "evaluator_sha256": sha256("scripts/evaluate_public_validation_spans.py"),
    "input_sha256": inputs,
    "prediction_sha256": sha256(predictions),
    "metric_sha256": sha256(metrics),
    "gpu_canary_sha256": sha256(canary),
    "prediction_rows": conversion["predictions"],
    "terminal_failures": conversion["terminal_failures"],
    "inference_gold_read": False,
    "test_gold_read": False,
    "finished_at": datetime.now(timezone.utc).isoformat(),
}
temporary = manifest.with_name(f".{manifest.name}.tmp")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
temporary.replace(manifest)
PY
  write_status_file "$current_dataset_status" complete complete "prediction, conversion, and validation evaluation complete"
done

current_dataset=""
current_dataset_status=""
write_runner_status complete complete "all three public validation datasets complete"
trap - EXIT
