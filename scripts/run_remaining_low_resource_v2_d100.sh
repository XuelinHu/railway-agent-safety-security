#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

protocol="configs/low_resource_protocol_v2.yaml"
matrix="data/processed/experiments/formal/low_resource_manifests_v2/run_matrix.jsonl"
expected_protocol_sha256="7ad5473c4dabbf7f3355ebf575209b041df9f47566e04c663ed6dc9a867405e5"
orchestration_python="/home/xuelin/miniconda3/bin/python"

if [[ "$(sha256sum "$protocol" | cut -d' ' -f1)" != "$expected_protocol_sha256" ]]; then
  printf '[%s] frozen protocol hash mismatch: %s\n' \
    "$(date --iso-8601=seconds)" "$protocol" >&2
  exit 2
fi

runs=(
  lr_v2_d100_seed20260830_kg_v2
  lr_v2_d100_seed20260831_baseline
  lr_v2_d100_seed20260831_kg_v1
  lr_v2_d100_seed20260831_kg_v2
  lr_v2_d100_seed20260901_baseline
  lr_v2_d100_seed20260901_kg_v1
  lr_v2_d100_seed20260901_kg_v2
)

audit_gate() {
  "$orchestration_python" scripts/audit_low_resource_v2_training_gate.py
}

for run_id in "${runs[@]}"; do
  output_directory="$(
    jq -r --arg run_id "$run_id" \
      'select(.run_id == $run_id) | .output_directory' "$matrix"
  )"
  if [[ -z "$output_directory" || "$output_directory" == "null" ]]; then
    printf '[%s] matrix row not found: %s\n' "$(date --iso-8601=seconds)" "$run_id" >&2
    exit 2
  fi
  if [[ -e "$output_directory" ]]; then
    printf '[%s] refusing existing output: %s\n' \
      "$(date --iso-8601=seconds)" "$output_directory" >&2
    exit 2
  fi

  printf '[%s] preflight: %s\n' "$(date --iso-8601=seconds)" "$run_id"
  if ! "$orchestration_python" scripts/run_low_resource_training_v2.py \
    --run-id "$run_id" --preflight-only; then
    audit_gate || true
    exit 1
  fi

  printf '[%s] training: %s\n' "$(date --iso-8601=seconds)" "$run_id"
  run_status=0
  "$orchestration_python" scripts/run_low_resource_training_v2.py \
    --run-id "$run_id" || run_status=$?
  audit_status=0
  audit_gate || audit_status=$?
  if (( run_status != 0 || audit_status != 0 )); then
    printf '[%s] stopped after failed row: %s (run=%d audit=%d)\n' \
      "$(date --iso-8601=seconds)" "$run_id" "$run_status" "$audit_status" >&2
    exit 1
  fi

  test -f "$output_directory/adapter_model.safetensors"
  test -f "$output_directory/training_metrics.json"
  test -f "$output_directory/telemetry.json"
  jq -e \
    --arg protocol_sha256 "$expected_protocol_sha256" \
    '.status == "complete"
      and .return_code == 0
      and .formal_test_read == false
      and .protocol_sha256 == $protocol_sha256' \
    "$output_directory/run_manifest.json" >/dev/null
  jq -e \
    '.train_examples == 1266
      and .steps == 317
      and .truncated_answers_with_eos == 0
      and .truncated_prompts == 0
      and .skipped_overlength == 0' \
    "$output_directory/training_metrics.json" >/dev/null
  printf '[%s] audited complete: %s\n' "$(date --iso-8601=seconds)" "$run_id"
done

audit_gate
jq -e \
  '.gate_status == "passed_all_d100"
    and .complete_runs == 9
    and .not_started_runs == 0
    and .lower_budgets_authorized == true
    and .formal_test_read == false
    and .validation_metrics_read == false' \
  data/processed/experiments/formal/low_resource_v2/d100_execution_gate.json >/dev/null
printf '[%s] all protocol v2 d100 rows passed; lower budgets are authorized but not started\n' \
  "$(date --iso-8601=seconds)"
