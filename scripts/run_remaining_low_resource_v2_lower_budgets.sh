#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

protocol="configs/low_resource_protocol_v2.yaml"
matrix="data/processed/experiments/formal/low_resource_manifests_v2/run_matrix.jsonl"
d100_gate="data/processed/experiments/formal/low_resource_v2/d100_execution_gate.json"
expected_protocol_sha256="7ad5473c4dabbf7f3355ebf575209b041df9f47566e04c663ed6dc9a867405e5"
expected_matrix_sha256="dbf9da4c78a9df3b79e674d61c4157c4c4b488dd91cdd42b4a2e744c34c7a84e"
orchestration_python="/home/xuelin/miniconda3/bin/python"
semantic_model="/ds2/xuelin/cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"

fail() {
  printf '[%s] ERROR: %s\n' "$(date --iso-8601=seconds)" "$*" >&2
  exit 2
}

check_frozen_inputs() {
  [[ "$(sha256sum "$protocol" | cut -d' ' -f1)" == "$expected_protocol_sha256" ]] ||
    fail "frozen protocol hash mismatch: $protocol"
  [[ "$(sha256sum "$matrix" | cut -d' ' -f1)" == "$expected_matrix_sha256" ]] ||
    fail "frozen run-matrix hash mismatch: $matrix"
  jq -e \
    --arg matrix_sha256 "$expected_matrix_sha256" \
    '.gate_status == "passed_all_d100"
      and .complete_runs == 9
      and .not_started_runs == 0
      and .lower_budgets_authorized == true
      and .formal_test_read == false
      and .validation_metrics_read == false
      and .matrix_sha256 == $matrix_sha256' \
    "$d100_gate" >/dev/null || fail "d100 authorization gate is not valid"
}

build_budget_assets() {
  local budget="$1"
  local budget_tag
  local document_manifest
  local asset_root
  budget_tag="$(printf '%03d' "$budget")"
  document_manifest="$(
    jq -r --argjson budget "$budget" \
      'select(.budget_documents == $budget) | .document_manifest' "$matrix" |
      sort -u
  )"
  asset_root="data/processed/experiments/formal/low_resource_v2/d${budget_tag}/assets"
  [[ "$(printf '%s\n' "$document_manifest" | sed '/^$/d' | wc -l | tr -d ' ')" == "1" ]] ||
    fail "budget d${budget_tag} does not resolve to one document manifest"
  [[ ! -e "$asset_root" ]] || fail "refusing existing asset directory: $asset_root"

  printf '[%s] assets: building d%s train/validation inputs\n' \
    "$(date --iso-8601=seconds)" "$budget_tag"
  "$orchestration_python" scripts/build_low_resource_v2_assets.py \
    --document-manifest "$document_manifest" \
    --output "$asset_root"

  printf '[%s] assets: building d%s KG V2 train prompts\n' \
    "$(date --iso-8601=seconds)" "$budget_tag"
  "$orchestration_python" scripts/build_kg_v2_jobs.py \
    --jobs "$asset_root/baseline_jobs.jsonl" \
    --concepts "$asset_root/knowledge_graph/concepts.jsonl" \
    --relations "$asset_root/knowledge_graph/relations.jsonl" \
    --output "$asset_root/kg_v2_jobs.jsonl" \
    --audit "$asset_root/kg_v2_audit.json" \
    --semantic-model "$semantic_model" \
    --device cuda --batch-size 16 --semantic-threshold 0.72 \
    --semantic-limit 4 --anchor-limit 12 --anchor-per-type 2 \
    --edge-limit 6 --min-type-purity 0.8 --min-en-chars 4 --min-zh-chars 2

  printf '[%s] assets: building d%s KG V2 validation prompts without metrics\n' \
    "$(date --iso-8601=seconds)" "$budget_tag"
  "$orchestration_python" scripts/build_kg_v2_jobs.py \
    --jobs "$asset_root/validation/baseline_jobs.jsonl" \
    --concepts "$asset_root/knowledge_graph/concepts.jsonl" \
    --relations "$asset_root/knowledge_graph/relations.jsonl" \
    --output "$asset_root/validation/kg_v2_jobs.jsonl" \
    --audit "$asset_root/validation/kg_v2_audit.json" \
    --semantic-model "$semantic_model" \
    --device cuda --batch-size 16 --semantic-threshold 0.72 \
    --semantic-limit 4 --anchor-limit 12 --anchor-per-type 2 \
    --edge-limit 6 --min-type-purity 0.8 --min-en-chars 4 --min-zh-chars 2

  # Refresh the immutable output-hash map now that KG V2 files exist.
  "$orchestration_python" scripts/build_low_resource_v2_assets.py \
    --document-manifest "$document_manifest" \
    --output "$asset_root"

  "$orchestration_python" - "$asset_root" "$budget" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
budget = int(sys.argv[2])
manifest_path = root / "asset_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
required = (
    "baseline_jobs.jsonl",
    "gold.jsonl",
    "index.jsonl",
    "kg_v1_jobs.jsonl",
    "kg_v2_jobs.jsonl",
    "kg_v2_audit.json",
    "validation/baseline_jobs.jsonl",
    "validation/gold.jsonl",
    "validation/index.jsonl",
    "validation/kg_v1_jobs.jsonl",
    "validation/kg_v2_jobs.jsonl",
    "validation/kg_v2_audit.json",
)
if manifest.get("protocol_id") != "low-resource-assets-v2":
    raise SystemExit("unexpected asset protocol")
if manifest.get("formal_test_read") is not False:
    raise SystemExit("asset build reports formal-test access")
if manifest.get("validation_used_for_selection") is not False:
    raise SystemExit("asset build reports validation-based selection")
if int(manifest.get("selected_documents", -1)) != budget:
    raise SystemExit("selected-document count differs from budget")
if int(manifest.get("window_jobs", 0)) <= 0:
    raise SystemExit("asset build contains no training windows")
outputs = manifest.get("outputs", {})
for relative in required:
    path = root / relative
    if not path.is_file():
        raise SystemExit(f"missing required asset: {path}")
    expected = outputs.get(relative)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"asset hash mismatch: {path}")
print(json.dumps({
    "budget_documents": budget,
    "train_windows": manifest["window_jobs"],
    "validation_windows": manifest["validation_jobs"],
    "formal_test_read": False,
    "status": "assets_audited",
}, indent=2))
PY
}

run_row() {
  local run_id="$1"
  local output_directory
  output_directory="$(
    jq -r --arg run_id "$run_id" \
      'select(.run_id == $run_id) | .output_directory' "$matrix"
  )"
  [[ -n "$output_directory" && "$output_directory" != "null" ]] ||
    fail "matrix row not found: $run_id"
  [[ ! -e "$output_directory" ]] || fail "refusing existing output: $output_directory"

  check_frozen_inputs
  printf '[%s] preflight: %s\n' "$(date --iso-8601=seconds)" "$run_id"
  "$orchestration_python" scripts/run_low_resource_training_v2.py \
    --run-id "$run_id" --preflight-only

  printf '[%s] training: %s\n' "$(date --iso-8601=seconds)" "$run_id"
  "$orchestration_python" scripts/run_low_resource_training_v2.py \
    --run-id "$run_id"

  test -f "$output_directory/adapter_model.safetensors"
  test -f "$output_directory/training_metrics.json"
  test -f "$output_directory/telemetry.json"
  jq -e \
    --arg protocol_sha256 "$expected_protocol_sha256" \
    '.status == "complete"
      and .return_code == 0
      and .formal_test_read == false
      and .protocol_sha256 == $protocol_sha256
      and .environment.sequence_audit.examples > 0
      and .environment.sequence_audit.overlength_examples == 0
      and .environment.sequence_audit.truncation_or_skip_allowed == false' \
    "$output_directory/run_manifest.json" >/dev/null
  expected_examples="$(
    jq -r '.environment.sequence_audit.examples' "$output_directory/run_manifest.json"
  )"
  jq -e --argjson expected_examples "$expected_examples" \
    '.train_examples == $expected_examples
      and .steps == ((($expected_examples + 3) / 4) | floor)
      and .truncated_answers_with_eos == 0
      and .truncated_prompts == 0
      and .skipped_overlength == 0' \
    "$output_directory/training_metrics.json" >/dev/null
  jq -e --arg run_id "$run_id" --argjson expected_examples "$expected_examples" \
    '.run_id == $run_id
      and .samples > 0
      and .wall_clock_seconds > 0
      and .train_examples == $expected_examples
      and .peak_device_memory_used_mib > 0' \
    "$output_directory/telemetry.json" >/dev/null
  printf '[%s] audited complete: %s\n' "$(date --iso-8601=seconds)" "$run_id"
}

check_frozen_inputs
for budget in 10 25 50; do
  build_budget_assets "$budget"
  while IFS= read -r run_id; do
    run_row "$run_id"
  done < <(
    jq -r --argjson budget "$budget" \
      'select(.budget_documents == $budget) | .run_id' "$matrix"
  )
done

complete_runs=0
while IFS= read -r output_directory; do
  if jq -e '.status == "complete" and .formal_test_read == false' \
    "$output_directory/run_manifest.json" >/dev/null 2>&1; then
    complete_runs=$((complete_runs + 1))
  fi
done < <(jq -r '.output_directory' "$matrix")
[[ "$complete_runs" == "36" ]] || fail "final matrix audit found $complete_runs/36 complete rows"
printf '[%s] all 36 protocol v2 training rows passed; validation metrics and formal test remain unread\n' \
  "$(date --iso-8601=seconds)"
