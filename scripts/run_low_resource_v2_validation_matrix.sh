#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

protocol="configs/low_resource_protocol_v2.yaml"
matrix="data/processed/experiments/formal/low_resource_manifests_v2/run_matrix.jsonl"
experiment_root="data/processed/experiments/formal/low_resource_v2"
runtime_python="/home/xuelin/miniconda3/envs/rc-llm-comet/bin/python"
model_path="/ds2/xuelin/cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
ontology="configs/risk_ontology.yaml"
source_gold="data/processed/reviewed/formal_split/validation.jsonl"
source_gold_index="data/processed/reviewed/formal_split/validation_index.jsonl"

expected_protocol_sha256="7ad5473c4dabbf7f3355ebf575209b041df9f47566e04c663ed6dc9a867405e5"

fail() {
  printf '[%s] ERROR: %s\n' "$(date --iso-8601=seconds)" "$*" >&2
  exit 2
}

require_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "$path" | cut -d' ' -f1)"
  [[ "$actual" == "$expected" ]] || fail "frozen hash mismatch: $path"
}

check_frozen_inputs() {
  require_sha256 "$protocol" "$expected_protocol_sha256"
  require_sha256 scripts/run_qlora_inference.py 44a573ec5037e22c137937beb0af31bf58c95f5e4e4afa97f6874dcb4841c486
  require_sha256 scripts/expand_compact_predictions.py e094d77dc47873645392818467df86729a06dd5886fb27c1338d028f1b9e8ad1
  require_sha256 scripts/verify_relations.py f5dd5a138d365b4592d1b9fb87ca4f2073c5d3dec4e63fc1ecb9faef6915588b
  require_sha256 scripts/fuse_kg_v1_v2_predictions.py b3f5ff78dcc76ad34e4e764c7e38bacb682888cc67a842c6a7c7dfbff3be2cf6
  require_sha256 scripts/evaluate_span_aware.py edc0805c273f177c92d5bf700a3a3ad92e0ded69f2c8c5c41b6e537f3a08fe0c
  require_sha256 scripts/evaluate_evidence_graph.py b502bba94d4230411ed1209aac47f550720806c2d9e5efdd16aa63c7eee56538
  require_sha256 scripts/bootstrap_span_compare.py 820ea9a26100fcea747885a8ce10fa78d6a71b09f30a67fc3021e4b9caa2bc66
  require_sha256 "$ontology" af6df8179a97156c513046427b16ca271b72c5c7ca88b4d0345ee56b635c4c10
  [[ -f "$source_gold" && -f "$source_gold_index" ]] || fail "validation source gold is missing"
  [[ "$source_gold" == *validation* && "$source_gold" != *test* ]] || fail "unsafe source-gold path"

  local complete_runs=0
  while IFS= read -r output_directory; do
    if jq -e \
      '.status == "complete" and .return_code == 0 and .formal_test_read == false' \
      "$output_directory/run_manifest.json" >/dev/null 2>&1; then
      complete_runs=$((complete_runs + 1))
    fi
  done < <(jq -r '.output_directory' "$matrix")
  [[ "$complete_runs" == "36" ]] || fail "training gate has only $complete_runs/36 complete rows"
}

audit_inference() {
  local jobs="$1"
  local predictions="$2"
  local inference_log="$3"
  local audit="$4"
  "$runtime_python" - "$jobs" "$predictions" "$inference_log" "$audit" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

jobs_path, predictions_path, log_path, audit_path = map(Path, sys.argv[1:])

def rows(path):
    values = []
    if not path.exists():
        return values
    # JSON strings may contain Unicode line separators (for example U+0085).
    # JSONL records are separated only by the physical LF byte, so splitting on
    # every Unicode line boundary can turn a valid record into two fragments.
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        values.append(json.loads(line))
    return values

jobs = rows(jobs_path)
predictions = rows(predictions_path)
logs = rows(log_path)
job_ids = [str(row["job_id"]) for row in jobs]
if len(job_ids) != 491 or len(set(job_ids)) != len(job_ids):
    raise SystemExit(f"unexpected validation job set: {len(job_ids)} rows")
terminal = {}
for row in logs:
    job_id = str(row.get("job_id", ""))
    if job_id not in set(job_ids):
        raise SystemExit(f"unknown inference-log job: {job_id}")
    terminal[job_id] = row
missing = sorted(set(job_ids) - set(terminal))
if missing:
    raise SystemExit(f"inference has {len(missing)} non-terminal jobs")
prediction_ids = [str(row.get("job_id", "")) for row in predictions]
if len(prediction_ids) != len(set(prediction_ids)):
    raise SystemExit("duplicate successful prediction IDs")
success_ids = {job_id for job_id, row in terminal.items() if row.get("status") == "success"}
if set(prediction_ids) != success_ids:
    raise SystemExit("successful prediction IDs differ from successful log IDs")

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

audit = {
    "status": "complete",
    "selection_split": "validation",
    "jobs": len(job_ids),
    "successes": len(success_ids),
    "failures_scored_as_empty": len(job_ids) - len(success_ids),
    "jobs_sha256": sha256(jobs_path),
    "predictions_sha256": sha256(predictions_path),
    "inference_log_sha256": sha256(log_path),
    "formal_test_read": False,
}
audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(audit, indent=2, sort_keys=True))
PY
}

run_inference_row() {
  local budget_tag="$1"
  local seed="$2"
  local system="$3"
  local run_root="$experiment_root/d${budget_tag}/seed${seed}/${system}"
  local assets="$experiment_root/d${budget_tag}/assets/validation"
  local jobs="$assets/${system}_jobs.jsonl"
  local output_root="$run_root/validation"
  local predictions="$output_root/predictions_compact.jsonl"
  local inference_log="$output_root/inference.log"
  local audit="$output_root/inference_audit.json"
  local -a instruction_args=()

  [[ -f "$run_root/adapter_model.safetensors" ]] || fail "missing adapter: $run_root"
  [[ -f "$jobs" && "$jobs" == *validation* && "$jobs" != *test* ]] || fail "unsafe jobs path: $jobs"
  mkdir -p "$output_root"
  if [[ "$system" != "baseline" ]]; then
    instruction_args+=(--use-job-instruction)
  fi

  if [[ -f "$audit" ]] && jq -e \
    '.status == "complete" and .jobs == 491 and .formal_test_read == false' \
    "$audit" >/dev/null 2>&1; then
    printf '[%s] inference already audited: d%s seed%s %s\n' \
      "$(date --iso-8601=seconds)" "$budget_tag" "$seed" "$system"
    return
  fi

  printf '[%s] inference start/resume: d%s seed%s %s\n' \
    "$(date --iso-8601=seconds)" "$budget_tag" "$seed" "$system"
  "$runtime_python" scripts/run_qlora_inference.py \
    --model-path "$model_path" \
    --adapter "$run_root" \
    --jobs "$jobs" \
    --output "$predictions" \
    --log "$inference_log" \
    --max-input-tokens 12288 \
    --max-new-tokens 4096 \
    --max-seconds-per-job 180 \
    --max-repeated-entity 6 \
    --stop-on-complete-json \
    --compact-target \
    --resume \
    "${instruction_args[@]}"
  audit_inference "$jobs" "$predictions" "$inference_log" "$audit"
}

materialize_empty_predictions() {
  local predictions="$1"
  local jobs="$2"
  local output="$3"
  "$runtime_python" - "$predictions" "$jobs" "$output" <<'PY'
import json
import sys
from pathlib import Path

predictions_path, jobs_path, output_path = map(Path, sys.argv[1:])

def rows(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]

predictions = {}
for row in rows(predictions_path):
    job_id = str(row["job_id"])
    if job_id in predictions:
        raise SystemExit(f"duplicate prediction ID: {job_id}")
    predictions[job_id] = row

jobs = rows(jobs_path)
job_ids = [str(job["job_id"]) for job in jobs]
unknown = sorted(set(predictions) - set(job_ids))
if unknown:
    raise SystemExit(f"predictions contain unknown jobs: {unknown[:3]}")

empty_count = 0
with output_path.open("w", encoding="utf-8") as stream:
    for job in jobs:
        job_id = str(job["job_id"])
        row = predictions.get(job_id)
        if row is None:
            empty_count += 1
            row = {
                "job_id": job_id,
                "annotation": {
                    "schema_version": "0.1.0",
                    "document_id": job["document_id"],
                    "language": job["language"],
                    "entities": [],
                    "relations": [],
                    "review": {
                        "status": "unreviewed",
                        "reviewers": [],
                        "notes": "empty prediction materialized from terminal inference failure",
                    },
                },
            }
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")

print(json.dumps({
    "jobs": len(jobs),
    "successful_prediction_rows": len(predictions),
    "failures_materialized_as_empty": empty_count,
    "output": str(output_path),
}, indent=2, sort_keys=True))
PY
}

evaluate_predictions() {
  local predictions="$1"
  local jobs="$2"
  local assets="$3"
  local output_prefix="$4"
  local complete_predictions="${output_prefix}_complete_predictions.jsonl"
  "$runtime_python" scripts/evaluate_span_aware.py \
    --source-gold "$source_gold" \
    --source-gold-index "$source_gold_index" \
    --gold "$assets/gold.jsonl" \
    --gold-index "$assets/index.jsonl" \
    --predictions "$predictions" \
    --jobs "$jobs" \
    --output "${output_prefix}_span_metrics.json"
  materialize_empty_predictions "$predictions" "$jobs" "$complete_predictions"
  "$runtime_python" scripts/evaluate_evidence_graph.py \
    --gold "$assets/gold.jsonl" \
    --gold-index "$assets/index.jsonl" \
    --predictions "$complete_predictions" \
    --jobs "$jobs" \
    --ontology "$ontology" \
    --output "${output_prefix}_graph_metrics.json"
}

process_group() {
  local budget_tag="$1"
  local seed="$2"
  local budget_root="$experiment_root/d${budget_tag}"
  local assets="$budget_root/assets/validation"
  local derived="$budget_root/seed${seed}/derived_validation"
  local completion="$derived/pipeline_complete.json"
  local system

  if [[ -f "$completion" ]] && jq -e \
    '.status == "complete" and .formal_test_read == false' "$completion" >/dev/null 2>&1; then
    printf '[%s] derived validation already complete: d%s seed%s\n' \
      "$(date --iso-8601=seconds)" "$budget_tag" "$seed"
    return
  fi
  mkdir -p "$derived"

  for system in baseline kg_v1 kg_v2; do
    local run_validation="$budget_root/seed${seed}/${system}/validation"
    "$runtime_python" scripts/expand_compact_predictions.py \
      --jobs "$assets/${system}_jobs.jsonl" \
      --predictions "$run_validation/predictions_compact.jsonl" \
      --output "$derived/${system}_expanded.jsonl" \
      --errors "$derived/${system}_expansion_errors.jsonl"
    evaluate_predictions \
      "$derived/${system}_expanded.jsonl" \
      "$assets/${system}_jobs.jsonl" \
      "$assets" \
      "$derived/${system}"
  done

  "$runtime_python" scripts/verify_relations.py \
    --annotations "$derived/kg_v2_complete_predictions.jsonl" \
    --ontology "$ontology" \
    --output "$derived/kg_v2_verified.jsonl" \
    --audit "$derived/kg_v2_verifier_audit.jsonl"

  "$runtime_python" scripts/fuse_kg_v1_v2_predictions.py \
    --v1 "$derived/kg_v1_complete_predictions.jsonl" \
    --v2 "$derived/kg_v2_complete_predictions.jsonl" \
    --verified "$derived/kg_v2_verified.jsonl" \
    --jobs "$assets/kg_v2_jobs.jsonl" \
    --output "$derived/kg_v3_raw.jsonl" \
    --audit "$derived/kg_v3_raw_audit.jsonl" \
    --relation-mode raw

  "$runtime_python" scripts/fuse_kg_v1_v2_predictions.py \
    --v1 "$derived/kg_v1_complete_predictions.jsonl" \
    --v2 "$derived/kg_v2_complete_predictions.jsonl" \
    --verified "$derived/kg_v2_verified.jsonl" \
    --jobs "$assets/kg_v2_jobs.jsonl" \
    --output "$derived/kg_v3_final.jsonl" \
    --audit "$derived/kg_v3_final_audit.jsonl" \
    --relation-mode verified

  evaluate_predictions "$derived/kg_v2_verified.jsonl" "$assets/kg_v2_jobs.jsonl" "$assets" "$derived/kg_v2_verified"
  evaluate_predictions "$derived/kg_v3_raw.jsonl" "$assets/kg_v2_jobs.jsonl" "$assets" "$derived/kg_v3_raw"
  evaluate_predictions "$derived/kg_v3_final.jsonl" "$assets/kg_v2_jobs.jsonl" "$assets" "$derived/kg_v3_final"

  "$runtime_python" scripts/bootstrap_span_compare.py \
    --left "$derived/kg_v1_span_metrics.json" \
    --right "$derived/kg_v2_span_metrics.json" \
    --output "$derived/kg_v1_vs_kg_v2_bootstrap.json" \
    --iterations 20000 --seed 20260830
  "$runtime_python" scripts/bootstrap_span_compare.py \
    --left "$derived/kg_v2_span_metrics.json" \
    --right "$derived/kg_v3_raw_span_metrics.json" \
    --output "$derived/kg_v2_vs_kg_v3_raw_bootstrap.json" \
    --iterations 20000 --seed 20260830
  "$runtime_python" scripts/bootstrap_span_compare.py \
    --left "$derived/kg_v2_span_metrics.json" \
    --right "$derived/kg_v3_final_span_metrics.json" \
    --output "$derived/kg_v2_vs_kg_v3_final_bootstrap.json" \
    --iterations 20000 --seed 20260830

  "$runtime_python" - "$derived" "$budget_tag" "$seed" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
budget = int(sys.argv[2])
seed = int(sys.argv[3])
required = sorted(
    path for path in root.iterdir()
    if path.is_file() and path.name != "pipeline_complete.json"
)
if not required:
    raise SystemExit("no derived validation outputs")
manifest = {
    "status": "complete",
    "selection_split": "validation",
    "budget_documents": budget,
    "seed": seed,
    "outputs": {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in required
    },
    "formal_test_read": False,
}
(root / "pipeline_complete.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(manifest, indent=2, sort_keys=True))
PY
}

check_frozen_inputs
printf '[%s] validation matrix controller started; formal test remains sealed\n' \
  "$(date --iso-8601=seconds)"

validation_lane="${VALIDATION_LANE:-all}"
case "$validation_lane" in
  a)
    priority_groups=(
      "100 20260830"
      "100 20260901"
      "050 20260830"
      "025 20260831"
      "010 20260901"
      "050 20260901"
    )
    ;;
  b)
    priority_groups=(
      "100 20260831"
      "010 20260830"
      "025 20260830"
      "050 20260831"
      "010 20260831"
      "025 20260901"
    )
    ;;
  smoke)
    # Run one already-inferred D100 group through the complete derivation chain
    # without advancing to another long-running validation group.
    priority_groups=(
      "100 20260830"
    )
    ;;
  all)
    priority_groups=(
      "100 20260830"
      "100 20260831"
      "100 20260901"
      "010 20260830"
      "050 20260830"
      "025 20260830"
      "010 20260831"
      "010 20260901"
      "050 20260831"
      "050 20260901"
      "025 20260831"
      "025 20260901"
    )
    ;;
  *)
    fail "unknown VALIDATION_LANE: $validation_lane"
    ;;
esac

for group in "${priority_groups[@]}"; do
  read -r budget_tag seed <<<"$group"
  for system in baseline kg_v1 kg_v2; do
    check_frozen_inputs
    run_inference_row "$budget_tag" "$seed" "$system"
  done
  process_group "$budget_tag" "$seed"
done

printf '[%s] validation lane %s complete; formal test remains sealed\n' \
  "$(date --iso-8601=seconds)" "$validation_lane"
