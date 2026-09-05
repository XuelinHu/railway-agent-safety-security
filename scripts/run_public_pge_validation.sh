#!/usr/bin/env bash
# Train public EAE/HRGE branches and derive EVGE/CFE/PGE on validation only.
set -euo pipefail

export PATH="/home/xuelin/miniconda3/bin:$PATH"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

model="/ds2/xuelin/cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
source_root="data/processed/public_benchmarks_full"
graph_root="data/processed/public_benchmarks_hrge_v1"
run_root="${PUBLIC_PGE_RUN_ROOT:-outputs/public_pge_validation_seed42}"
config="configs/public_pge_seed42.yaml"
seed="${PUBLIC_PGE_SEED:-42}"
status_file="$run_root/status.json"
gpu_lock="${PUBLIC_GPU_LOCK_FILE:-outputs/.locks/public-validation-gpu.lock}"
current_dataset=""
current_system=""
current_stage="initializing"
mkdir -p "$run_root" "$(dirname "$gpu_lock")"

write_status() {
  local state="$1"
  python3 - "$status_file" "$state" "$current_dataset" "$current_system" "$current_stage" "$run_root" "$seed" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
run_root = Path(sys.argv[6])
datasets = {}
for dataset in ("conll04", "scierc", "ade"):
    dataset_root = run_root / dataset
    datasets[dataset] = {
        "training": {
            branch: (dataset_root / f"{branch}_adapter" / "training_metrics.json").is_file()
            for branch in ("eae", "hrge")
        },
        "metrics": {
            system: (dataset_root / "metrics" / f"{system}_span.json").is_file()
            for system in ("soe", "eae", "hrge", "evge", "cfe", "pge")
        },
        "comparisons": {
            name: (dataset_root / "comparisons" / f"{name}.json").is_file()
            for name in ("eae_vs_hrge", "hrge_vs_cfe", "hrge_vs_pge")
        },
    }
payload = {
    "status": sys.argv[2],
    "selection_split": "validation",
    "formal_test_read": False,
    "seed": int(sys.argv[7]),
    "active_dataset": sys.argv[3] or None,
    "active_system": sys.argv[4] or None,
    "stage": sys.argv[5],
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "datasets": datasets,
}
temporary = path.with_name(f".{path.name}.tmp")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
temporary.replace(path)
PY
}

on_exit() {
  local code="$?"
  if (( code != 0 )); then
    write_status failed
  fi
}
trap on_exit EXIT

current_stage="waiting_for_exclusive_gpu_lock"
write_status waiting_for_gpu
exec 8>"$gpu_lock"
while ! flock -n 8; do
  sleep 30
  write_status waiting_for_gpu
done

current_stage="input_preflight"
write_status running
for dataset in conll04 scierc ade; do
  manifest="$graph_root/$dataset/preparation_manifest.json"
  soe_predictions="outputs/public_full_stage1/validation_analysis/completed_predictions/${dataset}_baseline_expanded.jsonl"
  soe_training="outputs/public_full_stage1/${dataset}_baseline/training_metrics.json"
  case "$dataset" in
    conll04) expected_validation_jobs=231 ;;
    scierc) expected_validation_jobs=275 ;;
    ade) expected_validation_jobs=384 ;;
  esac
  [[ -f "$manifest" ]]
  [[ -f "$soe_predictions" ]]
  [[ -f "$soe_training" ]]
  [[ "$(awk 'NF { count += 1 } END { print count + 0 }' "$soe_predictions")" -eq "$expected_validation_jobs" ]]
  jq -e '
    .status == "prepared_train_and_validation" and
    .test_job_file_read == false and
    .test_gold_read == false and
    .validation_gold_read == false and
    .pge_contract.ready_for_generator_training == true
  ' "$manifest" >/dev/null
  jq -e --arg model "$model" '
    .base_model == $model and
    .source_examples > 0 and
    .train_examples == .source_examples and
    .epochs == 1 and
    .max_length == 4096 and
    .lora_rank == 8 and
    .compact_target == true and
    .use_job_instruction == false and
    .seed == 42 and
    .truncated_prompts == 0 and
    .truncated_answers_with_eos == 0
  ' "$soe_training" >/dev/null
done

train_branch() {
  local dataset="$1" branch="$2"
  local base="$source_root/$dataset"
  local prepared="$graph_root/$dataset/jobs"
  local adapter="$run_root/$dataset/${branch}_adapter"
  current_dataset="$dataset"
  current_system="$branch"
  current_stage="training"
  write_status running
  if [[ ! -f "$adapter/training_metrics.json" ]]; then
    python3 scripts/train_qlora.py \
      --model-path "$model" \
      --gold "$base/train_gold.jsonl" \
      --index "$base/train_index.jsonl" \
      --jobs "$prepared/train_${branch}_jobs.jsonl" \
      --output "$adapter" \
      --epochs 1 \
      --batch-size 1 \
      --gradient-accumulation 4 \
      --learning-rate 0.0002 \
      --max-length 4096 \
      --lora-rank 8 \
      --lora-alpha 16 \
      --seed "$seed" \
      --compact-target \
      --use-job-instruction
  fi
  jq -e '
    .compact_target == true and
    .use_job_instruction == true and
    .truncated_prompts == 0 and
    .truncated_answers_with_eos == 0 and
    .seed == 42
  ' "$adapter/training_metrics.json" >/dev/null
}

infer_branch() {
  local dataset="$1" branch="$2"
  local base="$source_root/$dataset"
  local prepared="$graph_root/$dataset/jobs"
  local dataset_root="$run_root/$dataset"
  local adapter="$dataset_root/${branch}_adapter"
  local partial="$dataset_root/${branch}_validation_partial.jsonl"
  local complete="$dataset_root/${branch}_validation_complete.jsonl"
  local materialization="$dataset_root/${branch}_validation_materialization.json"
  current_dataset="$dataset"
  current_system="$branch"
  current_stage="validation_inference"
  write_status running
  if [[ ! -f "$materialization" ]] || \
     [[ "$(jq -r '.status // ""' "$materialization" 2>/dev/null || true)" != "complete" ]]; then
    bash scripts/run_qlora_inference_sharded.sh \
      --workers 2 \
      --model-path "$model" \
      --adapter "$adapter" \
      --jobs "$prepared/validation_${branch}_jobs.jsonl" \
      --output "$partial" \
      --log "$dataset_root/${branch}_validation.log" \
      --compact-target \
      --use-job-instruction \
      --resume \
      --max-input-tokens 4096 \
      --max-new-tokens 1024 \
      --max-seconds-per-job 90
    python3 scripts/materialize_missing_predictions.py \
      --jobs "$prepared/validation_${branch}_jobs.jsonl" \
      --predictions "$partial" \
      --output "$complete" \
      --summary "$materialization" \
      --reason "empty validation prediction materialized after terminal ${branch^^} inference failure"
  fi
  current_stage="source_expansion"
  write_status running
  python3 scripts/expand_compact_predictions.py \
    --jobs "$prepared/validation_${branch}_jobs.jsonl" \
    --predictions "$complete" \
    --output "$dataset_root/${branch}_validation_expanded.jsonl" \
    --errors "$dataset_root/${branch}_validation_expand_errors.jsonl"
}

evaluate_system() {
  local dataset="$1" system="$2" predictions="$3"
  local base="$source_root/$dataset"
  local jobs="$graph_root/$dataset/jobs/validation_hrge_jobs.jsonl"
  local metrics="$run_root/$dataset/metrics"
  current_dataset="$dataset"
  current_system="$system"
  current_stage="validation_evaluation"
  write_status running
  mkdir -p "$metrics"
  python3 scripts/evaluate_annotations.py \
    --gold "$base/validation_gold.jsonl" \
    --gold-index "$base/validation_index.jsonl" \
    --predictions "$predictions" \
    --jobs "$jobs" \
    --include-missing-as-empty \
    --output "$metrics/${system}_normalized_text.json"
  python3 scripts/evaluate_span_aware.py \
    --source-gold "$base/validation_gold.jsonl" \
    --source-gold-index "$base/validation_index.jsonl" \
    --gold "$base/validation_gold.jsonl" \
    --gold-index "$run_root/$dataset/validation_span_index.jsonl" \
    --predictions "$predictions" \
    --jobs "$jobs" \
    --output "$metrics/${system}_span.json"
  python3 scripts/evaluate_evidence_graph.py \
    --gold "$base/validation_gold.jsonl" \
    --gold-index "$base/validation_index.jsonl" \
    --predictions "$predictions" \
    --jobs "$jobs" \
    --ontology "$base/ontology.yaml" \
    --output "$metrics/${system}_evidence.json"
}

for dataset in conll04 scierc ade; do
  current_dataset="$dataset"
  dataset_root="$run_root/$dataset"
  base="$source_root/$dataset"
  prepared="$graph_root/$dataset/jobs"
  mkdir -p "$dataset_root/metrics" "$dataset_root/comparisons"

  tmp_index="$dataset_root/.validation_span_index.jsonl.tmp"
  jq -c '. + {parent_job_id: .job_id, split: "validation"}' \
    "$base/validation_index.jsonl" > "$tmp_index"
  mv "$tmp_index" "$dataset_root/validation_span_index.jsonl"

  train_branch "$dataset" eae
  train_branch "$dataset" hrge
  infer_branch "$dataset" eae
  infer_branch "$dataset" hrge

  current_system="evge"
  current_stage="relation_verification"
  write_status running
  python3 scripts/verify_relations.py \
    --annotations "$dataset_root/hrge_validation_expanded.jsonl" \
    --ontology "$base/ontology.yaml" \
    --output "$dataset_root/evge_validation.jsonl" \
    --audit "$dataset_root/evge_validation_audit.jsonl"

  current_system="cfe"
  current_stage="entity_gate"
  write_status running
  python3 scripts/fuse_kg_v1_v2_predictions.py \
    --v1 "$dataset_root/eae_validation_expanded.jsonl" \
    --v2 "$dataset_root/hrge_validation_expanded.jsonl" \
    --verified "$dataset_root/evge_validation.jsonl" \
    --jobs "$prepared/validation_hrge_jobs.jsonl" \
    --relation-mode raw \
    --output "$dataset_root/cfe_validation.jsonl" \
    --audit "$dataset_root/cfe_validation_audit.jsonl"

  current_system="pge"
  current_stage="entity_gate_and_relation_verification"
  write_status running
  python3 scripts/fuse_kg_v1_v2_predictions.py \
    --v1 "$dataset_root/eae_validation_expanded.jsonl" \
    --v2 "$dataset_root/hrge_validation_expanded.jsonl" \
    --verified "$dataset_root/evge_validation.jsonl" \
    --jobs "$prepared/validation_hrge_jobs.jsonl" \
    --relation-mode verified \
    --output "$dataset_root/pge_validation.jsonl" \
    --audit "$dataset_root/pge_validation_audit.jsonl"

  evaluate_system "$dataset" soe \
    "outputs/public_full_stage1/validation_analysis/completed_predictions/${dataset}_baseline_expanded.jsonl"
  evaluate_system "$dataset" eae "$dataset_root/eae_validation_expanded.jsonl"
  evaluate_system "$dataset" hrge "$dataset_root/hrge_validation_expanded.jsonl"
  evaluate_system "$dataset" evge "$dataset_root/evge_validation.jsonl"
  evaluate_system "$dataset" cfe "$dataset_root/cfe_validation.jsonl"
  evaluate_system "$dataset" pge "$dataset_root/pge_validation.jsonl"

  current_system="paired_statistics"
  current_stage="paired_bootstrap"
  write_status running
  python3 scripts/bootstrap_span_compare.py \
    --left "$dataset_root/metrics/eae_span.json" \
    --right "$dataset_root/metrics/hrge_span.json" \
    --iterations 20000 --seed 20260830 \
    --output "$dataset_root/comparisons/eae_vs_hrge.json"
  python3 scripts/bootstrap_span_compare.py \
    --left "$dataset_root/metrics/hrge_span.json" \
    --right "$dataset_root/metrics/cfe_span.json" \
    --iterations 20000 --seed 20260830 \
    --output "$dataset_root/comparisons/hrge_vs_cfe.json"
  python3 scripts/bootstrap_span_compare.py \
    --left "$dataset_root/metrics/hrge_span.json" \
    --right "$dataset_root/metrics/pge_span.json" \
    --iterations 20000 --seed 20260830 \
    --output "$dataset_root/comparisons/hrge_vs_pge.json"
done

current_dataset=""
current_system=""
current_stage="manifest"
write_status running
python3 - "$run_root" "$config" "$seed" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

run_root = Path(sys.argv[1])
config = Path(sys.argv[2])
implementation_files = [
    Path("scripts/run_public_pge_validation.sh"),
    Path("scripts/train_qlora.py"),
    Path("scripts/run_qlora_inference.py"),
    Path("scripts/run_qlora_inference_sharded.sh"),
    Path("scripts/materialize_missing_predictions.py"),
    Path("scripts/expand_compact_predictions.py"),
    Path("scripts/verify_relations.py"),
    Path("scripts/fuse_kg_v1_v2_predictions.py"),
    Path("scripts/evaluate_annotations.py"),
    Path("scripts/evaluate_span_aware.py"),
    Path("scripts/evaluate_evidence_graph.py"),
    Path("scripts/bootstrap_span_compare.py"),
]
files = [config, *implementation_files]
soe_lineage = {}
generator_coverage = {}
soe_completion_manifest = Path(
    "outputs/public_full_stage1/validation_analysis/missing_predictions.json"
)
files.append(soe_completion_manifest)
soe_completion = json.loads(soe_completion_manifest.read_text(encoding="utf-8"))
for dataset in ("conll04", "scierc", "ade"):
    files.append(Path("data/processed/public_benchmarks_hrge_v1") / dataset / "preparation_manifest.json")
    dataset_root = run_root / dataset
    generator_coverage[dataset] = {}
    files.extend(sorted((dataset_root / "metrics").glob("*.json")))
    files.extend(sorted((dataset_root / "comparisons").glob("*.json")))
    for branch in ("eae", "hrge"):
        materialization = dataset_root / f"{branch}_validation_materialization.json"
        files.append(dataset_root / f"{branch}_adapter" / "training_metrics.json")
        files.append(dataset_root / f"{branch}_validation_complete.jsonl")
        files.append(dataset_root / f"{branch}_validation_expanded.jsonl")
        files.append(materialization)
        coverage = json.loads(materialization.read_text(encoding="utf-8"))
        if coverage.get("status") != "complete" or coverage.get("gold_read") is not False:
            raise ValueError(f"invalid {dataset}/{branch} materialization summary")
        generator_coverage[dataset][branch] = {
            key: coverage[key]
            for key in (
                "jobs",
                "successful_prediction_rows",
                "failures_materialized_as_empty",
            )
        }
    for system in ("evge", "cfe", "pge"):
        files.append(dataset_root / f"{system}_validation.jsonl")
        files.append(dataset_root / f"{system}_validation_audit.jsonl")
    soe_predictions = (
        Path("outputs/public_full_stage1/validation_analysis/completed_predictions")
        / f"{dataset}_baseline_expanded.jsonl"
    )
    soe_training = Path("outputs/public_full_stage1") / f"{dataset}_baseline" / "training_metrics.json"
    files.extend((soe_predictions, soe_training))
    soe_lineage[dataset] = {
        "predictions": str(soe_predictions),
        "training_metrics": str(soe_training),
        "completion_manifest": str(soe_completion_manifest),
        "terminal_failures_materialized_as_empty": sum(
            row.get("dataset") == dataset and row.get("method") == "baseline"
            for row in soe_completion.get("records", [])
        ),
    }
missing = [str(path) for path in files if not path.is_file()]
if missing:
    raise FileNotFoundError(f"PGE run manifest inputs are missing: {missing}")
manifest = {
    "status": "complete",
    "selection_split": "validation",
    "formal_test_read": False,
    "seed": int(sys.argv[3]),
    "finished_at": datetime.now(timezone.utc).isoformat(),
    "implementation_files": [str(path) for path in implementation_files],
    "soe_lineage": soe_lineage,
    "generator_coverage": generator_coverage,
    "files": {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files
    },
}
target = run_root / "run_manifest.json"
temporary = target.with_name(f".{target.name}.tmp")
temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
temporary.replace(target)
PY

current_stage="complete"
write_status complete
trap - EXIT
