#!/usr/bin/env bash
# Run the promoted SOE/PGE public test and the two additional training seeds.
set -euo pipefail

export PATH="/home/xuelin/miniconda3/bin:$PATH"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

model="/ds2/xuelin/cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
source_root="data/processed/public_benchmarks_full"
validation_jobs_root="data/processed/public_benchmarks_hrge_v1"
test_jobs_root="data/processed/public_benchmarks_hrge_test_v1"
run_root="${PUBLIC_FORMAL_INTERNAL_ROOT:-outputs/public_formal_matrix/internal}"
promotion="outputs/public_formal_matrix/promotion.json"
status_file="$run_root/status.json"
gpu_lock="${PUBLIC_GPU_LOCK_FILE:-outputs/.locks/public-validation-gpu.lock}"
gpu_lock_held=false
current_seed=""
current_dataset=""
current_split=""
current_system=""
current_stage="initializing"
formal_test_read=false
mkdir -p "$run_root" "$(dirname "$gpu_lock")"

acquire_gpu_lock() {
  if [[ "$gpu_lock_held" == true ]]; then
    return 0
  fi
  current_stage="waiting_for_exclusive_gpu_lock"
  write_status waiting_for_gpu
  exec 8>"$gpu_lock"
  while ! flock -n 8; do
    sleep 30
    write_status waiting_for_gpu
  done
  gpu_lock_held=true
}

release_gpu_lock() {
  if [[ "$gpu_lock_held" == true ]]; then
    flock -u 8
    exec 8>&-
    gpu_lock_held=false
  fi
}

write_status() {
  local state="$1"
  python3 - "$status_file" "$state" "$current_seed" "$current_dataset" \
    "$current_split" "$current_system" "$current_stage" "$formal_test_read" \
    "$run_root" <<'PY'
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

target = Path(sys.argv[1])
run_root = Path(sys.argv[9])
seeds = {}
for seed in (42, 2026, 3407):
    datasets = {}
    for dataset in ("conll04", "scierc", "ade"):
        base = run_root / f"seed{seed}" / dataset
        datasets[dataset] = {
            split: {
                "complete": (base / split / "complete.json").is_file(),
                "soe_metrics": (base / split / "metrics" / "soe_span.json").is_file(),
                "pge_metrics": (base / split / "metrics" / "pge_span.json").is_file(),
            }
            for split in ("validation", "test")
        }
    seeds[str(seed)] = datasets
payload = {
    "status": sys.argv[2],
    "seed": int(sys.argv[3]) if sys.argv[3] else None,
    "active_dataset": sys.argv[4] or None,
    "active_split": sys.argv[5] or None,
    "active_system": sys.argv[6] or None,
    "stage": sys.argv[7],
    "formal_test_read": sys.argv[8].lower() == "true",
    "promoted_systems": ["soe", "pge"],
    "seeds": seeds,
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
target.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile(
    mode="w", encoding="utf-8", dir=target.parent,
    prefix=f".{target.name}.", delete=False,
) as stream:
    temporary = Path(stream.name)
    json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
os.replace(temporary, target)
PY
}

on_exit() {
  local code="$?"
  if (( code != 0 )); then
    current_stage="failed:${current_stage}"
    write_status failed || true
  fi
}
trap on_exit EXIT

require_release() {
  [[ -f "$promotion" ]] || { echo "missing formal promotion marker: $promotion" >&2; return 1; }
  jq -e '.status == "promoted" and .promoted_systems == ["soe", "pge"]' \
    "$promotion" >/dev/null
  for dataset in conll04 scierc ade; do
    jq -e '.status == "prepared_test" and .test_gold_read == false' \
      "$test_jobs_root/$dataset/preparation_manifest.json" >/dev/null
    [[ -s "$test_jobs_root/$dataset/jobs/test_eae_jobs.jsonl" ]]
    [[ -s "$test_jobs_root/$dataset/jobs/test_hrge_jobs.jsonl" ]]
  done
}

adapter_for() {
  local seed="$1" dataset="$2" branch="$3"
  if [[ "$seed" == 42 ]]; then
    case "$branch" in
      soe) printf 'outputs/public_full_stage1/%s_baseline\n' "$dataset" ;;
      eae|hrge) printf 'outputs/public_pge_validation_seed42/%s/%s_adapter\n' "$dataset" "$branch" ;;
      *) return 2 ;;
    esac
  else
    printf '%s/seed%s/%s/%s_adapter\n' "$run_root" "$seed" "$dataset" "$branch"
  fi
}

train_adapter() {
  local seed="$1" dataset="$2" branch="$3" adapter jobs instruction=()
  adapter="$(adapter_for "$seed" "$dataset" "$branch")"
  if [[ "$branch" == soe ]]; then
    jobs="$source_root/$dataset/train_baseline_jobs.jsonl"
  else
    jobs="$validation_jobs_root/$dataset/jobs/train_${branch}_jobs.jsonl"
    instruction=(--use-job-instruction)
  fi
  current_seed="$seed"
  current_dataset="$dataset"
  current_split="train"
  current_system="$branch"
  current_stage="training"
  write_status running
  if [[ ! -f "$adapter/training_metrics.json" ]]; then
    acquire_gpu_lock
    current_stage="training"
    write_status running
    python3 scripts/train_qlora.py \
      --model-path "$model" \
      --gold "$source_root/$dataset/train_gold.jsonl" \
      --index "$source_root/$dataset/train_index.jsonl" \
      --jobs "$jobs" \
      --output "$adapter" \
      --epochs 1 --batch-size 1 --gradient-accumulation 4 \
      --learning-rate 0.0002 --max-length 4096 \
      --lora-rank 8 --lora-alpha 16 --seed "$seed" \
      --compact-target "${instruction[@]}"
    release_gpu_lock
  fi
  jq -e --argjson seed "$seed" --arg model "$model" --argjson uses_instruction \
    "$([[ "$branch" == soe ]] && printf false || printf true)" '
      .seed == $seed and .base_model == $model and .epochs == 1 and
      .max_length == 4096 and .lora_rank == 8 and
      .compact_target == true and .use_job_instruction == $uses_instruction and
      .truncated_prompts == 0 and .truncated_answers_with_eos == 0
    ' "$adapter/training_metrics.json" >/dev/null
}

jobs_for() {
  local dataset="$1" split="$2" branch="$3"
  if [[ "$branch" == soe ]]; then
    printf '%s/%s/%s_baseline_jobs.jsonl\n' "$source_root" "$dataset" "$split"
  elif [[ "$split" == validation ]]; then
    printf '%s/%s/jobs/validation_%s_jobs.jsonl\n' "$validation_jobs_root" "$dataset" "$branch"
  else
    printf '%s/%s/jobs/test_%s_jobs.jsonl\n' "$test_jobs_root" "$dataset" "$branch"
  fi
}

infer_branch() {
  local seed="$1" dataset="$2" split="$3" branch="$4"
  local adapter jobs target raw complete expanded materialization instruction=()
  adapter="$(adapter_for "$seed" "$dataset" "$branch")"
  jobs="$(jobs_for "$dataset" "$split" "$branch")"
  target="$run_root/seed${seed}/${dataset}/${split}"
  raw="$target/${branch}_partial.jsonl"
  complete="$target/${branch}_complete.jsonl"
  expanded="$target/${branch}_expanded.jsonl"
  materialization="$target/${branch}_materialization.json"
  [[ "$branch" != soe ]] && instruction=(--use-job-instruction)
  mkdir -p "$target"
  current_seed="$seed"
  current_dataset="$dataset"
  current_split="$split"
  current_system="$branch"
  current_stage="${split}_inference"
  [[ "$split" == test ]] && formal_test_read=true
  write_status running
  if [[ ! -f "$materialization" ]] || \
     [[ "$(jq -r '.status // ""' "$materialization" 2>/dev/null || true)" != complete ]]; then
    acquire_gpu_lock
    current_stage="${split}_inference"
    write_status running
    bash scripts/run_qlora_inference_sharded.sh \
      --workers 2 --model-path "$model" --adapter "$adapter" \
      --jobs "$jobs" --output "$raw" --log "$target/${branch}.log" \
      --compact-target --resume --max-input-tokens 4096 \
      --max-new-tokens 1024 --max-seconds-per-job 90 "${instruction[@]}"
    release_gpu_lock
    python3 scripts/materialize_missing_predictions.py \
      --jobs "$jobs" --predictions "$raw" --output "$complete" \
      --summary "$materialization" \
      --reason "empty ${split} prediction materialized after terminal ${branch} inference failure"
  fi
  current_stage="${split}_source_expansion"
  write_status running
  if [[ ! -f "$expanded" || "$complete" -nt "$expanded" ]]; then
    python3 scripts/expand_compact_predictions.py \
      --jobs "$jobs" --predictions "$complete" --output "$expanded" \
      --errors "$target/${branch}_expand_errors.jsonl"
  fi
}

build_derived_systems() {
  local seed="$1" dataset="$2" split="$3" target hrge_jobs
  target="$run_root/seed${seed}/${dataset}/${split}"
  hrge_jobs="$(jobs_for "$dataset" "$split" hrge)"
  current_seed="$seed"
  current_dataset="$dataset"
  current_split="$split"
  current_system="pge"
  current_stage="${split}_deterministic_gates"
  write_status running
  python3 scripts/verify_relations.py \
    --annotations "$target/hrge_expanded.jsonl" \
    --ontology "$source_root/$dataset/ontology.yaml" \
    --output "$target/evge.jsonl" --audit "$target/evge_audit.jsonl"
  python3 scripts/fuse_kg_v1_v2_predictions.py \
    --v1 "$target/eae_expanded.jsonl" --v2 "$target/hrge_expanded.jsonl" \
    --verified "$target/evge.jsonl" --jobs "$hrge_jobs" --relation-mode raw \
    --output "$target/cfe.jsonl" --audit "$target/cfe_audit.jsonl"
  python3 scripts/fuse_kg_v1_v2_predictions.py \
    --v1 "$target/eae_expanded.jsonl" --v2 "$target/hrge_expanded.jsonl" \
    --verified "$target/evge.jsonl" --jobs "$hrge_jobs" --relation-mode verified \
    --output "$target/pge.jsonl" --audit "$target/pge_audit.jsonl"
}

evaluate_system() {
  local seed="$1" dataset="$2" split="$3" system="$4" predictions="$5"
  local target jobs allow=()
  target="$run_root/seed${seed}/${dataset}/${split}"
  jobs="$(jobs_for "$dataset" "$split" hrge)"
  [[ "$system" == soe ]] && jobs="$(jobs_for "$dataset" "$split" soe)"
  [[ "$split" == test ]] && allow=(--allow-non-validation)
  mkdir -p "$target/metrics"
  current_seed="$seed"
  current_dataset="$dataset"
  current_split="$split"
  current_system="$system"
  current_stage="${split}_evaluation"
  [[ "$split" == test ]] && formal_test_read=true
  write_status running
  python3 scripts/evaluate_annotations.py \
    --gold "$source_root/$dataset/${split}_gold.jsonl" \
    --gold-index "$source_root/$dataset/${split}_index.jsonl" \
    --predictions "$predictions" --jobs "$jobs" --include-missing-as-empty \
    --output "$target/metrics/${system}_normalized_text.json"
  python3 scripts/evaluate_span_aware.py \
    --source-gold "$source_root/$dataset/${split}_gold.jsonl" \
    --source-gold-index "$source_root/$dataset/${split}_index.jsonl" \
    --gold "$source_root/$dataset/${split}_gold.jsonl" \
    --gold-index "$target/${split}_span_index.jsonl" \
    --predictions "$predictions" --jobs "$jobs" \
    --output "$target/metrics/${system}_span.json" "${allow[@]}"
  python3 scripts/evaluate_evidence_graph.py \
    --gold "$source_root/$dataset/${split}_gold.jsonl" \
    --gold-index "$source_root/$dataset/${split}_index.jsonl" \
    --predictions "$predictions" --jobs "$jobs" \
    --ontology "$source_root/$dataset/ontology.yaml" \
    --output "$target/metrics/${system}_evidence.json"
}

evaluate_split() {
  local seed="$1" dataset="$2" split="$3" target tmp_index bootstrap_script
  target="$run_root/seed${seed}/${dataset}/${split}"
  bootstrap_script="scripts/bootstrap_span_compare.py"
  [[ "$split" == test ]] && bootstrap_script="scripts/bootstrap_span_compare_formal_test.py"
  tmp_index="$target/.${split}_span_index.jsonl.tmp"
  jq -c --arg split "$split" '. + {parent_job_id: .job_id, split: $split}' \
    "$source_root/$dataset/${split}_index.jsonl" > "$tmp_index"
  mv "$tmp_index" "$target/${split}_span_index.jsonl"
  evaluate_system "$seed" "$dataset" "$split" soe "$target/soe_expanded.jsonl"
  evaluate_system "$seed" "$dataset" "$split" eae "$target/eae_expanded.jsonl"
  evaluate_system "$seed" "$dataset" "$split" hrge "$target/hrge_expanded.jsonl"
  evaluate_system "$seed" "$dataset" "$split" evge "$target/evge.jsonl"
  evaluate_system "$seed" "$dataset" "$split" cfe "$target/cfe.jsonl"
  evaluate_system "$seed" "$dataset" "$split" pge "$target/pge.jsonl"
  current_system="paired_statistics"
  current_stage="${split}_paired_bootstrap"
  write_status running
  python3 "$bootstrap_script" \
    --left "$target/metrics/soe_span.json" --right "$target/metrics/pge_span.json" \
    --iterations 20000 --seed 20260830 --output "$target/soe_vs_pge.json"
  python3 - "$target/complete.json" "$seed" "$dataset" "$split" <<'PY'
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

target = Path(sys.argv[1])
payload = {
    "status": "complete",
    "seed": int(sys.argv[2]),
    "dataset": sys.argv[3],
    "split": sys.argv[4],
    "formal_test_read": sys.argv[4] == "test",
    "systems": ["soe", "eae", "hrge", "evge", "cfe", "pge"],
    "finished_at": datetime.now(timezone.utc).isoformat(),
}
with tempfile.NamedTemporaryFile(
    mode="w", encoding="utf-8", dir=target.parent,
    prefix=f".{target.name}.", delete=False,
) as stream:
    temporary = Path(stream.name)
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
os.replace(temporary, target)
PY
}

run_split() {
  local seed="$1" dataset="$2" split="$3" target
  target="$run_root/seed${seed}/${dataset}/${split}"
  if [[ -f "$target/complete.json" ]] && \
     [[ "$(jq -r '.status // ""' "$target/complete.json" 2>/dev/null || true)" == complete ]]; then
    echo "reuse complete formal split seed=$seed dataset=$dataset split=$split"
    return 0
  fi
  infer_branch "$seed" "$dataset" "$split" soe
  infer_branch "$seed" "$dataset" "$split" eae
  infer_branch "$seed" "$dataset" "$split" hrge
  build_derived_systems "$seed" "$dataset" "$split"
  evaluate_split "$seed" "$dataset" "$split"
}

current_stage="release_preflight"
write_status running
require_release

# Run the already-trained seed first so the primary test table is available
# before the additional-seed matrix finishes.
for dataset in conll04 scierc ade; do
  train_adapter 42 "$dataset" soe
  train_adapter 42 "$dataset" eae
  train_adapter 42 "$dataset" hrge
  run_split 42 "$dataset" test
done

for seed in 2026 3407; do
  for dataset in conll04 scierc ade; do
    train_adapter "$seed" "$dataset" soe
    train_adapter "$seed" "$dataset" eae
    train_adapter "$seed" "$dataset" hrge
    run_split "$seed" "$dataset" validation
    run_split "$seed" "$dataset" test
  done
done

current_seed=""
current_dataset=""
current_split=""
current_system=""
current_stage="complete"
formal_test_read=true
write_status complete
trap - EXIT
