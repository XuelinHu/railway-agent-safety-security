#!/usr/bin/env bash
# Reproducible fresh SpERT training/evaluation on public train/validation only.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${SPERT_PYTHON:-/home/xuelin/miniconda3/envs/rc-llm-comet/bin/python}"
HF_HUB_ROOT="${HF_HUB_ROOT:-/ds2/xuelin/cache/huggingface/hub}"
DATA_ROOT="${SPERT_DATA_ROOT:-$ROOT/data/processed/spert_fresh_train_v1}"
RUN_ROOT="${SPERT_RUN_ROOT:-$ROOT/outputs/public_horizontal_validation/spert_fresh}"
REPO="$ROOT/tools/external-baselines/spert"
SEED="${SPERT_SEED:-42}"
GPU_LOCK_FILE="${PUBLIC_GPU_LOCK_FILE:-$ROOT/outputs/.locks/public-validation-gpu.lock}"
GPU_LOCK_HELD=false

acquire_gpu_lock() {
  if [[ "$GPU_LOCK_HELD" == true ]]; then
    return 0
  fi
  mkdir -p "$(dirname "$GPU_LOCK_FILE")"
  exec 8>"$GPU_LOCK_FILE"
  while ! flock -n 8; do
    echo "$(date --iso-8601=seconds) waiting_for_public_validation_gpu_lock"
    sleep 30
  done
  GPU_LOCK_HELD=true
}

resolve_snapshot() {
  local repository="$1"
  local revision
  revision="$(<"$HF_HUB_ROOT/$repository/refs/main")"
  printf '%s/%s\n' "$HF_HUB_ROOT/$repository/snapshots" "$revision"
}

backbone_for() {
  case "$1" in
    conll04|ade) resolve_snapshot models--google-bert--bert-base-cased ;;
    scierc) resolve_snapshot models--allenai--scibert_scivocab_uncased ;;
    *) echo "unsupported dataset: $1" >&2; return 2 ;;
  esac
}

validate_dataset() {
  case "$1" in conll04|scierc|ade) ;; *) echo "dataset must be conll04, scierc, or ade" >&2; return 2 ;; esac
}

prepare() {
  "$PYTHON" scripts/prepare_spert_fresh_splits.py
}

preflight() {
  prepare
  CUDA_VISIBLE_DEVICES="" "$PYTHON" scripts/preflight_spert_fresh.py
}

train() {
  local dataset="$1" backbone label save_root log_root pointer checkpoint
  validate_dataset "$dataset"
  backbone="$(backbone_for "$dataset")"
  label="spert_${dataset}_seed${SEED}"
  save_root="$RUN_ROOT/$dataset/seed${SEED}/models"
  log_root="$RUN_ROOT/$dataset/seed${SEED}/train_logs"
  pointer="$RUN_ROOT/$dataset/seed${SEED}/final_model_path.txt"
  if [[ -s "$pointer" ]]; then
    checkpoint="$(<"$pointer")"
    if [[ -f "$checkpoint/pytorch_model.bin" ]]; then
      echo "$(date --iso-8601=seconds) reuse_spert_checkpoint dataset=$dataset checkpoint=$checkpoint"
      return 0
    fi
  fi
  acquire_gpu_lock
  mkdir -p "$save_root" "$log_root"
  "$PYTHON" scripts/run_spert_compat.py --repo "$REPO" -- train \
    --model_type spert --model_path "$backbone" --tokenizer_path "$backbone" \
    --train_path "$DATA_ROOT/$dataset/train.json" \
    --valid_path "$DATA_ROOT/$dataset/validation.json" \
    --types_path "$DATA_ROOT/$dataset/types.json" \
    --train_batch_size "${SPERT_TRAIN_BATCH_SIZE:-2}" --eval_batch_size 1 \
    --neg_entity_count 100 --neg_relation_count 100 --epochs "${SPERT_EPOCHS:-20}" \
    --lr 5e-5 --lr_warmup 0.1 --weight_decay 0.01 --max_grad_norm 1.0 \
    --rel_filter_threshold 0.4 --size_embedding 25 --prop_drop 0.1 \
    --max_span_size 10 --sampling_processes "${SPERT_SAMPLING_PROCESSES:-4}" \
    --max_pairs 1000 --seed "$SEED" --final_eval --store_predictions \
    --label "$label" --save_path "$save_root" --log_path "$log_root"
  checkpoint="$(find "$save_root/$label" -type d -name final_model -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
  [[ -n "$checkpoint" && -f "$checkpoint/pytorch_model.bin" ]] || {
    echo "fresh final checkpoint was not created" >&2
    return 1
  }
  printf '%s\n' "$checkpoint" > "$RUN_ROOT/$dataset/seed${SEED}/final_model_path.txt"
}

validation() {
  local dataset="$1" pointer checkpoint label log_root prediction source_root target_root
  validate_dataset "$dataset"
  pointer="$RUN_ROOT/$dataset/seed${SEED}/final_model_path.txt"
  [[ -s "$pointer" ]] || { echo "missing checkpoint pointer: $pointer" >&2; return 1; }
  checkpoint="$(<"$pointer")"
  [[ -f "$checkpoint/pytorch_model.bin" ]] || { echo "incomplete checkpoint: $checkpoint" >&2; return 1; }
  label="spert_${dataset}_seed${SEED}_validation"
  log_root="$RUN_ROOT/$dataset/seed${SEED}/validation_logs"
  target_root="$RUN_ROOT/$dataset/seed${SEED}"
  if [[ -f "$target_root/conversion_manifest.json" ]] && \
     [[ "$(jq -r '.status // ""' "$target_root/conversion_manifest.json" 2>/dev/null || true)" == "complete" ]] && \
     [[ -f "$target_root/validation_normalized_text_metrics.json" ]] && \
     [[ -f "$target_root/validation_character_span_metrics.json" ]]; then
    echo "$(date --iso-8601=seconds) reuse_spert_validation dataset=$dataset"
    return 0
  fi
  acquire_gpu_lock
  mkdir -p "$log_root"
  "$PYTHON" scripts/run_spert_compat.py --repo "$REPO" -- eval \
    --model_type spert --model_path "$checkpoint" --tokenizer_path "$checkpoint" \
    --dataset_path "$DATA_ROOT/$dataset/validation.json" \
    --types_path "$DATA_ROOT/$dataset/types.json" --eval_batch_size 1 \
    --rel_filter_threshold 0.4 --size_embedding 25 --prop_drop 0.1 \
    --max_span_size 10 --sampling_processes "${SPERT_SAMPLING_PROCESSES:-4}" \
    --max_pairs 1000 --seed "$SEED" --store_predictions \
    --label "$label" --log_path "$log_root"
  prediction="$(find "$log_root/$label" -type f -name 'predictions_test_epoch_0.json' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
  [[ -n "$prediction" && -s "$prediction" ]] || {
    echo "SpERT validation predictions were not created" >&2
    return 1
  }
  source_root="$ROOT/data/processed/public_benchmarks_full/$dataset"
  "$PYTHON" scripts/convert_spert_predictions.py \
    --dataset "$dataset" \
    --validation-data "$DATA_ROOT/$dataset/validation.json" \
    --predictions "$prediction" \
    --jobs "$source_root/validation_baseline_jobs.jsonl" \
    --output "$target_root/validation_predictions.jsonl" \
    --manifest "$target_root/conversion_manifest.json"
  "$PYTHON" scripts/evaluate_annotations.py \
    --gold "$source_root/validation_gold.jsonl" \
    --gold-index "$source_root/validation_index.jsonl" \
    --predictions "$target_root/validation_predictions.jsonl" \
    --jobs "$source_root/validation_baseline_jobs.jsonl" \
    --include-missing-as-empty \
    --output "$target_root/validation_normalized_text_metrics.json"
  "$PYTHON" scripts/evaluate_public_validation_spans.py \
    --gold "$source_root/validation_gold.jsonl" \
    --gold-index "$source_root/validation_index.jsonl" \
    --predictions "$target_root/validation_predictions.jsonl" \
    --jobs "$source_root/validation_baseline_jobs.jsonl" \
    --output "$target_root/validation_character_span_metrics.json"
}

all_validation() {
  local dataset status_file="$RUN_ROOT/status.json"
  preflight
  for dataset in conll04 scierc ade; do
    train "$dataset"
    validation "$dataset"
  done
  "$PYTHON" - "$RUN_ROOT" "$SEED" "$status_file" <<'PY'
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

root, seed, target = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
datasets = {}
for name in ("conll04", "scierc", "ade"):
    base = root / name / f"seed{seed}"
    conversion = json.loads((base / "conversion_manifest.json").read_text(encoding="utf-8"))
    datasets[name] = {
        "checkpoint": (base / "final_model_path.txt").read_text(encoding="utf-8").strip(),
        "prediction_rows": conversion["rows"],
        "normalized_text_metrics": str(base / "validation_normalized_text_metrics.json"),
        "character_span_metrics": str(base / "validation_character_span_metrics.json"),
    }
payload = {
    "status": "complete",
    "split": "validation",
    "seed": seed,
    "test_split_access": "forbidden-and-not-read",
    "finished_at": datetime.now(timezone.utc).isoformat(),
    "datasets": datasets,
}
target.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile(
    mode="w", encoding="utf-8", dir=target.parent, prefix=f".{target.name}.", delete=False
) as stream:
    temporary = Path(stream.name)
    json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
os.replace(temporary, target)
PY
}

status() {
  "$PYTHON" - "$DATA_ROOT" "$RUN_ROOT" <<'PY'
import json, sys
from pathlib import Path
data_root, run_root = map(Path, sys.argv[1:])
result = {"data": {}, "runs": {}}
for dataset in ("conll04", "scierc", "ade"):
    manifest = data_root / dataset / "manifest.json"
    result["data"][dataset] = json.loads(manifest.read_text())["rows"] if manifest.is_file() else "missing"
    pointers = sorted((run_root / dataset).glob("seed*/final_model_path.txt"))
    result["runs"][dataset] = [str(path) for path in pointers]
preflight = run_root / "preflight.json"
result["preflight"] = json.loads(preflight.read_text())["status"] if preflight.is_file() else "missing"
print(json.dumps(result, indent=2))
PY
}

command="${1:-}"
case "$command" in
  prepare) prepare ;;
  preflight) preflight ;;
  train) [[ $# -eq 2 ]] || { echo "usage: $0 train DATASET" >&2; exit 2; }; train "$2" ;;
  validation) [[ $# -eq 2 ]] || { echo "usage: $0 validation DATASET" >&2; exit 2; }; validation "$2" ;;
  all) all_validation ;;
  status) status ;;
  *) echo "usage: $0 prepare|preflight|train DATASET|validation DATASET|all|status" >&2; exit 2 ;;
esac
