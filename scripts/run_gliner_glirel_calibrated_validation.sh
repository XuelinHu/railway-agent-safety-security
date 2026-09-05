#!/usr/bin/env bash
# Execute the train-calibrated GLiNER + GLiREL validation protocol.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

export PATH="/home/xuelin/miniconda3/bin:$PATH"
export HF_HOME="${HF_HOME:-/ds2/xuelin/cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

run_root="outputs/public_horizontal_validation/gliner_glirel_calibrated"
t0_root="outputs/public_horizontal_validation/gliner_glirel_t0"
protocol="$run_root/protocol.json"
status_file="$run_root/status.json"
requested_workers="${GLINER_GLIREL_CALIBRATED_WORKERS:-2}"
parallel_min_free_mib="${GLINER_GLIREL_PARALLEL_MIN_FREE_MIB:-12288}"
workers="$requested_workers"
parallel_decision="pending_gpu_check"
current_dataset=""
mkdir -p "$run_root"

if [[ ! -f "$protocol" ]]; then
  echo "calibrated validation protocol is missing: $protocol" >&2
  exit 2
fi
if [[ ! "$requested_workers" =~ ^[12]$ ]]; then
  echo "GLINER_GLIREL_CALIBRATED_WORKERS must be 1 or 2" >&2
  exit 2
fi

write_status() {
  local state="$1" active_dataset="${2:-}"
  python3 - "$status_file" "$state" "$active_dataset" "$protocol" "$run_root" \
    "$workers" "$requested_workers" "$parallel_decision" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

target = Path(sys.argv[1])
state = sys.argv[2]
active_dataset = sys.argv[3] or None
protocol_path = Path(sys.argv[4])
run_root = Path(sys.argv[5])
workers = int(sys.argv[6])
requested_workers = int(sys.argv[7])
parallel_decision = sys.argv[8]
protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
datasets = {}
for name, configuration in protocol["datasets"].items():
    jobs = Path("data/processed/public_benchmarks_full") / name / "validation_baseline_jobs.jsonl"
    predictions = run_root / f"{name}_validation.jsonl"
    expected = sum(bool(line.strip()) for line in jobs.open(encoding="utf-8"))
    completed = sum(bool(line.strip()) for line in predictions.open(encoding="utf-8")) if predictions.is_file() else 0
    datasets[name] = {
        "configuration": configuration,
        "expected_jobs": expected,
        "prediction_rows": completed,
        "predictions_complete": completed == expected,
        "character_span_metrics_ready": (run_root / f"{name}_validation.character_span_metrics.json").is_file(),
        "lineage_ready": (run_root / f"{name}_lineage.json").is_file(),
    }
payload = {
    "status": state,
    "active_dataset": active_dataset,
    "selection_split": "train_inner_calibration",
    "validation_gold_used_for_selection": False,
    "test_gold_used_for_selection": False,
    "parallel_workers": workers,
    "requested_parallel_workers": requested_workers,
    "parallel_decision": parallel_decision,
    "protocol": str(protocol_path),
    "datasets": datasets,
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
temporary = target.with_name(f".{target.name}.tmp")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
temporary.replace(target)
PY
}

on_exit() {
  local code="$?"
  if (( code != 0 )); then
    write_status failed "$current_dataset" || true
  fi
}
trap on_exit EXIT

for dataset in conll04 scierc ade; do
  label_mode="$(jq -r ".datasets.${dataset}.label_mode" "$protocol")"
  python3 scripts/run_gliner_glirel_validation.py \
    --dataset "$dataset" --relation-label-mode "$label_mode" --preflight-only
done

gpu_lock="${PUBLIC_GPU_LOCK_FILE:-outputs/.locks/public-validation-gpu.lock}"
mkdir -p "$(dirname "$gpu_lock")"
exec 8>"$gpu_lock"
write_status waiting_for_gpu
while ! flock -n 8; do
  echo "$(date --iso-8601=seconds) waiting_for_public_validation_gpu_lock"
  sleep 30
done

free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1)"
if (( requested_workers == 2 )) && [[ "$free_mib" =~ ^[0-9]+$ ]] && \
   (( free_mib >= parallel_min_free_mib )); then
  workers=2
  parallel_decision="enabled_free_mib_${free_mib}"
else
  workers=1
  parallel_decision="single_worker_free_mib_${free_mib:-unknown}"
fi
write_status ready_for_inference

for dataset in conll04 scierc ade; do
  current_dataset="$dataset"
  write_status running "$dataset"
  base="data/processed/public_benchmarks_full/$dataset"
  predictions="$run_root/${dataset}_validation.jsonl"
  label_mode="$(jq -r ".datasets.${dataset}.label_mode" "$protocol")"
  relation_threshold="$(jq -r ".datasets.${dataset}.relation_threshold" "$protocol")"
  execution="$(jq -r ".datasets.${dataset}.execution" "$protocol")"

  if [[ "$execution" == reuse_canonical_t0_predictions ]]; then
    source_predictions="$t0_root/${dataset}_validation.jsonl"
    [[ -f "$source_predictions" ]] || { echo "missing t0 predictions: $source_predictions" >&2; exit 1; }
    if [[ -f "$predictions" ]]; then
      [[ "$(sha256sum "$source_predictions" | cut -d' ' -f1)" == "$(sha256sum "$predictions" | cut -d' ' -f1)" ]] || {
        echo "existing reused predictions differ from t0 source: $predictions" >&2
        exit 1
      }
    else
      cp --reflink=auto "$source_predictions" "$predictions"
    fi
    python3 - "$run_root/${dataset}_lineage.json" "$source_predictions" "$predictions" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

target, source, destination = map(Path, sys.argv[1:])
def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()
payload = {
    "execution": "reuse_canonical_t0_predictions",
    "source": str(source),
    "destination": str(destination),
    "source_sha256": digest(source),
    "destination_sha256": digest(destination),
    "recorded_at": datetime.now(timezone.utc).isoformat(),
}
temporary = target.with_name(f".{target.name}.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
temporary.replace(target)
PY
  else
    total="$(awk 'NF {count++} END {print count + 0}' "$base/validation_baseline_jobs.jsonl")"
    chunk=$(( (total + workers - 1) / workers ))
    pids=()
    shards=()
    for (( worker=0; worker<workers; worker++ )); do
      offset=$(( worker * chunk ))
      (( offset >= total )) && break
      shard="$run_root/${dataset}_validation.part${worker}.jsonl"
      shards+=("$shard")
      python3 scripts/run_gliner_glirel_validation.py \
        --dataset "$dataset" \
        --jobs "$base/validation_baseline_jobs.jsonl" \
        --output "$shard" \
        --offset "$offset" --limit "$chunk" \
        --entity-threshold 0.5 \
        --relation-threshold "$relation_threshold" \
        --relation-label-mode "$label_mode" \
        --device cuda:0 --dtype float16 --resume \
        >"$run_root/${dataset}_validation.part${worker}.log" 2>&1 &
      pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do
      wait "$pid" || failed=1
    done
    if (( failed != 0 )); then
      if rg -qi 'CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED' \
        "$run_root/${dataset}_validation.part"*.log; then
        workers=1
        parallel_decision="fallback_to_single_after_cuda_oom_${dataset}"
        write_status running "$dataset"
        python3 scripts/run_gliner_glirel_validation.py \
          --dataset "$dataset" --jobs "$base/validation_baseline_jobs.jsonl" \
          --output "$predictions" --entity-threshold 0.5 \
          --relation-threshold "$relation_threshold" --relation-label-mode "$label_mode" \
          --device cuda:0 --dtype float16 --resume
      else
        exit 1
      fi
    else
      python3 scripts/merge_gliner_glirel_shards.py \
        --jobs "$base/validation_baseline_jobs.jsonl" \
        --shards "${shards[@]}" --output "$predictions"
    fi
    python3 - "$run_root/${dataset}_lineage.json" "$label_mode" "$relation_threshold" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
target = Path(sys.argv[1])
payload = {
    "execution": "fresh_inference",
    "label_mode": sys.argv[2],
    "relation_threshold": float(sys.argv[3]),
    "recorded_at": datetime.now(timezone.utc).isoformat(),
}
temporary = target.with_name(f".{target.name}.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
temporary.replace(target)
PY
  fi

  python3 scripts/evaluate_annotations.py \
    --gold "$base/validation_gold.jsonl" --gold-index "$base/validation_index.jsonl" \
    --predictions "$predictions" --jobs "$base/validation_baseline_jobs.jsonl" \
    --include-missing-as-empty \
    --output "$run_root/${dataset}_validation.normalized_text_metrics.json"
  python3 scripts/evaluate_public_validation_spans.py \
    --gold "$base/validation_gold.jsonl" --gold-index "$base/validation_index.jsonl" \
    --predictions "$predictions" --jobs "$base/validation_baseline_jobs.jsonl" \
    --output "$run_root/${dataset}_validation.character_span_metrics.json"
done

current_dataset=""
write_status complete
trap - EXIT
