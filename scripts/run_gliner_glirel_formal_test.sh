#!/usr/bin/env bash
# Run the train-calibrated GLiNER + GLiREL baseline on the promoted test split.
set -euo pipefail

export PATH="/home/xuelin/miniconda3/bin:$PATH"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

release_status="outputs/public_formal_matrix/release_status.json"
promotion="outputs/public_formal_matrix/promotion.json"
prepared_root="data/processed/public_benchmarks_hrge_test_v1"
protocol="outputs/public_horizontal_validation/gliner_glirel_calibrated/protocol.json"
data_root="data/processed/public_benchmarks_full"
run_root="outputs/public_formal_matrix/horizontal/gliner_glirel_calibrated"
status_file="$run_root/status.json"
python="${PUBLIC_FORMAL_PYTHON:-python3}"
release_contract="scripts/qwen_zeroshot_formal_contract.py"
formal_contract="scripts/gliner_glirel_formal_contract.py"
gpu_lock="${PUBLIC_GPU_LOCK_FILE:-outputs/.locks/public-validation-gpu.lock}"
poll_seconds="${PUBLIC_FORMAL_RELEASE_POLL_SECONDS:-30}"
requested_workers="${GLINER_GLIREL_FORMAL_WORKERS:-2}"
workers="$requested_workers"
current_dataset=""
current_stage="waiting_for_formal_release"
formal_test_read=false
gpu_lock_held=false
release_sha256=""
release_fingerprint=""
protocol_sha256=""
declare -A expected_jobs=([conll04]=288 [scierc]=551 [ade]=427)
mkdir -p "$run_root" "$(dirname "$gpu_lock")"

if [[ ! "$requested_workers" =~ ^[12]$ ]]; then
  echo "GLINER_GLIREL_FORMAL_WORKERS must be 1 or 2" >&2
  exit 2
fi

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
  local state="$1" error="${2:-}"
  local args=(
    status --output "$status_file" --state "$state" --stage "$current_stage"
    --formal-test-read "$formal_test_read" --workers "$workers"
    --requested-workers "$requested_workers" --data-root "$data_root"
    --run-root "$run_root" --protocol "$protocol"
    --release-status "$release_status" --promotion "$promotion"
    --prepared-root "$prepared_root" --gpu-lock "$gpu_lock"
  )
  [[ -n "$current_dataset" ]] && args+=(--active-dataset "$current_dataset")
  [[ -n "$release_sha256" ]] && args+=(--release-sha256 "$release_sha256")
  [[ -n "$release_fingerprint" ]] && args+=(--release-fingerprint "$release_fingerprint")
  [[ -n "$protocol_sha256" ]] && args+=(--protocol-sha256 "$protocol_sha256")
  [[ -n "$error" ]] && args+=(--error "$error")
  "$python" "$formal_contract" "${args[@]}" >/dev/null
}

on_exit() {
  local code="$?"
  if (( code != 0 )); then
    trap - EXIT
    release_gpu_lock || true
    write_status failed "stage=${current_stage}; exit_code=${code}" || true
  fi
}
trap on_exit EXIT

write_status waiting_for_release
while ! jq -e '.status == "complete"' "$release_status" >/dev/null 2>&1; do
  sleep "$poll_seconds"
  write_status waiting_for_release
done

current_stage="canonical_release_verification"
write_status running
release_verification="$("$python" "$formal_contract" verify-release \
  --release-status "$release_status" --promotion "$promotion" \
  --prepared-root "$prepared_root")"
release_sha256="$(jq -er '.release_status.sha256' <<<"$release_verification")"
release_fingerprint="$(jq -er '.canonical_fingerprint' <<<"$release_verification")"
protocol_verification="$("$python" "$formal_contract" validate-protocol \
  --protocol "$protocol")"
protocol_sha256="$(jq -er '.sha256' <<<"$protocol_verification")"

current_stage="cpu_preflight"
formal_test_read=true
write_status running
for dataset in conll04 scierc ade; do
  "$python" "$release_contract" preflight --data-root "$data_root" \
    --dataset "$dataset" --expected "${expected_jobs[$dataset]}" >/dev/null
  "$python" scripts/run_gliner_glirel_validation.py \
    --dataset "$dataset" --split test \
    --jobs "$data_root/$dataset/test_baseline_jobs.jsonl" \
    --relation-label-mode "$(jq -r --arg d "$dataset" '.datasets[$d].label_mode' "$protocol")" \
    --preflight-only >/dev/null
done

for dataset in conll04 scierc ade; do
  current_dataset="$dataset"
  current_stage="dataset_resume_validation"
  workers="$requested_workers"
  write_status running
  base="$data_root/$dataset"
  jobs="$base/test_baseline_jobs.jsonl"
  predictions="$run_root/${dataset}_test.jsonl"
  span_metrics="$run_root/${dataset}_test.character_span_metrics.json"
  normalized_metrics="$run_root/${dataset}_test.normalized_text_metrics.json"
  label_mode="$(jq -r --arg d "$dataset" '.datasets[$d].label_mode' "$protocol")"
  relation_threshold="$(jq -r --arg d "$dataset" '.datasets[$d].relation_threshold' "$protocol")"
  entity_threshold="$(jq -r --arg d "$dataset" '.datasets[$d].entity_threshold' "$protocol")"

  if "$python" "$formal_contract" validate --data-root "$data_root" \
    --run-root "$run_root" --dataset "$dataset" \
    --expected "${expected_jobs[$dataset]}" --quiet >/dev/null 2>&1; then
    current_stage="reused_verified_dataset"
    write_status running
    echo "reuse complete GLiNER/GLiREL formal test dataset=$dataset"
    continue
  fi

  acquire_gpu_lock
  current_stage="test_inference"
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1)"
  if [[ "$workers" == 2 && ( ! "$free_mib" =~ ^[0-9]+$ || "$free_mib" -lt 12288 ) ]]; then
    workers=1
  fi
  inference_exit=0
  merge_required=false
  if (( workers == 1 )); then
      "$python" scripts/run_gliner_glirel_validation.py \
        --dataset "$dataset" --split test --jobs "$jobs" --output "$predictions" \
        --entity-threshold "$entity_threshold" \
        --relation-threshold "$relation_threshold" \
        --relation-label-mode "$label_mode" --device cuda:0 --dtype float16 --resume \
        || inference_exit=$?
  else
    total="$(awk 'NF {count++} END {print count + 0}' "$jobs")"
    chunk=$(( (total + workers - 1) / workers ))
    pids=()
    shards=()
    for (( worker=0; worker<workers; worker++ )); do
      offset=$(( worker * chunk ))
      (( offset >= total )) && break
      shard="$run_root/${dataset}_test.part${worker}.jsonl"
      shards+=("$shard")
      "$python" scripts/run_gliner_glirel_validation.py \
        --dataset "$dataset" --split test --jobs "$jobs" --output "$shard" \
        --offset "$offset" --limit "$chunk" \
        --entity-threshold "$entity_threshold" \
        --relation-threshold "$relation_threshold" \
        --relation-label-mode "$label_mode" --device cuda:0 --dtype float16 --resume \
        >"$run_root/${dataset}_test.part${worker}.log" 2>&1 &
      pids+=("$!")
    done
    worker_failed=0
    for pid in "${pids[@]}"; do
      wait "$pid" || worker_failed=1
    done
    if (( worker_failed != 0 )); then
      if rg -qi 'CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED' \
        "$run_root/${dataset}_test.part"*.log; then
        workers=1
        "$python" scripts/run_gliner_glirel_validation.py \
          --dataset "$dataset" --split test --jobs "$jobs" --output "$predictions" \
          --entity-threshold "$entity_threshold" \
          --relation-threshold "$relation_threshold" \
          --relation-label-mode "$label_mode" --device cuda:0 --dtype float16 --resume \
          || inference_exit=$?
      else
        inference_exit=1
      fi
    else
      merge_required=true
    fi
  fi

  release_gpu_lock
  if (( inference_exit != 0 )); then
    exit "$inference_exit"
  fi
  if [[ "$merge_required" == true ]]; then
    "$python" scripts/merge_gliner_glirel_shards.py \
      --jobs "$jobs" --shards "${shards[@]}" --output "$predictions"
  fi

  current_stage="test_evaluation"
  write_status running
  "$python" scripts/evaluate_annotations.py \
    --gold "$base/test_gold.jsonl" --gold-index "$base/test_index.jsonl" \
    --predictions "$predictions" --jobs "$jobs" --include-missing-as-empty \
    --output "$normalized_metrics"
  "$python" scripts/evaluate_public_validation_spans.py \
    --gold "$base/test_gold.jsonl" --gold-index "$base/test_index.jsonl" \
    --predictions "$predictions" --jobs "$jobs" --output "$span_metrics" \
    --allow-non-validation

  current_stage="dataset_contract_validation"
  "$python" "$formal_contract" validate --data-root "$data_root" \
    --run-root "$run_root" --dataset "$dataset" \
    --expected "${expected_jobs[$dataset]}" >/dev/null
  write_status running
done

current_dataset=""
current_stage="final_release_identity_verification"
current_release_verification="$("$python" "$formal_contract" verify-release \
  --release-status "$release_status" --promotion "$promotion" \
  --prepared-root "$prepared_root")"
current_release_sha256="$(jq -er '.release_status.sha256' <<<"$current_release_verification")"
current_release_fingerprint="$(jq -er '.canonical_fingerprint' <<<"$current_release_verification")"
if [[ "$current_release_sha256" != "$release_sha256" || \
      "$current_release_fingerprint" != "$release_fingerprint" ]]; then
  echo "formal release/promotion/preparation identity changed during GLiNER/GLiREL run" >&2
  exit 1
fi
current_protocol_verification="$("$python" "$formal_contract" validate-protocol \
  --protocol "$protocol")"
current_protocol_sha256="$(jq -er '.sha256' <<<"$current_protocol_verification")"
if [[ "$current_protocol_sha256" != "$protocol_sha256" ]]; then
  echo "calibrated GLiNER/GLiREL protocol changed during formal test" >&2
  exit 1
fi
current_stage="complete"
write_status complete
trap - EXIT
