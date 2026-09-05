#!/usr/bin/env bash
# Run the independent Qwen3-4B zero-shot control on the released public test split.
set -euo pipefail

export PATH="/home/xuelin/miniconda3/bin:$PATH"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

python="${PUBLIC_FORMAL_PYTHON:-python3}"
contract="scripts/qwen_zeroshot_formal_contract.py"
model="/ds2/xuelin/cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
data_root="data/processed/public_benchmarks_full"
prepared_root="data/processed/public_benchmarks_hrge_test_v1"
run_root="${QWEN_ZERO_SHOT_FORMAL_ROOT:-outputs/public_formal_matrix/horizontal/qwen3_4b_zero_shot}"
release_status="outputs/public_formal_matrix/release_status.json"
promotion="outputs/public_formal_matrix/promotion.json"
status_file="$run_root/status.json"
gpu_lock="${PUBLIC_GPU_LOCK_FILE:-outputs/.locks/public-validation-gpu.lock}"
poll_seconds="${PUBLIC_FORMAL_RELEASE_POLL_SECONDS:-30}"

declare -A expected_jobs=(
  [conll04]=288
  [scierc]=551
  [ade]=427
)

current_stage="initializing"
current_dataset=""
formal_test_read=false
release_sha256=""
release_fingerprint=""
gpu_lock_held=false
mkdir -p "$run_root" "$(dirname "$gpu_lock")"

acquire_gpu_lock() {
  if [[ "$gpu_lock_held" == true ]]; then
    return 0
  fi
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
    status
    --output "$status_file"
    --state "$state"
    --stage "$current_stage"
    --formal-test-read "$formal_test_read"
    --data-root "$data_root"
    --run-root "$run_root"
    --release-status "$release_status"
    --promotion "$promotion"
    --prepared-root "$prepared_root"
    --model-path "$model"
    --gpu-lock "$gpu_lock"
  )
  [[ -n "$current_dataset" ]] && args+=(--active-dataset "$current_dataset")
  [[ -n "$release_sha256" ]] && args+=(--release-sha256 "$release_sha256")
  [[ -n "$release_fingerprint" ]] && args+=(--release-fingerprint "$release_fingerprint")
  [[ -n "$error" ]] && args+=(--error "$error")
  "$python" "$contract" "${args[@]}" >/dev/null
}

on_exit() {
  local code="$?"
  if (( code != 0 )); then
    trap - EXIT
    write_status failed "stage=${current_stage}; exit_code=${code}" || true
  fi
}
trap on_exit EXIT

# This is the only operation before the release gate: update a sealed status
# and inspect release_status.json itself.  Test jobs and gold stay unopened.
current_stage="waiting_for_formal_release"
write_status waiting_for_formal_release
while ! jq -e '.status == "complete"' "$release_status" >/dev/null 2>&1; do
  sleep "$poll_seconds"
  write_status waiting_for_formal_release
done

current_stage="release_contract_verification"
write_status running
release_verification="$("$python" "$contract" verify-release \
  --release-status "$release_status" \
  --promotion "$promotion" \
  --prepared-root "$prepared_root")"
release_sha256="$(jq -er '.release_status.sha256' <<<"$release_verification")"
release_fingerprint="$(jq -er '.canonical_fingerprint' <<<"$release_verification")"
write_status running

for dataset in conll04 scierc ade; do
  current_dataset="$dataset"
  base="$data_root/$dataset"
  jobs="$base/test_baseline_jobs.jsonl"
  gold="$base/test_gold.jsonl"
  gold_index="$base/test_index.jsonl"
  ontology="$base/ontology.yaml"
  partial="$run_root/${dataset}_test_partial.jsonl"
  terminal_log="$run_root/${dataset}_test.log"
  materialization="$run_root/${dataset}_test_materialization.json"
  complete="$run_root/${dataset}_test_complete.jsonl"
  expanded="$run_root/${dataset}_test_expanded.jsonl"
  expand_errors="$run_root/${dataset}_test_expand_errors.jsonl"
  verified="$run_root/${dataset}_test_verified.jsonl"
  verification_audit="$run_root/${dataset}_test_verification_audit.jsonl"
  span_index="$run_root/${dataset}_test_span_index.jsonl"
  normalized_metrics="$run_root/${dataset}_test_normalized_text_metrics.json"
  character_span_metrics="$run_root/${dataset}_test_character_span_metrics.json"

  # Set the audit bit before the first command that opens test jobs or gold.
  formal_test_read=true
  current_stage="test_input_preflight"
  write_status running
  "$python" "$contract" preflight \
    --data-root "$data_root" \
    --dataset "$dataset" \
    --expected "${expected_jobs[$dataset]}" >/dev/null

  # A resume skips work only after validating every coverage and metric
  # invariant; the existence of a materialization or metrics file is not enough.
  if "$python" "$contract" validate \
    --data-root "$data_root" \
    --run-root "$run_root" \
    --dataset "$dataset" \
    --expected "${expected_jobs[$dataset]}" \
    --quiet >/dev/null 2>&1; then
    current_stage="reused_verified_dataset"
    write_status running
    echo "reuse complete Qwen zero-shot formal test dataset=$dataset"
    continue
  fi

  # Only GPU inference is serialized.  Release the lock immediately afterward
  # so materialization, verification, and metrics do not block queued GPU work.
  current_stage="waiting_for_exclusive_gpu_lock"
  write_status waiting_for_gpu
  acquire_gpu_lock
  current_stage="test_inference"
  write_status running
  inference_exit=0
  bash scripts/run_qlora_inference_sharded.sh \
    --workers 2 \
    --model-path "$model" \
    --jobs "$jobs" \
    --output "$partial" \
    --log "$terminal_log" \
    --compact-target \
    --resume \
    --max-input-tokens 4096 \
    --max-new-tokens 1024 \
    --max-seconds-per-job 90 || inference_exit=$?
  release_gpu_lock
  if (( inference_exit != 0 )); then
    exit "$inference_exit"
  fi
  current_stage="test_inference_complete"
  write_status running

  current_stage="terminal_failure_materialization"
  write_status running
  "$python" scripts/materialize_missing_predictions.py \
    --jobs "$jobs" \
    --predictions "$partial" \
    --output "$complete" \
    --summary "$materialization" \
    --reason "empty formal-test prediction materialized after terminal zero-shot inference failure"

  current_stage="compact_prediction_expansion"
  write_status running
  "$python" scripts/expand_compact_predictions.py \
    --jobs "$jobs" \
    --predictions "$complete" \
    --output "$expanded" \
    --errors "$expand_errors"

  current_stage="relation_verification"
  write_status running
  "$python" scripts/verify_relations.py \
    --annotations "$expanded" \
    --ontology "$ontology" \
    --output "$verified" \
    --audit "$verification_audit"

  current_stage="formal_span_index"
  write_status running
  "$python" "$contract" span-index \
    --source "$gold_index" \
    --output "$span_index" >/dev/null

  current_stage="normalized_text_evaluation"
  write_status running
  "$python" scripts/evaluate_annotations.py \
    --gold "$gold" \
    --gold-index "$gold_index" \
    --predictions "$verified" \
    --jobs "$jobs" \
    --include-missing-as-empty \
    --output "$normalized_metrics"

  current_stage="strict_character_span_evaluation"
  write_status running
  "$python" scripts/evaluate_span_aware.py \
    --source-gold "$gold" \
    --source-gold-index "$gold_index" \
    --gold "$gold" \
    --gold-index "$span_index" \
    --predictions "$verified" \
    --jobs "$jobs" \
    --output "$character_span_metrics" \
    --allow-non-validation

  current_stage="dataset_contract_validation"
  "$python" "$contract" validate \
    --data-root "$data_root" \
    --run-root "$run_root" \
    --dataset "$dataset" \
    --expected "${expected_jobs[$dataset]}" >/dev/null
  write_status running
done

# Bind the final status to the exact release marker observed before test access.
current_dataset=""
current_stage="final_release_identity_verification"
current_release_verification="$("$python" "$contract" verify-release \
  --release-status "$release_status" \
  --promotion "$promotion" \
  --prepared-root "$prepared_root")"
current_release_sha256="$(jq -er '.release_status.sha256' <<<"$current_release_verification")"
current_release_fingerprint="$(jq -er '.canonical_fingerprint' <<<"$current_release_verification")"
if [[ "$current_release_sha256" != "$release_sha256" ]]; then
  echo "formal release marker changed during Qwen zero-shot test run" >&2
  exit 1
fi
if [[ "$current_release_fingerprint" != "$release_fingerprint" ]]; then
  echo "canonical release/promotion/preparation fingerprint changed during Qwen zero-shot test run" >&2
  exit 1
fi

current_stage="complete"
write_status complete
trap - EXIT
