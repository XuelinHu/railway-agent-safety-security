#!/usr/bin/env bash
# Evaluate only the three frozen SpERT seed-42 checkpoints on promoted test data.
set -euo pipefail

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

python="${SPERT_PYTHON:-/home/xuelin/miniconda3/envs/rc-llm-comet/bin/python}"
contract="scripts/spert_fresh_formal_contract.py"
release_contract="scripts/qwen_zeroshot_formal_contract.py"
compat="scripts/run_spert_compat.py"
spert_repo="$root/tools/external-baselines/spert"
release_status="outputs/public_formal_matrix/release_status.json"
promotion="outputs/public_formal_matrix/promotion.json"
public_prepared_root="data/processed/public_benchmarks_hrge_test_v1"
prepared_root="data/processed/spert_fresh_test_v1"
data_root="data/processed/public_benchmarks_full"
run_root="outputs/public_formal_matrix/horizontal/spert_fresh_seed42"
status_file="$run_root/status.json"
gpu_lock="${PUBLIC_GPU_LOCK_FILE:-$root/outputs/.locks/public-validation-gpu.lock}"
poll_seconds="${PUBLIC_FORMAL_RELEASE_POLL_SECONDS:-30}"

current_stage="initializing"
current_dataset=""
formal_test_read=false
release_sha256=""
release_fingerprint=""
gpu_lock_held=false
mkdir -p "$run_root" "$(dirname "$gpu_lock")"

write_status() {
  local state="$1" error="${2:-}"
  local args=(
    status --output "$status_file" --state "$state" --stage "$current_stage"
    --formal-test-read "$formal_test_read" --run-root "$run_root"
    --gpu-lock "$gpu_lock"
  )
  [[ -n "$current_dataset" ]] && args+=(--active-dataset "$current_dataset")
  [[ -n "$release_sha256" ]] && args+=(--release-sha256 "$release_sha256")
  [[ -n "$release_fingerprint" ]] && args+=(--release-fingerprint "$release_fingerprint")
  [[ -n "$error" ]] && args+=(--error "$error")
  "$python" "$contract" "${args[@]}" >/dev/null
}

acquire_gpu_lock() {
  if [[ "$gpu_lock_held" == true ]]; then
    return 0
  fi
  exec 8>"$gpu_lock"
  while ! flock -n 8; do
    current_stage="waiting_for_exclusive_gpu_lock"
    write_status waiting_for_gpu
    sleep 30
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

on_exit() {
  local code="$?"
  if (( code != 0 )); then
    trap - EXIT
    release_gpu_lock || true
    write_status failed "stage=${current_stage}; exit_code=${code}" || true
  fi
}
trap on_exit EXIT

verify_release() {
  "$python" "$release_contract" verify-release \
    --release-status "$release_status" \
    --promotion "$promotion" \
    --prepared-root "$public_prepared_root"
}

verify_release_unchanged() {
  local verification current_sha current_fingerprint
  verification="$(verify_release)"
  current_sha="$(jq -er '.release_status.sha256' <<<"$verification")"
  current_fingerprint="$(jq -er '.canonical_fingerprint' <<<"$verification")"
  if [[ "$current_sha" != "$release_sha256" || \
        "$current_fingerprint" != "$release_fingerprint" ]]; then
    echo "canonical formal release changed during SpERT seed-42 test evaluation" >&2
    return 1
  fi
}

# Before the release gate, touch only status and release_status.json.  In
# particular, do not construct, validate, or open a test dataset path here.
current_stage="waiting_for_formal_release"
write_status waiting_for_formal_release
while ! jq -e '.status == "complete" and .stage == "complete"' \
  "$release_status" >/dev/null 2>&1; do
  sleep "$poll_seconds"
  write_status waiting_for_formal_release
done

current_stage="canonical_release_verification"
write_status running
release_verification="$(verify_release)"
release_sha256="$(jq -er '.release_status.sha256' <<<"$release_verification")"
release_fingerprint="$(jq -er '.canonical_fingerprint' <<<"$release_verification")"

# Checkpoint verification is eval-only and hashes the exact existing model,
# tokenizer, training-argument, and extra.state files.  There is no backbone
# or missing-checkpoint fallback.
current_stage="frozen_checkpoint_verification"
write_status running
"$python" "$contract" verify-checkpoints >/dev/null

# Set the audit bit before the first operation that may open any public or
# native test row.  The preparer independently repeats canonical release and
# promotion verification before access.
formal_test_read=true
current_stage="promoted_test_input_preparation"
write_status running
"$python" scripts/prepare_spert_fresh_test.py >/dev/null
"$python" scripts/prepare_spert_fresh_test.py --verify >/dev/null
verify_release_unchanged

for dataset in conll04 scierc ade; do
  current_dataset="$dataset"
  base="$data_root/$dataset"
  target="$run_root/$dataset"
  prepared_manifest="$prepared_root/$dataset/manifest.json"
  jobs="$base/test_baseline_jobs.jsonl"
  gold="$base/test_gold.jsonl"
  gold_index="$base/test_index.jsonl"
  log_root="$target/upstream_logs"
  snapshot="$target/eval_snapshot.json"
  raw_predictions="$target/raw_predictions.json"
  eval_args="$target/eval_args.json"
  capture_manifest="$target/raw_capture_manifest.json"
  predictions="$target/test_predictions.jsonl"
  conversion_manifest="$target/conversion_manifest.json"
  normalized_metrics="$target/test_normalized_text_metrics.json"
  span_metrics="$target/test_character_span_metrics.json"
  label="spert_${dataset}_seed42_formal_test"
  mkdir -p "$target"

  current_stage="content_validated_resume_check"
  write_status running
  if "$python" "$contract" validate \
    --dataset "$dataset" --data-root "$data_root" \
    --prepared-root "$prepared_root" --run-root "$run_root" \
    --release-status "$release_status" --promotion "$promotion" \
    --public-prepared-root "$public_prepared_root" --quiet \
    >/dev/null 2>&1; then
    current_stage="reused_verified_dataset"
    write_status running
    echo "reuse content-verified SpERT formal test dataset=$dataset"
    continue
  fi

  current_stage="frozen_checkpoint_verification"
  write_status running
  checkpoint_json="$("$python" "$contract" verify-checkpoint --dataset "$dataset")"
  checkpoint="$(jq -er '.checkpoint.path' <<<"$checkpoint_json")"

  current_stage="waiting_for_exclusive_gpu_lock"
  write_status waiting_for_gpu
  acquire_gpu_lock

  # Snapshot the complete set of existing upstream run directories before
  # evaluation.  Capture later requires exactly one set-difference entry; it
  # never searches for or trusts a "newest" file.
  current_stage="exact_output_discovery_snapshot"
  write_status running
  "$python" "$contract" begin-eval \
    --dataset "$dataset" --prepared-manifest "$prepared_manifest" \
    --log-root "$log_root" --snapshot "$snapshot" \
    --release-status "$release_status" --promotion "$promotion" \
    --public-prepared-root "$public_prepared_root" >/dev/null

  current_stage="frozen_seed42_test_evaluation"
  write_status running
  evaluation_exit=0
  "$python" "$compat" --repo "$spert_repo" -- eval \
    --model_type spert --model_path "$checkpoint" --tokenizer_path "$checkpoint" \
    --dataset_path "$root/$prepared_root/$dataset/test.json" \
    --types_path "$root/$prepared_root/$dataset/types.json" \
    --eval_batch_size 1 --rel_filter_threshold 0.4 --size_embedding 25 \
    --prop_drop 0.1 --max_span_size 10 --sampling_processes 4 \
    --max_pairs 1000 --seed 42 --store_predictions \
    --label "$label" --log_path "$root/$log_root" || evaluation_exit=$?
  release_gpu_lock
  if (( evaluation_exit != 0 )); then
    exit "$evaluation_exit"
  fi
  verify_release_unchanged

  current_stage="exact_new_output_capture"
  write_status running
  "$python" "$contract" capture-eval \
    --dataset "$dataset" --prepared-manifest "$prepared_manifest" \
    --jobs "$jobs" --log-root "$log_root" --snapshot "$snapshot" \
    --raw-output "$raw_predictions" --args-output "$eval_args" \
    --capture-manifest "$capture_manifest" \
    --release-status "$release_status" --promotion "$promotion" \
    --public-prepared-root "$public_prepared_root" >/dev/null

  current_stage="ordered_prediction_conversion"
  write_status running
  "$python" scripts/convert_spert_test_predictions.py \
    --dataset "$dataset" --prepared-manifest "$prepared_manifest" \
    --predictions "$raw_predictions" --jobs "$jobs" \
    --inference-manifest "$capture_manifest" \
    --output "$predictions" --manifest "$conversion_manifest" >/dev/null

  current_stage="normalized_text_evaluation"
  write_status running
  "$python" scripts/evaluate_annotations.py \
    --gold "$gold" --gold-index "$gold_index" \
    --predictions "$predictions" --jobs "$jobs" \
    --include-missing-as-empty --output "$normalized_metrics" >/dev/null

  current_stage="strict_character_span_evaluation"
  write_status running
  "$python" scripts/evaluate_public_validation_spans.py \
    --gold "$gold" --gold-index "$gold_index" \
    --predictions "$predictions" --jobs "$jobs" \
    --output "$span_metrics" --allow-non-validation >/dev/null

  current_stage="content_validated_completion"
  write_status running
  "$python" "$contract" finalize \
    --dataset "$dataset" --data-root "$data_root" \
    --prepared-root "$prepared_root" --run-root "$run_root" \
    --release-status "$release_status" --promotion "$promotion" \
    --public-prepared-root "$public_prepared_root" --quiet >/dev/null
  "$python" "$contract" validate \
    --dataset "$dataset" --data-root "$data_root" \
    --prepared-root "$prepared_root" --run-root "$run_root" \
    --release-status "$release_status" --promotion "$promotion" \
    --public-prepared-root "$public_prepared_root" --quiet >/dev/null
  verify_release_unchanged
done

current_dataset=""
current_stage="final_release_and_artifact_verification"
write_status running
verify_release_unchanged
"$python" scripts/prepare_spert_fresh_test.py --verify >/dev/null
"$python" "$contract" verify-checkpoints >/dev/null
for dataset in conll04 scierc ade; do
  "$python" "$contract" validate \
    --dataset "$dataset" --data-root "$data_root" \
    --prepared-root "$prepared_root" --run-root "$run_root" \
    --release-status "$release_status" --promotion "$promotion" \
    --public-prepared-root "$public_prepared_root" --quiet >/dev/null
done

current_stage="complete"
write_status complete
trap - EXIT
