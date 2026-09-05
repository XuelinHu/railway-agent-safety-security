#!/usr/bin/env bash
# Keep the single GPU occupied with ordered, resumable public validation jobs.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

poll_seconds="${GPU_QUEUE_POLL_SECONDS:-30}"
heartbeat_seconds="${GPU_QUEUE_HEARTBEAT_SECONDS:-30}"
gliner_workers="${GLINER_GLIREL_WORKERS:-2}"
state_root="outputs/public_gpu_validation_queue"
status_file="$state_root/status.json"
qwen_status="outputs/public_horizontal_validation/qwen3_4b_zero_shot/status.json"
qwen_retry_status="outputs/public_horizontal_validation/qwen3_4b_zero_shot/retry_status.json"
glirel_status="outputs/public_horizontal_validation/gliner_glirel/status.json"
stage="initializing"
child_pid=""
qwen_retry_state=""
qwen_remaining_failed_jobs=0
mkdir -p "$state_root"

if [[ ! "$poll_seconds" =~ ^[1-9][0-9]*$ ]] || \
   [[ ! "$heartbeat_seconds" =~ ^[1-9][0-9]*$ ]]; then
  echo "GPU queue poll and heartbeat intervals must be positive integers" >&2
  exit 2
fi
if [[ ! "$gliner_workers" =~ ^[12]$ ]]; then
  echo "GLINER_GLIREL_WORKERS must be 1 or 2" >&2
  exit 2
fi

write_status() {
  local state="$1"
  python3 - "$status_file" "$state" "$stage" "$poll_seconds" "$heartbeat_seconds" \
    "$gliner_workers" "$qwen_retry_state" "$qwen_remaining_failed_jobs" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "status": sys.argv[2],
    "stage": sys.argv[3],
    "poll_seconds": int(sys.argv[4]),
    "heartbeat_seconds": int(sys.argv[5]),
    "gliner_glirel_requested_workers": int(sys.argv[6]),
    "parallel_policy": "two inference workers with automatic single-worker fallback; training remains exclusive",
    "qwen_retry_status": sys.argv[7] or None,
    "qwen_remaining_failed_jobs": int(sys.argv[8]),
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "order": [
        "qwen3-4b-zero-shot-validation",
        "qwen3-4b-zero-shot-failed-job-retry",
        "gliner-glirel-validation",
    ],
}
temporary = path.with_name(f".{path.name}.tmp")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
temporary.replace(path)
PY
}

on_exit() {
  local code="$?"
  trap - EXIT
  if (( code != 0 )); then
    write_status failed || true
  fi
  exit "$code"
}
trap on_exit EXIT
trap 'exit 143' TERM INT

run_with_heartbeat() {
  local code=0
  "$@" &
  child_pid="$!"
  while kill -0 "$child_pid" 2>/dev/null; do
    write_status running
    sleep "$heartbeat_seconds"
  done
  if wait "$child_pid"; then
    code=0
  else
    code="$?"
  fi
  child_pid=""
  return "$code"
}

retry_is_terminal() {
  local retry_state
  retry_state="$(jq -r '.status // ""' "$qwen_retry_status" 2>/dev/null || true)"
  [[ "$retry_state" == "complete" || \
     "$retry_state" == "complete_with_terminal_failures" ]]
}

stage="waiting_for_qwen_validation"
write_status waiting
while [[ ! -f "$qwen_status" ]] || \
      [[ "$(jq -r '.status // ""' "$qwen_status" 2>/dev/null || true)" != "validation_complete" ]]; do
  write_status waiting
  sleep "$poll_seconds"
done

stage="qwen_failed_job_retry"
write_status running
if [[ ! -f "$qwen_retry_status" ]] || ! retry_is_terminal; then
  run_with_heartbeat bash scripts/repair_qwen_zeroshot_validation.sh
fi
qwen_retry_state="$(jq -r '.status // ""' "$qwen_retry_status" 2>/dev/null || true)"
qwen_remaining_failed_jobs="$(jq -r '.remaining_failed_jobs // ""' "$qwen_retry_status" 2>/dev/null || true)"
if [[ ! "$qwen_remaining_failed_jobs" =~ ^[0-9]+$ ]]; then
  echo "Qwen retry status has an invalid remaining_failed_jobs value" >&2
  exit 1
fi
case "$qwen_retry_state" in
  complete)
    if (( qwen_remaining_failed_jobs != 0 )); then
      echo "Qwen retry reports complete with remaining failures" >&2
      exit 1
    fi
    ;;
  complete_with_terminal_failures)
    if (( qwen_remaining_failed_jobs == 0 )); then
      echo "Qwen retry reports terminal failures but the remaining count is zero" >&2
      exit 1
    fi
    ;;
  *)
    echo "Qwen retry did not reach a terminal state: $qwen_retry_state" >&2
    exit 1
    ;;
esac

stage="gliner_glirel_validation"
write_status running
if [[ ! -f "$glirel_status" ]] || \
   [[ "$(jq -r '.status // ""' "$glirel_status" 2>/dev/null || true)" != "complete" ]]; then
  if ! run_with_heartbeat env \
      GPU_POLL_SECONDS="$poll_seconds" \
      GLINER_GLIREL_WORKERS="$gliner_workers" \
      bash scripts/run_gliner_glirel_validation_all.sh; then
    if (( gliner_workers == 1 )); then
      exit 1
    fi
    stage="gliner_glirel_validation_single_worker_fallback"
    write_status running
    run_with_heartbeat env \
      GPU_POLL_SECONDS="$poll_seconds" \
      GLINER_GLIREL_WORKERS=1 \
      bash scripts/run_gliner_glirel_validation_all.sh
  fi
fi

stage="complete"
if (( qwen_remaining_failed_jobs > 0 )); then
  write_status complete_with_terminal_failures
else
  write_status complete
fi
trap - EXIT
