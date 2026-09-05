#!/usr/bin/env bash
# Queue-safe OneKE load+generation canary.  This file is created but not started
# by setup/preflight; the shared public GPU lock remains authoritative.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

runtime="${PUBLIC_ONEKE_PYTHON:-/ds2/xuelin/envs/public-oneke-formal/bin/python}"
model="${PUBLIC_ONEKE_MODEL:-/ds2/xuelin/cache/huggingface/hub/models--zjunlp--OneKE/snapshots/696148c0581b29f530af738ddab500deaa8fe8f2}"
state_root="outputs/public_external_formal/oneke"
marker="$state_root/gpu_canary.json"
launcher_status="$state_root/canary_launcher_status.json"
environment_status="$state_root/environment_status.json"
gpu_lock="${PUBLIC_GPU_LOCK_FILE:-outputs/.locks/public-validation-gpu.lock}"
poll_seconds="${PUBLIC_ONEKE_CANARY_POLL_SECONDS:-30}"
mkdir -p "$state_root" "$(dirname "$gpu_lock")"

write_status() {
  local state="$1" stage="$2" detail="$3"
  python3 - "$launcher_status" "$state" "$stage" "$detail" "$gpu_lock" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema_version": "public-oneke-canary-launcher-v1",
    "status": sys.argv[2],
    "stage": sys.argv[3],
    "detail": sys.argv[4],
    "gpu_lock": sys.argv[5],
    "test_gold_read": False,
    "terminal": sys.argv[2] in {"complete", "blocked_with_reason"},
    "retry_ready": sys.argv[2] == "retry_ready",
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
temporary = path.with_name(f".{path.name}.tmp")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
temporary.replace(path)
PY
}

canary_marker_passed() {
  [[ -f "$marker" ]] && jq -e \
    --arg revision "696148c0581b29f530af738ddab500deaa8fe8f2" '
      .schema_version == "public-external-gpu-canary-v1" and
      .baseline == "oneke" and
      .gpu == "RTX 3090" and
      .capacity_gib == 24 and
      .status == "passed" and
      .terminal == true and
      .runtime_compatible == true and
      .exit_code == 0 and
      .model_revision == $revision and
      .quantization == "bitsandbytes-nf4-double-quantization" and
      .prompt_version == "oneke-upstream-jsonlike-batched-v3" and
      .test_gold_read == false and
      ((.actual_gpu_name | type) == "string") and
      (.actual_gpu_name | contains("3090")) and
      ((.actual_total_memory_bytes | type) == "number") and
      .actual_total_memory_bytes >= (20 * 1024 * 1024 * 1024) and
      .actual_total_memory_bytes <= (24 * 1024 * 1024 * 1024) and
      ((.peak_allocated_bytes | type) == "number") and
      .peak_allocated_bytes > 0 and
      .peak_allocated_bytes <= .actual_total_memory_bytes and
      ((.peak_reserved_bytes | type) == "number") and
      .peak_reserved_bytes > 0 and
      .peak_reserved_bytes <= .actual_total_memory_bytes and
      .synthetic_relation_signature.source == "Adverse-Effect" and
      .synthetic_relation_signature.target == "Drug"
    ' "$marker" >/dev/null 2>&1
}

on_exit() {
  local code="$?"
  if (( code != 0 )); then
    write_status blocked_with_reason failed "canary runner exited with code $code"
  fi
}
trap on_exit EXIT

if [[ "${1:-}" == "--rearm" ]]; then
  write_status retry_ready repaired "CPU contract repair passed; GPU canary may be retried"
  trap - EXIT
  exit 0
elif [[ $# -ne 0 ]]; then
  write_status blocked_with_reason preflight "unsupported canary launcher argument"
  exit 2
fi

if [[ ! -x "$runtime" ]]; then
  write_status blocked_with_reason preflight "isolated OneKE runtime is missing"
  exit 2
fi
expected_environment="$(cd "$(dirname "$runtime")/.." 2>/dev/null && pwd -L || true)"
if [[ ! -f "$environment_status" ]] || ! jq -e \
  --arg environment "$expected_environment" '
    .schema_version == "public-oneke-environment-v1" and
    .status == "complete" and
    .environment == $environment and
    .cuda_visible_devices == "" and
    .gpu_process_started == false
  ' "$environment_status" >/dev/null 2>&1; then
  write_status blocked_with_reason preflight "isolated OneKE runtime is not verified"
  exit 2
fi
if canary_marker_passed; then
  write_status complete already_passed "existing verified canary marker retained"
  trap - EXIT
  exit 0
fi

write_status waiting_for_gpu waiting_for_exclusive_gpu_lock "internal public GPU work has priority"
exec 8>"$gpu_lock"
while ! flock -n 8; do
  sleep "$poll_seconds"
  write_status waiting_for_gpu waiting_for_exclusive_gpu_lock "internal public GPU work has priority"
done

write_status running gpu_canary "loading OneKE with NF4 and running synthetic NER+RE"
CUDA_VISIBLE_DEVICES=0 "$runtime" scripts/run_public_oneke_formal.py \
  --model "$model" \
  --max-input-tokens 3072 \
  --max-new-tokens 128 \
  canary \
  --marker "$marker"

if ! canary_marker_passed; then
  write_status blocked_with_reason failed "canary command returned without a valid passed marker"
  exit 3
fi

write_status complete gpu_canary_passed "OneKE is eligible for the formal validation queue"
trap - EXIT
