#!/usr/bin/env bash
# Run the official-threshold GLiNER + GLiREL sensitivity channel after PGE and fresh SpERT.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

poll_seconds="${GLIREL_T0_QUEUE_POLL_SECONDS:-30}"
spert_status="outputs/public_horizontal_validation/spert_fresh/status.json"
run_root="outputs/public_horizontal_validation/gliner_glirel_t0"
launcher_status="$run_root/launcher_status.json"
canary_status="$run_root/compatibility_canary.json"
requested_workers="${GLINER_GLIREL_T0_WORKERS:-2}"
mkdir -p "$run_root"

if [[ ! "$poll_seconds" =~ ^[1-9][0-9]*$ ]]; then
  echo "GLIREL_T0_QUEUE_POLL_SECONDS must be a positive integer" >&2
  exit 2
fi
if [[ ! "$requested_workers" =~ ^[12]$ ]]; then
  echo "GLINER_GLIREL_T0_WORKERS must be 1 or 2" >&2
  exit 2
fi

write_status() {
  local state="$1"
  python3 - "$launcher_status" "$state" "$poll_seconds" "$spert_status" \
    "$run_root" "$requested_workers" "$canary_status" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

target = Path(sys.argv[1])
state = sys.argv[2]
poll_seconds = int(sys.argv[3])
upstream_path = Path(sys.argv[4])
output_root = Path(sys.argv[5])
requested_workers = int(sys.argv[6])
canary_path = Path(sys.argv[7])
upstream = None
if upstream_path.is_file():
    try:
        upstream = json.loads(upstream_path.read_text(encoding="utf-8")).get("status")
    except (OSError, json.JSONDecodeError):
        pass
canary = None
if canary_path.is_file():
    try:
        report = json.loads(canary_path.read_text(encoding="utf-8"))
        canary = {
            "status": report.get("status"),
            "runtime_compatible": report.get("runtime_compatible"),
            "checked_at": report.get("checked_at"),
            "path": str(canary_path),
        }
    except (OSError, json.JSONDecodeError):
        canary = {"status": "invalid", "runtime_compatible": False, "path": str(canary_path)}
payload = {
    "status": state,
    "queue_order": ["public_pge_seed42", "spert_fresh_seed42", "gliner_glirel_t0"],
    "waits_for": "outputs/public_horizontal_validation/spert_fresh/status.json:complete",
    "upstream_status": upstream,
    "poll_seconds": poll_seconds,
    "output_root": str(output_root),
    "parameters": {
        "entity_threshold": 0.5,
        "relation_threshold": 0.0,
        "dtype": "float16",
        "requested_workers": requested_workers,
    },
    "preserves": "outputs/public_horizontal_validation/gliner_glirel",
    "compatibility_canary": canary,
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
    write_status failed || true
  fi
}
trap on_exit EXIT

write_status checking_compatibility_canary
if [[ "${GLIREL_T0_FORCE_CANARY:-0}" == "1" || ! -f "$canary_status" ]]; then
  if ! env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    /home/xuelin/miniconda3/bin/python scripts/check_glirel_compatibility.py \
      --output "$canary_status"; then
    write_status blocked_incompatible_runtime
    trap - EXIT
    exit 0
  fi
fi
if [[ "$(jq -r '.status // ""' "$canary_status" 2>/dev/null || true)" != "passed" ]] || \
   [[ "$(jq -r '.runtime_compatible // false' "$canary_status" 2>/dev/null || true)" != "true" ]]; then
  write_status blocked_incompatible_runtime
  trap - EXIT
  exit 0
fi

write_status waiting_for_spert_fresh
while [[ ! -f "$spert_status" ]] || \
      [[ "$(jq -r '.status // ""' "$spert_status" 2>/dev/null || true)" != "complete" ]]; do
  sleep "$poll_seconds"
  write_status waiting_for_spert_fresh
done

write_status starting_gliner_glirel_t0
exec env \
  OUTPUT_ROOT="$run_root" \
  ENTITY_THRESHOLD=0.5 \
  RELATION_THRESHOLD=0.0 \
  MODEL_DTYPE=float16 \
  GLINER_GLIREL_WORKERS="$requested_workers" \
  GPU_POLL_SECONDS="$poll_seconds" \
  bash scripts/run_gliner_glirel_validation_all.sh
