#!/usr/bin/env bash
# Wait for public PGE validation, then run the fresh SpERT validation baseline.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

poll_seconds="${SPERT_QUEUE_POLL_SECONDS:-30}"
pge_status="outputs/public_pge_validation_seed42/status.json"
run_root="outputs/public_horizontal_validation/spert_fresh"
launcher_status="$run_root/launcher_status.json"
mkdir -p "$run_root"

write_status() {
  local state="$1"
  python3 - "$launcher_status" "$state" "$poll_seconds" "$pge_status" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

target = Path(sys.argv[1])
state = sys.argv[2]
poll_seconds = int(sys.argv[3])
upstream_path = Path(sys.argv[4])
upstream = None
if upstream_path.is_file():
    try:
        upstream = json.loads(upstream_path.read_text(encoding="utf-8")).get("status")
    except (OSError, json.JSONDecodeError):
        pass
payload = {
    "status": state,
    "waits_for": "outputs/public_pge_validation_seed42/status.json:complete",
    "upstream_status": upstream,
    "poll_seconds": poll_seconds,
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

write_status waiting_for_pge
while [[ ! -f "$pge_status" ]] || \
      [[ "$(jq -r '.status // ""' "$pge_status" 2>/dev/null || true)" != "complete" ]]; do
  sleep "$poll_seconds"
  write_status waiting_for_pge
done

write_status running_spert_fresh
exec bash scripts/run_spert_fresh_baseline.sh all
