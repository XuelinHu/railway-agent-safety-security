#!/usr/bin/env bash
# Wait for horizontal validation, then hand the idle GPU to public PGE training.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

poll_seconds="${PUBLIC_PGE_QUEUE_POLL_SECONDS:-30}"
queue_status="outputs/public_gpu_validation_queue/status.json"
run_root="outputs/public_pge_validation_seed42"
launcher_status="$run_root/launcher_status.json"
mkdir -p "$run_root"

write_status() {
  local state="$1"
  python3 - "$launcher_status" "$state" "$poll_seconds" "$queue_status" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
queue_path = Path(sys.argv[4])
upstream = {}
if queue_path.is_file():
    try:
        upstream = json.loads(queue_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        upstream = {}
payload = {
    "status": sys.argv[2],
    "waits_for": "outputs/public_gpu_validation_queue/status.json:terminal",
    "poll_seconds": int(sys.argv[3]),
    "accepted_upstream_statuses": ["complete", "complete_with_terminal_failures"],
    "upstream_status": upstream.get("status"),
    "upstream_qwen_remaining_failed_jobs": upstream.get("qwen_remaining_failed_jobs", 0),
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
temporary = path.with_name(f".{path.name}.tmp")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
temporary.replace(path)
PY
}

horizontal_is_terminal() {
  local state
  state="$(jq -r '.status // ""' "$queue_status" 2>/dev/null || true)"
  [[ "$state" == "complete" || "$state" == "complete_with_terminal_failures" ]]
}

write_status waiting_for_horizontal_validation
while [[ ! -f "$queue_status" ]] || ! horizontal_is_terminal; do
  sleep "$poll_seconds"
  write_status waiting_for_horizontal_validation
done

write_status starting_public_pge
exec bash scripts/run_public_pge_validation.sh
