#!/usr/bin/env bash
# Keep the formal internal matrix queued until the audited test release is ready.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

launcher_status="outputs/public_formal_matrix/internal_launcher_status.json"
poll_seconds="${PUBLIC_FORMAL_RELEASE_POLL_SECONDS:-30}"
guard_seconds="${PUBLIC_FORMAL_RELEASE_GUARD_SECONDS:-60}"
release_snapshot="outputs/public_formal_matrix/internal_release_attestation.json"
resume_attestation="outputs/public_formal_matrix/internal/resume_attestation.json"
release_status_sha=""
canonical_release_fingerprint=""
prepared_release_sha=""
matrix_pid=""
mkdir -p "$(dirname "$launcher_status")"

write_launcher() {
  local state="$1"
  python3 - "$launcher_status" "$state" "$poll_seconds" \
    "$release_status_sha" "$canonical_release_fingerprint" \
    "$prepared_release_sha" <<'PY'
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

target = Path(sys.argv[1])
payload = {
    "status": sys.argv[2],
    "poll_seconds": int(sys.argv[3]),
    "release_status_sha256": sys.argv[4] or None,
    "canonical_release_fingerprint": sys.argv[5] or None,
    "prepared_release_sha256": sys.argv[6] or None,
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
target.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile(
    mode="w", encoding="utf-8", dir=target.parent,
    prefix=f".{target.name}.", delete=False,
) as stream:
    temporary = Path(stream.name)
    json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
os.replace(temporary, target)
PY
}

# This gate intentionally waits for the publisher's closed, completed release
# record and its exact persisted attestation.  Prepared outputs can become
# valid before the publisher performs its second replay and marks complete.
while ! python3 scripts/verify_public_formal_internal_state.py \
  --mode release-only \
  >"${release_snapshot}.tmp" 2>/dev/null; do
  write_launcher waiting_for_formal_release
  sleep "$poll_seconds"
done
mv "${release_snapshot}.tmp" "$release_snapshot"
release_status_sha="$(jq -er '.release.release_status.sha256' "$release_snapshot")"
canonical_release_fingerprint="$(jq -er '.release.canonical_fingerprint' "$release_snapshot")"
prepared_release_sha="$(jq -er '.release.prepared_release_sha256' "$release_snapshot")"

python3 scripts/verify_public_formal_internal_state.py \
  --mode preflight --quarantine-invalid \
  --expected-release-status-sha256 "$release_status_sha" \
  --expected-canonical-fingerprint "$canonical_release_fingerprint" \
  --expected-prepared-release-sha256 "$prepared_release_sha" \
  --output "$resume_attestation" \
  >outputs/public_formal_matrix/internal_preflight.json

terminate_matrix() {
  if [[ -n "$matrix_pid" ]] && kill -0 "$matrix_pid" 2>/dev/null; then
    kill -TERM -- "-$matrix_pid" 2>/dev/null || kill -TERM "$matrix_pid" 2>/dev/null || true
    wait "$matrix_pid" 2>/dev/null || true
  fi
}
trap terminate_matrix EXIT INT TERM

write_launcher running_internal_matrix
setsid scripts/run_public_formal_internal_matrix.sh &
matrix_pid="$!"
while kill -0 "$matrix_pid" 2>/dev/null; do
  sleep "$guard_seconds" &
  wait "$!"
  if ! python3 scripts/verify_public_formal_internal_state.py \
    --mode release-only \
    --expected-release-status-sha256 "$release_status_sha" \
    --expected-canonical-fingerprint "$canonical_release_fingerprint" \
    --expected-prepared-release-sha256 "$prepared_release_sha" \
    >outputs/public_formal_matrix/internal_release_guard.json; then
    write_launcher release_changed_fail_closed
    terminate_matrix
    exit 1
  fi
done
if wait "$matrix_pid"; then
  matrix_status=0
else
  matrix_status="$?"
fi
matrix_pid=""
if (( matrix_status != 0 )); then
  write_launcher internal_matrix_failed
  exit "$matrix_status"
fi

python3 scripts/verify_public_formal_internal_state.py \
  --mode postflight \
  --expected-release-status-sha256 "$release_status_sha" \
  --expected-canonical-fingerprint "$canonical_release_fingerprint" \
  --expected-prepared-release-sha256 "$prepared_release_sha" \
  --output "$resume_attestation" \
  >outputs/public_formal_matrix/internal_postflight.json
write_launcher complete
trap - EXIT INT TERM
