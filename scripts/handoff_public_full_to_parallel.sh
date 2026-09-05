#!/usr/bin/env bash
# Wait for the in-flight CoNLL04 stage, then replace the serial continuation.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
marker="outputs/public_full_stage1/conll04_kg_validation_metrics.json"
while [[ ! -f "$marker" ]]; do sleep 20; done
systemctl --user stop public-full-stage1.service || true
while systemctl --user is-active --quiet public-full-stage1.service; do sleep 2; done
systemd-run --user --unit=public-full-stage1-parallel --collect --no-block \
  /usr/bin/bash -lc "cd '$ROOT'; exec > outputs/public_full_stage1/parallel_runner.log 2>&1; exec scripts/run_public_full_remaining_parallel.sh"
