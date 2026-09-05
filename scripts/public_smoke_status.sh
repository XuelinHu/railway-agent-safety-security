#!/usr/bin/env bash
# Read-only progress snapshot for a running public smoke experiment.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
RUN_ROOT="outputs/public_smoke_40_8_8"

printf '%s\n' '--- services ---'
(systemctl --user --no-pager --full status public-smoke-qwen-40-8-8.service 2>&1 || true) | sed -n '1,14p'
(systemctl --user --no-pager --full status public-smoke-external-preflight.service 2>&1 || true) | sed -n '1,14p'
printf '%s\n' '--- gpu ---'
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
printf '%s\n' '--- latest runner log ---'
tail -30 "$RUN_ROOT/runner.log" 2>/dev/null || true
printf '%s\n' '--- completed training metrics ---'
find "$RUN_ROOT" -name training_metrics.json -print 2>/dev/null | sort
printf '%s\n' '--- completed evaluation metrics ---'
find "$RUN_ROOT" -name '*_metrics.json' -print 2>/dev/null | sort
