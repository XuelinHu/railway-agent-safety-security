#!/usr/bin/env bash
# Non-invasive 10-minute watchdog for the ADE portion of the full public run.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
RUN_ROOT="outputs/public_full_stage1"
LOG="$RUN_ROOT/ade_monitor.log"
mkdir -p "$RUN_ROOT"

while true; do
  now="$(date --iso-8601=seconds)"
  service="inactive"
  if systemctl --user is-active --quiet public-full-stage1-parallel-resume.service; then
    service="active"
  fi
  ade_train=0
  ade_validation=0
  [[ -f "$RUN_ROOT/ade_baseline/training_metrics.json" ]] && ade_train=$((ade_train + 1))
  [[ -f "$RUN_ROOT/ade_kg/training_metrics.json" ]] && ade_train=$((ade_train + 1))
  [[ -f "$RUN_ROOT/ade_baseline_validation_metrics.json" ]] && ade_validation=$((ade_validation + 1))
  [[ -f "$RUN_ROOT/ade_kg_validation_metrics.json" ]] && ade_validation=$((ade_validation + 1))
  gpu="$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | head -1 || true)"
  printf '{"checked_at":"%s","main_service":"%s","ade_train_completed":%s,"ade_validation_completed":%s,"gpu":"%s"}\n' \
    "$now" "$service" "$ade_train" "$ade_validation" "$gpu" >> "$LOG"
  if [[ "$service" != "active" && "$ade_validation" -lt 2 ]]; then
    printf '{"checked_at":"%s","alert":"main service inactive before ADE validation completed"}\n' "$now" >> "$LOG"
    exit 1
  fi
  [[ "$ade_validation" -eq 2 ]] && exit 0
  sleep 600
done
