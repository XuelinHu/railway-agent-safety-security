#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT="$ROOT/outputs/public_full_stage1/validation_analysis"
mkdir -p "$OUTPUT"

exec 9>"$OUTPUT/.runner.lock"
exec >>"$OUTPUT/runner.log" 2>&1
if ! flock -n 9; then
  printf '[%s] public validation analysis is already running; leaving it untouched\n' \
    "$(date --iso-8601=seconds)"
  exit 0
fi

cd "$ROOT"
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${PUBLIC_VALIDATION_OMP_THREADS:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

printf '[%s] starting validation-only CPU post-processing\n' \
  "$(date --iso-8601=seconds)"
python3 scripts/analyze_public_full_validation.py \
  --data-root data/processed/public_benchmarks_full \
  --run-root outputs/public_full_stage1 \
  --output outputs/public_full_stage1/validation_analysis \
  --iterations 20000 \
  --seed 20260904 \
  --expected-missing 13
printf '[%s] validation-only CPU post-processing complete\n' \
  "$(date --iso-8601=seconds)"
