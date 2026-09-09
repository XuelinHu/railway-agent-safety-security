#!/usr/bin/env bash
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT="$ROOT/paper/cross-residual/results"
mkdir -p "$OUT"
STATUS="$OUT/experiment_status.json"
if [[ ! -f "$ROOT/data/processed/reviewed/gold/train.jsonl" ]]; then
  printf '{"status":"waiting_for_ignored_reviewed_corpus","message":"Restore data/processed/reviewed before launching model jobs."}\n' > "$STATUS"
  exit 0
fi
printf '{"status":"ready","message":"Reviewed corpus detected; launch the matrix from experiment_matrix.yaml."}\n' > "$STATUS"
