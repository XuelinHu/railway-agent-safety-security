#!/usr/bin/env bash
# Non-blocking matrix runner. It deliberately stays alive until the deadline.
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT="$ROOT/paper/cross-residual/results"
mkdir -p "$OUT"
STATUS="$OUT/experiment_status.json"
LOG="$OUT/experiment_runner.log"
HOURS="${CROSS_RESIDUAL_HOURS:-3}"
DEADLINE=$(( $(date +%s) + HOURS*3600 ))
MODEL="${CROSS_RESIDUAL_MODEL_PATH:-}"
GOLD="$ROOT/data/processed/reviewed/gold/train.jsonl"
INDEX="$ROOT/data/processed/reviewed/index.jsonl"
JOBS="$ROOT/data/processed/reviewed/jobs.jsonl"

echo "$(date -Is) matrix start; deadline=$(date -d @$DEADLINE -Is)" >> "$LOG"
while (( $(date +%s) < DEADLINE )); do
  now=$(date +%s); remaining=$((DEADLINE-now))
  if [[ ! -s "$GOLD" || ! -s "$INDEX" || ! -s "$JOBS" || -z "$MODEL" || ! -d "$MODEL" ]]; then
    printf '{"status":"waiting","remaining_seconds":%s,"reason":"data_or_model_unavailable"}\n' "$remaining" > "$STATUS"
    echo "$(date -Is) waiting for reviewed data/model; remaining=${remaining}s" >> "$LOG"
    sleep 60
    continue
  fi
  for variant in source_only graph_concat cross_residual; do
    (( $(date +%s) >= DEADLINE )) && break 2
    out="$OUT/$variant"
    mkdir -p "$out"
    printf '{"status":"running","variant":"%s","remaining_seconds":%s}\n' "$variant" "$((DEADLINE-$(date +%s)))" > "$STATUS"
    # Small, repeatable run; each invocation is bounded by the global deadline.
    timeout "$((DEADLINE-$(date +%s)))" python3 "$ROOT/scripts/train_qlora.py" \
      --model-path "$MODEL" --gold "$GOLD" --index "$INDEX" --jobs "$JOBS" \
      --output "$out" --epochs 1 --batch-size 1 --gradient-accumulation 4 \
      --max-length 512 --compact-target >> "$LOG" 2>&1 || true
  done
done
printf '{"status":"finished","finished_at":"%s"}\n' "$(date -Is)" > "$STATUS"
echo "$(date -Is) matrix finished" >> "$LOG"
