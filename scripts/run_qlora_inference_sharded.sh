#!/usr/bin/env bash
# Run independent QLoRA inference shards, then merge them in source-job order.
# Each worker owns its output files so concurrent writers never touch one JSONL.
set -euo pipefail

workers=2
if [[ "${1:-}" == "--workers" ]]; then
  workers="$2"
  shift 2
fi
if (( workers < 1 )); then
  echo "--workers must be positive" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

args=()
jobs=""
output=""
log=""
while (( $# )); do
  case "$1" in
    --jobs) jobs="$2"; args+=("$1" "$2"); shift 2 ;;
    --output) output="$2"; shift 2 ;;
    --log) log="$2"; shift 2 ;;
    *) args+=("$1"); shift ;;
  esac
done
if [[ -z "$jobs" || -z "$output" || -z "$log" ]]; then
  echo "--jobs, --output, and --log are required" >&2
  exit 2
fi

total="$(awk 'NF {count++} END {print count + 0}' "$jobs")"
chunk=$(( (total + workers - 1) / workers ))
mkdir -p "$(dirname "$output")"
pids=()
for (( worker=0; worker<workers; worker++ )); do
  offset=$(( worker * chunk ))
  (( offset >= total )) && break
  part_output="${output}.part${worker}"
  part_log="${log}.part${worker}"
  python3 scripts/run_qlora_inference.py "${args[@]}" --output "$part_output" --log "$part_log" \
    --offset "$offset" --limit "$chunk" &
  pids+=("$!")
done
worker_failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    worker_failed=1
  fi
done
(( worker_failed == 0 )) || exit 1

python3 scripts/merge_qlora_shards.py \
  --jobs "$jobs" \
  --output "$output" \
  --log "$log" \
  --workers "$workers"
