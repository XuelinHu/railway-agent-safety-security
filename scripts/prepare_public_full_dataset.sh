#!/usr/bin/env bash
set -euo pipefail
dataset="$1"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
base="data/processed/public_benchmarks_full/$dataset"
mkdir -p outputs/public_full_stage1
for split in train validation test; do
  for mode in baseline kg_constrained; do
    suffix=baseline
    [[ "$mode" == kg_constrained ]] && suffix=kg
    python3 scripts/build_experiment_jobs.py --jobs "$base/jobs.jsonl" --manifest "$base/split_manifest.jsonl" \
      --mentions "$base/knowledge_graph/mentions.jsonl" --ontology "$base/ontology.yaml" --split "$split" --mode "$mode" \
      --output "$base/${split}_${suffix}_jobs.jsonl"
  done
done
echo "prep_complete dataset=$dataset"
