#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/xuelin/miniconda3/bin:$PATH"
export HF_ENDPOINT="https://hf-mirror.com"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
while systemctl --user is-active --quiet public-smoke-qwen-40-8-8.service; do sleep 30; done
for dataset in conll04 scierc ade; do
  base="data/processed/public_benchmarks_smoke_40_8_8/$dataset"
  for split in validation test; do
    python3 scripts/run_gliner_public_smoke.py --jobs "$base/${split}_baseline_jobs.jsonl" --ontology "$base/ontology.yaml" --output "outputs/public_smoke_40_8_8/${dataset}_gliner_${split}.jsonl"
    python3 scripts/evaluate_annotations.py --gold "$base/${split}_gold.jsonl" --gold-index "$base/${split}_index.jsonl" --predictions "outputs/public_smoke_40_8_8/${dataset}_gliner_${split}.jsonl" --include-missing-as-empty --output "outputs/public_smoke_40_8_8/${dataset}_gliner_${split}_metrics.json"
  done
done
