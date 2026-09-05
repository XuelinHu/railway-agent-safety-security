#!/usr/bin/env bash
# Low-priority CPU CRF companion job for the D100 external-baseline protocol.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

runtime_python="/home/xuelin/miniconda3/envs/rc-llm-comet/bin/python"
assets="data/processed/experiments/formal/low_resource_v2/d100/assets"
validation_assets="$assets/validation"
source_gold="data/processed/reviewed/formal_split/validation.jsonl"
source_gold_index="data/processed/reviewed/formal_split/validation_index.jsonl"
xlmr="/ds2/xuelin/cache/huggingface/hub/models--xlm-roberta-large/snapshots/c23d21b0620b635a76227c604d44e43a9f0ee389"
result_root="paper/results/external_baselines_d100"

for seed in 20260830 20260831 20260901; do
  root="$result_root/crf/seed${seed}"
  mkdir -p "$root"
  CUDA_VISIBLE_DEVICES="" nice -n 15 "$runtime_python" scripts/train_xlmr_ner_baseline.py \
    --train-gold "$assets/gold.jsonl" \
    --train-index "$assets/index.jsonl" \
    --train-jobs "$assets/baseline_jobs.jsonl" \
    --validation-jobs "$validation_assets/baseline_jobs.jsonl" \
    --model "$xlmr" \
    --output "$root/predictions.jsonl" \
    --model-output "$root/model" \
    --summary "$root/summary.json" \
    --seed "$seed" \
    --decoder crf \
    --epochs 3 \
    --device cpu
  "$runtime_python" scripts/evaluate_document_level_spans.py \
    --source-gold "$source_gold" \
    --source-gold-index "$source_gold_index" \
    --gold "$validation_assets/gold.jsonl" \
    --gold-index "$validation_assets/index.jsonl" \
    --predictions "$root/predictions.jsonl" \
    --jobs "$validation_assets/baseline_jobs.jsonl" \
    --output "$root/document_level_metrics.json"
done
