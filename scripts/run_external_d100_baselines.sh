#!/usr/bin/env bash
# Run validation-only, protocol-matched external baselines without reading test data.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

runtime_python="/home/xuelin/miniconda3/envs/rc-llm-comet/bin/python"
assets="data/processed/experiments/formal/low_resource_v2/d100/assets"
validation_assets="$assets/validation"
source_gold="data/processed/reviewed/formal_split/validation.jsonl"
source_gold_index="data/processed/reviewed/formal_split/validation_index.jsonl"
ontology="configs/risk_ontology.yaml"
xlmr="/ds2/xuelin/cache/huggingface/hub/models--xlm-roberta-large/snapshots/c23d21b0620b635a76227c604d44e43a9f0ee389"
bge="/ds2/xuelin/cache/huggingface/hub/models--BAAI--bge-m3/snapshots/9a0624b896d81da7492a910ffa53731274b6cf3d"
result_root="paper/results/external_baselines_d100"
seeds=(20260830 20260831 20260901)

mkdir -p "$result_root/logs" "$result_root/mbe" "$result_root/crf" "$result_root/bge_m3"

evaluate() {
  local predictions="$1"
  local output="$2"
  "$runtime_python" scripts/evaluate_document_level_spans.py \
    --source-gold "$source_gold" \
    --source-gold-index "$source_gold_index" \
    --gold "$validation_assets/gold.jsonl" \
    --gold-index "$validation_assets/index.jsonl" \
    --predictions "$predictions" \
    --jobs "$validation_assets/baseline_jobs.jsonl" \
    --output "$output"
}

run_mbe() {
  for seed in "${seeds[@]}"; do
    local root="$result_root/mbe/seed${seed}"
    mkdir -p "$root"
    CUDA_VISIBLE_DEVICES=0 "$runtime_python" scripts/train_xlmr_ner_baseline.py \
      --train-gold "$assets/gold.jsonl" \
      --train-index "$assets/index.jsonl" \
      --train-jobs "$assets/baseline_jobs.jsonl" \
      --validation-jobs "$validation_assets/baseline_jobs.jsonl" \
      --ontology "$ontology" \
      --model "$xlmr" \
      --output "$root/predictions.jsonl" \
      --model-output "$root/model" \
      --summary "$root/summary.json" \
      --seed "$seed" \
      --decoder greedy \
      --epochs 3 \
      --device cuda
    evaluate "$root/predictions.jsonl" "$root/document_level_metrics.json"
  done
}

run_crf() {
  for seed in "${seeds[@]}"; do
    local root="$result_root/crf/seed${seed}"
    mkdir -p "$root"
    CUDA_VISIBLE_DEVICES="" nice -n 15 "$runtime_python" scripts/train_xlmr_ner_baseline.py \
      --train-gold "$assets/gold.jsonl" \
      --train-index "$assets/index.jsonl" \
      --train-jobs "$assets/baseline_jobs.jsonl" \
      --validation-jobs "$validation_assets/baseline_jobs.jsonl" \
      --ontology "$ontology" \
      --model "$xlmr" \
      --output "$root/predictions.jsonl" \
      --model-output "$root/model" \
      --summary "$root/summary.json" \
      --seed "$seed" \
      --decoder crf \
      --epochs 3 \
      --device cpu
    evaluate "$root/predictions.jsonl" "$root/document_level_metrics.json"
  done
}

run_bge() {
  for seed in "${seeds[@]}"; do
    local mbe_root="$result_root/mbe/seed${seed}"
    local root="$result_root/bge_m3/seed${seed}"
    mkdir -p "$root"
    test -f "$mbe_root/predictions.jsonl"
    CUDA_VISIBLE_DEVICES=0 "$runtime_python" scripts/train_v3_relation_classifier.py \
      --train-gold "$assets/gold.jsonl" \
      --train-index "$assets/index.jsonl" \
      --train-jobs "$assets/baseline_jobs.jsonl" \
      --validation-entities "$mbe_root/predictions.jsonl" \
      --validation-jobs "$validation_assets/baseline_jobs.jsonl" \
      --ontology "$ontology" \
      --embedding-model "$bge" \
      --output "$root" \
      --seed "$seed" \
      --thresholds 0.5 \
      --device cuda
    evaluate "$root/predictions_t0p50.jsonl" "$root/document_level_metrics.json"
  done
}

run_mbe 2>&1 | tee "$result_root/logs/mbe.log"
run_bge 2>&1 | tee "$result_root/logs/bge_m3.log"
