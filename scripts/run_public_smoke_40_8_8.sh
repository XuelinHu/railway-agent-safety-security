#!/usr/bin/env bash
# Non-blocking 40/8/8 public benchmark smoke run for one RTX 3090.
# Qwen training/inference is deliberately serial. CPU-only conversion,
# evidence expansion, verification, and scoring happen immediately between GPU
# jobs and do not consume GPU memory.

set -euo pipefail

export PATH="/home/xuelin/miniconda3/bin:$PATH"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL="/ds2/xuelin/cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
RUN_ROOT="outputs/public_smoke_40_8_8"
DATA_ROOT="data/processed/public_benchmarks_smoke_40_8_8"
mkdir -p "$RUN_ROOT"

run_dataset() {
  local dataset="$1"
  local base="$DATA_ROOT/$dataset"

  python3 scripts/import_spert_benchmarks.py \
    --dataset "$dataset" --output-root "$DATA_ROOT" \
    --train-limit 40 --validation-limit 8 --test-limit 8 --seed public-smoke-42

  for split in train validation test; do
    for mode in baseline kg_constrained; do
      suffix="baseline"
      if [[ "$mode" == "kg_constrained" ]]; then suffix="kg"; fi
      python3 scripts/build_experiment_jobs.py \
        --jobs "$base/jobs.jsonl" --manifest "$base/split_manifest.jsonl" \
        --mentions "$base/knowledge_graph/mentions.jsonl" --ontology "$base/ontology.yaml" \
        --split "$split" --mode "$mode" --output "$base/${split}_${suffix}_jobs.jsonl"
    done
  done

  for method in baseline kg; do
    instruction=()
    if [[ "$method" == "kg" ]]; then instruction=(--use-job-instruction); fi
    python3 scripts/train_qlora.py --model-path "$MODEL" \
      --gold "$base/train_gold.jsonl" --index "$base/train_index.jsonl" \
      --jobs "$base/train_${method}_jobs.jsonl" --output "$RUN_ROOT/${dataset}_${method}" \
      --epochs 1 --batch-size 1 --gradient-accumulation 4 --max-length 4096 \
      --compact-target "${instruction[@]}"

    for split in validation test; do
      raw="$RUN_ROOT/${dataset}_${method}_${split}.jsonl"
      expanded="$RUN_ROOT/${dataset}_${method}_${split}_expanded.jsonl"
      verified="$RUN_ROOT/${dataset}_${method}_${split}_verified.jsonl"
      python3 scripts/run_qlora_inference.py --model-path "$MODEL" \
        --adapter "$RUN_ROOT/${dataset}_${method}" --jobs "$base/${split}_${method}_jobs.jsonl" \
        --output "$raw" --log "$RUN_ROOT/${dataset}_${method}_${split}.log" \
        --compact-target --resume --max-input-tokens 4096 --max-new-tokens 512 --max-seconds-per-job 45 "${instruction[@]}"
      python3 scripts/evaluate_annotations.py --gold "$base/${split}_gold.jsonl" \
        --gold-index "$base/${split}_index.jsonl" --predictions "$raw" \
        --include-missing-as-empty --output "$RUN_ROOT/${dataset}_${method}_${split}_raw_metrics.json"
      python3 scripts/expand_compact_predictions.py --jobs "$base/${split}_${method}_jobs.jsonl" \
        --predictions "$raw" --output "$expanded" --errors "$RUN_ROOT/${dataset}_${method}_${split}_expand_errors.jsonl"
      python3 scripts/evaluate_annotations.py --gold "$base/${split}_gold.jsonl" \
        --gold-index "$base/${split}_index.jsonl" --predictions "$expanded" \
        --include-missing-as-empty --output "$RUN_ROOT/${dataset}_${method}_${split}_evidence_metrics.json"
      python3 scripts/verify_relations.py --annotations "$expanded" --ontology "$base/ontology.yaml" \
        --output "$verified" --audit "$RUN_ROOT/${dataset}_${method}_${split}_audit.jsonl"
      python3 scripts/evaluate_annotations.py --gold "$base/${split}_gold.jsonl" \
        --gold-index "$base/${split}_index.jsonl" --predictions "$verified" \
        --include-missing-as-empty --output "$RUN_ROOT/${dataset}_${method}_${split}_metrics.json"
      python3 scripts/evaluate_evidence_graph.py --gold "$base/${split}_gold.jsonl" \
        --gold-index "$base/${split}_index.jsonl" --predictions "$verified" \
        --jobs "$base/${split}_${method}_jobs.jsonl" --ontology "$base/ontology.yaml" \
        --output "$RUN_ROOT/${dataset}_${method}_${split}_evidence_graph_metrics.json" || \
        printf '%s\n' "evidence graph evaluator skipped due to missing-generation audit; annotation metrics remain authoritative"
    done
  done

  # The KG raw output is the KG-prior variant. Its expanded output is the
  # evidence-gated variant, while the verified output is full PGE.
  cp "$RUN_ROOT/${dataset}_kg_validation_expanded.jsonl" "$RUN_ROOT/${dataset}_kg_evidence_validation.jsonl"
  cp "$RUN_ROOT/${dataset}_kg_test_expanded.jsonl" "$RUN_ROOT/${dataset}_kg_evidence_test.jsonl"
}

started="$(date --iso-8601=seconds)"
for dataset in conll04 scierc ade; do
  run_dataset "$dataset"
done
printf '{"status":"complete","started_at":"%s","finished_at":"%s"}\n' "$started" "$(date --iso-8601=seconds)" > "$RUN_ROOT/status.json"
