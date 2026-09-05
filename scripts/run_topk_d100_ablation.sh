#!/usr/bin/env bash
# Queue a validation-only Top-k semantic-pattern ablation after the external GPU lane.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

runtime_python="/home/xuelin/miniconda3/envs/rc-llm-comet/bin/python"
qwen="/ds2/xuelin/cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
bge="/ds2/xuelin/cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"
assets="data/processed/experiments/formal/low_resource_v2/d100/assets"
validation_assets="$assets/validation"
source_gold="data/processed/reviewed/formal_split/validation.jsonl"
source_gold_index="data/processed/reviewed/formal_split/validation_index.jsonl"
result_root="paper/results/topk_d100_ablation"
upstream_gpu_pid="${UPSTREAM_GPU_PID:?UPSTREAM_GPU_PID must identify the external-baseline GPU lane}"
seeds=(20260830 20260831 20260901)

mkdir -p "$result_root/logs"

# Do not contend with the active MBE/BGE-M3 comparison lane.
while kill -0 "$upstream_gpu_pid" 2>/dev/null; do
  printf '[%s] waiting for external GPU lane pid=%s\n' "$(date --iso-8601=seconds)" "$upstream_gpu_pid"
  sleep 60
done

build_jobs() {
  local k="$1"
  local root="$result_root/k${k}"
  mkdir -p "$root/jobs"
  CUDA_VISIBLE_DEVICES=0 "$runtime_python" scripts/build_kg_v2_jobs.py \
    --jobs "$assets/baseline_jobs.jsonl" \
    --concepts "$assets/knowledge_graph/concepts.jsonl" \
    --relations "$assets/knowledge_graph/relations.jsonl" \
    --output "$root/jobs/train.jsonl" \
    --audit "$root/jobs/train_audit.json" \
    --semantic-model "$bge" --device cuda --batch-size 16 \
    --semantic-threshold 0.72 --semantic-limit "$k" \
    --anchor-limit 12 --anchor-per-type 2 --edge-limit 6 \
    --min-type-purity 0.8 --min-en-chars 4 --min-zh-chars 2
  CUDA_VISIBLE_DEVICES=0 "$runtime_python" scripts/build_kg_v2_jobs.py \
    --jobs "$validation_assets/baseline_jobs.jsonl" \
    --concepts "$assets/knowledge_graph/concepts.jsonl" \
    --relations "$assets/knowledge_graph/relations.jsonl" \
    --output "$root/jobs/validation.jsonl" \
    --audit "$root/jobs/validation_audit.json" \
    --semantic-model "$bge" --device cuda --batch-size 16 \
    --semantic-threshold 0.72 --semantic-limit "$k" \
    --anchor-limit 12 --anchor-per-type 2 --edge-limit 6 \
    --min-type-purity 0.8 --min-en-chars 4 --min-zh-chars 2
}

assert_no_truncation() {
  local jobs="$1"
  "$runtime_python" - "$jobs" "$qwen" <<'PY'
import sys
from pathlib import Path
from scripts.train_qlora import build_examples
from transformers import AutoTokenizer

jobs, model = map(Path, sys.argv[1:])
root = jobs.parent.parent
assets = Path("data/processed/experiments/formal/low_resource_v2/d100/assets")
examples = build_examples(assets / "gold.jsonl", assets / "index.jsonl", jobs, compact_target=True, use_job_instruction=True)
tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=True)
lengths = []
for item in examples:
    prompt = tokenizer.apply_chat_template([
        {"role": "system", "content": item["system"]},
        {"role": "user", "content": item["user"]},
    ], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    lengths.append(len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) + len(tokenizer(item["target"] + "<|im_end|>", add_special_tokens=False)["input_ids"]))
maximum = max(lengths, default=0)
if maximum > 4096:
    raise SystemExit(f"Top-k token gate failed: max complete sequence {maximum} > 4096")
print(f"Top-k token gate passed: {len(lengths)} examples, max complete sequence {maximum}")
PY
}

evaluate() {
  local predictions="$1"
  local jobs="$2"
  local output="$3"
  "$runtime_python" scripts/evaluate_document_level_spans.py \
    --source-gold "$source_gold" --source-gold-index "$source_gold_index" \
    --gold "$validation_assets/gold.jsonl" --gold-index "$validation_assets/index.jsonl" \
    --predictions "$predictions" --jobs "$jobs" --output "$output"
}

run_setting() {
  local k="$1"
  local root="$result_root/k${k}"
  build_jobs "$k"
  assert_no_truncation "$root/jobs/train.jsonl"
  for seed in "${seeds[@]}"; do
    local run="$root/seed${seed}"
    mkdir -p "$run/validation"
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$runtime_python" scripts/train_qlora.py \
      --model-path "$qwen" --gold "$assets/gold.jsonl" --index "$assets/index.jsonl" \
      --jobs "$root/jobs/train.jsonl" --output "$run" --epochs 1 --batch-size 1 \
      --gradient-accumulation 4 --learning-rate 0.0002 --max-length 4096 \
      --lora-rank 8 --lora-alpha 16 --seed "$seed" --bucket-by-length \
      --length-bucket-size 32 --compact-target --use-job-instruction
    CUDA_VISIBLE_DEVICES=0 "$runtime_python" scripts/run_qlora_inference.py \
      --model-path "$qwen" --adapter "$run" --jobs "$root/jobs/validation.jsonl" \
      --output "$run/validation/predictions_compact.jsonl" --log "$run/validation/inference.log" \
      --max-input-tokens 12288 --max-new-tokens 4096 --max-seconds-per-job 180 \
      --max-repeated-entity 6 --stop-on-complete-json --compact-target --use-job-instruction
    "$runtime_python" scripts/expand_compact_predictions.py \
      --jobs "$root/jobs/validation.jsonl" --predictions "$run/validation/predictions_compact.jsonl" \
      --output "$run/validation/predictions_expanded.jsonl" --errors "$run/validation/expansion_errors.jsonl"
    evaluate "$run/validation/predictions_expanded.jsonl" "$root/jobs/validation.jsonl" "$run/validation/document_level_metrics.json"
  done
}

run_setting 2 2>&1 | tee "$result_root/logs/k2.log"
run_setting 8 2>&1 | tee "$result_root/logs/k8.log"
run_setting 16 2>&1 | tee "$result_root/logs/k16.log"
