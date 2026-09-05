#!/usr/bin/env python3
"""Train a small QLoRA adapter for evidence-grounded safety annotation."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


COMPACT_INSTRUCTION = """
COMPACT OUTPUT MODE: Return only one JSON object with exactly these top-level fields:
schema_version, document_id, language, entities, relations. Each entity contains only
id, text, type. Each relation contains only id, source_id, type, target_id, claim_status.
Do not include evidence, confidence, review, created_by, explanations, markdown, or any
other fields. Stop immediately after the closing brace of this compact object.
""".strip()


COMPACT_SYSTEM_INSTRUCTION = """
You are an evidence-grounded risk annotation model. Read the supplied document segments
and ontology, then identify only entities and relations supported by the segment text.
Return only one valid JSON object in COMPACT OUTPUT MODE with exactly these top-level
fields: schema_version, document_id, language, entities, relations. Each entity contains
only id, text, type. Each relation contains only id, source_id, type, target_id,
claim_status. Entity text must be copied exactly from a segment. Use only ontology entity
types and legal ontology relation signatures. Do not include evidence, confidence,
review, created_by, explanations, markdown, or any other fields. Stop immediately after
the closing brace of the compact object.
""".strip()


def compact_system(job: dict[str, Any]) -> str:
    """Keep only the KG suffix when constructing the compact training prompt."""
    kg_rules = job["system_instruction"].split("\n\nKG_RULES:", 1)
    if len(kg_rules) == 2:
        return f"{COMPACT_SYSTEM_INSTRUCTION}\n\nKG_RULES:{kg_rules[1]}\n\n{COMPACT_INSTRUCTION}"
    return f"{job['system_instruction']}\n\n{COMPACT_INSTRUCTION}"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_examples(
    gold: Path,
    index: Path,
    jobs: Path,
    compact_target: bool = False,
    use_job_instruction: bool = False,
) -> list[dict[str, str]]:
    annotations = load_jsonl(gold)
    indexes = load_jsonl(index)
    jobs_by_id = {row["job_id"]: row for row in load_jsonl(jobs)}
    if len(annotations) != len(indexes):
        raise ValueError("gold and index record counts differ")
    examples = []
    for annotation, record in zip(annotations, indexes):
        job = jobs_by_id[record["job_id"]]
        user = json.dumps(
            {
                "document_id": job["document_id"],
                "language": job["language"],
                "teacher_model": job.get("teacher_model", "qwen3-4b-qlora"),
                "ontology": job["ontology"],
                "segments": job["segments"],
            },
            ensure_ascii=False,
        )
        if compact_target:
            compact = {
                "schema_version": annotation["schema_version"],
                "document_id": annotation["document_id"],
                "language": annotation["language"],
                "entities": [
                    {"id": e["id"], "text": e["text"], "type": e["type"]} for e in annotation.get("entities", [])
                ],
                "relations": [
                    {
                        "id": r["id"],
                        "source_id": r["source_id"],
                        "type": r["type"],
                        "target_id": r["target_id"],
                        "claim_status": r["claim_status"],
                    }
                    for r in annotation.get("relations", [])
                ],
            }
            target = json.dumps(compact, ensure_ascii=False)
        else:
            target = json.dumps(annotation, ensure_ascii=False)
        if compact_target and use_job_instruction:
            system = compact_system(job)
        else:
            system = COMPACT_SYSTEM_INSTRUCTION if compact_target else job["system_instruction"]
        examples.append({"system": system, "user": user, "target": target})
    return examples


def main(args: argparse.Namespace) -> int:
    import torch
    from accelerate import Accelerator
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    process_started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA training requires CUDA")
    torch.cuda.reset_peak_memory_stats()
    examples = build_examples(
        args.gold,
        args.index,
        args.jobs,
        args.compact_target,
        args.use_job_instruction,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    torch.manual_seed(args.seed)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        ),
    )
    model.print_trainable_parameters()

    device = torch.device("cuda:0")
    encoded = []
    prompt_token_count = 0
    target_token_count = 0
    truncated_answers = 0
    truncated_prompts = 0
    skipped_overlength = 0
    for example in examples:
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": example["system"]},
                {"role": "user", "content": example["user"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        full = prompt + example["target"] + "<|im_end|>"
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(example["target"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
        if len(answer_ids) >= args.max_length:
            end_ids = tokenizer("<|im_end|>", add_special_tokens=False)["input_ids"]
            answer_ids = answer_ids[: args.max_length - len(end_ids)] + end_ids
            truncated_answers += 1
        prompt_budget = args.max_length - len(answer_ids)
        if prompt_budget < 1:
            continue
        if len(prompt_ids) > prompt_budget and args.skip_overlength:
            skipped_overlength += 1
            continue
        # Keep the beginning of the prompt and the complete available answer so
        # truncation never creates a batch with zero supervised tokens.
        if len(prompt_ids) > prompt_budget:
            truncated_prompts += 1
        prompt_ids = prompt_ids[:prompt_budget]
        full_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        encoded.append({"input_ids": full_ids, "labels": labels})
        prompt_token_count += len(prompt_ids)
        target_token_count += len(answer_ids)

    if not encoded:
        raise ValueError("no train examples contain supervised answer tokens")
    print(json.dumps({
        "prepared_examples": len(encoded),
        "max_sequence": max(len(row["input_ids"]) for row in encoded),
        "truncated_answers": truncated_answers,
        "truncated_prompts": truncated_prompts,
        "skipped_overlength": skipped_overlength,
    }, ensure_ascii=False), flush=True)

    def collate(batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(row["input_ids"]) for row in batch)
        input_ids = [row["input_ids"] + [tokenizer.pad_token_id] * (max_len - len(row["input_ids"])) for row in batch]
        labels = [row["labels"] + [-100] * (max_len - len(row["labels"])) for row in batch]
        attention = [[1] * len(row["input_ids"]) + [0] * (max_len - len(row["input_ids"])) for row in batch]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }

    generator = torch.Generator().manual_seed(args.seed)
    shuffle = True
    if args.bucket_by_length:
        ordered = sorted(encoded, key=lambda row: len(row["input_ids"]), reverse=True)
        buckets = [ordered[index : index + args.length_bucket_size] for index in range(0, len(ordered), args.length_bucket_size)]
        rng = random.Random(args.seed)
        for bucket in buckets:
            rng.shuffle(bucket)
        if len(buckets) > 1:
            tail = buckets[1:]
            rng.shuffle(tail)
            buckets = [buckets[0], *tail]
        encoded = [row for bucket in buckets for row in bucket]
        shuffle = False
    loader = DataLoader(
        encoded,
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        generator=generator,
    )
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    model.train()
    torch.cuda.synchronize(device)
    training_started = time.perf_counter()
    steps = 0
    losses = []
    for epoch in range(args.epochs):
        for batch in loader:
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
                loss = outputs.loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite training loss; check truncation and labels")
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                steps += 1
                losses.append(float(loss.detach().cpu()))
                if steps % 5 == 0:
                    print(json.dumps({"epoch": epoch + 1, "step": steps, "loss": round(losses[-1], 5)}, ensure_ascii=False), flush=True)

    torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - training_started

    args.output.mkdir(parents=True, exist_ok=True)
    accelerator.unwrap_model(model).save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    torch.cuda.synchronize(device)
    wall_clock_seconds = time.perf_counter() - process_started
    metrics = {
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": str(args.model_path),
        "source_examples": len(examples),
        "train_examples": len(encoded),
        "epochs": args.epochs,
        "steps": steps,
        "final_loss": losses[-1] if losses else None,
        "mean_loss": sum(losses) / len(losses) if losses else None,
        "perplexity_estimate": math.exp(min(sum(losses) / len(losses), 20)) if losses else None,
        "max_length": args.max_length,
        "lora_rank": args.lora_rank,
        "compact_target": args.compact_target,
        "use_job_instruction": args.use_job_instruction,
        "seed": args.seed,
        "bucket_by_length": args.bucket_by_length,
        "truncated_answers_with_eos": truncated_answers,
        "truncated_prompts": truncated_prompts,
        "skipped_overlength": skipped_overlength,
        "prompt_tokens": prompt_token_count,
        "target_tokens": target_token_count,
        "total_tokens": prompt_token_count + target_token_count,
        "training_wall_clock_seconds": round(training_seconds, 3),
        "process_wall_clock_seconds": round(wall_clock_seconds, 3),
        "examples_per_training_second": round(len(encoded) / training_seconds, 6),
        "tokens_per_training_second": round(
            (prompt_token_count + target_token_count) / training_seconds, 3
        ),
        "device_name": torch.cuda.get_device_name(device),
        "peak_cuda_memory_allocated_mib": round(
            torch.cuda.max_memory_allocated(device) / (1024**2), 2
        ),
        "peak_cuda_memory_reserved_mib": round(
            torch.cuda.max_memory_reserved(device) / (1024**2), 2
        ),
        "software": {
            "python": __import__("sys").version.split()[0],
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "peft": __import__("peft").__version__,
            "accelerate": __import__("accelerate").__version__,
            "bitsandbytes": __import__("bitsandbytes").__version__,
            "cuda": torch.version.cuda,
        },
    }
    (args.output / "training_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--gold", type=Path, default=Path("data/processed/reviewed/gold/train.jsonl"))
    parser.add_argument("--index", type=Path, default=Path("data/processed/reviewed/gold/train_index.jsonl"))
    parser.add_argument("--jobs", type=Path, default=Path("data/processed/experiments/train_kg_jobs.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/experiments/qwen3_4b_kg_qlora"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--bucket-by-length",
        action="store_true",
        help="Process the longest length bucket first and shuffle within/between buckets",
    )
    parser.add_argument("--length-bucket-size", type=int, default=32)
    parser.add_argument("--compact-target", action="store_true", help="Train a short entity/relation intermediate format")
    parser.add_argument(
        "--skip-overlength",
        action="store_true",
        help="Skip examples whose complete prompt and answer exceed max-length instead of truncating source context",
    )
    parser.add_argument(
        "--use-job-instruction",
        action="store_true",
        help="Keep each job's system instruction, including KG constraints, for compact-target training",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
