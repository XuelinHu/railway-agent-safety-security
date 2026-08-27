#!/usr/bin/env python3
"""Train a small QLoRA adapter for evidence-grounded safety annotation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_examples(gold: Path, index: Path, jobs: Path, compact_target: bool = False) -> list[dict[str, str]]:
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
                "teacher_model": "qlora-student",
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
        examples.append({"system": job["system_instruction"], "user": user, "target": target})
    return examples


def main(args: argparse.Namespace) -> int:
    import torch
    from accelerate import Accelerator
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    examples = build_examples(args.gold, args.index, args.jobs, args.compact_target)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
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
    for example in examples:
        prompt = (
            "<|im_start|>system\n"
            + example["system"]
            + "<|im_end|>\n<|im_start|>user\n"
            + example["user"]
            + "<|im_end|>\n<|im_start|>assistant\n"
        )
        full = prompt + example["target"] + "<|im_end|>"
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(example["target"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
        if len(answer_ids) >= args.max_length:
            answer_ids = answer_ids[: args.max_length - 1]
        prompt_budget = args.max_length - len(answer_ids)
        if prompt_budget < 1:
            continue
        # Keep the beginning of the prompt and the complete available answer so
        # truncation never creates a batch with zero supervised tokens.
        prompt_ids = prompt_ids[:prompt_budget]
        full_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        encoded.append({"input_ids": full_ids, "labels": labels})

    if not encoded:
        raise ValueError("no train examples contain supervised answer tokens")
    print(json.dumps({"prepared_examples": len(encoded), "max_sequence": max(len(row["input_ids"]) for row in encoded)}, ensure_ascii=False), flush=True)

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

    loader = DataLoader(encoded, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    model.train()
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
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                steps += 1
                losses.append(float(loss.detach().cpu()))
                if steps % 5 == 0:
                    print(json.dumps({"epoch": epoch + 1, "step": steps, "loss": round(losses[-1], 5)}, ensure_ascii=False), flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    accelerator.unwrap_model(model).save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    metrics = {
        "base_model": str(args.model_path),
        "train_examples": len(examples),
        "epochs": args.epochs,
        "steps": steps,
        "final_loss": losses[-1] if losses else None,
        "mean_loss": sum(losses) / len(losses) if losses else None,
        "perplexity_estimate": math.exp(min(sum(losses) / len(losses), 20)) if losses else None,
        "max_length": args.max_length,
        "lora_rank": args.lora_rank,
        "compact_target": args.compact_target,
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
    parser.add_argument("--compact-target", action="store_true", help="Train a short entity/relation intermediate format")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
