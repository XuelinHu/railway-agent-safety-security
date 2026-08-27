#!/usr/bin/env python3
"""Run local QLoRA adapter inference for experiment jobs."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_json(text: str) -> dict[str, Any]:
    starts = [match.start() for match in re.finditer(r"\{", text)]
    for start in starts:
        try:
            value, _ = json.JSONDecoder().raw_decode(text[start:])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    raise ValueError("model output did not contain a complete JSON object")


def payload(job: dict[str, Any]) -> str:
    return json.dumps(
        {
            "document_id": job["document_id"],
            "language": job["language"],
            "teacher_model": job.get("teacher_model", "qwen3-4b-qlora"),
            "ontology": job["ontology"],
            "segments": job["segments"],
        },
        ensure_ascii=False,
    )


def main(args: argparse.Namespace) -> int:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.adapter, local_files_only=True, trust_remote_code=True)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, args.adapter, local_files_only=True)
    model.eval()
    device = next(model.parameters()).device
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output, args.log.open("w", encoding="utf-8") as log:
        jobs = load_jsonl(args.jobs)
        for index, job in enumerate(jobs, 1):
            started = time.monotonic()
            messages = [
                {"role": "system", "content": job["system_instruction"]},
                {"role": "user", "content": payload(job)},
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_input_tokens).to(device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            try:
                annotation = parse_json(text)
            except Exception as error:
                log.write(json.dumps({"job_id": job["job_id"], "status": "failed", "error": str(error), "raw_tail": text[-500:]}, ensure_ascii=False) + "\n")
                log.flush()
                print(f"{index}/{len(jobs)} {job['job_id']} failed: {error}", flush=True)
                continue
            annotation.setdefault("schema_version", "0.1.0")
            annotation.setdefault("document_id", job["document_id"])
            annotation.setdefault("language", job["language"])
            annotation.setdefault("review", {"status": "unreviewed", "reviewers": [], "notes": "qlora inference"})
            for entity in annotation.get("entities", []):
                entity.setdefault("created_by", "qwen3-4b-qlora")
                entity.setdefault("review_status", "pending")
            for relation in annotation.get("relations", []):
                relation.setdefault("created_by", "qwen3-4b-qlora")
                relation.setdefault("review_status", "pending")
            output.write(json.dumps({"job_id": job["job_id"], "annotation": annotation}, ensure_ascii=False) + "\n")
            output.flush()
            log.write(json.dumps({"job_id": job["job_id"], "status": "success", "elapsed_seconds": round(time.monotonic() - started, 3)}, ensure_ascii=False) + "\n")
            log.flush()
            print(f"{index}/{len(jobs)} {job['job_id']} success", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--max-input-tokens", type=int, default=12288)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
