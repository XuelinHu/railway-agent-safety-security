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
    candidates: list[dict[str, Any]] = []
    starts = [match.start() for match in re.finditer(r"\{", text)]
    for start in starts:
        try:
            value, _ = json.JSONDecoder().raw_decode(text[start:])
            if isinstance(value, dict):
                candidates.append(value)
        except json.JSONDecodeError:
            repaired = repair_unclosed_json(text[start:])
            if repaired is None:
                continue
            try:
                value = json.loads(repaired)
                if isinstance(value, dict):
                    candidates.append(value)
            except json.JSONDecodeError:
                continue
    if candidates:
        # Generated answers can contain JSON examples or nested evidence objects
        # before the actual annotation. Prefer the complete annotation envelope.
        complete = [item for item in candidates if "entities" in item and "relations" in item]
        if complete:
            return complete[-1]
        raise ValueError("model output contained JSON but no complete annotation envelope")
    raise ValueError("model output did not contain a complete JSON object")


def repair_unclosed_json(text: str) -> str | None:
    """Close only unclosed JSON containers; do not alter parsed content."""
    stack: list[str] = []
    repaired_chars: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            repaired_chars.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            repaired_chars.append(char)
        elif char in "[{":
            stack.append(char)
            repaired_chars.append(char)
        elif char in "]}":
            expected = "[" if char == "]" else "{"
            if stack and stack[-1] == expected:
                stack.pop()
                repaired_chars.append(char)
                continue
            # A truncated array can be followed by the enclosing object's
            # closing brace. Insert only the missing array closers in that case.
            if char == "}" and "{" in stack:
                while stack and stack[-1] == "[":
                    repaired_chars.append("]")
                    stack.pop()
                if stack and stack[-1] == "{":
                    stack.pop()
                    repaired_chars.append("}")
                    continue
            return None
        else:
            repaired_chars.append(char)
    if in_string or not stack:
        return "".join(repaired_chars) if not in_string and not stack else None
    repaired = re.sub(r",\s*$", "", "".join(repaired_chars))
    return repaired + "".join("]" if opener == "[" else "}" for opener in reversed(stack))


def complete_annotation_generated(text: str) -> bool:
    for start in (match.start() for match in re.finditer(r"\{", text)):
        try:
            value, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "entities" in value and "relations" in value:
            return True
    return False


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


COMPACT_INSTRUCTION = """
COMPACT OUTPUT MODE: Return only one JSON object with exactly these top-level fields:
schema_version, document_id, language, entities, relations. Each entity contains only
id, text, type. Each relation contains only id, source_id, type, target_id, claim_status.
Do not include evidence, confidence, review, created_by, explanations, markdown, or any
other fields. Stop immediately after the closing brace of this compact object.
""".strip()


def main(args: argparse.Namespace) -> int:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList

    class CompleteAnnotationCriteria(StoppingCriteria):
        def __init__(self, prompt_length: int) -> None:
            self.prompt_length = prompt_length

        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            generated_text = tokenizer.decode(
                input_ids[0][self.prompt_length :], skip_special_tokens=True
            )
            return complete_annotation_generated(generated_text)

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
        if args.job_id:
            jobs = [job for job in jobs if job["job_id"] == args.job_id]
            if not jobs:
                raise SystemExit(f"job not found: {args.job_id}")
        for index, job in enumerate(jobs, 1):
            started = time.monotonic()
            generated_text = ""
            messages = [
                {
                    "role": "system",
                    "content": job["system_instruction"]
                    + ("\n\n" + COMPACT_INSTRUCTION if args.compact_target else ""),
                },
                {"role": "user", "content": payload(job)},
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_input_tokens).to(device)
            stopping = StoppingCriteriaList()
            if args.stop_on_complete_json:
                stopping.append(CompleteAnnotationCriteria(inputs["input_ids"].shape[1]))
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.eos_token_id,
                    stopping_criteria=stopping,
                )
            generated_text = tokenizer.decode(generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            try:
                annotation = parse_json(generated_text)
            except Exception as error:
                log.write(json.dumps({"job_id": job["job_id"], "status": "failed", "error": str(error), "raw_output": generated_text[:12000]}, ensure_ascii=False) + "\n")
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
    parser.add_argument("--job-id", help="Run one job for a focused generation diagnostic")
    parser.add_argument(
        "--stop-on-complete-json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop as soon as a complete entities/relations JSON object is generated",
    )
    parser.add_argument(
        "--compact-target",
        action="store_true",
        help="Prompt the adapter to emit the compact intermediate schema used during training",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
