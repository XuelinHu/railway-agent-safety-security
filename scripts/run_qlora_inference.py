#!/usr/bin/env python3
"""Run local QLoRA adapter inference for experiment jobs."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prepare_jsonl_for_append(path: Path) -> None:
    """Drop only a truncated final record and ensure appends start on a new line."""
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    if not text:
        return
    lines = text.splitlines()
    nonempty = [index for index, line in enumerate(lines) if line.strip()]
    if not nonempty:
        path.write_text("", encoding="utf-8")
        return
    last_nonempty = nonempty[-1]
    truncate_at: int | None = None
    for index in nonempty:
        try:
            json.loads(lines[index])
        except json.JSONDecodeError as error:
            if index != last_nonempty:
                raise ValueError(f"{path}:{index + 1}: invalid JSON before the final row") from error
            truncate_at = index
    if truncate_at is not None:
        lines = lines[:truncate_at]
    normalized = "\n".join(line for line in lines if line.strip())
    temporary = path.with_name(f".{path.name}.resume.tmp")
    temporary.write_text(normalized + ("\n" if normalized else ""), encoding="utf-8")
    temporary.replace(path)


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
    # Small causal decoders sometimes omit the closing bracket of the first
    # top-level array immediately before emitting the next top-level key.
    text = re.sub(r'}\s*,\s*("relations"\s*:)', r'}], \1', text, count=1)
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


def repeated_entity_text(text: str, threshold: int) -> str | None:
    """Return an entity text repeated enough to indicate a generation loop."""
    if threshold <= 0:
        return None
    values = [
        bytes(value, "utf-8").decode("unicode_escape") if "\\" in value else value
        for value in re.findall(r'"text"\s*:\s*"((?:\\.|[^"\\])*)"', text)
    ]
    counts = Counter(values)
    repeated = [value for value, count in counts.items() if count >= threshold]
    return repeated[0] if repeated else None


def completed_job_ids(output_path: Path, log_path: Path, retry_failed: bool) -> set[str]:
    """Collect terminal jobs without failing on a truncated final JSONL line."""
    completed: set[str] = set()
    for path, is_log in ((output_path, False), (log_path, True)):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            job_id = row.get("job_id")
            if not job_id:
                continue
            if not is_log or row.get("status") == "success" or not retry_failed:
                completed.add(job_id)
    return completed


def system_instruction(job: dict[str, Any], compact_target: bool, use_job_instruction: bool) -> str:
    if not compact_target:
        return job["system_instruction"]
    if use_job_instruction:
        kg_rules = job["system_instruction"].split("\n\nKG_RULES:", 1)
        if len(kg_rules) == 2:
            return f"{COMPACT_SYSTEM_INSTRUCTION}\n\nKG_RULES:{kg_rules[1]}\n\n{COMPACT_INSTRUCTION}"
        return f"{job['system_instruction']}\n\n{COMPACT_INSTRUCTION}"
    return COMPACT_SYSTEM_INSTRUCTION


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

    class ElapsedTimeCriteria(StoppingCriteria):
        def __init__(self, max_seconds: float) -> None:
            self.started = time.monotonic()
            self.max_seconds = max_seconds
            self.triggered = False

        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            self.triggered = time.monotonic() - self.started >= self.max_seconds
            return self.triggered

    class RepeatedEntityCriteria(StoppingCriteria):
        def __init__(self, prompt_length: int, threshold: int) -> None:
            self.prompt_length = prompt_length
            self.threshold = threshold
            self.triggered = False
            self.repeated_text: str | None = None

        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            generated_tokens = input_ids.shape[1] - self.prompt_length
            if generated_tokens < 128 or generated_tokens % 16:
                return False
            generated_text = tokenizer.decode(
                input_ids[0][self.prompt_length :], skip_special_tokens=True
            )
            self.repeated_text = repeated_entity_text(generated_text, self.threshold)
            self.triggered = self.repeated_text is not None
            return self.triggered

    tokenizer_source = args.adapter or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, local_files_only=True, trust_remote_code=True
    )
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
    model = (
        PeftModel.from_pretrained(base, args.adapter, local_files_only=True)
        if args.adapter
        else base
    )
    model.eval()
    created_by = "qwen3-4b-qlora" if args.adapter else "qwen3-4b-zero-shot"
    device = next(model.parameters()).device
    args.output.parent.mkdir(parents=True, exist_ok=True)
    jobs = load_jsonl(args.jobs)
    if args.job_id:
        jobs = [job for job in jobs if job["job_id"] == args.job_id]
        if not jobs:
            raise SystemExit(f"job not found: {args.job_id}")
    if args.offset:
        jobs = jobs[args.offset :]
    if args.limit:
        jobs = jobs[: args.limit]
    if args.resume:
        prepare_jsonl_for_append(args.output)
        prepare_jsonl_for_append(args.log)
    completed = completed_job_ids(args.output, args.log, args.retry_failed) if args.resume else set()
    pending_jobs = [job for job in jobs if job["job_id"] not in completed]
    print(
        f"loaded={len(jobs)} completed={len(jobs) - len(pending_jobs)} pending={len(pending_jobs)}",
        flush=True,
    )
    mode = "a" if args.resume else "w"
    with args.output.open(mode, encoding="utf-8") as output, args.log.open(mode, encoding="utf-8") as log:
        for index, job in enumerate(pending_jobs, 1):
            started = time.monotonic()
            generated_text = ""
            messages = [
                {
                    "role": "system",
                    "content": system_instruction(
                        job, args.compact_target, args.use_job_instruction
                    ),
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
            timeout_criteria = None
            repetition_criteria = None
            if args.stop_on_complete_json:
                stopping.append(CompleteAnnotationCriteria(inputs["input_ids"].shape[1]))
            if args.max_seconds_per_job > 0:
                timeout_criteria = ElapsedTimeCriteria(args.max_seconds_per_job)
                stopping.append(timeout_criteria)
            if args.max_repeated_entity > 0:
                repetition_criteria = RepeatedEntityCriteria(
                    inputs["input_ids"].shape[1], args.max_repeated_entity
                )
                stopping.append(repetition_criteria)
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
            elapsed_seconds = round(time.monotonic() - started, 3)
            generated_tokens = int(generated.shape[1] - inputs["input_ids"].shape[1])
            stop_reason = "generation_complete"
            if timeout_criteria and timeout_criteria.triggered:
                stop_reason = "time_limit"
            elif repetition_criteria and repetition_criteria.triggered:
                stop_reason = "repeated_entity"
            elif generated_tokens >= args.max_new_tokens:
                stop_reason = "token_limit"
            try:
                annotation = parse_json(generated_text)
            except Exception as error:
                log.write(json.dumps({
                    "job_id": job["job_id"],
                    "status": "failed",
                    "error": str(error),
                    "stop_reason": stop_reason,
                    "repeated_entity_text": repetition_criteria.repeated_text if repetition_criteria else None,
                    "generated_tokens": generated_tokens,
                    "elapsed_seconds": elapsed_seconds,
                    "raw_output": generated_text[:12000],
                }, ensure_ascii=False) + "\n")
                log.flush()
                print(
                    f"{index}/{len(pending_jobs)} {job['job_id']} failed "
                    f"({stop_reason}, {generated_tokens} tokens): {error}",
                    flush=True,
                )
                continue
            annotation.setdefault("schema_version", "0.1.0")
            annotation.setdefault("document_id", job["document_id"])
            annotation.setdefault("language", job["language"])
            annotation.setdefault("review", {"status": "unreviewed", "reviewers": [], "notes": "qlora inference"})
            for entity in annotation.get("entities", []):
                entity.setdefault("created_by", created_by)
                entity.setdefault("review_status", "pending")
            for relation in annotation.get("relations", []):
                relation.setdefault("created_by", created_by)
                relation.setdefault("review_status", "pending")
            output.write(json.dumps({"job_id": job["job_id"], "annotation": annotation}, ensure_ascii=False) + "\n")
            output.flush()
            log.write(json.dumps({
                "job_id": job["job_id"],
                "status": "success",
                "stop_reason": stop_reason,
                "generated_tokens": generated_tokens,
                "elapsed_seconds": elapsed_seconds,
            }, ensure_ascii=False) + "\n")
            log.flush()
            print(f"{index}/{len(pending_jobs)} {job['job_id']} success", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--adapter",
        type=Path,
        help="Optional QLoRA adapter. Omit it to run the base model as a zero-shot control.",
    )
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--max-input-tokens", type=int, default=12288)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-seconds-per-job", type=float, default=180.0)
    parser.add_argument(
        "--max-repeated-entity",
        type=int,
        default=6,
        help="Stop when the same entity text occurs this many times; use 0 to disable",
    )
    parser.add_argument("--limit", type=int, help="Run only the first N selected jobs")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many selected jobs")
    parser.add_argument("--job-id", help="Run one job for a focused generation diagnostic")
    parser.add_argument("--resume", action="store_true", help="Append and skip terminal jobs")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="With --resume, retry failed jobs while retaining successful jobs",
    )
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
    parser.add_argument(
        "--use-job-instruction",
        action="store_true",
        help="Prepend each job's KG/context instruction in compact output mode",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
