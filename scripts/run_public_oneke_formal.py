#!/usr/bin/env python3
"""Frozen OneKE 4-bit worker for a queued canary or public validation run.

The module is intentionally not imported by CPU preflight.  It imports Torch,
Transformers, and bitsandbytes only after validating that the requested mode is
``canary`` or that every input row is a canonical validation request.  It has no
test mode and accepts no gold path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from public_external_adapters.oneke import (
    RAW_SCHEMA,
    REQUEST_SCHEMA,
    SPLIT,
    canonical_label,
    load_jsonl,
    reject_test_path,
    unwrap_json,
    write_json_atomic,
)


MODEL_REVISION = "696148c0581b29f530af738ddab500deaa8fe8f2"
DEFAULT_MODEL = Path(
    "/ds2/xuelin/cache/huggingface/hub/models--zjunlp--OneKE/"
    f"snapshots/{MODEL_REVISION}"
)
PROMPT_VERSION = "oneke-upstream-jsonlike-batched-v3"
INSTRUCTIONS = {
    "NER": (
        "You are an expert in named entity recognition. Please extract entities "
        "that match the schema definition from the input. Return an empty list if "
        "the entity type does not exist. Please return your final extraction results "
        "as a JSON object without escape characters or line breaks, wrapped in triple "
        "backticks (```). Use standard double quotes (\"\") for JSON structure "
    ),
    "RE": (
        "You are an expert in relationship extraction. Please extract relationship "
        "triples that match the schema definition from the input. "
        "Return an empty list for relationships that do not exist. Please return your "
        "final extraction results as a JSON object without escape characters or line "
        "breaks, wrapped in triple backticks (```). Use standard double quotes (\"\") "
        "for JSON structure"
    ),
}
SCHEMA_BATCH_SIZE = {"NER": 6, "RE": 4}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def request_sha256(request: dict[str, Any]) -> str:
    frozen = json.dumps(
        request, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(frozen).hexdigest()


def build_prompt(task: str, constraint: list[str], text: str) -> str:
    if task not in INSTRUCTIONS:
        raise ValueError(f"unsupported OneKE task: {task}")
    # Preserve the exact string-substitution semantics of upstream
    # ``EXTRACT_INSTRUCTION_JSON``.  It is intentionally JSON-like rather than
    # serialized with json.dumps: instruction and input are inserted raw and
    # the Python list representation of ``constraint`` is retained.
    prompt = (
        "\n{\n"
        f'    "instruction": {INSTRUCTIONS[task]},\n'
        f'    "schema": {constraint},\n'
        f'    "input": {text},\n'
        "}\n"
    )
    system = (
        "<<SYS>>\nYou are a helpful assistant. "
        "你是一个乐于助人的助手。\n<</SYS>>\n\n"
    )
    return f"[INST] {system}{prompt}[/INST]"


def validate_model_path(path: Path) -> None:
    required = (
        "config.json",
        "pytorch_model.bin.index.json",
        "tokenizer.model",
    )
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise ValueError(f"OneKE snapshot is incomplete: {missing}")
    if path.name != MODEL_REVISION:
        raise ValueError(
            f"OneKE revision must be {MODEL_REVISION}, found directory {path.name}"
        )
    try:
        index = json.loads((path / "pytorch_model.bin.index.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"OneKE weight index is invalid: {exc}") from exc
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("OneKE weight index has no weight map")
    shards = {name for name in weight_map.values() if isinstance(name, str) and name}
    missing_shards = sorted(
        name for name in shards if not (path / name).is_file() or (path / name).stat().st_size == 0
    )
    if not shards or missing_shards:
        raise ValueError(f"OneKE snapshot has missing weight shards: {missing_shards}")


def validate_requests(rows: list[dict[str, Any]], path: Path) -> None:
    reject_test_path(path, "OneKE requests")
    identifiers: list[str] = []
    datasets: set[str] = set()
    for row in rows:
        if row.get("schema_version") != REQUEST_SCHEMA:
            raise ValueError(f"{row.get('job_id')}: request schema mismatch")
        if row.get("split") != SPLIT:
            raise ValueError(f"{row.get('job_id')}: only validation is allowed")
        if row.get("seed") != 42:
            raise ValueError(f"{row.get('job_id')}: only frozen seed 42 is allowed")
        dataset = row.get("dataset")
        if dataset not in {"conll04", "scierc", "ade"}:
            raise ValueError(f"{row.get('job_id')}: unsupported validation dataset")
        datasets.add(str(dataset))
        provenance = row.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError(f"{row.get('job_id')}: provenance is missing")
        if provenance.get("test_gold_read") is not False:
            raise ValueError(f"{row.get('job_id')}: invalid test access declaration")
        if provenance.get("prompt_uses_gold") is not False:
            raise ValueError(f"{row.get('job_id')}: gold-conditioned prompt is forbidden")
        source = row.get("source")
        if not isinstance(source, dict) or not isinstance(source.get("text"), str):
            raise ValueError(f"{row.get('job_id')}: source text is invalid")
        start, end = source.get("start"), source.get("end")
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or end - start != len(source["text"])
        ):
            raise ValueError(f"{row.get('job_id')}: source offsets are invalid")
        tasks = row.get("tasks")
        if not isinstance(tasks, dict):
            raise ValueError(f"{row.get('job_id')}: tasks are invalid")
        for key, expected_task, expected_key in (
            ("ner", "NER", "entity_list"),
            ("re", "RE", "relation_list"),
        ):
            task = tasks.get(key)
            if (
                not isinstance(task, dict)
                or task.get("task") != expected_task
                or task.get("expected_key") != expected_key
                or not isinstance(task.get("constraint"), list)
                or not task["constraint"]
                or any(not isinstance(item, str) or not item for item in task["constraint"])
            ):
                raise ValueError(f"{row.get('job_id')}: {key} task contract is invalid")
        identifier = row.get("job_id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("request job_id is invalid")
        identifiers.append(identifier)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("request job_id values are not unique")
    if len(datasets) > 1:
        raise ValueError("one validation invocation may contain only one dataset")


def validate_resumable_raw(
    rows: list[dict[str, Any]], requests: list[dict[str, Any]], path: Path
) -> set[str]:
    """Validate every persisted terminal row before trusting it for resume."""

    reject_test_path(path, "OneKE raw responses")
    requested = {str(row["job_id"]): row for row in requests}
    identifiers: list[str] = []
    for row in rows:
        identifier = row.get("job_id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"{path}: raw response job_id is invalid")
        if identifier not in requested:
            raise ValueError(f"resume output contains unknown job: {identifier}")
        request = requested[identifier]
        expected = {
            "schema_version": RAW_SCHEMA,
            "dataset": request["dataset"],
            "split": SPLIT,
            "seed": 42,
            "model_revision": MODEL_REVISION,
            "request_sha256": request_sha256(request),
            "test_gold_read": False,
        }
        for field, value in expected.items():
            if row.get(field) != value:
                raise ValueError(f"{identifier}: invalid resumable raw field {field}")
        if row.get("status") not in {"complete", "terminal_failure"}:
            raise ValueError(f"{identifier}: raw response is not terminal")
        identifiers.append(identifier)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"{path}: duplicate raw response job_id")
    return set(identifiers)


def load_runtime(model_path: Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("OneKE GPU worker requires CUDA")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"OneKE formal worker requires exactly one visible GPU, found {torch.cuda.device_count()}"
        )
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        device_map={"": 0},
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return torch, tokenizer, model


def generate(
    torch: Any,
    tokenizer: Any,
    model: Any,
    prompt: str,
    max_input_tokens: int,
    max_new_tokens: int,
) -> tuple[str, dict[str, Any], str | None]:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        # Upstream ``tokenizer.encode`` keeps the model's BOS token.
        add_special_tokens=True,
        truncation=False,
    )
    input_tokens = int(encoded["input_ids"].shape[-1])
    if input_tokens > max_input_tokens:
        raise ValueError(
            f"OneKE prompt has {input_tokens} tokens, above frozen cap {max_input_tokens}"
        )
    encoded = {name: tensor.to("cuda:0") for name, tensor in encoded.items()}
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=(
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id
            ),
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    torch.cuda.synchronize(0)
    generated = output[0, input_tokens:]
    raw_text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    parsed, parse_error = unwrap_json(raw_text)
    telemetry = {
        "input_tokens": input_tokens,
        "output_tokens": int(generated.shape[-1]),
        "seconds": round(time.perf_counter() - started, 4),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
    }
    return raw_text, {"parsed": parsed, "telemetry": telemetry}, parse_error


def schema_batches(task: str, constraint: list[str]) -> list[list[str]]:
    size = SCHEMA_BATCH_SIZE[task]
    return [constraint[index : index + size] for index in range(0, len(constraint), size)]


def normalize_task_payload(
    task: str, parsed: dict[str, Any], constraint: list[str]
) -> tuple[dict[str, list[dict[str, str]]], str | None, str]:
    """Translate the checkpoint's documented schema-keyed JSON to OneKE's newer lists."""

    expected_key = "entity_list" if task == "NER" else "relation_list"
    if expected_key in parsed:
        value = parsed[expected_key]
        if not isinstance(value, list):
            return {expected_key: []}, f"expected {expected_key} list", "canonical_list"
        return {expected_key: value}, None, "canonical_list"

    normalized: list[dict[str, str]] = []
    for raw_label, values in parsed.items():
        label, known = canonical_label(raw_label, constraint, "")
        if not known:
            return (
                {expected_key: []},
                f"native response contains out-of-schema key {raw_label!r}",
                "native_schema_map",
            )
        if not isinstance(values, list):
            return (
                {expected_key: []},
                f"native response value for {raw_label!r} is not a list",
                "native_schema_map",
            )
        for position, item in enumerate(values):
            if task == "NER":
                if not isinstance(item, str) or not item.strip():
                    return (
                        {expected_key: []},
                        f"native NER item {raw_label!r}[{position}] is not text",
                        "native_schema_map",
                    )
                normalized.append({"name": item.strip(), "type": label})
                continue
            if not isinstance(item, dict):
                return (
                    {expected_key: []},
                    f"native RE item {raw_label!r}[{position}] is not an object",
                    "native_schema_map",
                )
            head = item.get("subject", item.get("head"))
            tail = item.get("object", item.get("tail"))
            if not isinstance(head, str) or not head.strip() or not isinstance(tail, str) or not tail.strip():
                return (
                    {expected_key: []},
                    f"native RE item {raw_label!r}[{position}] lacks text subject/object",
                    "native_schema_map",
                )
            normalized.append(
                {"head": head.strip(), "tail": tail.strip(), "relation": label}
            )
    return {expected_key: normalized}, None, "native_schema_map"


def infer_request(
    torch: Any,
    tokenizer: Any,
    model: Any,
    request: dict[str, Any],
    max_input_tokens: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    task_errors: list[str] = []
    for key in ("ner", "re"):
        task = request["tasks"][key]
        task_name = task["task"]
        expected_key = "entity_list" if key == "ner" else "relation_list"
        merged: list[dict[str, str]] = []
        batch_records: list[dict[str, Any]] = []
        for batch_number, constraint_batch in enumerate(
            schema_batches(task_name, list(task["constraint"])), start=1
        ):
            prompt = build_prompt(task_name, constraint_batch, request["source"]["text"])
            try:
                raw_text, result, parse_error = generate(
                    torch,
                    tokenizer,
                    model,
                    prompt,
                    max_input_tokens,
                    max_new_tokens,
                )
                normalized, contract_error, response_format = normalize_task_payload(
                    task_name, result["parsed"], constraint_batch
                )
                record = {
                    "batch": batch_number,
                    "constraint": constraint_batch,
                    "raw_text": raw_text,
                    "parsed_native": result["parsed"],
                    "parsed": normalized,
                    "response_format": response_format,
                    "telemetry": result["telemetry"],
                    "parse_error": parse_error,
                    "contract_error": contract_error,
                }
                batch_records.append(record)
                if parse_error:
                    task_errors.append(f"{key}[{batch_number}]:{parse_error}")
                if contract_error:
                    task_errors.append(f"{key}[{batch_number}]:{contract_error}")
                if not parse_error and not contract_error:
                    merged.extend(normalized[expected_key])
            except Exception as exc:  # retain a reportable row for per-batch failures
                batch_records.append(
                    {
                        "batch": batch_number,
                        "constraint": constraint_batch,
                        "raw_text": "",
                        "parsed_native": {},
                        "parsed": {expected_key: []},
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                task_errors.append(f"{key}[{batch_number}]:{type(exc).__name__}:{exc}")
                if "out of memory" in str(exc).casefold():
                    torch.cuda.empty_cache()
        telemetry_rows = [
            record["telemetry"] for record in batch_records if "telemetry" in record
        ]
        tasks[key] = {
            "task": task_name,
            "prompt_version": PROMPT_VERSION,
            "schema_batch_size": SCHEMA_BATCH_SIZE[task_name],
            "batches": batch_records,
            "parsed": {expected_key: merged},
            "telemetry": {
                "input_tokens": sum(row["input_tokens"] for row in telemetry_rows),
                "output_tokens": sum(row["output_tokens"] for row in telemetry_rows),
                "seconds": round(sum(row["seconds"] for row in telemetry_rows), 4),
                "peak_allocated_bytes": max(
                    (row["peak_allocated_bytes"] for row in telemetry_rows), default=0
                ),
                "peak_reserved_bytes": max(
                    (row["peak_reserved_bytes"] for row in telemetry_rows), default=0
                ),
            },
        }
    return {
        "schema_version": RAW_SCHEMA,
        "job_id": request["job_id"],
        "dataset": request["dataset"],
        "split": SPLIT,
        "seed": 42,
        "status": "complete" if not task_errors else "terminal_failure",
        "tasks": tasks,
        "errors": task_errors,
        "model_revision": MODEL_REVISION,
        "request_sha256": request_sha256(request),
        "quantization": "bitsandbytes-nf4-double-quantization",
        "test_gold_read": False,
        "finished_at": utc_now(),
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def canary_diagnostics(result: dict[str, Any]) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    for name, task in result.get("tasks", {}).items():
        batches: list[dict[str, Any]] = []
        for record in task.get("batches", []):
            raw_text = str(record.get("raw_text", ""))
            native = record.get("parsed_native")
            parsed = record.get("parsed")
            batches.append(
                {
                    "batch": record.get("batch"),
                    "constraint": record.get("constraint"),
                    "raw_text_sha256": hashlib.sha256(
                        raw_text.encode("utf-8")
                    ).hexdigest(),
                    "raw_text_preview": raw_text[:512],
                    "raw_text_truncated": len(raw_text) > 512,
                    "parsed_native_keys": (
                        sorted(str(key) for key in native) if isinstance(native, dict) else []
                    ),
                    "normalized_keys": (
                        sorted(str(key) for key in parsed) if isinstance(parsed, dict) else []
                    ),
                    "response_format": record.get("response_format"),
                    "parse_error": record.get("parse_error"),
                    "contract_error": record.get("contract_error"),
                    "error": record.get("error"),
                    "telemetry": record.get("telemetry"),
                }
            )
        diagnostics[name] = {
            "parsed_counts": {
                key: len(value) if isinstance(value, list) else None
                for key, value in task.get("parsed", {}).items()
            },
            "batches": batches,
        }
    return diagnostics


def validate_canary_extraction(result: dict[str, Any]) -> None:
    entities = result["tasks"]["ner"]["parsed"]["entity_list"]
    relations = result["tasks"]["re"]["parsed"]["relation_list"]
    entity_pairs = {
        (
            str(item.get("name", item.get("text", ""))).strip().casefold(),
            canonical_label(
                item.get("type", item.get("label")),
                ["Drug", "Adverse-Effect", "Person"],
                "",
            )[0],
        )
        for item in entities
        if isinstance(item, dict)
    }
    relation_triples = {
        (
            str(item.get("head", item.get("source", ""))).strip().casefold(),
            str(item.get("tail", item.get("target", ""))).strip().casefold(),
            canonical_label(
                item.get("relation", item.get("type")), ["Adverse-Effect"], ""
            )[0],
        )
        for item in relations
        if isinstance(item, dict)
    }
    if ("aspirin", "Drug") not in entity_pairs or (
        "nausea",
        "Adverse-Effect",
    ) not in entity_pairs:
        raise RuntimeError("canary NER did not recover the synthetic Drug/effect pair")
    # The frozen ADE ontology defines source=Adverse-Effect and target=Drug.
    # This is also the direction used in validation gold; do not reverse it to
    # match the surface causative wording of the synthetic sentence.
    if ("nausea", "aspirin", "Adverse-Effect") not in relation_triples:
        raise RuntimeError("canary RE did not recover the frozen ADE relation signature")


def revalidate_canary(args: argparse.Namespace) -> int:
    """Re-audit a hash-bound failed marker after a CPU-only assertion repair."""

    validate_model_path(args.model)
    marker = args.marker
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"canary marker is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("canary marker must be an object")
    expected = {
        "schema_version": "public-external-gpu-canary-v1",
        "baseline": "oneke",
        "gpu": "RTX 3090",
        "capacity_gib": 24.0,
        "status": "failed",
        "exit_code": 1,
        "model_revision": MODEL_REVISION,
        "quantization": "bitsandbytes-nf4-double-quantization",
        "prompt_version": PROMPT_VERSION,
        "test_gold_read": False,
        "terminal": True,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"canary revalidation field mismatch: {field}")
    if "directed synthetic relation" not in str(payload.get("error", "")):
        raise ValueError("canary marker is not the known pre-fix direction assertion failure")
    actual_gpu = payload.get("actual_gpu_name")
    total_memory = payload.get("actual_total_memory_bytes")
    peak_allocated = payload.get("peak_allocated_bytes")
    peak_reserved = payload.get("peak_reserved_bytes")
    if not isinstance(actual_gpu, str) or "3090" not in actual_gpu:
        raise ValueError("canary marker does not bind an RTX 3090")
    if not isinstance(total_memory, int) or not 20 * 1024**3 <= total_memory <= 24 * 1024**3:
        raise ValueError("canary marker has invalid GPU capacity")
    for name, value in (("allocated", peak_allocated), ("reserved", peak_reserved)):
        if not isinstance(value, int) or not 0 < value <= total_memory:
            raise ValueError(f"canary marker has invalid peak {name} memory")

    diagnostics = payload.get("canary_task_diagnostics")
    if not isinstance(diagnostics, dict):
        raise ValueError("canary marker has no task diagnostics")
    tasks: dict[str, Any] = {}
    canary_tasks: dict[str, Any] = {}
    for key, task_name, expected_key in (
        ("ner", "NER", "entity_list"),
        ("re", "RE", "relation_list"),
    ):
        record = diagnostics.get(key)
        batches = record.get("batches") if isinstance(record, dict) else None
        if not isinstance(batches, list) or len(batches) != 1:
            raise ValueError(f"canary {key} diagnostics must contain one batch")
        batch = batches[0]
        raw_text = batch.get("raw_text_preview")
        if not isinstance(raw_text, str) or batch.get("raw_text_truncated") is not False:
            raise ValueError(f"canary {key} raw response is not fully retained")
        if hashlib.sha256(raw_text.encode("utf-8")).hexdigest() != batch.get(
            "raw_text_sha256"
        ):
            raise ValueError(f"canary {key} raw response hash mismatch")
        native, parse_error = unwrap_json(raw_text)
        if parse_error:
            raise ValueError(f"canary {key} raw response is invalid: {parse_error}")
        constraint = batch.get("constraint")
        if not isinstance(constraint, list) or not constraint:
            raise ValueError(f"canary {key} constraint is invalid")
        normalized, contract_error, response_format = normalize_task_payload(
            task_name, native, constraint
        )
        if contract_error:
            raise ValueError(f"canary {key} contract failed: {contract_error}")
        tasks[key] = {"parsed": normalized}
        telemetry = batch.get("telemetry")
        if not isinstance(telemetry, dict):
            raise ValueError(f"canary {key} telemetry is invalid")
        canary_tasks[key] = {
            "input_tokens": telemetry.get("input_tokens"),
            "output_tokens": telemetry.get("output_tokens"),
            "response_format": response_format,
        }
    validate_canary_extraction({"tasks": tasks})

    prior_error = payload.pop("error")
    payload.update(
        {
            "status": "passed",
            "exit_code": 0,
            "runtime_compatible": True,
            "canary_tasks": canary_tasks,
            "synthetic_relation_signature": {
                "type": "Adverse-Effect",
                "source": "Adverse-Effect",
                "target": "Drug",
                "observed": "nausea->Aspirin",
            },
            "revalidated_without_gpu": True,
            "revalidation": {
                "raw_text_sha256_verified": True,
                "reason": "fixed canary assertion to match frozen ADE source/target signature",
                "prior_assertion_error": prior_error,
                "revalidated_at": utc_now(),
            },
        }
    )
    write_json_atomic(marker, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0


def run_validation(args: argparse.Namespace) -> int:
    validate_model_path(args.model)
    reject_test_path(args.output, "OneKE raw output")
    rows = load_jsonl(args.requests)
    validate_requests(rows, args.requests)
    completed: set[str] = set()
    if args.resume and args.output.is_file():
        existing = load_jsonl(args.output)
        completed = validate_resumable_raw(existing, rows, args.output)
    elif args.output.exists():
        raise FileExistsError(f"refusing to overwrite raw OneKE output: {args.output}")

    torch, tokenizer, model = load_runtime(args.model)
    for request in rows:
        if request["job_id"] in completed:
            continue
        row = infer_request(
            torch,
            tokenizer,
            model,
            request,
            args.max_input_tokens,
            args.max_new_tokens,
        )
        append_jsonl(args.output, row)
        print(
            json.dumps(
                {
                    "job_id": row["job_id"],
                    "status": row["status"],
                    "completed": len(completed) + 1,
                    "total": len(rows),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        completed.add(request["job_id"])
    return 0


def run_canary(args: argparse.Namespace) -> int:
    marker = args.marker
    started = utc_now()
    payload: dict[str, Any] = {
        "schema_version": "public-external-gpu-canary-v1",
        "baseline": "oneke",
        "gpu": "RTX 3090",
        "capacity_gib": 24.0,
        "status": "failed",
        "exit_code": 1,
        "model_revision": MODEL_REVISION,
        "quantization": "bitsandbytes-nf4-double-quantization",
        "prompt_version": PROMPT_VERSION,
        "test_gold_read": False,
        "started_at": started,
    }
    try:
        validate_model_path(args.model)
        torch, tokenizer, model = load_runtime(args.model)
        actual_name = torch.cuda.get_device_name(0)
        total_memory = int(torch.cuda.get_device_properties(0).total_memory)
        if "3090" not in actual_name:
            raise RuntimeError(f"registered canary requires RTX 3090, found {actual_name}")
        payload.update(
            {
                "actual_gpu_name": actual_name,
                "actual_total_memory_bytes": total_memory,
            }
        )
        request = {
            "job_id": "oneke_gpu_canary",
            "dataset": "synthetic",
            "source": {"text": "Aspirin caused nausea in Alice."},
            "tasks": {
                "ner": {"task": "NER", "constraint": ["Drug", "Adverse-Effect", "Person"]},
                "re": {"task": "RE", "constraint": ["Adverse-Effect"]},
            },
        }
        result = infer_request(
            torch,
            tokenizer,
            model,
            request,
            args.max_input_tokens,
            min(args.max_new_tokens, 128),
        )
        payload["canary_task_diagnostics"] = canary_diagnostics(result)
        if result["status"] != "complete":
            raise RuntimeError(f"canary inference contract failed: {result['errors']}")
        validate_canary_extraction(result)
        payload.update(
            {
                "status": "passed",
                "exit_code": 0,
                "runtime_compatible": True,
                "synthetic_relation_signature": {
                    "type": "Adverse-Effect",
                    "source": "Adverse-Effect",
                    "target": "Drug",
                    "observed": "nausea->Aspirin",
                },
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
                "canary_tasks": {
                    name: {
                        "input_tokens": item["telemetry"]["input_tokens"],
                        "output_tokens": item["telemetry"]["output_tokens"],
                    }
                    for name, item in result["tasks"].items()
                },
            }
        )
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
    if "torch" in locals() and torch.cuda.is_available():
        payload.setdefault("peak_allocated_bytes", int(torch.cuda.max_memory_allocated(0)))
        payload.setdefault("peak_reserved_bytes", int(torch.cuda.max_memory_reserved(0)))
    payload["terminal"] = True
    payload["finished_at"] = utc_now()
    write_json_atomic(marker, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return int(payload["exit_code"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-input-tokens", type=int, default=3072)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    canary = subparsers.add_parser("canary")
    canary.add_argument(
        "--marker",
        type=Path,
        default=Path("outputs/public_external_formal/oneke/gpu_canary.json"),
    )

    validation = subparsers.add_parser("validation")
    validation.add_argument("--requests", type=Path, required=True)
    validation.add_argument("--output", type=Path, required=True)
    validation.add_argument("--resume", action="store_true")

    revalidate = subparsers.add_parser("revalidate-canary")
    revalidate.add_argument(
        "--marker",
        type=Path,
        default=Path("outputs/public_external_formal/oneke/gpu_canary.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.model = args.model.resolve()
    if args.max_input_tokens <= 0 or args.max_new_tokens <= 0:
        raise ValueError("token limits must be positive")
    if args.mode == "canary":
        return run_canary(args)
    if args.mode == "revalidate-canary":
        return revalidate_canary(args)
    return run_validation(args)


if __name__ == "__main__":
    raise SystemExit(main())
