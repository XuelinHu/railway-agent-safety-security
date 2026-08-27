#!/usr/bin/env python3
"""Run compact safety annotation inference through a local llama.cpp GGUF model."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def payload(job: dict[str, Any]) -> str:
    return json.dumps(
        {
            "document_id": job["document_id"],
            "language": job["language"],
            "teacher_model": "qwen3.8-27b-gguf",
            "ontology": job["ontology"],
            "segments": job["segments"],
        },
        ensure_ascii=False,
    )


def prompt(job: dict[str, Any]) -> str:
    return (
        "You are an evidence-grounded risk annotation model. Read the supplied document segments and ontology. "
        "Return only one valid JSON object with exactly these top-level fields: schema_version, document_id, "
        "language, entities, relations. Each entity contains only id, text, type. Each relation contains only "
        "id, source_id, type, target_id, claim_status. Entity text must be copied exactly from a segment. Use "
        "only ontology entity types and legal ontology relation signatures. Do not include evidence, confidence, "
        "review, created_by, explanations, markdown, or any other fields. Do not repeat entities.\n\n"
        + payload(job)
    )


def normalize_ids(annotation: dict[str, Any]) -> dict[str, Any]:
    """Normalize model ID casing and make entity/relation references contiguous."""
    entities = annotation.get("entities", [])
    entity_map = {str(entity.get("id", "")).upper(): f"E{index}" for index, entity in enumerate(entities, 1)}
    normalized_entities = []
    for index, entity in enumerate(entities, 1):
        normalized_entities.append({**entity, "id": f"E{index}"})
    normalized_relations = []
    for index, relation in enumerate(annotation.get("relations", []), 1):
        source_id = entity_map.get(str(relation.get("source_id", "")).upper(), relation.get("source_id"))
        target_id = entity_map.get(str(relation.get("target_id", "")).upper(), relation.get("target_id"))
        normalized_relations.append({**relation, "id": f"R{index}", "source_id": source_id, "target_id": target_id})
    return {**annotation, "schema_version": "0.1.0", "entities": normalized_entities, "relations": normalized_relations}


def run_one(args: argparse.Namespace, job: dict[str, Any]) -> tuple[str, str, float, int]:
    started = time.monotonic()
    command = [
        str(args.llama_cli),
        "--model", str(args.model),
        "--gpu-layers", str(args.gpu_layers),
        "--ctx-size", str(args.context_size),
        "--predict", str(args.max_new_tokens),
        "--batch-size", str(args.batch_size),
        "--ubatch-size", str(args.ubatch_size),
        "--flash-attn", "auto",
        "--reasoning", "off",
        "--single-turn",
        "--simple-io",
        "--no-display-prompt",
        "--temp", "0.7",
        "--top-p", "0.8",
        "--seed", "42",
        "--prompt", prompt(job),
    ]
    if args.constrained_json:
        if not args.schema:
            raise ValueError("--schema is required with --constrained-json")
        prompt_index = command.index("--prompt")
        command[prompt_index:prompt_index] = ["--json-schema-file", str(args.schema)]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    output = completed.stdout.strip()
    if completed.stderr.strip():
        output += "\n" + completed.stderr.strip()
    return output, "" if completed.returncode == 0 else f"llama.cpp exit code {completed.returncode}", time.monotonic() - started, completed.returncode


def main(args: argparse.Namespace) -> int:
    jobs = load_jsonl(args.jobs)
    if args.job_id:
        jobs = [job for job in jobs if job["job_id"] == args.job_id]
        if not jobs:
            raise SystemExit(f"job not found: {args.job_id}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_stream, args.log.open("w", encoding="utf-8") as log_stream:
        for index, job in enumerate(jobs, 1):
            raw, error, elapsed, returncode = run_one(args, job)
            record = {"job_id": job["job_id"], "status": "success" if not error else "failed", "elapsed_seconds": round(elapsed, 3), "returncode": returncode}
            if error:
                record["error"] = error
                record["raw_output"] = raw[:12000]
                log_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"{index}/{len(jobs)} {job['job_id']} failed: {error}", flush=True)
                continue
            try:
                from run_qlora_inference import parse_json

                annotation = normalize_ids(parse_json(raw))
            except Exception as exc:
                record["status"] = "failed"
                record["error"] = str(exc)
                record["raw_output"] = raw[:12000]
                log_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"{index}/{len(jobs)} {job['job_id']} failed: {exc}", flush=True)
                continue
            output_stream.write(json.dumps({"job_id": job["job_id"], "annotation": annotation}, ensure_ascii=False) + "\n")
            output_stream.flush()
            log_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_stream.flush()
            print(f"{index}/{len(jobs)} {job['job_id']} success", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-cli", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=False)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--job-id")
    parser.add_argument("--gpu-layers", default="all")
    parser.add_argument("--context-size", type=int, default=12288)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--ubatch-size", type=int, default=256)
    parser.add_argument("--constrained-json", action="store_true", help="Use llama.cpp grammar; disabled by default for this Qwen3.8 build")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
