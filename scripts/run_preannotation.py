#!/usr/bin/env python3
"""Run provider-neutral pre-annotation jobs through Ollama or OpenAI."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


def load_jobs(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def user_payload(job: dict[str, Any]) -> str:
    payload = {
        "document_id": job["document_id"],
        "language": job["language"],
        "teacher_model": job["teacher_model"],
        "ontology": job["ontology"],
        "segments": job["segments"],
    }
    return json.dumps(payload, ensure_ascii=False)


def run_ollama(job: dict[str, Any], model: str, schema: dict[str, Any], timeout: int, think: bool) -> tuple[str, Any]:
    body = {
        "model": model,
        "stream": False,
        "think": think,
        "format": schema,
        "messages": [
            {"role": "system", "content": job["system_instruction"]},
            {"role": "user", "content": user_payload(job)},
        ],
        "options": {"temperature": 0, "seed": 42},
    }
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        envelope = json.loads(response.read().decode("utf-8"))
    return envelope["message"]["content"], {
        "prompt_eval_count": envelope.get("prompt_eval_count"),
        "eval_count": envelope.get("eval_count"),
        "total_duration": envelope.get("total_duration"),
    }


def run_openai(
    job: dict[str, Any],
    model: str,
    schema: dict[str, Any],
    timeout: int,
    _: bool,
    base_url: str | None,
    api_key_env: str,
) -> tuple[str, Any]:
    from openai import OpenAI

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"missing API key environment variable: {api_key_env}")
    client_kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout}
    if base_url:
        client_kwargs["base_url"] = base_url.rstrip("/")
    client = OpenAI(**client_kwargs)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": job["system_instruction"]},
            {"role": "user", "content": user_payload(job)},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "safety_risk_preannotation",
                "strict": False,
                "schema": schema,
            },
        },
        temperature=0,
        max_tokens=12000,
    )
    usage = response.usage.model_dump() if response.usage else None
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("chat completion returned empty content")
    return content, usage


def run(args: argparse.Namespace) -> int:
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    jobs = load_jobs(args.jobs)
    existing_ids: set[str] = set()
    if args.output.exists() and args.resume:
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing_ids.add(json.loads(line)["job_id"])
    pending = [job for job in jobs if job["job_id"] not in existing_ids]
    if args.job_id:
        pending = [job for job in pending if job["job_id"] == args.job_id]
        if not pending:
            raise SystemExit(f"job not found or already completed: {args.job_id}")
    if args.max_jobs:
        pending = pending[: args.max_jobs]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    provider_runner = run_ollama if args.provider == "ollama" else run_openai
    failures = 0
    with args.output.open("a", encoding="utf-8") as output_stream, args.log.open("a", encoding="utf-8") as log_stream:
        for index, job in enumerate(pending, 1):
            started = time.monotonic()
            log_record: dict[str, Any] = {
                "job_id": job["job_id"],
                "provider": args.provider,
                "model": args.model,
                "prompt_version": job.get("prompt_version"),
                "ontology_version": job.get("ontology", {}).get("version"),
            }
            try:
                call_job = job
                if args.repair_schema:
                    call_job = dict(job)
                    call_job["system_instruction"] = (
                        job["system_instruction"]
                        + "\n\nREPAIR CONSTRAINTS: Keep the exact schema field meanings. "
                        + "Use only the listed relation_types as the relation 'type'. "
                        + "Use only explicit, inferred, normative, or uncertain as relation claim_status. "
                        + "Use only pending for entity/relation review_status and unreviewed for review.status. "
                        + "Never put a relation type in claim_status or review.status. "
                        + "If a claim cannot be represented by the listed ontology, omit it."
                    )
                last_error: Exception | None = None
                for attempt in range(args.retries + 1):
                    try:
                        if args.provider == "openai":
                            content, usage = provider_runner(
                                call_job,
                                args.model,
                                schema,
                                args.timeout,
                                args.think,
                                args.base_url,
                                args.api_key_env,
                            )
                        else:
                            content, usage = provider_runner(call_job, args.model, schema, args.timeout, args.think)
                        break
                    except Exception as error:
                        last_error = error
                        transient = any(token in str(error).lower() for token in (" 429", " 500", " 502", " 503", "timeout", "temporarily unavailable"))
                        if not transient or attempt >= args.retries:
                            raise
                        time.sleep(min(30, 5 * (attempt + 1)))
                else:
                    raise last_error or RuntimeError("provider call failed")
                annotation = json.loads(content)
                annotation.setdefault("schema_version", "0.1.0")
                annotation.setdefault("document_id", job["document_id"])
                annotation.setdefault("language", job["language"])
                annotation.setdefault("review", {"status": "unreviewed", "reviewers": [], "notes": "teacher pre-annotation"})
                for entity in annotation.get("entities", []):
                    entity.setdefault("created_by", job["teacher_model"])
                    entity.setdefault("review_status", "pending")
                for relation in annotation.get("relations", []):
                    relation.setdefault("created_by", job["teacher_model"])
                    relation.setdefault("review_status", "pending")
                schema_errors = [error.message for error in validator.iter_errors(annotation)]
                if annotation.get("document_id") != job["document_id"]:
                    schema_errors.append("document_id does not match job")
                if schema_errors:
                    raise ValueError("; ".join(schema_errors[:10]))
                output_stream.write(json.dumps({"job_id": job["job_id"], "annotation": annotation}, ensure_ascii=False) + "\n")
                output_stream.flush()
                log_record.update(status="success", usage=usage)
            except Exception as error:
                failures += 1
                log_record.update(status="failed", error=str(error)[:2000])
            log_record["elapsed_seconds"] = round(time.monotonic() - started, 3)
            log_stream.write(json.dumps(log_record, ensure_ascii=False) + "\n")
            log_stream.flush()
            print(f"{index}/{len(pending)} {job['job_id']} {log_record['status']}")
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["ollama", "openai"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL, such as http://127.0.0.1:8999/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Environment variable containing the provider API key")
    parser.add_argument("--jobs", type=Path, default=Path("data/processed/preannotation/jobs.jsonl"))
    parser.add_argument("--schema", type=Path, default=Path("schemas/preannotation_candidate.schema.json"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/preannotation/candidates.jsonl"))
    parser.add_argument("--log", type=Path, default=Path("outputs/preannotation_run.jsonl"))
    parser.add_argument("--max-jobs", type=int, default=1)
    parser.add_argument("--job-id")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--retries", type=int, default=3, help="Retries for transient provider errors")
    parser.add_argument("--repair-schema", action="store_true", help="Append strict field and enum repair constraints")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--think", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
