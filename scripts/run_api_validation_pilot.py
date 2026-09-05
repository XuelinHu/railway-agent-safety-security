#!/usr/bin/env python3
"""Run a validation-only API extraction pilot with strict compact outputs.

The runner deliberately refuses any input outside the frozen validation assets.
It stores no credentials and makes response rows resumable by job ID.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml


VALIDATION_MARKER = "low_resource_v2/d100/assets/validation"
PROVIDERS = {
    "deepseek": ("https://api.deepseek.com/chat/completions", "DEEPSEEK_API_KEY", "deepseek-chat"),
    "bailian": ("https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions", "BAILIAN_API_KEY", "qwen-plus"),
    "sub2api": ("http://127.0.0.1:8999/v1/chat/completions", "SUB2API_API_KEY", "gpt-5.4-mini"),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_annotation(text: str) -> dict[str, Any]:
    for match in re.finditer(r"\{", text):
        try:
            value, _ = json.JSONDecoder().raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("entities"), list) and isinstance(value.get("relations"), list):
            return value
    raise ValueError("response did not contain an entities/relations JSON object")


def request_annotation(endpoint: str, api_key: str, model: str, system: str, user: str) -> tuple[dict[str, Any], float]:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "max_tokens": 1800,
            "response_format": {"type": "json_object"},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    started = time.perf_counter()
    request = Request(endpoint, data=body, headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    try:
        with urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"network error: {error.reason}") from error
    latency = time.perf_counter() - started
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"missing choices: {json.dumps(payload, ensure_ascii=False)[:500]}")
    content = choices[0].get("message", {}).get("content", "")
    return {"annotation": parse_annotation(content), "model": payload.get("model", model), "usage": payload.get("usage", {})}, latency


def compact_system() -> str:
    return (
        "You are an evidence-grounded safety information extraction service. "
        "Return exactly one valid JSON object with schema_version, document_id, language, entities, relations. "
        "Every entity must have id, text, type. Every relation must have id, source_id, type, target_id, claim_status. "
        "Copy each entity text exactly as one contiguous substring from the supplied segments. "
        "Use only supplied ontology types, relations, and legal relation signatures. "
        "Create a relation only when directly supported by the supplied text. "
        "claim_status must be explicit, inferred, normative, or uncertain. "
        "Do not include explanations, Markdown, evidence objects, confidence, or additional fields."
    )


def normalise(annotation: dict[str, Any], job: dict[str, Any], ontology: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    entity_types = set(ontology["entity_types"])
    relation_types = set(ontology["relation_types"])
    claims = set(ontology["claim_statuses"])
    signatures = ontology["allowed_relation_signatures"]
    source = "\n".join(segment["text"] for segment in job["segments"])
    entities: list[dict[str, Any]] = []
    id_map: dict[str, str] = {}
    errors: list[str] = []
    seen: set[tuple[str, str]] = set()
    for raw in annotation.get("entities", []):
        text, kind = raw.get("text"), raw.get("type")
        old_id = str(raw.get("id", ""))
        if not isinstance(text, str) or not text or kind not in entity_types:
            errors.append(f"rejected entity {old_id}: missing text or illegal type")
            continue
        if text not in source:
            errors.append(f"rejected entity {old_id}: text not in source")
            continue
        key = (text, kind)
        if key in seen:
            continue
        seen.add(key)
        new_id = f"E{len(entities) + 1}"
        id_map[old_id] = new_id
        entities.append({"id": new_id, "text": text, "type": kind})
    by_id = {item["id"]: item for item in entities}
    relations: list[dict[str, Any]] = []
    for raw in annotation.get("relations", []):
        source_id = id_map.get(str(raw.get("source_id", "")))
        target_id = id_map.get(str(raw.get("target_id", "")))
        relation_type, claim = raw.get("type"), raw.get("claim_status")
        if not source_id or not target_id or relation_type not in relation_types or claim not in claims:
            errors.append(f"rejected relation {raw.get('id', '')}: missing endpoint/type/status")
            continue
        signature = signatures[relation_type]
        if by_id[source_id]["type"] not in signature["source"] or by_id[target_id]["type"] not in signature["target"]:
            errors.append(f"rejected relation {raw.get('id', '')}: illegal signature")
            continue
        relations.append({"id": f"R{len(relations) + 1}", "source_id": source_id, "type": relation_type, "target_id": target_id, "claim_status": claim})
    return {
        "schema_version": "0.1.0",
        "document_id": job["document_id"],
        "language": job["language"],
        "entities": entities,
        "relations": relations,
    }, errors


def main(args: argparse.Namespace) -> int:
    if VALIDATION_MARKER not in str(args.jobs):
        raise SystemExit("refusing non-validation job input")
    endpoint, env_name, default_model = PROVIDERS[args.provider]
    api_key = os.environ.get(env_name)
    if not api_key:
        raise SystemExit(f"missing {env_name}")
    model = args.model or default_model
    ontology = yaml.safe_load(args.ontology.read_text(encoding="utf-8"))
    jobs = [job for job in load_jsonl(args.jobs) if job["document_id"] in set(args.document_ids)]
    if not jobs:
        raise SystemExit("no validation jobs matched document IDs")
    args.output.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output / "predictions_compact.jsonl"
    audit_path = args.output / "responses.jsonl"
    complete = {row.get("job_id") for row in load_jsonl(audit_path) if row.get("status") in {"success", "failed"}} if audit_path.exists() else set()
    pending = [job for job in jobs if job["job_id"] not in complete]
    system = compact_system()

    def one(job: dict[str, Any]) -> dict[str, Any]:
        user_payload = {"document_id": job["document_id"], "language": job["language"], "ontology": {**job["ontology"], "allowed_relation_signatures": ontology["allowed_relation_signatures"]}, "segments": job["segments"]}
        try:
            response, latency = request_annotation(endpoint, api_key, model, system, json.dumps(user_payload, ensure_ascii=False))
            annotation, warnings = normalise(response["annotation"], job, ontology)
            return {"job_id": job["job_id"], "status": "success", "annotation": annotation, "model": response["model"], "usage": response["usage"], "latency_seconds": round(latency, 3), "normalisation_warnings": warnings}
        except Exception as error:
            return {"job_id": job["job_id"], "status": "failed", "error": str(error)}

    with audit_path.open("a", encoding="utf-8") as stream, ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(one, job) for job in pending]
        for future in as_completed(futures):
            row = future.result()
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(json.dumps({"provider": args.provider, "job_id": row["job_id"], "status": row["status"]}, ensure_ascii=False), flush=True)

    rows = load_jsonl(audit_path)
    by_id = {row["job_id"]: row for row in rows if row.get("status") == "success"}
    with predictions_path.open("w", encoding="utf-8") as stream:
        for job in jobs:
            annotation = by_id.get(job["job_id"], {}).get("annotation", {"schema_version": "0.1.0", "document_id": job["document_id"], "language": job["language"], "entities": [], "relations": []})
            stream.write(json.dumps({"job_id": job["job_id"], "annotation": annotation}, ensure_ascii=False) + "\n")
    summary = {"provider": args.provider, "requested_jobs": len(jobs), "successes": sum(row.get("status") == "success" for row in rows), "failures": sum(row.get("status") == "failed" for row in rows), "documents": sorted(set(args.document_ids)), "model_requested": model, "validation_only": True}
    (args.output / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=sorted(PROVIDERS), required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, default=Path("configs/risk_ontology.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--document-id", dest="document_ids", action="append", required=True)
    parser.add_argument("--model")
    parser.add_argument("--workers", type=int, default=2)
    raise SystemExit(main(parser.parse_args()))
