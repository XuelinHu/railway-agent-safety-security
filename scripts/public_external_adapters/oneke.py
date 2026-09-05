#!/usr/bin/env python3
"""Leakage-safe schema adapter for OneKE public validation inference.

The adapter has two deliberately separate operations:

``prepare``
    Convert canonical public *validation jobs* into frozen OneKE NER and RE
    requests.  The operation never accepts a test path and never reads gold.

``convert``
    Convert raw OneKE NER/RE responses into the canonical evidence-bearing
    prediction schema.  Missing or malformed responses become explicit empty
    or unresolved predictions and remain in the evaluation denominator.

No function in this module imports OneKE, Transformers, Torch, or CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REQUEST_SCHEMA = "public-oneke-request-v1"
RAW_SCHEMA = "public-oneke-raw-response-v1"
CONVERSION_SCHEMA = "public-oneke-canonical-conversion-v1"
DATASETS = {"conll04": 231, "scierc": 275, "ade": 384}
SPLIT = "validation"
SEED = 42
UNKNOWN_TYPE = "__ONEKE_UNTYPED_ENDPOINT__"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: invalid JSON on line {line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}: line {line_number} is not an object")
        rows.append(value)
    return rows


def reject_test_path(path: Path, role: str) -> None:
    # Inspect every component, not just the basename.  Otherwise a seemingly
    # harmless ``validation_jobs.jsonl`` below a ``.../test/...`` directory
    # could bypass the split seal.  Match ``test`` as a path-name token so
    # ordinary names such as ``pytest`` or ``contest`` are not false positives.
    has_test_namespace = any(
        "test" in re.split(r"[^a-z0-9]+", component.casefold())
        for component in path.parts
    )
    if has_test_namespace:
        raise ValueError(f"{role} must not reference a test namespace: {path}")


def keys(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [str(item) for item in value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def require_unique_job_ids(rows: list[dict[str, Any]], path: Path) -> None:
    identifiers = [row.get("job_id") for row in rows]
    if any(not isinstance(value, str) or not value for value in identifiers):
        raise ValueError(f"{path}: every row must have a non-empty string job_id")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"{path}: duplicate job_id")


def one_source_segment(job: dict[str, Any]) -> dict[str, Any]:
    segments = job.get("segments")
    if not isinstance(segments, list) or len(segments) != 1:
        raise ValueError(f"{job.get('job_id')}: public OneKE jobs require exactly one segment")
    segment = segments[0]
    if not isinstance(segment, dict):
        raise ValueError(f"{job.get('job_id')}: segment is not an object")
    text = segment.get("text")
    start = segment.get("start")
    end = segment.get("end")
    if not isinstance(text, str) or not isinstance(start, int) or not isinstance(end, int):
        raise ValueError(f"{job.get('job_id')}: invalid source segment")
    if end - start != len(text):
        raise ValueError(f"{job.get('job_id')}: source segment offsets do not match text length")
    return segment


def request_from_job(job: dict[str, Any], dataset: str) -> dict[str, Any]:
    if job.get("category") != dataset:
        raise ValueError(
            f"{job.get('job_id')}: category {job.get('category')!r} is not {dataset!r}"
        )
    source_path = str(job.get("source_path", ""))
    if not source_path.startswith(f"public:{dataset}:validation:"):
        raise ValueError(f"{job.get('job_id')}: source_path is not frozen validation data")
    segment = one_source_segment(job)
    ontology = job.get("ontology")
    if not isinstance(ontology, dict):
        raise ValueError(f"{job.get('job_id')}: ontology is missing")
    entity_types = keys(ontology.get("entity_types"))
    relation_types = keys(ontology.get("relation_types"))
    if not entity_types or not relation_types:
        raise ValueError(f"{job.get('job_id')}: entity/relation constraints are empty")
    signatures = ontology.get("allowed_relation_signatures", {})
    if not isinstance(signatures, dict):
        raise ValueError(f"{job.get('job_id')}: relation signatures are invalid")
    return {
        "schema_version": REQUEST_SCHEMA,
        "job_id": job["job_id"],
        "document_id": job.get("document_id"),
        "dataset": dataset,
        "split": SPLIT,
        "seed": SEED,
        "source": {
            "segment_id": segment.get("segment_id"),
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"],
        },
        "tasks": {
            "ner": {
                "task": "NER",
                "constraint": entity_types,
                "expected_key": "entity_list",
            },
            "re": {
                "task": "RE",
                "constraint": relation_types,
                "expected_key": "relation_list",
            },
        },
        "canonical_schema": {
            "entity_types": entity_types,
            "relation_types": relation_types,
            "allowed_relation_signatures": signatures,
            "claim_status": "explicit",
        },
        "provenance": {
            "source_path": source_path,
            "prompt_uses_gold": False,
            "test_gold_read": False,
        },
    }


def prepare_requests(
    jobs_path: Path, dataset: str, expected_rows: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if dataset not in DATASETS:
        raise ValueError(f"unsupported public dataset: {dataset}")
    reject_test_path(jobs_path, "jobs")
    jobs = load_jsonl(jobs_path)
    require_unique_job_ids(jobs, jobs_path)
    if len(jobs) != expected_rows:
        raise ValueError(
            f"{dataset}: expected {expected_rows} complete validation jobs, found {len(jobs)}"
        )
    requests = [request_from_job(job, dataset) for job in jobs]
    return requests, {
        "schema_version": REQUEST_SCHEMA,
        "status": "prepared",
        "dataset": dataset,
        "split": SPLIT,
        "jobs": len(requests),
        "jobs_sha256": sha256(jobs_path),
        "gold_read": False,
        "test_gold_read": False,
        "generated_at": utc_now(),
    }


def unwrap_json(value: Any) -> tuple[dict[str, Any], str | None]:
    if isinstance(value, dict):
        return value, None
    if not isinstance(value, str):
        return {}, f"response is {type(value).__name__}, not object/string"
    text = value.strip()
    fenced = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates = [*reversed(fenced), text]
    first, last = text.find("{"), text.rfind("}")
    if 0 <= first < last:
        candidates.append(text[first : last + 1])
    errors: list[str] = []
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(str(exc))
            continue
        if isinstance(parsed, dict):
            return parsed, None
        errors.append("decoded JSON is not an object")
    return {}, "; ".join(errors[-2:]) or "no JSON object found"


def response_payload(raw: dict[str, Any], task: str) -> tuple[dict[str, Any], str | None]:
    lowered = task.casefold()
    tasks = raw.get("tasks", {})
    if isinstance(tasks, dict) and isinstance(tasks.get(lowered), dict):
        task_record = tasks[lowered]
        for field in ("parsed", "result", "raw_text"):
            if field in task_record:
                return unwrap_json(task_record[field])
    for field in (f"{lowered}_result", f"{lowered}_response"):
        if field in raw:
            return unwrap_json(raw[field])
    return {}, f"{task} response is missing"


def canonical_label(value: Any, allowed: list[str], fallback: str) -> tuple[str, bool]:
    raw = str(value or "").strip()
    def key(label: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", label.casefold())

    mapping = {key(item): item for item in allowed}
    if key(raw) in mapping:
        return mapping[key(raw)], True
    return raw or fallback, False


def source_occurrences(text: str, mention: str) -> list[tuple[int, int]]:
    if not mention:
        return []
    exact: list[tuple[int, int]] = []
    cursor = 0
    while True:
        position = text.find(mention, cursor)
        if position < 0:
            break
        exact.append((position, position + len(mention)))
        cursor = position + max(1, len(mention))
    if exact:
        return exact
    return [match.span() for match in re.finditer(re.escape(mention), text, re.IGNORECASE)]


def entity_evidence(
    source: dict[str, Any], mention: str, used: set[tuple[int, int]]
) -> tuple[dict[str, Any] | None, str | None]:
    for local_start, local_end in source_occurrences(source["text"], mention):
        span = (local_start, local_end)
        if span in used:
            continue
        used.add(span)
        exact = source["text"][local_start:local_end]
        return {
            "segment_id": source["segment_id"],
            "start": source["start"] + local_start,
            "end": source["start"] + local_end,
            "text": exact,
        }, None
    return None, "mention_not_found_or_occurrences_exhausted"


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def normalize_mention(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def relation_evidence(
    source: dict[str, Any], head: dict[str, Any], tail: dict[str, Any]
) -> dict[str, Any] | None:
    first = head.get("evidence")
    second = tail.get("evidence")
    if not isinstance(first, dict) or not isinstance(second, dict):
        return None
    if first.get("segment_id") != source["segment_id"] or second.get("segment_id") != source["segment_id"]:
        return None
    start = min(first["start"], second["start"])
    end = max(first["end"], second["end"])
    local_start, local_end = start - source["start"], end - source["start"]
    return {
        "segment_id": source["segment_id"],
        "start": start,
        "end": end,
        "text": source["text"][local_start:local_end],
    }


def convert_one(
    request: dict[str, Any], raw: dict[str, Any] | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = request["source"]
    schema = request["canonical_schema"]
    entity_types = list(schema["entity_types"])
    relation_types = list(schema["relation_types"])
    issues: list[dict[str, Any]] = []
    if raw is None:
        raw = {"status": "terminal_failure"}
        issues.append({"kind": "missing_raw_response"})
    elif raw.get("job_id") != request["job_id"]:
        raise ValueError(f"raw/request job mismatch for {request['job_id']}")

    ner, ner_error = response_payload(raw, "NER")
    re_payload, re_error = response_payload(raw, "RE")
    if ner_error:
        issues.append({"kind": "ner_parse_error", "detail": ner_error})
    if re_error:
        issues.append({"kind": "re_parse_error", "detail": re_error})
    raw_entities = as_list(ner.get("entity_list"))
    raw_relations = as_list(re_payload.get("relation_list"))
    if ner and not isinstance(ner.get("entity_list", []), list):
        issues.append({"kind": "ner_entity_list_not_list"})
    if re_payload and not isinstance(re_payload.get("relation_list", []), list):
        issues.append({"kind": "re_relation_list_not_list"})

    entities: list[dict[str, Any]] = []
    by_text: dict[str, list[dict[str, Any]]] = defaultdict(list)
    used_spans: set[tuple[int, int]] = set()

    def add_entity(name: Any, entity_type: Any, origin: str) -> dict[str, Any]:
        mention = str(name or "").strip()
        canonical_type, known_type = canonical_label(entity_type, entity_types, UNKNOWN_TYPE)
        evidence, evidence_error = entity_evidence(source, mention, used_spans)
        entity = {
            "id": f"E{len(entities) + 1}",
            "text": evidence["text"] if evidence else mention,
            "type": canonical_type,
            "evidence": evidence,
            "claim_status": "explicit",
            "review_status": "unreviewed",
            "created_by": "OneKE",
        }
        entities.append(entity)
        by_text[normalize_mention(mention)].append(entity)
        if not known_type:
            issues.append(
                {
                    "kind": "unknown_entity_type",
                    "entity_id": entity["id"],
                    "value": canonical_type,
                    "origin": origin,
                }
            )
        if evidence_error:
            issues.append(
                {
                    "kind": "unresolved_entity_span",
                    "entity_id": entity["id"],
                    "text": mention,
                    "origin": origin,
                }
            )
        return entity

    for position, item in enumerate(raw_entities):
        if not isinstance(item, dict):
            issues.append({"kind": "invalid_entity_item", "position": position})
            continue
        add_entity(item.get("name", item.get("text")), item.get("type", item.get("label")), "ner")

    def endpoint(name: Any, relation_position: int, role: str) -> dict[str, Any]:
        normalized = normalize_mention(name)
        if by_text.get(normalized):
            return by_text[normalized][0]
        issues.append(
            {
                "kind": "relation_endpoint_missing_from_ner",
                "relation_position": relation_position,
                "role": role,
                "text": str(name or ""),
            }
        )
        return add_entity(name, UNKNOWN_TYPE, f"re_{role}")

    relations: list[dict[str, Any]] = []
    for position, item in enumerate(raw_relations):
        if not isinstance(item, dict):
            issues.append({"kind": "invalid_relation_item", "position": position})
            continue
        relation_type, known_relation = canonical_label(
            item.get("relation", item.get("type")), relation_types, "__ONEKE_UNKNOWN_RELATION__"
        )
        head = endpoint(item.get("head", item.get("source")), position, "head")
        tail = endpoint(item.get("tail", item.get("target")), position, "tail")
        if not known_relation:
            issues.append(
                {
                    "kind": "unknown_relation_type",
                    "relation_position": position,
                    "value": relation_type,
                }
            )
        relations.append(
            {
                "id": f"R{len(relations) + 1}",
                "source_id": head["id"],
                "target_id": tail["id"],
                "type": relation_type,
                "claim_status": "explicit",
                "evidence": relation_evidence(source, head, tail),
                "review_status": "unreviewed",
                "created_by": "OneKE",
            }
        )

    annotation = {
        "document_id": request.get("document_id"),
        "language": "en",
        "category": request["dataset"],
        "entities": entities,
        "relations": relations,
        "review": {"status": "unreviewed"},
    }
    prediction = {
        "job_id": request["job_id"],
        "annotation": annotation,
        "external_baseline": {
            "name": "OneKE",
            "checkpoint_mode": "published_checkpoint_4bit",
            "raw_status": raw.get("status", "unknown"),
            "terminal_failure": raw.get("status") == "terminal_failure",
        },
    }
    audit = {
        "job_id": request["job_id"],
        "schema_version": CONVERSION_SCHEMA,
        "raw_status": raw.get("status", "unknown"),
        "raw_entities": len(raw_entities),
        "raw_relations": len(raw_relations),
        "canonical_entities": len(entities),
        "canonical_relations": len(relations),
        "issues": issues,
        "gold_read": False,
        "test_gold_read": False,
    }
    return prediction, audit


def convert_responses(
    requests_path: Path, raw_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    reject_test_path(requests_path, "requests")
    reject_test_path(raw_path, "raw responses")
    requests = load_jsonl(requests_path)
    raw_rows = load_jsonl(raw_path) if raw_path.is_file() else []
    require_unique_job_ids(requests, requests_path)
    require_unique_job_ids(raw_rows, raw_path) if raw_rows else None
    raw_by_job = {row["job_id"]: row for row in raw_rows}
    unknown = sorted(set(raw_by_job) - {row["job_id"] for row in requests})
    if unknown:
        raise ValueError(f"raw responses contain unknown job IDs: {unknown[:3]}")
    predictions: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for request in requests:
        if request.get("schema_version") != REQUEST_SCHEMA or request.get("split") != SPLIT:
            raise ValueError(f"{request.get('job_id')}: request contract mismatch")
        prediction, audit = convert_one(request, raw_by_job.get(request["job_id"]))
        predictions.append(prediction)
        audits.append(audit)
    terminal_failures = sum(
        item["external_baseline"]["terminal_failure"] for item in predictions
    )
    return predictions, audits, {
        "schema_version": CONVERSION_SCHEMA,
        "status": "complete_with_terminal_failures" if terminal_failures else "complete",
        "requests": len(requests),
        "raw_responses": len(raw_rows),
        "predictions": len(predictions),
        "terminal_failures": terminal_failures,
        "gold_read": False,
        "test_gold_read": False,
        "generated_at": utc_now(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    prepare.add_argument("--jobs", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--summary", type=Path, required=True)
    prepare.add_argument("--expected-rows", type=int)

    convert = subparsers.add_parser("convert")
    convert.add_argument("--requests", type=Path, required=True)
    convert.add_argument("--raw", type=Path, required=True)
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--audit", type=Path, required=True)
    convert.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        expected = args.expected_rows if args.expected_rows is not None else DATASETS[args.dataset]
        requests, summary = prepare_requests(args.jobs, args.dataset, expected)
        write_jsonl_atomic(args.output, requests)
        summary["output"] = str(args.output)
        summary["output_sha256"] = sha256(args.output)
        write_json_atomic(args.summary, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    predictions, audits, summary = convert_responses(args.requests, args.raw)
    write_jsonl_atomic(args.output, predictions)
    write_jsonl_atomic(args.audit, audits)
    summary.update(
        {
            "output": str(args.output),
            "output_sha256": sha256(args.output),
            "audit": str(args.audit),
            "audit_sha256": sha256(args.audit),
        }
    )
    write_json_atomic(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
