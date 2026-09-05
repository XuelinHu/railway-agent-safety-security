#!/usr/bin/env python3
"""Convert ordered SpERT validation predictions to project annotation JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


DATASETS = ("conll04", "scierc", "ade")


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def token_offsets(tokens: list[str]) -> list[tuple[int, int]]:
    offsets = []
    position = 0
    for token in tokens:
        offsets.append((position, position + len(token)))
        position += len(token) + 1
    return offsets


def validate_span(entity: dict[str, Any], token_count: int, context: str) -> tuple[int, int]:
    start, end = entity.get("start"), entity.get("end")
    if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= token_count:
        raise ValueError(f"{context}: invalid entity token span ({start!r}, {end!r})")
    return start, end


def convert_document(
    dataset: str,
    source: dict[str, Any],
    prediction: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    source_tokens = source.get("tokens")
    predicted_tokens = prediction.get("tokens")
    if not isinstance(source_tokens, list) or not all(isinstance(token, str) for token in source_tokens):
        raise ValueError(f"{job.get('job_id')}: source tokens are invalid")
    if predicted_tokens != source_tokens:
        raise ValueError(f"{job.get('job_id')}: prediction tokens do not match validation input")
    segments = job.get("segments", [])
    if len(segments) != 1:
        raise ValueError(f"{job.get('job_id')}: expected exactly one source segment")
    segment = segments[0]
    text = " ".join(source_tokens)
    if segment.get("text") != text or segment.get("start") != 0:
        raise ValueError(f"{job.get('job_id')}: token text does not match the public source segment")
    expected_document_id = f"{dataset}_validation_{source.get('orig_id')}"
    if job.get("document_id") != expected_document_id:
        raise ValueError(
            f"{job.get('job_id')}: source orig_id maps to {expected_document_id!r}, "
            f"not {job.get('document_id')!r}"
        )

    ontology = job.get("ontology", {})
    entity_types = set(ontology.get("entity_types", {}))
    relation_types = set(ontology.get("relation_types", {}))
    offsets = token_offsets(source_tokens)
    entities = []
    spans: list[tuple[int, int]] = []
    for index, entity in enumerate(prediction.get("entities", []), 1):
        start_token, end_token = validate_span(entity, len(source_tokens), job["job_id"])
        entity_type = entity.get("type")
        if entity_type not in entity_types:
            raise ValueError(f"{job['job_id']}: unknown predicted entity type {entity_type!r}")
        start = offsets[start_token][0]
        end = offsets[end_token - 1][1]
        entity_text = text[start:end]
        spans.append((start, end))
        entities.append(
            {
                "id": f"E{index}",
                "text": entity_text,
                "type": entity_type,
                "normalized_name": None,
                "evidence": {
                    "text": entity_text,
                    "segment_id": segment["segment_id"],
                    "page": segment.get("page"),
                    "start": start,
                    "end": end,
                },
                "confidence": 1.0,
                "review_status": "pending",
                "created_by": "spert-fresh-seed42",
            }
        )

    relations = []
    for index, relation in enumerate(prediction.get("relations", []), 1):
        head, tail = relation.get("head"), relation.get("tail")
        if not isinstance(head, int) or not isinstance(tail, int):
            raise ValueError(f"{job['job_id']}: relation endpoints must be integers")
        if not 0 <= head < len(entities) or not 0 <= tail < len(entities):
            raise ValueError(f"{job['job_id']}: relation endpoint is outside predicted entities")
        relation_type = relation.get("type")
        if relation_type not in relation_types:
            raise ValueError(f"{job['job_id']}: unknown predicted relation type {relation_type!r}")
        evidence_start = min(spans[head][0], spans[tail][0])
        evidence_end = max(spans[head][1], spans[tail][1])
        relations.append(
            {
                "id": f"R{index}",
                "source_id": f"E{head + 1}",
                "type": relation_type,
                "target_id": f"E{tail + 1}",
                "claim_status": "explicit",
                "evidence": [
                    {
                        "text": text[evidence_start:evidence_end],
                        "segment_id": segment["segment_id"],
                        "page": segment.get("page"),
                        "start": evidence_start,
                        "end": evidence_end,
                    }
                ],
                "confidence": 1.0,
                "review_status": "pending",
                "created_by": "spert-fresh-seed42",
            }
        )

    return {
        "job_id": job["job_id"],
        "annotation": {
            "schema_version": "0.1.0",
            "document_id": job["document_id"],
            "language": job.get("language", "en"),
            "entities": entities,
            "relations": relations,
            "review": {
                "status": "unreviewed",
                "reviewers": [],
                "notes": "Fresh SpERT validation prediction; discrete labels do not export confidence scores.",
            },
        },
    }


def convert(args: argparse.Namespace) -> dict[str, Any]:
    source_rows = read_json(args.validation_data)
    prediction_rows = read_json(args.predictions)
    jobs = read_jsonl(args.jobs)
    if not isinstance(source_rows, list) or not isinstance(prediction_rows, list):
        raise ValueError("SpERT validation data and predictions must be JSON arrays")
    if not len(source_rows) == len(prediction_rows) == len(jobs):
        raise ValueError(
            "row-count mismatch: "
            f"validation={len(source_rows)} predictions={len(prediction_rows)} jobs={len(jobs)}"
        )
    if len({job.get("job_id") for job in jobs}) != len(jobs):
        raise ValueError("public validation jobs contain missing or duplicate job IDs")

    converted = [
        convert_document(args.dataset, source, prediction, job)
        for source, prediction, job in zip(source_rows, prediction_rows, jobs, strict=True)
    ]
    atomic_write_jsonl(args.output, converted)
    summary = {
        "status": "complete",
        "dataset": args.dataset,
        "split": "validation",
        "rows": len(converted),
        "entities": sum(len(row["annotation"]["entities"]) for row in converted),
        "relations": sum(len(row["annotation"]["relations"]) for row in converted),
        "test_split_access": "forbidden-and-not-read",
        "inputs": {
            "validation_data": {"path": str(args.validation_data), "sha256": sha256_file(args.validation_data)},
            "predictions": {"path": str(args.predictions), "sha256": sha256_file(args.predictions)},
            "jobs": {"path": str(args.jobs), "sha256": sha256_file(args.jobs)},
        },
        "output": {"path": str(args.output), "sha256": sha256_file(args.output)},
    }
    atomic_write_json(args.manifest, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = convert(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
