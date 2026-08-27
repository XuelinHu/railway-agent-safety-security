#!/usr/bin/env python3
"""Validate annotation JSON/JSONL against schema, source offsets, and ontology signatures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import yaml
from jsonschema import Draft202012Validator


def records(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    if path.suffix.lower() == ".jsonl":
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                yield line_number, json.loads(line)
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            for index, item in enumerate(value, 1):
                yield index, item
        else:
            yield 1, value


def semantic_errors(annotation: dict[str, Any], document: dict[str, Any], ontology: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    text = document["text"]
    segment_ids = {segment["segment_id"] for segment in document["segments"]}
    entities = annotation.get("entities", [])
    entity_by_id = {entity.get("id"): entity for entity in entities}
    if len(entity_by_id) != len(entities):
        errors.append("duplicate entity IDs")

    def check_evidence(evidence: dict[str, Any], label: str) -> None:
        start, end = evidence.get("start"), evidence.get("end")
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end > len(text) or start >= end:
            errors.append(f"{label}: invalid offsets")
            return
        if text[start:end] != evidence.get("text"):
            errors.append(f"{label}: evidence text does not match document offsets")
        base_segment_id = str(evidence.get("segment_id", "")).split(".", 1)[0]
        if base_segment_id not in segment_ids:
            errors.append(f"{label}: unknown segment_id {evidence.get('segment_id')}")

    for entity in entities:
        check_evidence(entity.get("evidence", {}), f"entity {entity.get('id')}")
        evidence = entity.get("evidence", {})
        entity_text = entity.get("text", "")
        if entity_text and entity_text not in evidence.get("text", ""):
            errors.append(f"entity {entity.get('id')}: entity text is not inside evidence")

    relation_ids: set[str] = set()
    signatures = ontology.get("allowed_relation_signatures", {})
    for relation in annotation.get("relations", []):
        relation_id = relation.get("id")
        if relation_id in relation_ids:
            errors.append(f"duplicate relation ID {relation_id}")
        relation_ids.add(relation_id)
        source = entity_by_id.get(relation.get("source_id"))
        target = entity_by_id.get(relation.get("target_id"))
        if not source or not target:
            errors.append(f"relation {relation_id}: unknown source or target entity")
            continue
        signature = signatures.get(relation.get("type"))
        if signature:
            if source.get("type") not in signature.get("source", []):
                errors.append(f"relation {relation_id}: illegal source type {source.get('type')}")
            if target.get("type") not in signature.get("target", []):
                errors.append(f"relation {relation_id}: illegal target type {target.get('type')}")
        for evidence_index, evidence in enumerate(relation.get("evidence", []), 1):
            check_evidence(evidence, f"relation {relation_id} evidence {evidence_index}")
    return errors


def validate(args: argparse.Namespace) -> int:
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    ontology = yaml.safe_load(args.ontology.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    failures = 0
    total = 0
    for record_number, record in records(args.annotations):
        total += 1
        annotation = record.get("annotation", record)
        record_errors = [error.message for error in validator.iter_errors(annotation)]
        document_id = annotation.get("document_id", "")
        document_path = args.document_root / f"{document_id}.json"
        if not document_path.exists():
            record_errors.append(f"source document not found: {document_path}")
        else:
            document = json.loads(document_path.read_text(encoding="utf-8"))
            record_errors.extend(semantic_errors(annotation, document, ontology))
        if record_errors:
            failures += 1
            print(f"record {record_number} ({document_id}):")
            for error in record_errors:
                print(f"  - {error}")
    print(json.dumps({"records": total, "failed": failures, "passed": total - failures}, indent=2))
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--schema", type=Path, default=Path("schemas/risk_annotation.schema.json"))
    parser.add_argument("--ontology", type=Path, default=Path("configs/risk_ontology.yaml"))
    parser.add_argument("--document-root", type=Path, default=Path("data/processed/corpus/documents"))
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(validate(parse_args()))
