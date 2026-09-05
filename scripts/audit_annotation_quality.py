#!/usr/bin/env python3
"""Audit annotation and source-evidence quality without changing annotations."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def annotation_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("annotation", row)


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def matching_spans(source: str, value: str) -> list[tuple[int, int]]:
    if not value or not value.strip():
        return []
    exact = [(match.start(), match.end()) for match in re.finditer(re.escape(value), source)]
    if exact:
        return exact
    tokens = re.split(r"\s+", value.strip())
    pattern = r"\s+".join(re.escape(token) for token in tokens if token)
    return [(match.start(), match.end()) for match in re.finditer(pattern, source)] if pattern else []


def evidence_items(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def evidence_quote_is_source(evidence: dict[str, Any], segment_by_id: dict[str, dict[str, Any]]) -> bool:
    segment = segment_by_id.get(evidence.get("segment_id"))
    return bool(segment and matching_spans(segment.get("text", ""), evidence.get("text", "")))


def evidence_contains(evidence: dict[str, Any], text: str) -> bool:
    return bool(normalize_text(text) and normalize_text(text) in normalize_text(evidence.get("text")))


def add_issue(issues: list[dict[str, Any]], counts: Counter[str], job_id: str | None, kind: str, item_id: Any, detail: str) -> None:
    counts[kind] += 1
    issues.append({"job_id": job_id, "kind": kind, "item_id": item_id, "detail": detail})


def audit_annotation(
    annotation: dict[str, Any],
    job: dict[str, Any] | None,
    ontology: dict[str, Any],
    job_id: str | None,
) -> tuple[list[dict[str, Any]], Counter[str], dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    if not job:
        add_issue(issues, counts, job_id, "job_not_found", None, "annotation has no matching source job")
        return issues, counts, {"job_id": job_id, "language": annotation.get("language")}

    segments = job.get("segments", [])
    segment_by_id = {segment.get("segment_id"): segment for segment in segments}
    diagnostics = {
        "job_id": job_id,
        "document_id": annotation.get("document_id"),
        "language": annotation.get("language", job.get("language")),
        "segment_count": len(segments),
        "character_count": sum(len(segment.get("text", "")) for segment in segments),
        "max_segment_characters": max((len(segment.get("text", "")) for segment in segments), default=0),
    }
    entity_by_id = {entity.get("id"): entity for entity in annotation.get("entities", []) if entity.get("id")}
    entity_types = set(ontology.get("entity_types", {}))
    relation_types = set(ontology.get("relation_types", {}))
    claim_statuses = set(ontology.get("claim_statuses", {}))
    signatures = ontology.get("allowed_relation_signatures", {})

    for entity in annotation.get("entities", []):
        entity_id = entity.get("id")
        entity_text = entity.get("text", "")
        if entity.get("type") not in entity_types:
            add_issue(issues, counts, job_id, "unknown_entity_type", entity_id, str(entity.get("type")))
        matches = [
            (segment.get("segment_id"), span)
            for segment in segments
            for span in matching_spans(segment.get("text", ""), entity_text)
        ]
        entity_evidence = entity.get("evidence")
        has_explicit_offset = isinstance(entity_evidence, dict) and entity_evidence.get("start") is not None
        if not matches:
            add_issue(issues, counts, job_id, "entity_text_not_found", entity_id, entity_text[:120])
        elif len(matches) > 1 and not has_explicit_offset:
            add_issue(issues, counts, job_id, "entity_text_ambiguous", entity_id, f"{len(matches)} occurrences: {entity_text[:100]}")
        evidence = entity.get("evidence")
        if not isinstance(evidence, dict) or not evidence_quote_is_source(evidence, segment_by_id):
            add_issue(issues, counts, job_id, "entity_evidence_invalid", entity_id, "evidence is missing or not a source quote")

    for relation in annotation.get("relations", []):
        relation_id = relation.get("id")
        source = entity_by_id.get(relation.get("source_id"))
        target = entity_by_id.get(relation.get("target_id"))
        if not source or not target:
            add_issue(issues, counts, job_id, "relation_entity_reference_invalid", relation_id, "source or target entity is missing")
            continue
        relation_type = relation.get("type")
        if relation_type not in relation_types:
            add_issue(issues, counts, job_id, "unknown_relation_type", relation_id, str(relation_type))
        else:
            signature = signatures.get(relation_type, {})
            if source.get("type") not in signature.get("source", []) or target.get("type") not in signature.get("target", []):
                add_issue(issues, counts, job_id, "illegal_relation_signature", relation_id, f"{source.get('type')} -{relation_type}-> {target.get('type')}")
        if relation.get("claim_status") not in claim_statuses:
            add_issue(issues, counts, job_id, "claim_status_invalid", relation_id, str(relation.get("claim_status")))
        evidence = evidence_items(relation.get("evidence"))
        if not evidence or not all(evidence_quote_is_source(item, segment_by_id) for item in evidence):
            add_issue(issues, counts, job_id, "relation_evidence_invalid", relation_id, "missing or non-source relation evidence")
        if evidence and not any(
            evidence_contains(item, source.get("text", "")) and evidence_contains(item, target.get("text", ""))
            for item in evidence
        ):
            add_issue(issues, counts, job_id, "relation_evidence_missing_entity", relation_id, "no relation quote contains both endpoints")

    return issues, counts, diagnostics


def run(args: argparse.Namespace) -> int:
    jobs = {job["job_id"]: job for job in load_jsonl(args.jobs)}
    rows = load_jsonl(args.annotations)
    index = load_jsonl(args.index) if args.index and args.index.exists() else []
    ontology = yaml.safe_load(args.ontology.read_text(encoding="utf-8"))
    all_issues: list[dict[str, Any]] = []
    total_counts: Counter[str] = Counter()
    diagnostics: list[dict[str, Any]] = []

    for position, row in enumerate(rows):
        annotation = annotation_from_row(row)
        job_id = row.get("job_id")
        if job_id is None and position < len(index):
            job_id = index[position].get("job_id")
        issues, counts, job_diagnostics = audit_annotation(annotation, jobs.get(job_id), ontology, job_id)
        all_issues.extend(issues)
        total_counts.update(counts)
        diagnostics.append(job_diagnostics)

    summary = {
        "records": len(rows),
        "entities": sum(len(annotation_from_row(row).get("entities", [])) for row in rows),
        "relations": sum(len(annotation_from_row(row).get("relations", [])) for row in rows),
        "issue_count": len(all_issues),
        "issue_counts": dict(sorted(total_counts.items())),
        "source_diagnostics": diagnostics,
        "annotations": str(args.annotations),
        "issues": str(args.issues),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.issues.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with args.issues.open("w", encoding="utf-8") as stream:
        for issue in all_issues:
            stream.write(json.dumps(issue, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--issues", type=Path, required=True)
    parser.add_argument("--index", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
