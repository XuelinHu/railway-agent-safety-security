#!/usr/bin/env python3
"""Resolve teacher evidence quotes to deterministic source offsets and pages."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def flexible_span(source: str, value: str) -> tuple[int, int]:
    exact_start = source.find(value)
    if exact_start >= 0:
        if source.find(value, exact_start + 1) >= 0:
            raise ValueError(f"ambiguous repeated text: {value[:80]!r}")
        return exact_start, exact_start + len(value)
    tokens = re.split(r"\s+", value.strip())
    if not tokens or not all(tokens):
        raise ValueError("empty text")
    pattern = r"\s+".join(re.escape(token) for token in tokens)
    matches = list(re.finditer(pattern, source))
    if len(matches) != 1:
        reason = "not found" if not matches else "ambiguous"
        raise ValueError(f"{reason} after whitespace normalization: {value[:80]!r}")
    return matches[0].start(), matches[0].end()


def resolve_evidence(evidence: dict[str, Any], segment_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    segment = segment_by_id.get(evidence["segment_id"])
    if not segment:
        raise ValueError(f"unknown segment_id: {evidence['segment_id']}")
    quote = evidence["text"]
    # Expanded compact predictions already carry deterministic offsets. Prefer
    # them when the quoted text matches the source at that offset; this also
    # handles repeated mentions that cannot be resolved by quote uniqueness.
    if "start" in evidence and "end" in evidence:
        local_start = int(evidence["start"]) - int(segment["start"])
        local_end = int(evidence["end"]) - int(segment["start"])
        if 0 <= local_start <= local_end <= len(segment["text"]):
            source_quote = segment["text"][local_start:local_end]
            if source_quote == quote:
                return {
                    "text": source_quote,
                    "segment_id": evidence["segment_id"],
                    "page": segment.get("page"),
                    "start": segment["start"] + local_start,
                    "end": segment["start"] + local_end,
                }
    try:
        local_start, local_end = flexible_span(segment["text"], quote)
    except ValueError as error:
        raise ValueError(f"invalid evidence in {evidence['segment_id']}: {error}") from error
    start = segment["start"] + local_start
    return {
        "text": segment["text"][local_start:local_end],
        "segment_id": evidence["segment_id"],
        "page": segment.get("page"),
        "start": start,
        "end": segment["start"] + local_end,
    }


def normalize(args: argparse.Namespace) -> int:
    jobs = {job["job_id"]: job for job in load_jsonl(args.jobs)}
    candidates = load_jsonl(args.candidates)
    ontology = yaml.safe_load(args.ontology.read_text(encoding="utf-8"))
    signatures = ontology.get("allowed_relation_signatures", {})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.errors.parent.mkdir(parents=True, exist_ok=True)
    successes = 0
    failures = 0
    invalid_candidates = 0
    with args.output.open("w", encoding="utf-8") as output_stream, args.errors.open("w", encoding="utf-8") as error_stream:
        for item in candidates:
            job_id = item["job_id"]
            candidate = item["annotation"]
            job = jobs.get(job_id)
            if not job:
                error_stream.write(json.dumps({"job_id": job_id, "error": "job not found"}) + "\n")
                failures += 1
                continue
            segment_by_id = {segment["segment_id"]: segment for segment in job["segments"]}
            candidate_errors: list[str] = []
            try:
                annotation = dict(candidate)
                normalized_entities: list[dict[str, Any]] = []
                valid_entity_ids: set[str] = set()
                for entity in candidate["entities"]:
                    try:
                        evidence = resolve_evidence(entity["evidence"], segment_by_id)
                        local_start, local_end = flexible_span(evidence["text"], entity["text"])
                        source_entity_text = evidence["text"][local_start:local_end]
                        normalized_entity = {**entity, "text": source_entity_text, "evidence": evidence}
                        if source_entity_text != entity["text"]:
                            normalized_entity["normalized_name"] = entity["text"]
                        normalized_entities.append(normalized_entity)
                        valid_entity_ids.add(entity["id"])
                    except Exception as error:
                        candidate_errors.append(f"entity {entity.get('id')}: {error}")
                normalized_relations: list[dict[str, Any]] = []
                for relation in candidate["relations"]:
                    try:
                        if relation["source_id"] not in valid_entity_ids or relation["target_id"] not in valid_entity_ids:
                            raise ValueError("source or target entity was rejected")
                        normalized_relations.append(
                            {
                                **relation,
                                "evidence": [
                                    resolve_evidence(evidence, segment_by_id) for evidence in relation["evidence"]
                                ],
                            }
                        )
                    except Exception as error:
                        candidate_errors.append(f"relation {relation.get('id')}: {error}")
                annotation["entities"] = normalized_entities
                entity_by_id = {entity["id"]: entity for entity in normalized_entities}
                annotation["relations"] = normalized_relations
                constrained_relations: list[dict[str, Any]] = []
                for relation in normalized_relations:
                    signature = signatures.get(relation["type"])
                    source_type = entity_by_id[relation["source_id"]]["type"]
                    target_type = entity_by_id[relation["target_id"]]["type"]
                    if signature and (
                        source_type not in signature.get("source", [])
                        or target_type not in signature.get("target", [])
                    ):
                        candidate_errors.append(
                            f"relation {relation['id']}: illegal signature {source_type} -{relation['type']}-> {target_type}"
                        )
                        continue
                    constrained_relations.append(relation)
                annotation["relations"] = constrained_relations
                output_record: dict[str, Any] = {"job_id": job_id, "annotation": annotation} if args.include_job_id else annotation
                output_stream.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                successes += 1
                if candidate_errors:
                    invalid_candidates += len(candidate_errors)
                    error_stream.write(
                        json.dumps({"job_id": job_id, "candidate_errors": candidate_errors}, ensure_ascii=False) + "\n"
                    )
            except Exception as error:
                error_stream.write(json.dumps({"job_id": job_id, "error": str(error)}, ensure_ascii=False) + "\n")
                failures += 1
    print(json.dumps({"normalized": successes, "failed": failures, "invalid_candidates": invalid_candidates}, indent=2))
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, default=Path("data/processed/preannotation/jobs.jsonl"))
    parser.add_argument("--candidates", type=Path, default=Path("data/processed/preannotation/candidates.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/preannotation/normalized.jsonl"))
    parser.add_argument("--errors", type=Path, default=Path("outputs/preannotation_normalization_errors.jsonl"))
    parser.add_argument("--ontology", type=Path, default=Path("configs/risk_ontology.yaml"))
    parser.add_argument("--include-job-id", action="store_true", help="Write {job_id, annotation} envelopes for experiment alignment")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(normalize(parse_args()))
