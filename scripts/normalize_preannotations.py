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


def matching_spans(source: str, value: str) -> list[tuple[int, int]]:
    """Return all exact matches, falling back to whitespace-normalized matches."""
    if not value.strip():
        return []
    exact_matches = [
        (match.start(), match.end()) for match in re.finditer(re.escape(value), source)
    ]
    if exact_matches:
        return exact_matches
    tokens = re.split(r"\s+", value.strip())
    if not tokens or not all(tokens):
        return []
    pattern = r"\s+".join(re.escape(token) for token in tokens)
    return [(match.start(), match.end()) for match in re.finditer(pattern, source)]


def locate_unique_entity_evidence(
    entity_text: str, segment_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Recover evidence only when the entity text has one source occurrence."""
    matches: list[tuple[dict[str, Any], int, int]] = []
    for segment in segment_by_id.values():
        for local_start, local_end in matching_spans(segment["text"], entity_text):
            matches.append((segment, local_start, local_end))
    if len(matches) != 1:
        reason = "not found" if not matches else f"ambiguous ({len(matches)} occurrences)"
        raise ValueError(f"entity text {reason} across supplied segments: {entity_text[:80]!r}")
    segment, local_start, local_end = matches[0]
    return {
        "text": segment["text"][local_start:local_end],
        "segment_id": segment["segment_id"],
        "page": segment.get("page"),
        "start": segment["start"] + local_start,
        "end": segment["start"] + local_end,
    }


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


def relation_signature_is_legal(
    relation_type: str,
    source_type: str,
    target_type: str,
    signatures: dict[str, Any],
) -> bool:
    signature = signatures.get(relation_type)
    if not signature:
        return False
    return source_type in signature.get("source", []) and target_type in signature.get("target", [])


def constrain_relation_direction(
    relation: dict[str, Any],
    entity_by_id: dict[str, dict[str, Any]],
    signatures: dict[str, Any],
    repair_inverse: bool = False,
) -> tuple[dict[str, Any] | None, str | None]:
    source_type = entity_by_id[relation["source_id"]]["type"]
    target_type = entity_by_id[relation["target_id"]]["type"]
    if relation_signature_is_legal(relation["type"], source_type, target_type, signatures):
        return relation, None
    if repair_inverse and relation_signature_is_legal(
        relation["type"], target_type, source_type, signatures
    ):
        repaired = {
            **relation,
            "source_id": relation["target_id"],
            "target_id": relation["source_id"],
        }
        message = (
            f"relation {relation['id']}: swapped direction from "
            f"{source_type} -{relation['type']}-> {target_type}"
        )
        return repaired, message
    return None, (
        f"relation {relation['id']}: illegal signature "
        f"{source_type} -{relation['type']}-> {target_type}"
    )


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
    repaired_candidates = 0
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
            candidate_repairs: list[str] = []
            try:
                annotation = dict(candidate)
                normalized_entities: list[dict[str, Any]] = []
                valid_entity_ids: set[str] = set()
                for entity in candidate["entities"]:
                    try:
                        try:
                            evidence = resolve_evidence(entity["evidence"], segment_by_id)
                            local_start, local_end = flexible_span(evidence["text"], entity["text"])
                        except ValueError as original_error:
                            if not args.repair_unique_entity_evidence:
                                raise
                            evidence = locate_unique_entity_evidence(entity["text"], segment_by_id)
                            local_start, local_end = 0, len(evidence["text"])
                            candidate_repairs.append(
                                f"entity {entity['id']}: replaced invalid evidence with unique source span "
                                f"({original_error})"
                            )
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
                        normalized_relation = {
                            **relation,
                            "evidence": [
                                resolve_evidence(evidence, segment_by_id) for evidence in relation["evidence"]
                            ],
                        }
                        if "claim_status" not in normalized_relation:
                            if not args.default_missing_claim_status:
                                raise ValueError("missing claim_status")
                            normalized_relation["claim_status"] = args.default_missing_claim_status
                            candidate_repairs.append(
                                f"relation {relation['id']}: defaulted missing claim_status to "
                                f"{args.default_missing_claim_status}"
                            )
                        normalized_relations.append(normalized_relation)
                    except Exception as error:
                        candidate_errors.append(f"relation {relation.get('id')}: {error}")
                annotation["entities"] = normalized_entities
                entity_by_id = {entity["id"]: entity for entity in normalized_entities}
                annotation["relations"] = normalized_relations
                constrained_relations: list[dict[str, Any]] = []
                for relation in normalized_relations:
                    constrained, repair = constrain_relation_direction(
                        relation,
                        entity_by_id,
                        signatures,
                        repair_inverse=args.repair_inverse_relations,
                    )
                    if constrained is None:
                        candidate_errors.append(repair or f"relation {relation['id']}: rejected")
                        continue
                    constrained_relations.append(constrained)
                    if repair:
                        candidate_repairs.append(repair)
                annotation["relations"] = constrained_relations
                output_record: dict[str, Any] = {"job_id": job_id, "annotation": annotation} if args.include_job_id else annotation
                output_stream.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                successes += 1
                if candidate_errors or candidate_repairs:
                    invalid_candidates += len(candidate_errors)
                    repaired_candidates += len(candidate_repairs)
                    error_stream.write(
                        json.dumps(
                            {
                                "job_id": job_id,
                                "candidate_errors": candidate_errors,
                                "candidate_repairs": candidate_repairs,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            except Exception as error:
                error_stream.write(json.dumps({"job_id": job_id, "error": str(error)}, ensure_ascii=False) + "\n")
                failures += 1
    print(
        json.dumps(
            {
                "normalized": successes,
                "failed": failures,
                "invalid_candidates": invalid_candidates,
                "candidate_repairs": repaired_candidates,
            },
            indent=2,
        )
    )
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, default=Path("data/processed/preannotation/jobs.jsonl"))
    parser.add_argument("--candidates", type=Path, default=Path("data/processed/preannotation/candidates.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/preannotation/normalized.jsonl"))
    parser.add_argument("--errors", type=Path, default=Path("outputs/preannotation_normalization_errors.jsonl"))
    parser.add_argument("--ontology", type=Path, default=Path("configs/risk_ontology.yaml"))
    parser.add_argument("--include-job-id", action="store_true", help="Write {job_id, annotation} envelopes for experiment alignment")
    parser.add_argument(
        "--repair-unique-entity-evidence",
        action="store_true",
        help="Use an entity's unique source occurrence when teacher evidence is invalid",
    )
    parser.add_argument(
        "--repair-inverse-relations",
        action="store_true",
        help="Swap relation endpoints only when the inverse uniquely satisfies the ontology signature",
    )
    parser.add_argument(
        "--default-missing-claim-status",
        choices=("explicit", "inferred", "normative", "uncertain"),
        help="Explicitly repair a missing relation claim status; omitted values are rejected by default",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(normalize(parse_args()))
