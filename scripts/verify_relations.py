#!/usr/bin/env python3
"""Apply deterministic second-stage validation to extracted relations.

The verifier keeps the annotation record intact except for relations that do
not pass structural, ontology, evidence, or claim-status checks. Every
decision is written to a separate JSONL audit file so the filter is
reproducible and reviewable.
"""

from __future__ import annotations

import argparse
import copy
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


def contains_text(container: Any, value: Any) -> bool:
    """Match mentions while tolerating line-break and spacing differences."""
    haystack = normalize_text(container)
    needle = normalize_text(value)
    return bool(needle) and needle in haystack


def evidence_items(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def evidence_segment_ids(value: Any) -> set[str]:
    return {
        str(item["segment_id"])
        for item in evidence_items(value)
        if item.get("segment_id") is not None
    }


def evidence_is_well_formed(value: Any) -> bool:
    items = evidence_items(value)
    return bool(items) and all(
        normalize_text(item.get("text")) and item.get("segment_id") is not None
        for item in items
    )


def relation_evidence_contains_both(relation: dict[str, Any], source: dict[str, Any], target: dict[str, Any]) -> bool:
    for evidence in evidence_items(relation.get("evidence")):
        if contains_text(evidence.get("text"), source.get("text")) and contains_text(
            evidence.get("text"), target.get("text")
        ):
            return True
    return False


def verify_annotation(
    annotation: dict[str, Any],
    ontology: dict[str, Any],
    *,
    require_local_cooccurrence: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return a relation-filtered annotation and one audit record per relation."""
    verified = copy.deepcopy(annotation)
    entities = verified.get("entities", [])
    entity_by_id = {entity.get("id"): entity for entity in entities if entity.get("id")}
    relation_types = set(ontology.get("relation_types", {}))
    claim_statuses = set(ontology.get("claim_statuses", {}))
    signatures = ontology.get("allowed_relation_signatures", {})
    accepted_relations: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []

    for relation in verified.get("relations", []):
        relation_id = relation.get("id")
        source_id = relation.get("source_id")
        target_id = relation.get("target_id")
        source = entity_by_id.get(source_id)
        target = entity_by_id.get(target_id)
        reasons: list[str] = []

        if not source or not target:
            reasons.append("unknown_entity_reference")
        if source and target and source_id == target_id:
            reasons.append("self_relation")

        relation_type = relation.get("type")
        if relation_type not in relation_types:
            reasons.append("unknown_relation_type")
        elif source and target:
            signature = signatures.get(relation_type, {})
            if source.get("type") not in signature.get("source", []) or target.get("type") not in signature.get("target", []):
                reasons.append("illegal_entity_type_signature")

        claim_status = relation.get("claim_status")
        if claim_status not in claim_statuses:
            reasons.append("missing_or_unknown_claim_status")

        relation_evidence = relation.get("evidence")
        if not evidence_is_well_formed(relation_evidence):
            reasons.append("missing_or_malformed_relation_evidence")
        if source and not evidence_is_well_formed(source.get("evidence")):
            reasons.append("malformed_source_entity_evidence")
        if target and not evidence_is_well_formed(target.get("evidence")):
            reasons.append("malformed_target_entity_evidence")

        if source and target and evidence_is_well_formed(relation_evidence):
            if require_local_cooccurrence and not relation_evidence_contains_both(relation, source, target):
                reasons.append("entities_not_co_present_in_relation_evidence")

        accepted = not reasons
        if accepted:
            accepted_relations.append(relation)

        audit.append(
            {
                "relation_id": relation_id,
                "accepted": accepted,
                "reasons": reasons or ["accepted"],
                "source_id": source_id,
                "target_id": target_id,
                "relation_type": relation_type,
                "claim_status": claim_status,
                "source_segment_ids": sorted(evidence_segment_ids(source.get("evidence"))) if source else [],
                "target_segment_ids": sorted(evidence_segment_ids(target.get("evidence"))) if target else [],
                "relation_segment_ids": sorted(evidence_segment_ids(relation_evidence)),
            }
        )

    verified["relations"] = accepted_relations
    return verified, audit


def run(args: argparse.Namespace) -> int:
    ontology = yaml.safe_load(args.ontology.read_text(encoding="utf-8"))
    rows = load_jsonl(args.annotations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    relation_counts = Counter()
    output_rows: list[dict[str, Any]] = []

    with args.audit.open("w", encoding="utf-8") as audit_stream:
        for row in rows:
            annotation = annotation_from_row(row)
            verified, audit = verify_annotation(
                annotation,
                ontology,
                require_local_cooccurrence=not args.allow_cross_segment,
            )
            relation_counts["input"] += len(annotation.get("relations", []))
            relation_counts["accepted"] += sum(item["accepted"] for item in audit)
            relation_counts["rejected"] += sum(not item["accepted"] for item in audit)
            for item in audit:
                for reason in item["reasons"]:
                    relation_counts[f"reason:{reason}"] += 1
                audit_stream.write(
                    json.dumps(
                        {
                            "job_id": row.get("job_id"),
                            "document_id": annotation.get("document_id"),
                            **item,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            if "annotation" in row:
                output_rows.append({**row, "annotation": verified})
            else:
                output_rows.append(verified)

    with args.output.open("w", encoding="utf-8") as output_stream:
        for row in output_rows:
            output_stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "jobs": len(rows),
        "input_relations": relation_counts["input"],
        "accepted_relations": relation_counts["accepted"],
        "rejected_relations": relation_counts["rejected"],
        "rejection_reasons": {
            key.removeprefix("reason:"): value
            for key, value in sorted(relation_counts.items())
            if key.startswith("reason:")
        },
        "output": str(args.output),
        "audit": str(args.audit),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument(
        "--allow-cross-segment",
        action="store_true",
        help="Skip the local evidence co-occurrence check while retaining other checks",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
