#!/usr/bin/env python3
"""Fuse conservative KG-v1 agreement with KG-v2 recall candidates.

The fusion gate never reads gold annotations.  A KG-v2 entity is retained when
at least one auditable signal is present:

* KG-v1 predicts the same normalized text and entity type;
* the KG-v2 prompt contains a matching source-gated entity anchor; or
* the entity is an endpoint of a relation accepted by the deterministic
  evidence/signature verifier.

Relations can be omitted (entity-only M2), filtered from raw KG-v2 output, or
taken from the independently verified KG-v2 ablation.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def entity_key(entity: dict[str, Any]) -> tuple[str, str]:
    return normalize(entity.get("text")), str(entity.get("type", ""))


def keyed_annotations(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        job_id = row.get("job_id")
        if not job_id:
            raise ValueError("prediction row is missing job_id")
        if job_id in result:
            raise ValueError(f"duplicate prediction job_id: {job_id}")
        result[job_id] = row.get("annotation", row)
    return result


def verified_endpoints(annotation: dict[str, Any]) -> set[tuple[str, str]]:
    entities = {
        entity.get("id"): entity
        for entity in annotation.get("entities", [])
        if entity.get("id")
    }
    endpoints = set()
    for relation in annotation.get("relations", []):
        for field in ("source_id", "target_id"):
            entity = entities.get(relation.get(field))
            if entity:
                endpoints.add(entity_key(entity))
    return endpoints


def anchor_keys(job: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        (normalize(anchor.get("text")), str(anchor.get("type", "")))
        for anchor in job.get("kg_v2_context", {}).get("anchors", [])
        if normalize(anchor.get("text")) and anchor.get("type")
    }


def fuse_job(
    job_id: str,
    v1: dict[str, Any],
    v2: dict[str, Any],
    verified: dict[str, Any],
    job: dict[str, Any],
    relation_mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], Counter[str]]:
    v1_keys = {entity_key(entity) for entity in v1.get("entities", [])}
    anchors = anchor_keys(job)
    endpoints = verified_endpoints(verified)
    accepted_entities = []
    accepted_ids = set()
    seen_keys = set()
    audit = []
    counts: Counter[str] = Counter()

    for entity in v2.get("entities", []):
        key = entity_key(entity)
        reasons = []
        if key in v1_keys:
            reasons.append("v1_v2_exact_agreement")
        if key in anchors:
            reasons.append("source_gated_anchor_type_match")
        if key in endpoints:
            reasons.append("verified_relation_endpoint")
        accepted = bool(reasons) and key not in seen_keys
        if accepted:
            accepted_entities.append(entity)
            accepted_ids.add(entity.get("id"))
            seen_keys.add(key)
            counts["entities_accepted"] += 1
            for reason in reasons:
                counts[f"entity_reason:{reason}"] += 1
        else:
            counts["entities_rejected"] += 1
            if key in seen_keys:
                reasons.append("duplicate_normalized_text_and_type")
        audit.append(
            {
                "job_id": job_id,
                "entity_id": entity.get("id"),
                "text": entity.get("text"),
                "type": entity.get("type"),
                "accepted": accepted,
                "reasons": reasons or ["no_acceptance_signal"],
            }
        )

    if relation_mode == "none":
        source_relations = []
    elif relation_mode == "raw":
        source_relations = v2.get("relations", [])
    elif relation_mode == "verified":
        source_relations = verified.get("relations", [])
    else:
        raise ValueError(f"unknown relation mode: {relation_mode}")

    accepted_relations = []
    for relation in source_relations:
        if (
            relation.get("source_id") in accepted_ids
            and relation.get("target_id") in accepted_ids
        ):
            accepted_relations.append(relation)
            counts["relations_accepted"] += 1
        else:
            counts["relations_rejected_missing_gated_endpoint"] += 1

    result = {
        **v2,
        "entities": accepted_entities,
        "relations": accepted_relations,
        "review": {
            "status": "unreviewed",
            "reviewers": [],
            "notes": f"KG-v1/v2 auditable fusion; relation_mode={relation_mode}",
        },
    }
    return result, audit, counts


def run(args: argparse.Namespace) -> int:
    v1 = keyed_annotations(load_jsonl(args.v1))
    v2 = keyed_annotations(load_jsonl(args.v2))
    verified = keyed_annotations(load_jsonl(args.verified))
    jobs = {row["job_id"]: row for row in load_jsonl(args.jobs)}
    requested = list(jobs)
    missing = {
        "v1": sorted(set(requested) - set(v1)),
        "v2": sorted(set(requested) - set(v2)),
        "verified": sorted(set(requested) - set(verified)),
    }
    if any(missing.values()):
        raise ValueError(f"missing requested predictions: {missing}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    totals: Counter[str] = Counter()
    with args.output.open("w", encoding="utf-8") as output_stream, args.audit.open(
        "w", encoding="utf-8"
    ) as audit_stream:
        for job_id in requested:
            annotation, audit, counts = fuse_job(
                job_id,
                v1[job_id],
                v2[job_id],
                verified[job_id],
                jobs[job_id],
                args.relation_mode,
            )
            totals.update(counts)
            output_stream.write(
                json.dumps({"job_id": job_id, "annotation": annotation}, ensure_ascii=False)
                + "\n"
            )
            for item in audit:
                audit_stream.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "version": "kg-v3-conservative-fusion-1.0",
        "jobs": len(requested),
        "relation_mode": args.relation_mode,
        "acceptance_rule": [
            "v1_v2_exact_agreement",
            "source_gated_anchor_type_match",
            "verified_relation_endpoint",
        ],
        **dict(sorted(totals.items())),
        "output": str(args.output),
        "audit": str(args.audit),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--verified", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument(
        "--relation-mode", choices=("none", "raw", "verified"), default="none"
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
