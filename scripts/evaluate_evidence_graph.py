#!/usr/bin/env python3
"""Evaluate evidence quality, causal edges, and graph overlap for annotations."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


CAUSAL_RELATIONS = {"causes", "contributes_to", "results_in"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def annotation_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("annotation", row)


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def evidence_items(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def source_quote_is_valid(evidence: dict[str, Any], segments: dict[str, dict[str, Any]]) -> bool:
    segment = segments.get(evidence.get("segment_id"))
    quote = evidence.get("text")
    if not segment or not normalize_text(quote):
        return False
    return normalize_text(quote) in normalize_text(segment.get("text"))


def evidence_contains(evidence: dict[str, Any], entity_text: Any) -> bool:
    return bool(normalize_text(entity_text)) and normalize_text(entity_text) in normalize_text(evidence.get("text"))


def valid_entity_evidence(entity: dict[str, Any], segments: dict[str, dict[str, Any]]) -> bool:
    items = evidence_items(entity.get("evidence"))
    return bool(items) and any(
        source_quote_is_valid(item, segments) and evidence_contains(item, entity.get("text"))
        for item in items
    )


def valid_relation_evidence(
    relation: dict[str, Any], source: dict[str, Any] | None, target: dict[str, Any] | None, segments: dict[str, dict[str, Any]]
) -> bool:
    if not source or not target:
        return False
    return any(
        source_quote_is_valid(item, segments)
        and evidence_contains(item, source.get("text"))
        and evidence_contains(item, target.get("text"))
        for item in evidence_items(relation.get("evidence"))
    )


def entity_key(entity: dict[str, Any]) -> tuple[str, str]:
    return normalize_text(entity.get("text")), str(entity.get("type", ""))


def relation_key(relation: dict[str, Any], entities: dict[str, dict[str, Any]]) -> tuple[str, str, str]:
    source = entities.get(relation.get("source_id"), {})
    target = entities.get(relation.get("target_id"), {})
    return normalize_text(source.get("text")), str(relation.get("type", "")), normalize_text(target.get("text"))


def f1(gold: set[Any], predicted: set[Any]) -> dict[str, int | float]:
    correct = len(gold & predicted)
    precision = correct / len(predicted) if predicted else 0.0
    recall = correct / len(gold) if gold else 0.0
    score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "gold": len(gold),
        "predicted": len(predicted),
        "correct": correct,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(score, 4),
    }


def graph_metrics(gold: dict[str, Any], predicted: dict[str, Any]) -> dict[str, Any]:
    gold_entities = {entity_key(entity) for entity in gold.get("entities", [])}
    predicted_entities = {entity_key(entity) for entity in predicted.get("entities", [])}
    gold_entity_by_id = {entity.get("id"): entity for entity in gold.get("entities", [])}
    predicted_entity_by_id = {entity.get("id"): entity for entity in predicted.get("entities", [])}
    gold_edges = {
        (*relation_key(relation, gold_entity_by_id),)
        for relation in gold.get("relations", [])
    }
    predicted_edges = {
        (*relation_key(relation, predicted_entity_by_id),)
        for relation in predicted.get("relations", [])
    }
    return {
        "node": f1(gold_entities, predicted_entities),
        "edge": f1(gold_edges, predicted_edges),
        "node_jaccard": round(len(gold_entities & predicted_entities) / len(gold_entities | predicted_entities), 4)
        if gold_entities | predicted_entities
        else 1.0,
        "edge_jaccard": round(len(gold_edges & predicted_edges) / len(gold_edges | predicted_edges), 4)
        if gold_edges | predicted_edges
        else 1.0,
    }


def relation_subset(annotation: dict[str, Any], relation_types: set[str]) -> set[tuple[str, str, str]]:
    entities = {entity.get("id"): entity for entity in annotation.get("entities", [])}
    return {
        relation_key(relation, entities)
        for relation in annotation.get("relations", [])
        if relation.get("type") in relation_types
    }


def evaluate_job(
    gold: dict[str, Any], predicted: dict[str, Any], job: dict[str, Any] | None, ontology: dict[str, Any]
) -> dict[str, Any]:
    segments = {segment.get("segment_id"): segment for segment in (job or {}).get("segments", [])}
    predicted_entities = predicted.get("entities", [])
    predicted_relations = predicted.get("relations", [])
    entities = {entity.get("id"): entity for entity in predicted_entities}
    relation_types = set(ontology.get("relation_types", {}))
    signatures = ontology.get("allowed_relation_signatures", {})
    claim_statuses = set(ontology.get("claim_statuses", {}))

    entity_evidence_present = sum(bool(evidence_items(entity.get("evidence"))) for entity in predicted_entities)
    entity_evidence_valid = sum(valid_entity_evidence(entity, segments) for entity in predicted_entities)
    relation_evidence_valid = 0
    relation_evidence_present = 0
    unsupported_claims = 0
    invalid_relations = 0
    for relation in predicted_relations:
        source = entities.get(relation.get("source_id"))
        target = entities.get(relation.get("target_id"))
        relation_evidence_present += bool(evidence_items(relation.get("evidence")))
        evidence_ok = valid_relation_evidence(relation, source, target, segments)
        relation_evidence_valid += evidence_ok
        unsupported_claims += not evidence_ok
        signature = signatures.get(relation.get("type"), {})
        structurally_valid = bool(
            source
            and target
            and relation.get("type") in relation_types
            and source.get("type") in signature.get("source", [])
            and target.get("type") in signature.get("target", [])
            and relation.get("claim_status") in claim_statuses
            and evidence_ok
        )
        invalid_relations += not structurally_valid

    gold_causal = relation_subset(gold, CAUSAL_RELATIONS)
    predicted_causal = relation_subset(predicted, CAUSAL_RELATIONS)
    graph = graph_metrics(gold, predicted)
    return {
        "language": gold.get("language", "unknown"),
        "entity_count": len(predicted_entities),
        "relation_count": len(predicted_relations),
        "entity_evidence_present_count": entity_evidence_present,
        "entity_evidence_valid_count": entity_evidence_valid,
        "relation_evidence_present_count": relation_evidence_present,
        "relation_evidence_valid_count": relation_evidence_valid,
        "unsupported_claim_count": unsupported_claims,
        "invalid_relation_count": invalid_relations,
        "entity_evidence_coverage": round(entity_evidence_present / len(predicted_entities), 4)
        if predicted_entities
        else 0.0,
        "entity_evidence_correctness": round(entity_evidence_valid / entity_evidence_present, 4)
        if entity_evidence_present
        else 0.0,
        "relation_evidence_coverage": round(relation_evidence_present / len(predicted_relations), 4)
        if predicted_relations
        else 0.0,
        "relation_evidence_correctness": round(relation_evidence_valid / relation_evidence_present, 4)
        if relation_evidence_present
        else 0.0,
        "unsupported_claim_rate": round(unsupported_claims / len(predicted_relations), 4)
        if predicted_relations
        else 0.0,
        "invalid_relation_rate": round(invalid_relations / len(predicted_relations), 4)
        if predicted_relations
        else 0.0,
        "causal_edge": f1(gold_causal, predicted_causal),
        "graph": graph,
    }


def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {}
    numeric_rates = (
        "entity_evidence_coverage",
        "entity_evidence_correctness",
        "relation_evidence_coverage",
        "relation_evidence_correctness",
        "unsupported_claim_rate",
        "invalid_relation_rate",
    )
    graph_fields = ("node_jaccard", "edge_jaccard")
    counts = {
        field: sum(int(item[field]) for item in items)
        for field in (
            "entity_count",
            "relation_count",
            "entity_evidence_present_count",
            "entity_evidence_valid_count",
            "relation_evidence_present_count",
            "relation_evidence_valid_count",
            "unsupported_claim_count",
            "invalid_relation_count",
        )
    }

    def rate(numerator: str, denominator: str) -> float:
        return round(counts[numerator] / counts[denominator], 4) if counts[denominator] else 0.0

    result = {
        "entity_evidence_coverage": rate(
            "entity_evidence_present_count", "entity_count"
        ),
        "entity_evidence_correctness": rate(
            "entity_evidence_valid_count", "entity_evidence_present_count"
        ),
        "relation_evidence_coverage": rate(
            "relation_evidence_present_count", "relation_count"
        ),
        "relation_evidence_correctness": rate(
            "relation_evidence_valid_count", "relation_evidence_present_count"
        ),
        "unsupported_claim_rate": rate("unsupported_claim_count", "relation_count"),
        "invalid_relation_rate": rate("invalid_relation_count", "relation_count"),
        "counts": counts,
        "macro_by_job": {
            field: round(sum(item[field] for item in items) / len(items), 4)
            for field in numeric_rates
        },
    }
    result.update({field: round(sum(item["graph"][field] for item in items) / len(items), 4) for field in graph_fields})
    causal_gold = sum(int(item["causal_edge"]["gold"]) for item in items)
    causal_predicted = sum(int(item["causal_edge"]["predicted"]) for item in items)
    causal_correct = sum(int(item["causal_edge"]["correct"]) for item in items)
    result["causal_edge"] = f1(
        {f"gold_{index}" for index in range(causal_gold)},
        {f"predicted_{index}" for index in range(causal_predicted)},
    )
    result["causal_edge"]["correct"] = causal_correct
    precision = causal_correct / causal_predicted if causal_predicted else 0.0
    recall = causal_correct / causal_gold if causal_gold else 0.0
    result["causal_edge"].update(
        {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0}
    )
    result["jobs"] = len(items)
    return result


def run(args: argparse.Namespace) -> int:
    ontology = yaml.safe_load(args.ontology.read_text(encoding="utf-8"))
    gold_rows = load_jsonl(args.gold)
    gold_index = load_jsonl(args.gold_index)
    predicted_rows = load_jsonl(args.predictions)
    jobs = {row["job_id"]: row for row in load_jsonl(args.jobs)}
    gold = {row["job_id"]: gold_rows[row["record_index"]] for row in gold_index}
    predicted = {row["job_id"]: annotation_from_row(row) for row in predicted_rows}
    job_ids = [row["job_id"] for row in gold_index]
    missing = [job_id for job_id in job_ids if job_id not in predicted]
    if missing:
        raise ValueError(f"missing predictions for {len(missing)} jobs; first: {missing[:3]}")

    items = {
        job_id: evaluate_job(gold[job_id], predicted[job_id], jobs.get(job_id), ontology)
        for job_id in job_ids
    }
    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items.values():
        by_language[item["language"]].append(item)
    result = {
        "jobs": len(job_ids),
        "overall": aggregate(list(items.values())),
        "by_language": {language: aggregate(rows) for language, rows in sorted(by_language.items())},
        "definitions": {
            "entity_evidence_coverage": "predicted entities carrying at least one evidence item / predicted entities",
            "entity_evidence_correctness": "predicted entities whose supplied evidence is a source quote containing the entity text / entities carrying evidence",
            "relation_evidence_coverage": "predicted relations carrying at least one evidence item / predicted relations",
            "relation_evidence_correctness": "predicted relations whose supplied evidence is a source quote containing both predicted endpoints / relations carrying evidence",
            "unsupported_claim_rate": "predicted relations without source-valid evidence containing both endpoints / predicted relations",
            "invalid_relation_rate": "predicted relations failing entity reference, ontology signature, claim-status, or evidence checks / predicted relations",
            "causal_edge": "strict endpoint-text, direction, and relation-type F1 for causes, contributes_to, and results_in",
            "graph_jaccard": "macro per-job Jaccard overlap for typed entity nodes and directed typed relation edges",
            "macro_by_job": "diagnostic unweighted mean of per-job evidence and validity rates, including zero-denominator jobs as zero",
        },
        "per_job": items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["overall"], ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--gold-index", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, default=Path("configs/risk_ontology.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
