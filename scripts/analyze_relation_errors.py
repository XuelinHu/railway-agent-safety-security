#!/usr/bin/env python3
"""Diagnose entity and relation bottlenecks in merged extraction predictions."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def annotation(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("annotation", row)


def keyed_rows(rows: list[dict[str, Any]], index: list[dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    result = {}
    for position, row in enumerate(rows):
        if "annotation" in row:
            result[row["job_id"]] = row["annotation"]
        elif index:
            result[index[position]["job_id"]] = row
        else:
            raise ValueError("annotation rows without job_id require an index file")
    return result


def entity_key(entity: dict[str, Any]) -> tuple[str, str]:
    return norm(entity.get("text")), str(entity.get("type", ""))


def relation_parts(
    relation: dict[str, Any], entities: dict[str, dict[str, Any]]
) -> tuple[tuple[str, str], str, tuple[str, str]] | None:
    source = entities.get(relation.get("source_id"))
    target = entities.get(relation.get("target_id"))
    if not source or not target:
        return None
    return entity_key(source), str(relation.get("type", "")), entity_key(target)


def relation_key(parts: tuple[tuple[str, str], str, tuple[str, str]]) -> tuple[str, str, str]:
    source, relation_type, target = parts
    return source[0], relation_type, target[0]


def endpoint_key(parts: tuple[tuple[str, str], str, tuple[str, str]]) -> tuple[str, str]:
    source, _, target = parts
    return source[0], target[0]


def undirected_endpoint_key(parts: tuple[tuple[str, str], str, tuple[str, str]]) -> tuple[str, str]:
    source, _, target = parts
    return tuple(sorted((source[0], target[0])))


def empty_summary() -> dict[str, Any]:
    return {
        "gold_entities": 0,
        "predicted_entities": 0,
        "correct_entities": 0,
        "gold_relations": 0,
        "predicted_relations": 0,
        "strict_relation_matches": 0,
        "gold_relations_with_both_endpoints_predicted": 0,
        "gold_directed_endpoint_pairs": 0,
        "predicted_directed_endpoint_pairs": 0,
        "correct_directed_endpoint_pairs": 0,
        "gold_undirected_endpoint_pairs": 0,
        "predicted_undirected_endpoint_pairs": 0,
        "correct_undirected_endpoint_pairs": 0,
        "predicted_relation_categories": Counter(),
        "missed_relation_categories": Counter(),
        "gold_relation_types": Counter(),
        "predicted_relation_types": Counter(),
    }


def merge_summary(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, Counter):
            target[key].update(value)
        elif isinstance(value, int):
            target[key] += value


def classify_predicted_relation(
    parts: tuple[tuple[str, str], str, tuple[str, str]],
    gold_parts: list[tuple[tuple[str, str], str, tuple[str, str]]],
    gold_entities: set[tuple[str, str]],
) -> str:
    key = relation_key(parts)
    gold_keys = {relation_key(item) for item in gold_parts}
    if key in gold_keys:
        return "strict_match"
    source, relation_type, target = parts
    directed = endpoint_key(parts)
    directed_gold = {endpoint_key(item) for item in gold_parts}
    reversed_gold = {(item[2][0], item[0][0]) for item in gold_parts}
    if source not in gold_entities or target not in gold_entities:
        if source[0] in {entity[0] for entity in gold_entities} or target[0] in {entity[0] for entity in gold_entities}:
            return "entity_type_or_boundary_error"
        return "unsupported_or_hallucinated_endpoint"
    if directed in directed_gold:
        return "wrong_relation_type"
    if directed in reversed_gold:
        reverse_types = {
            item[1]
            for item in gold_parts
            if endpoint_key(item) == (target[0], source[0])
        }
        return "reversed_direction" if relation_type in reverse_types else "reversed_direction_and_type"
    return "unsupported_entity_pair"


def classify_missed_relation(
    parts: tuple[tuple[str, str], str, tuple[str, str]],
    predicted_parts: list[tuple[tuple[str, str], str, tuple[str, str]]],
    predicted_entities: set[tuple[str, str]],
) -> str:
    if parts[0] in predicted_entities and parts[2] in predicted_entities:
        predicted_directed = {endpoint_key(item) for item in predicted_parts}
        predicted_undirected = {undirected_endpoint_key(item) for item in predicted_parts}
        if endpoint_key(parts) in predicted_directed:
            return "wrong_relation_type_or_claim"
        if undirected_endpoint_key(parts) in predicted_undirected:
            return "direction_error"
        return "relation_type_or_pair_error_with_recalled_entities"
    if parts[0][0] in {entity[0] for entity in predicted_entities} or parts[2][0] in {entity[0] for entity in predicted_entities}:
        return "partial_entity_boundary_or_type_error"
    return "entity_recall_bottleneck"


def analyze_job(gold: dict[str, Any], predicted: dict[str, Any]) -> dict[str, Any]:
    summary = empty_summary()
    gold_entities = {entity_key(entity) for entity in gold.get("entities", [])}
    predicted_entities = {entity_key(entity) for entity in predicted.get("entities", [])}
    summary["gold_entities"] = len(gold_entities)
    summary["predicted_entities"] = len(predicted_entities)
    summary["correct_entities"] = len(gold_entities & predicted_entities)
    gold_by_id = {entity["id"]: entity for entity in gold.get("entities", []) if entity.get("id")}
    predicted_by_id = {entity["id"]: entity for entity in predicted.get("entities", []) if entity.get("id")}
    gold_parts = [
        parts
        for relation in gold.get("relations", [])
        if (parts := relation_parts(relation, gold_by_id)) is not None
    ]
    predicted_parts = [
        parts
        for relation in predicted.get("relations", [])
        if (parts := relation_parts(relation, predicted_by_id)) is not None
    ]
    gold_keys = {relation_key(parts) for parts in gold_parts}
    predicted_keys = {relation_key(parts) for parts in predicted_parts}
    gold_directed = {endpoint_key(parts) for parts in gold_parts}
    predicted_directed = {endpoint_key(parts) for parts in predicted_parts}
    gold_undirected = {undirected_endpoint_key(parts) for parts in gold_parts}
    predicted_undirected = {undirected_endpoint_key(parts) for parts in predicted_parts}
    summary.update(
        {
            "gold_relations": len(gold_keys),
            "predicted_relations": len(predicted_keys),
            "strict_relation_matches": len(gold_keys & predicted_keys),
            "gold_relations_with_both_endpoints_predicted": sum(
                parts[0] in predicted_entities and parts[2] in predicted_entities for parts in gold_parts
            ),
            "gold_directed_endpoint_pairs": len(gold_directed),
            "predicted_directed_endpoint_pairs": len(predicted_directed),
            "correct_directed_endpoint_pairs": len(gold_directed & predicted_directed),
            "gold_undirected_endpoint_pairs": len(gold_undirected),
            "predicted_undirected_endpoint_pairs": len(predicted_undirected),
            "correct_undirected_endpoint_pairs": len(gold_undirected & predicted_undirected),
        }
    )
    summary["gold_relation_types"].update(parts[1] for parts in gold_parts)
    summary["predicted_relation_types"].update(parts[1] for parts in predicted_parts)
    for parts in predicted_parts:
        summary["predicted_relation_categories"][classify_predicted_relation(parts, gold_parts, gold_entities)] += 1
    for parts in gold_parts:
        if relation_key(parts) not in predicted_keys:
            summary["missed_relation_categories"][classify_missed_relation(parts, predicted_parts, predicted_entities)] += 1
    return summary


def finalize(summary: dict[str, Any]) -> dict[str, Any]:
    gold_relations = summary["gold_relations"]
    predicted_relations = summary["predicted_relations"]
    def ratio(numerator: int, denominator: int) -> float:
        return round(numerator / denominator, 4) if denominator else 0.0
    return {
        "entities": {
            "gold": summary["gold_entities"],
            "predicted": summary["predicted_entities"],
            "correct": summary["correct_entities"],
            "strict_recall": ratio(summary["correct_entities"], summary["gold_entities"]),
        },
        "relations": {
            "gold": gold_relations,
            "predicted": predicted_relations,
            "strict_correct": summary["strict_relation_matches"],
            "strict_recall": ratio(summary["strict_relation_matches"], gold_relations),
            "endpoint_reachable_upper_bound_recall": ratio(
                summary["gold_relations_with_both_endpoints_predicted"], gold_relations
            ),
            "directed_endpoint_pair_recall": ratio(
                summary["correct_directed_endpoint_pairs"], summary["gold_directed_endpoint_pairs"]
            ),
            "undirected_endpoint_pair_recall": ratio(
                summary["correct_undirected_endpoint_pairs"], summary["gold_undirected_endpoint_pairs"]
            ),
            "predicted_relation_categories": dict(summary["predicted_relation_categories"]),
            "missed_relation_categories": dict(summary["missed_relation_categories"]),
            "gold_relation_types": dict(summary["gold_relation_types"]),
            "predicted_relation_types": dict(summary["predicted_relation_types"]),
        },
    }


def run(args: argparse.Namespace) -> int:
    gold_rows = load_jsonl(args.gold)
    gold_index = load_jsonl(args.gold_index)
    requested = [row["job_id"] for row in load_jsonl(args.jobs)]
    gold = keyed_rows(gold_rows, gold_index)
    predicted = keyed_rows(load_jsonl(args.predictions))
    per_language: dict[str, dict[str, Any]] = defaultdict(empty_summary)
    overall = empty_summary()
    per_relation_type: dict[str, dict[str, int]] = defaultdict(lambda: {"gold": 0, "predicted": 0, "strict_correct": 0})
    missing = []
    for job_id in requested:
        if job_id not in gold or job_id not in predicted:
            missing.append(job_id)
            continue
        job_summary = analyze_job(gold[job_id], predicted[job_id])
        merge_summary(overall, job_summary)
        language = gold[job_id].get("language", "unknown")
        merge_summary(per_language[language], job_summary)
        gold_by_id = {entity["id"]: entity for entity in gold[job_id].get("entities", []) if entity.get("id")}
        predicted_by_id = {entity["id"]: entity for entity in predicted[job_id].get("entities", []) if entity.get("id")}
        gold_keys = {relation_key(parts) for relation in gold[job_id].get("relations", []) if (parts := relation_parts(relation, gold_by_id)) is not None}
        predicted_keys = {relation_key(parts) for relation in predicted[job_id].get("relations", []) if (parts := relation_parts(relation, predicted_by_id)) is not None}
        for key in gold_keys:
            per_relation_type[key[1]]["gold"] += 1
            if key in predicted_keys:
                per_relation_type[key[1]]["strict_correct"] += 1
        for key in predicted_keys:
            per_relation_type[key[1]]["predicted"] += 1
    if missing:
        raise ValueError(f"missing requested jobs: {missing}")
    result = {
        "unit": "merged_text_block",
        "requested_jobs": len(requested),
        "documents": len({gold[job_id]["document_id"] for job_id in requested}),
        "overall": finalize(overall),
        "by_language": {language: finalize(summary) for language, summary in sorted(per_language.items())},
        "by_relation_type": {
            relation_type: {
                **values,
                "recall": round(values["strict_correct"] / values["gold"], 4) if values["gold"] else 0.0,
                "precision": round(values["strict_correct"] / values["predicted"], 4) if values["predicted"] else 0.0,
            }
            for relation_type, values in sorted(per_relation_type.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--gold-index", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
