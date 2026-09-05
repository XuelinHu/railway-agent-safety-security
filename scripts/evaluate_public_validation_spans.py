#!/usr/bin/env python3
"""Score public benchmark predictions by exact source character spans.

Unlike the project's window-aware evaluator, public benchmark jobs already
contain one source sentence with stable global offsets. This evaluator trusts
only validated ``evidence.start/end`` intervals and never recovers spans from
normalized entity text.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Hashable


FIELDS = ("entity_strict", "relation_strict", "relation_with_claim_status")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def annotation(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("annotation", row)
    if not isinstance(value, dict):
        raise ValueError("annotation row must contain a JSON object")
    return value


def indexed_gold(rows: list[dict[str, Any]], index: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in index:
        job_id = item.get("job_id")
        position = item.get("record_index")
        if not isinstance(job_id, str) or not isinstance(position, int) or not 0 <= position < len(rows):
            raise ValueError(f"invalid gold index row: {item}")
        if job_id in result:
            raise ValueError(f"duplicate gold job_id {job_id!r}")
        result[job_id] = annotation(rows[position])
    return result


def prediction_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        job_id = row.get("job_id")
        if not isinstance(job_id, str):
            raise ValueError("every prediction row must contain job_id")
        if job_id in result:
            raise ValueError(f"duplicate prediction job_id {job_id!r}")
        result[job_id] = annotation(row)
    return result


def exact_entity_key(
    entity: dict[str, Any],
    job: dict[str, Any],
    side: str,
    position: int,
) -> tuple[tuple[Hashable, ...], bool]:
    evidence = entity.get("evidence")
    if not isinstance(evidence, dict):
        return (side, "invalid", position, entity.get("id"), entity.get("type")), False
    start, end = evidence.get("start"), evidence.get("end")
    segment_id = evidence.get("segment_id")
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
        return (side, "invalid", position, entity.get("id"), entity.get("type")), False
    for segment in job.get("segments", []):
        if segment.get("segment_id") != segment_id:
            continue
        segment_start = segment.get("start")
        segment_end = segment.get("end")
        if not isinstance(segment_start, int) or not isinstance(segment_end, int):
            break
        if not segment_start <= start < end <= segment_end:
            break
        local_start, local_end = start - segment_start, end - segment_start
        source_text = str(segment.get("text", ""))[local_start:local_end]
        if source_text != entity.get("text") or evidence.get("text") != source_text:
            break
        return (start, end, str(entity.get("type"))), True
    return (side, "invalid", position, entity.get("id"), entity.get("type")), False


def annotation_keys(
    value: dict[str, Any], job: dict[str, Any], side: str
) -> tuple[dict[str, set[tuple[Hashable, ...]]], int]:
    entities = value.get("entities", [])
    if not isinstance(entities, list):
        raise ValueError("annotation entities must be a list")
    entity_keys: dict[Any, tuple[Hashable, ...]] = {}
    entity_set: set[tuple[Hashable, ...]] = set()
    unresolved = 0
    for position, entity in enumerate(entities):
        if not isinstance(entity, dict):
            raise ValueError("entity must be an object")
        key, resolved = exact_entity_key(entity, job, side, position)
        entity_set.add(key)
        if entity.get("id") in entity_keys:
            raise ValueError(f"duplicate entity ID {entity.get('id')!r}")
        entity_keys[entity.get("id")] = key
        unresolved += int(not resolved)

    relations = value.get("relations", [])
    if not isinstance(relations, list):
        raise ValueError("annotation relations must be a list")
    relation_set: set[tuple[Hashable, ...]] = set()
    claim_set: set[tuple[Hashable, ...]] = set()
    for position, relation in enumerate(relations):
        if not isinstance(relation, dict):
            raise ValueError("relation must be an object")
        source = entity_keys.get(relation.get("source_id"))
        target = entity_keys.get(relation.get("target_id"))
        if source is None or target is None:
            key = (side, "invalid_relation", position, relation.get("id"))
        else:
            key = (source, str(relation.get("type")), target)
        relation_set.add(key)
        claim_set.add((*key, str(relation.get("claim_status"))))
    return {
        "entity_strict": entity_set,
        "relation_strict": relation_set,
        "relation_with_claim_status": claim_set,
    }, unresolved


def score(gold: set[Any], predicted: set[Any]) -> dict[str, int | float]:
    correct = len(gold & predicted)
    precision = correct / len(predicted) if predicted else 0.0
    recall = correct / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "gold": len(gold),
        "predicted": len(predicted),
        "correct": correct,
    }


def aggregate(items: list[dict[str, Any]], field: str) -> dict[str, int | float]:
    gold = sum(item[field]["gold"] for item in items)
    predicted = sum(item[field]["predicted"] for item in items)
    correct = sum(item[field]["correct"] for item in items)
    precision = correct / predicted if predicted else 0.0
    recall = correct / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "gold": gold,
        "predicted": predicted,
        "correct": correct,
    }


def macro(items: list[dict[str, Any]], field: str) -> dict[str, float]:
    if not items:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        metric: round(sum(float(item[field][metric]) for item in items) / len(items), 4)
        for metric in ("precision", "recall", "f1")
    }


def run(args: argparse.Namespace) -> int:
    allow_non_validation = bool(getattr(args, "allow_non_validation", False))
    split = "test" if allow_non_validation else "validation"
    expected_jobs_name = f"{split}_baseline_jobs.jsonl"
    if args.jobs.name != expected_jobs_name:
        raise ValueError(f"--jobs must point to {expected_jobs_name}")
    job_rows = load_jsonl(args.jobs)
    jobs = {row["job_id"]: row for row in job_rows}
    requested = [row["job_id"] for row in job_rows]
    if len(jobs) != len(requested):
        raise ValueError("jobs contain duplicate job_id values")
    gold = indexed_gold(load_jsonl(args.gold), load_jsonl(args.gold_index))
    predictions = prediction_rows(load_jsonl(args.predictions))
    if set(requested) - set(gold):
        raise ValueError(f"gold is missing requested jobs: {sorted(set(requested) - set(gold))[:3]}")
    if set(predictions) - set(requested):
        raise ValueError(f"predictions contain jobs outside {split} input: {sorted(set(predictions) - set(requested))[:3]}")

    per_job: dict[str, dict[str, Any]] = {}
    items: list[dict[str, Any]] = []
    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job_id in requested:
        gold_annotation = gold[job_id]
        predicted_annotation = predictions.get(
            job_id,
            {
                "document_id": gold_annotation.get("document_id"),
                "language": gold_annotation.get("language", "unknown"),
                "entities": [],
                "relations": [],
            },
        )
        gold_keys, unresolved_gold = annotation_keys(gold_annotation, jobs[job_id], f"gold:{job_id}")
        predicted_keys, unresolved_predicted = annotation_keys(
            predicted_annotation, jobs[job_id], f"prediction:{job_id}"
        )
        item: dict[str, Any] = {
            "job_id": job_id,
            "language": gold_annotation.get("language", "unknown"),
            "unresolved_gold_entities": unresolved_gold,
            "unresolved_predicted_entities": unresolved_predicted,
        }
        for field in FIELDS:
            item[field] = score(gold_keys[field], predicted_keys[field])
        per_job[job_id] = item
        items.append(item)
        by_language[item["language"]].append(item)

    result: dict[str, Any] = {
        "metric": "strict-source-character-span",
        "selection_split": split,
        "formal_test_read": allow_non_validation,
        "jobs_gold": len(requested),
        "jobs_predicted": len(predictions),
        "jobs_evaluated": len(requested),
        "jobs_missing_predictions": len(set(requested) - set(predictions)),
        "generation_success_rate": round(len(set(requested) & set(predictions)) / len(requested), 4)
        if requested
        else 0.0,
        **{field: aggregate(items, field) for field in FIELDS},
        "macro_by_job": {field: macro(items, field) for field in FIELDS},
        "resolution": {
            "unresolved_gold_entities": sum(item["unresolved_gold_entities"] for item in items),
            "unresolved_predicted_entities": sum(
                item["unresolved_predicted_entities"] for item in items
            ),
        },
        "by_language": {},
        "per_job": per_job,
    }
    for language, language_items in sorted(by_language.items()):
        result["by_language"][language] = {
            "jobs": len(language_items),
            **{field: aggregate(language_items, field) for field in FIELDS},
            "macro_by_job": {field: macro(language_items, field) for field in FIELDS},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_job"}, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--gold-index", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-non-validation",
        action="store_true",
        help="Explicitly score the promoted formal-test split",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
