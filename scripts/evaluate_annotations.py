#!/usr/bin/env python3
"""Evaluate strict entity and relation extraction metrics for JSONL annotations."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def norm(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def unpack(rows: list[dict[str, Any]], index: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    result = {}
    for position, row in enumerate(rows):
        if "annotation" in row:
            job_id = row["job_id"]
            annotation = row["annotation"]
        elif index:
            job_id = index[position]["job_id"]
            annotation = row
        else:
            job_id = f"record_{position + 1}"
            annotation = row
        result[job_id] = annotation
    return result


def annotation_items(annotation: Any, field: str) -> list[Any]:
    if not isinstance(annotation, dict):
        return []
    value = annotation.get(field, [])
    return value if isinstance(value, list) else []


def entity_key(entity: Any) -> tuple[str, str]:
    if not isinstance(entity, dict):
        return "__invalid_entity_text__", "__invalid_entity_type__"
    text = entity.get("text")
    entity_type = entity.get("type")
    return (
        norm(text) if isinstance(text, str) else "__invalid_entity_text__",
        entity_type if isinstance(entity_type, str) else "__invalid_entity_type__",
    )


def relation_key(relation: Any, entities: dict[str, dict[str, Any]]) -> tuple[str, str, str]:
    if not isinstance(relation, dict):
        marker = json.dumps(relation, ensure_ascii=False, sort_keys=True)
        return f"__invalid_relation__:{marker}", "__invalid_relation__", ""
    source_id = relation.get("source_id")
    target_id = relation.get("target_id")
    relation_type = relation.get("type")
    source = entities.get(source_id, {}) if isinstance(source_id, str) else {}
    target = entities.get(target_id, {}) if isinstance(target_id, str) else {}
    if not (
        isinstance(source_id, str)
        and isinstance(target_id, str)
        and isinstance(relation_type, str)
        and source
        and target
    ):
        # Generated baselines may emit a syntactically valid but incomplete
        # relation. Keep each such item in the prediction denominator while
        # making it impossible for it to match a valid gold relation.
        marker = json.dumps(
            {
                "id": relation.get("id"),
                "source_id": source_id,
                "target_id": target_id,
                "type": relation_type,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return f"__invalid_relation__:{marker}", "__invalid_relation__", ""
    return norm(str(source.get("text", ""))), relation_type, norm(str(target.get("text", "")))


def scores(gold: set[Any], predicted: set[Any]) -> dict[str, float]:
    true_positive = len(gold & predicted)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4), "gold": len(gold), "predicted": len(predicted), "correct": true_positive}


def aggregate(values: list[dict[str, float]]) -> dict[str, float]:
    gold = sum(value["gold"] for value in values)
    predicted = sum(value["predicted"] for value in values)
    correct = sum(value["correct"] for value in values)
    precision = correct / predicted if predicted else 0.0
    recall = correct / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4), "gold": gold, "predicted": predicted, "correct": correct}


def run(args: argparse.Namespace) -> int:
    gold_rows = load_jsonl(args.gold)
    gold_index = load_jsonl(args.gold_index)
    predicted_rows = load_jsonl(args.predictions)
    predicted_index = load_jsonl(args.pred_index) if args.pred_index and args.pred_index.exists() else None
    gold = unpack(gold_rows, gold_index)
    predicted = unpack(predicted_rows, predicted_index)
    requested_job_ids: list[str] | None = None
    if args.jobs:
        requested_job_ids = [row["job_id"] for row in load_jsonl(args.jobs)]
        if args.offset:
            requested_job_ids = requested_job_ids[args.offset :]
        if args.limit:
            requested_job_ids = requested_job_ids[: args.limit]
        requested = set(requested_job_ids)
        gold = {job_id: annotation for job_id, annotation in gold.items() if job_id in requested}
        predicted = {
            job_id: annotation for job_id, annotation in predicted.items() if job_id in requested
        }
    elif args.limit:
        raise SystemExit("--limit requires --jobs so the selected job order is explicit")
    if args.include_missing_as_empty:
        job_ids = sorted(gold)
    else:
        job_ids = sorted(set(gold) & set(predicted))
    entity_scores = []
    relation_scores = []
    claim_scores = []
    by_language: dict[str, list[str]] = defaultdict(list)
    for job_id in job_ids:
        g = gold[job_id]
        p = predicted.get(
            job_id,
            {"language": g.get("language", "unknown"), "entities": [], "relations": []},
        )
        g_entity_rows = annotation_items(g, "entities")
        p_entity_rows = annotation_items(p, "entities")
        g_relation_rows = annotation_items(g, "relations")
        p_relation_rows = annotation_items(p, "relations")
        g_entities = {entity_key(entity) for entity in g_entity_rows}
        p_entities = {entity_key(entity) for entity in p_entity_rows}
        entity_scores.append(scores(g_entities, p_entities))
        g_by_id = {
            entity["id"]: entity
            for entity in g_entity_rows
            if isinstance(entity, dict) and isinstance(entity.get("id"), str)
        }
        p_by_id = {
            entity["id"]: entity
            for entity in p_entity_rows
            if isinstance(entity, dict) and isinstance(entity.get("id"), str)
        }
        g_relations = {relation_key(relation, g_by_id) for relation in g_relation_rows}
        p_relations = {relation_key(relation, p_by_id) for relation in p_relation_rows}
        relation_scores.append(scores(g_relations, p_relations))
        g_claims = {
            (relation_key(relation, g_by_id), relation.get("claim_status", "unknown"))
            for relation in g_relation_rows
            if isinstance(relation, dict)
        }
        # Older/generated predictions may omit claim_status; treat those as
        # unknown rather than aborting the whole dataset evaluation.
        p_claims = {
            (relation_key(relation, p_by_id), relation.get("claim_status", "unknown"))
            for relation in p_relation_rows
            if isinstance(relation, dict)
        }
        claim_scores.append(scores(g_claims, p_claims))
        by_language[g.get("language", "unknown")].append(job_id)

    def average(values: list[dict[str, float]]) -> dict[str, float]:
        if not values:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "gold": 0, "predicted": 0, "correct": 0}
        return {key: round(sum(value[key] for value in values) / len(values), 4) for key in ("precision", "recall", "f1")}

    result: dict[str, Any] = {
        "jobs_gold": len(gold),
        "jobs_predicted": len(predicted),
        "jobs_evaluated": len(job_ids),
        "jobs_missing_predictions": len(set(gold) - set(predicted)),
        "generation_success_rate": round(len(set(gold) & set(predicted)) / len(gold), 4)
        if gold
        else 0.0,
        "entity_strict": aggregate(entity_scores),
        "relation_strict": aggregate(relation_scores),
        "relation_with_claim_status": aggregate(claim_scores),
        "macro_by_job": {"entity_strict": average(entity_scores), "relation_strict": average(relation_scores), "relation_with_claim_status": average(claim_scores)},
        "by_language": {},
    }
    for language, language_jobs in by_language.items():
        language_entity = [entity_scores[job_ids.index(job_id)] for job_id in language_jobs]
        language_relation = [relation_scores[job_ids.index(job_id)] for job_id in language_jobs]
        language_claim = [claim_scores[job_ids.index(job_id)] for job_id in language_jobs]
        result["by_language"][language] = {
            "jobs": len(language_jobs),
            "entity_strict": aggregate(language_entity),
            "relation_strict": aggregate(language_relation),
            "relation_with_claim_status": aggregate(language_claim),
            "macro_by_job": {
                "entity_strict": average(language_entity),
                "relation_strict": average(language_relation),
                "relation_with_claim_status": average(language_claim),
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--gold-index", type=Path, default=Path("data/processed/reviewed/gold/record_index.jsonl"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--pred-index", type=Path)
    parser.add_argument("--jobs", type=Path, help="Restrict evaluation to job IDs in this file")
    parser.add_argument("--limit", type=int, help="Use the first N rows from --jobs")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many rows from --jobs")
    parser.add_argument(
        "--include-missing-as-empty",
        action="store_true",
        help="Count selected gold jobs without predictions as empty predictions",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
