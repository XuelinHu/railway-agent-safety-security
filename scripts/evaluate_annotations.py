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


def entity_key(entity: dict[str, Any]) -> tuple[str, str]:
    return norm(entity["text"]), entity["type"]


def relation_key(relation: dict[str, Any], entities: dict[str, dict[str, Any]]) -> tuple[str, str, str]:
    source = entities.get(relation["source_id"], {})
    target = entities.get(relation["target_id"], {})
    return norm(source.get("text", "")), relation["type"], norm(target.get("text", ""))


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
    job_ids = sorted(set(gold) & set(predicted))
    entity_scores = []
    relation_scores = []
    claim_scores = []
    by_language: dict[str, list[str]] = defaultdict(list)
    for job_id in job_ids:
        g = gold[job_id]
        p = predicted[job_id]
        g_entities = {entity_key(entity) for entity in g.get("entities", [])}
        p_entities = {entity_key(entity) for entity in p.get("entities", [])}
        entity_scores.append(scores(g_entities, p_entities))
        g_by_id = {entity["id"]: entity for entity in g.get("entities", [])}
        p_by_id = {entity["id"]: entity for entity in p.get("entities", [])}
        g_relations = {relation_key(relation, g_by_id) for relation in g.get("relations", [])}
        p_relations = {relation_key(relation, p_by_id) for relation in p.get("relations", [])}
        relation_scores.append(scores(g_relations, p_relations))
        g_claims = {(relation_key(r, g_by_id), r["claim_status"]) for r in g.get("relations", [])}
        p_claims = {(relation_key(r, p_by_id), r["claim_status"]) for r in p.get("relations", [])}
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
        "entity_strict": aggregate(entity_scores),
        "relation_strict": aggregate(relation_scores),
        "relation_with_claim_status": aggregate(claim_scores),
        "macro_by_job": {"entity_strict": average(entity_scores), "relation_strict": average(relation_scores), "relation_with_claim_status": average(claim_scores)},
        "by_language": {},
    }
    for language, language_jobs in by_language.items():
        language_entity = [entity_scores[job_ids.index(job_id)] for job_id in language_jobs]
        language_relation = [relation_scores[job_ids.index(job_id)] for job_id in language_jobs]
        result["by_language"][language] = {"jobs": len(language_jobs), "entity_strict": average(language_entity), "relation_strict": average(language_relation)}
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
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
