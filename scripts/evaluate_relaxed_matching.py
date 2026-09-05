#!/usr/bin/env python3
"""Evaluate strict and diagnostic relaxed entity/relation matching modes."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


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


def boundary_overlap(left: str, right: str, threshold: float = 0.5) -> bool:
    if left == right:
        return True
    if not left or not right:
        return False
    shorter, longer = sorted((left, right), key=len)
    return shorter in longer and len(shorter) / len(longer) >= threshold


OUTER_MARKUP = {
    "《": "》",
    "（": "）",
    "(": ")",
    "【": "】",
    "[": "]",
    "<": ">",
    "“": "”",
    "「": "」",
    "『": "』",
}


def strip_outer_markup(value: Any) -> str:
    """Remove only balanced presentation wrappers, preserving inner content."""
    text = norm(value)
    changed = True
    while changed and len(text) >= 2:
        changed = False
        closing = OUTER_MARKUP.get(text[0])
        if closing and text.endswith(closing):
            text = text[1:-1].strip()
            changed = True
    return text


def entity_matches(predicted: dict[str, Any], gold: dict[str, Any], mode: str) -> bool:
    predicted_text = norm(predicted.get("text"))
    gold_text = norm(gold.get("text"))
    if not predicted_text or not gold_text:
        return False
    if mode == "strict":
        return predicted_text == gold_text and predicted.get("type") == gold.get("type")
    if mode == "text":
        return predicted_text == gold_text
    if mode == "outer_markup":
        return strip_outer_markup(predicted.get("text")) == strip_outer_markup(gold.get("text")) and predicted.get("type") == gold.get("type")
    if mode == "boundary_type":
        return predicted.get("type") == gold.get("type") and boundary_overlap(predicted_text, gold_text)
    if mode == "boundary_text":
        return boundary_overlap(predicted_text, gold_text)
    raise ValueError(f"unknown matching mode: {mode}")


def unique_entities(entities: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    seen = set()
    result = []
    for entity in entities:
        key = norm(entity.get("text")) if mode in {"text", "boundary_text"} else (norm(entity.get("text")), entity.get("type"))
        if key not in seen:
            seen.add(key)
            result.append(entity)
    return result


def max_match_count(
    predicted: list[Any],
    gold: list[Any],
    predicate: Callable[[Any, Any], bool],
) -> int:
    matches: dict[int, int] = {}

    def visit(predicted_index: int, visited: set[int]) -> bool:
        for gold_index, gold_item in enumerate(gold):
            if gold_index in visited or not predicate(predicted[predicted_index], gold_item):
                continue
            visited.add(gold_index)
            if gold_index not in matches or visit(matches[gold_index], visited):
                matches[gold_index] = predicted_index
                return True
        return False

    return sum(visit(index, set()) for index in range(len(predicted)))


def relation_items(annotation_row: dict[str, Any]) -> list[dict[str, Any]]:
    entities = {entity.get("id"): entity for entity in annotation_row.get("entities", [])}
    result = []
    for relation in annotation_row.get("relations", []):
        source = entities.get(relation.get("source_id"), {})
        target = entities.get(relation.get("target_id"), {})
        result.append({"relation": relation, "source": source, "target": target})
    return result


def relation_matches(predicted: dict[str, Any], gold: dict[str, Any], mode: str) -> bool:
    return (
        predicted["relation"].get("type") == gold["relation"].get("type")
        and entity_matches(predicted["source"], gold["source"], mode)
        and entity_matches(predicted["target"], gold["target"], mode)
    )


def score(gold_count: int, predicted_count: int, correct: int) -> dict[str, int | float]:
    precision = correct / predicted_count if predicted_count else 0.0
    recall = correct / gold_count if gold_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "gold": gold_count,
        "predicted": predicted_count,
        "correct": correct,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def analyze_job(gold: dict[str, Any], predicted: dict[str, Any], mode: str) -> dict[str, dict[str, int | float]]:
    gold_entities = unique_entities(gold.get("entities", []), mode)
    predicted_entities = unique_entities(predicted.get("entities", []), mode)
    gold_relations = relation_items(gold)
    predicted_relations = relation_items(predicted)
    entity_correct = max_match_count(
        predicted_entities, gold_entities, lambda left, right: entity_matches(left, right, mode)
    )
    relation_correct = max_match_count(
        predicted_relations, gold_relations, lambda left, right: relation_matches(left, right, mode)
    )
    return {
        "entity": score(len(gold_entities), len(predicted_entities), entity_correct),
        "relation": score(len(gold_relations), len(predicted_relations), relation_correct),
    }


def aggregate(items: list[dict[str, int | float]], field: str) -> dict[str, int | float]:
    gold = sum(int(item[field]["gold"]) for item in items)
    predicted = sum(int(item[field]["predicted"]) for item in items)
    correct = sum(int(item[field]["correct"]) for item in items)
    return score(gold, predicted, correct)


def run(args: argparse.Namespace) -> int:
    gold = keyed_rows(load_jsonl(args.gold), load_jsonl(args.gold_index))
    predicted = keyed_rows(load_jsonl(args.predictions))
    jobs = [row["job_id"] for row in load_jsonl(args.jobs)]
    by_language: dict[str, list[dict[str, dict[str, int | float]]]] = defaultdict(list)
    overall: list[dict[str, dict[str, int | float]]] = []
    for job_id in jobs:
        if job_id not in gold or job_id not in predicted:
            raise ValueError(f"missing job: {job_id}")
        item = analyze_job(gold[job_id], predicted[job_id], args.mode)
        overall.append(item)
        by_language[gold[job_id].get("language", "unknown")].append(item)
    result = {
        "matching_mode": args.mode,
        "jobs": len(jobs),
        "documents": len({gold[job_id]["document_id"] for job_id in jobs}),
        "overall": {field: aggregate(overall, field) for field in ("entity", "relation")},
        "by_language": {
            language: {field: aggregate(items, field) for field in ("entity", "relation")}
            for language, items in sorted(by_language.items())
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
    parser.add_argument("--mode", choices=("strict", "text", "outer_markup", "boundary_type", "boundary_text"), default="strict")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
