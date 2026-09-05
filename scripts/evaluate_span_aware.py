#!/usr/bin/env python3
"""Evaluate window predictions with one-to-one global character spans.

Window gold keeps the original entity IDs but intentionally omits evidence.
This evaluator joins each window entity back to its provenance-bearing parent
annotation, resolves the entity inside its evidence quote, and scores repeated
mentions as separate objects.  Missing predictions and unresolved spans remain
in the denominators.  Non-validation inputs require an explicit opt-in.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def annotation(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("annotation", row)


def indexed_rows(
    rows: list[dict[str, Any]], index: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    return {item["job_id"]: rows[item["record_index"]] for item in index}


def evidence_items(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def matching_spans(source: str, value: str) -> list[tuple[int, int]]:
    """Find exact occurrences, falling back only to whitespace normalization."""
    if not value.strip():
        return []
    exact = [
        (match.start(), match.end())
        for match in re.finditer(re.escape(value), source)
    ]
    if exact:
        return exact
    tokens = re.split(r"\s+", value.strip())
    if not tokens or not all(tokens):
        return []
    pattern = r"\s+".join(re.escape(token) for token in tokens)
    return [
        (match.start(), match.end()) for match in re.finditer(pattern, source)
    ]


def span_in_job(
    start: int,
    end: int,
    segment_id: Any,
    segments: list[dict[str, Any]],
) -> bool:
    for segment in segments:
        original_id = segment.get("original_segment_id", segment.get("segment_id"))
        if segment_id not in {segment.get("segment_id"), original_id}:
            continue
        segment_start = segment.get("start")
        segment_end = segment.get("end")
        if not isinstance(segment_start, int):
            continue
        if not isinstance(segment_end, int):
            segment_end = segment_start + len(str(segment.get("text", "")))
        if segment_start <= start < end <= segment_end:
            return True
    return False


def source_text_for_span(
    start: int,
    end: int,
    segment_id: Any,
    segments: list[dict[str, Any]],
) -> str | None:
    for segment in segments:
        original_id = segment.get("original_segment_id", segment.get("segment_id"))
        if segment_id not in {segment.get("segment_id"), original_id}:
            continue
        segment_start = segment.get("start")
        if not isinstance(segment_start, int):
            continue
        local_start = start - segment_start
        local_end = end - segment_start
        text = str(segment.get("text", ""))
        if 0 <= local_start < local_end <= len(text):
            return text[local_start:local_end]
    return None


def resolve_entity_span(
    entity: dict[str, Any], job: dict[str, Any]
) -> tuple[tuple[int, int] | None, str]:
    """Resolve one entity to a unique global span within the supplied window."""
    entity_text = str(entity.get("text", ""))
    candidates: set[tuple[int, int]] = set()
    for evidence in evidence_items(entity.get("evidence")):
        quote = str(evidence.get("text", ""))
        quote_start = evidence.get("start")
        quote_end = evidence.get("end")
        if not isinstance(quote_start, int) or not isinstance(quote_end, int):
            continue
        # Some extractors carry a full source segment in evidence.text while
        # start/end point directly to the entity.  Prefer that unambiguous
        # interval when its source slice equals entity.text.
        direct_text = source_text_for_span(
            quote_start,
            quote_end,
            evidence.get("segment_id"),
            job.get("segments", []),
        )
        if direct_text == entity_text:
            candidates.add((quote_start, quote_end))
            continue
        # Provenance-bearing gold instead stores the quote interval.  Only use
        # quote-relative offsets when start/end length agrees with the quote.
        if quote_end - quote_start != len(quote):
            continue
        for local_start, local_end in matching_spans(quote, entity_text):
            global_span = quote_start + local_start, quote_start + local_end
            if span_in_job(
                *global_span, evidence.get("segment_id"), job.get("segments", [])
            ):
                candidates.add(global_span)

    if not candidates:
        for segment in job.get("segments", []):
            segment_start = segment.get("start")
            if not isinstance(segment_start, int):
                continue
            for local_start, local_end in matching_spans(
                str(segment.get("text", "")), entity_text
            ):
                candidates.add(
                    (segment_start + local_start, segment_start + local_end)
                )
    if len(candidates) == 1:
        return next(iter(candidates)), "resolved"
    if not candidates:
        return None, "not_found"
    return None, "ambiguous"


def located_entity(
    entity: dict[str, Any], job: dict[str, Any]
) -> dict[str, Any]:
    span, status = resolve_entity_span(entity, job)
    return {"entity": entity, "span": span, "resolution": status}


def max_match_count(
    predicted: list[Any],
    gold: list[Any],
    predicate: Callable[[Any, Any], bool],
) -> int:
    matches: dict[int, int] = {}

    def visit(predicted_index: int, visited: set[int]) -> bool:
        for gold_index, gold_item in enumerate(gold):
            if gold_index in visited or not predicate(
                predicted[predicted_index], gold_item
            ):
                continue
            visited.add(gold_index)
            if gold_index not in matches or visit(matches[gold_index], visited):
                matches[gold_index] = predicted_index
                return True
        return False

    return sum(visit(index, set()) for index in range(len(predicted)))


def entity_matches(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return bool(
        left["span"]
        and left["span"] == right["span"]
        and left["entity"].get("type") == right["entity"].get("type")
    )


def relation_items(
    annotation_row: dict[str, Any],
    located: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    entity_by_id = {
        item["entity"].get("id"): item
        for item in located
        if item["entity"].get("id")
    }
    return [
        {
            "relation": relation,
            "source": entity_by_id.get(relation.get("source_id")),
            "target": entity_by_id.get(relation.get("target_id")),
        }
        for relation in annotation_row.get("relations", [])
    ]


def relation_matches(
    left: dict[str, Any], right: dict[str, Any], include_claim_status: bool
) -> bool:
    if not left["source"] or not left["target"] or not right["source"] or not right["target"]:
        return False
    return bool(
        left["relation"].get("type") == right["relation"].get("type")
        and (
            not include_claim_status
            or left["relation"].get("claim_status")
            == right["relation"].get("claim_status")
        )
        and entity_matches(left["source"], right["source"])
        and entity_matches(left["target"], right["target"])
    )


def score(gold: int, predicted: int, correct: int) -> dict[str, int | float]:
    precision = correct / predicted if predicted else 0.0
    recall = correct / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "gold": gold,
        "predicted": predicted,
        "correct": correct,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def analyze_job(
    gold: dict[str, Any],
    predicted: dict[str, Any],
    parent_gold: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    parent_entities = {
        entity.get("id"): entity for entity in parent_gold.get("entities", [])
    }
    gold_located = []
    gold_resolution = Counter()
    for entity in gold.get("entities", []):
        source_entity = parent_entities.get(entity.get("id"))
        if not source_entity:
            item = {"entity": entity, "span": None, "resolution": "missing_parent"}
        else:
            item = located_entity(source_entity, job)
            item["entity"] = entity
        gold_located.append(item)
        gold_resolution[item["resolution"]] += 1

    predicted_located = [
        located_entity(entity, job) for entity in predicted.get("entities", [])
    ]
    predicted_resolution = Counter(
        item["resolution"] for item in predicted_located
    )
    entity_correct = max_match_count(
        predicted_located, gold_located, entity_matches
    )
    gold_relations = relation_items(gold, gold_located)
    predicted_relations = relation_items(predicted, predicted_located)
    relation_correct = max_match_count(
        predicted_relations,
        gold_relations,
        lambda left, right: relation_matches(left, right, False),
    )
    claim_correct = max_match_count(
        predicted_relations,
        gold_relations,
        lambda left, right: relation_matches(left, right, True),
    )
    return {
        "language": gold.get("language", "unknown"),
        "document_id": gold.get("document_id"),
        "entity": score(
            len(gold_located), len(predicted_located), entity_correct
        ),
        "relation": score(
            len(gold_relations), len(predicted_relations), relation_correct
        ),
        "relation_with_claim_status": score(
            len(gold_relations), len(predicted_relations), claim_correct
        ),
        "resolution": {
            "gold_entities": dict(sorted(gold_resolution.items())),
            "predicted_entities": dict(sorted(predicted_resolution.items())),
            "gold_relation_endpoints_resolved": sum(
                bool(item["source"] and item["source"]["span"] and item["target"] and item["target"]["span"])
                for item in gold_relations
            ),
            "predicted_relation_endpoints_resolved": sum(
                bool(item["source"] and item["source"]["span"] and item["target"] and item["target"]["span"])
                for item in predicted_relations
            ),
        },
    }


def aggregate(items: list[dict[str, Any]], field: str) -> dict[str, int | float]:
    return score(
        sum(int(item[field]["gold"]) for item in items),
        sum(int(item[field]["predicted"]) for item in items),
        sum(int(item[field]["correct"]) for item in items),
    )


def macro(items: list[dict[str, Any]], field: str) -> dict[str, float]:
    return {
        metric: round(
            sum(float(item[field][metric]) for item in items) / len(items), 4
        )
        if items
        else 0.0
        for metric in ("precision", "recall", "f1")
    }


def aggregate_resolution(items: list[dict[str, Any]]) -> dict[str, Any]:
    gold = Counter()
    predicted = Counter()
    for item in items:
        gold.update(item["resolution"]["gold_entities"])
        predicted.update(item["resolution"]["predicted_entities"])
    return {
        "gold_entities": dict(sorted(gold.items())),
        "predicted_entities": dict(sorted(predicted.items())),
        "gold_relation_endpoints_resolved": sum(
            item["resolution"]["gold_relation_endpoints_resolved"] for item in items
        ),
        "predicted_relation_endpoints_resolved": sum(
            item["resolution"]["predicted_relation_endpoints_resolved"] for item in items
        ),
    }


def run(args: argparse.Namespace) -> int:
    source_index = load_jsonl(args.source_gold_index)
    source_gold = indexed_rows(load_jsonl(args.source_gold), source_index)
    window_index = load_jsonl(args.gold_index)
    if not args.allow_non_validation:
        non_validation = sorted(
            {item.get("split") for item in window_index if item.get("split") != "validation"}
        )
        if non_validation:
            raise ValueError(
                f"refusing non-validation split(s) {non_validation}; pass --allow-non-validation explicitly"
            )
    window_gold = indexed_rows(load_jsonl(args.gold), window_index)
    predictions = {
        row["job_id"]: annotation(row) for row in load_jsonl(args.predictions)
    }
    jobs = {row["job_id"]: row for row in load_jsonl(args.jobs)}
    parent_by_window = {
        item["job_id"]: item.get("parent_job_id") for item in window_index
    }
    requested = [item["job_id"] for item in window_index]
    missing_parents = sorted(
        {
            parent_by_window[job_id]
            for job_id in requested
            if parent_by_window[job_id] not in source_gold
        }
    )
    missing_jobs = sorted(set(requested) - set(jobs))
    if missing_parents or missing_jobs:
        raise ValueError(
            f"missing parent gold={missing_parents[:3]} or jobs={missing_jobs[:3]}"
        )

    per_job = {}
    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    items = []
    for job_id in requested:
        gold = window_gold[job_id]
        predicted = predictions.get(
            job_id,
            {
                "document_id": gold.get("document_id"),
                "language": gold.get("language", "unknown"),
                "entities": [],
                "relations": [],
            },
        )
        item = analyze_job(
            gold,
            predicted,
            source_gold[parent_by_window[job_id]],
            jobs[job_id],
        )
        per_job[job_id] = item
        items.append(item)
        by_language[item["language"]].append(item)

    fields = ("entity", "relation", "relation_with_claim_status")
    result = {
        "metric": "strict-global-character-span-one-to-one",
        "selection_split": "validation"
        if not args.allow_non_validation
        else "explicit-non-validation-opt-in",
        "formal_test_read": bool(args.allow_non_validation),
        "jobs": len(requested),
        "documents": len({item["document_id"] for item in items}),
        "generation_success_rate": round(
            len(set(requested) & set(predictions)) / len(requested), 4
        )
        if requested
        else 0.0,
        "overall": {field: aggregate(items, field) for field in fields},
        "macro_by_job": {field: macro(items, field) for field in fields},
        "resolution": aggregate_resolution(items),
        "by_language": {},
        "per_job": per_job,
    }
    for language, language_items in sorted(by_language.items()):
        result["by_language"][language] = {
            "jobs": len(language_items),
            **{field: aggregate(language_items, field) for field in fields},
            "macro_by_job": {
                field: macro(language_items, field) for field in fields
            },
            "resolution": aggregate_resolution(language_items),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "per_job"},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-gold", type=Path, required=True)
    parser.add_argument("--source-gold-index", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--gold-index", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-non-validation", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
