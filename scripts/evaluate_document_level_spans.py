#!/usr/bin/env python3
"""Evaluate unique global spans after merging all windows of each document.

The window builder deliberately overlaps source regions and may also create
relation-rescue windows.  Scoring each window independently therefore gives
some source objects more than one vote.  This evaluator resolves window
predictions to global character spans, merges them by document, and only then
computes strict entity, relation, and claim-status-aware relation metrics.

The script is validation-safe by default and writes no experiment artifacts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Hashable

from evaluate_span_aware import (
    annotation,
    evidence_items,
    indexed_rows,
    load_jsonl,
    matching_spans,
    resolve_entity_span,
    score,
)


FIELDS = ("entity", "relation", "relation_with_claim_status")


def unresolved_entity_key(
    document_id: str, parent_job_id: str, entity: dict[str, Any]
) -> tuple[Hashable, ...]:
    return (
        document_id,
        "unresolved-gold",
        parent_job_id,
        str(entity.get("id")),
        str(entity.get("type")),
    )


def resolved_entity_key(
    document_id: str, span: tuple[int, int], entity_type: Any
) -> tuple[Hashable, ...]:
    return (document_id, span[0], span[1], str(entity_type))


def resolve_parent_gold_span(
    entity: dict[str, Any],
) -> tuple[tuple[int, int] | None, str]:
    """Resolve reviewed gold from its authoritative global evidence interval."""
    entity_text = str(entity.get("text", ""))
    candidates: set[tuple[int, int]] = set()
    for evidence in evidence_items(entity.get("evidence")):
        quote = str(evidence.get("text", ""))
        quote_start = evidence.get("start")
        quote_end = evidence.get("end")
        if not isinstance(quote_start, int) or not isinstance(quote_end, int):
            continue
        if quote_end - quote_start == len(entity_text) and quote == entity_text:
            candidates.add((quote_start, quote_end))
            continue
        if quote_end - quote_start != len(quote):
            continue
        candidates.update(
            (quote_start + local_start, quote_start + local_end)
            for local_start, local_end in matching_spans(quote, entity_text)
        )
    if len(candidates) == 1:
        return next(iter(candidates)), "resolved"
    if not candidates:
        return None, "not_found"
    return None, "ambiguous"


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


def run(args: argparse.Namespace) -> int:
    source_index = load_jsonl(args.source_gold_index)
    source_gold = indexed_rows(load_jsonl(args.source_gold), source_index)
    window_index = load_jsonl(args.gold_index)
    if not args.allow_non_validation:
        non_validation = sorted(
            {
                item.get("split")
                for item in window_index
                if item.get("split") != "validation"
            }
        )
        if non_validation:
            raise ValueError(
                f"refusing non-validation split(s) {non_validation}; "
                "pass --allow-non-validation explicitly"
            )

    window_gold = indexed_rows(load_jsonl(args.gold), window_index)
    jobs = {row["job_id"]: row for row in load_jsonl(args.jobs)}
    predictions = {
        row["job_id"]: annotation(row) for row in load_jsonl(args.predictions)
    }
    requested = [item["job_id"] for item in window_index]
    parent_by_window = {
        item["job_id"]: str(item.get("parent_job_id")) for item in window_index
    }
    windows_by_parent: dict[str, list[str]] = defaultdict(list)
    for job_id in requested:
        windows_by_parent[parent_by_window[job_id]].append(job_id)

    missing_parents = sorted(set(windows_by_parent) - set(source_gold))
    missing_jobs = sorted(set(requested) - set(jobs))
    unexpected_predictions = sorted(set(predictions) - set(requested))
    if missing_parents or missing_jobs or unexpected_predictions:
        raise ValueError(
            "input mismatch: "
            f"missing parent gold={missing_parents[:3]}, "
            f"missing jobs={missing_jobs[:3]}, "
            f"unexpected predictions={unexpected_predictions[:3]}"
        )

    language_by_document: dict[str, str] = {}
    parent_records_by_document: dict[str, int] = Counter()
    for parent_job_id, parent in source_gold.items():
        document_id = str(parent.get("document_id"))
        parent_records_by_document[document_id] += 1
        child_languages = {
            str(window_gold[job_id].get("language", jobs[job_id].get("language", "unknown")))
            for job_id in windows_by_parent.get(parent_job_id, [])
        }
        if len(child_languages) != 1:
            raise ValueError(
                f"inconsistent languages for parent record {parent_job_id}: "
                f"{sorted(child_languages)}"
            )
        language = next(iter(child_languages))
        previous = language_by_document.setdefault(document_id, language)
        if previous != language:
            raise ValueError(f"inconsistent language for document {document_id}")

    gold_entities_by_document: dict[str, set[tuple[Hashable, ...]]] = defaultdict(set)
    gold_relations_by_document: dict[str, set[tuple[Hashable, ...]]] = defaultdict(set)
    gold_claim_relations_by_document: dict[str, set[tuple[Hashable, ...]]] = defaultdict(set)
    gold_resolution = Counter()
    gold_window_entity_occurrences = 0
    gold_window_relation_occurrences = 0

    for parent_job_id, parent in source_gold.items():
        document_id = str(parent.get("document_id"))
        child_job_ids = windows_by_parent.get(parent_job_id, [])
        child_entity_ids = {
            str(entity.get("id"))
            for job_id in child_job_ids
            for entity in window_gold[job_id].get("entities", [])
        }
        gold_window_entity_occurrences += sum(
            len(window_gold[job_id].get("entities", [])) for job_id in child_job_ids
        )
        gold_window_relation_occurrences += sum(
            len(window_gold[job_id].get("relations", [])) for job_id in child_job_ids
        )

        entity_keys: dict[Any, tuple[Hashable, ...]] = {}
        for entity in parent.get("entities", []):
            entity_id = entity.get("id")
            if str(entity_id) not in child_entity_ids:
                raise ValueError(
                    f"gold entity {parent_job_id}/{entity_id} is absent from all windows"
                )
            span, resolution = resolve_parent_gold_span(entity)
            if span is not None:
                key = resolved_entity_key(
                    document_id, span, entity.get("type")
                )
            else:
                key = unresolved_entity_key(document_id, parent_job_id, entity)
            entity_keys[entity_id] = key
            gold_entities_by_document[document_id].add(key)
            gold_resolution[resolution] += 1

        for relation_index, relation in enumerate(parent.get("relations", [])):
            source_key = entity_keys.get(relation.get("source_id"))
            target_key = entity_keys.get(relation.get("target_id"))
            if source_key is None or target_key is None:
                relation_key = (
                    document_id,
                    "unresolved-gold-relation",
                    parent_job_id,
                    relation_index,
                )
            else:
                relation_key = (
                    document_id,
                    source_key,
                    str(relation.get("type")),
                    target_key,
                )
            gold_relations_by_document[document_id].add(relation_key)
            gold_claim_relations_by_document[document_id].add(
                (*relation_key, str(relation.get("claim_status")))
            )

    predicted_entities_by_document: dict[str, set[tuple[Hashable, ...]]] = defaultdict(set)
    predicted_relations_by_document: dict[str, set[tuple[Hashable, ...]]] = defaultdict(set)
    predicted_claim_relations_by_document: dict[str, set[tuple[Hashable, ...]]] = defaultdict(set)
    predicted_resolution = Counter()
    raw_predicted_entities = 0
    raw_predicted_relations = 0

    for job_id in requested:
        job = jobs[job_id]
        document_id = str(job.get("document_id"))
        predicted = predictions.get(
            job_id,
            {
                "document_id": document_id,
                "language": job.get("language", "unknown"),
                "entities": [],
                "relations": [],
            },
        )
        if str(predicted.get("document_id")) != document_id:
            raise ValueError(f"prediction document mismatch for {job_id}")
        raw_predicted_entities += len(predicted.get("entities", []))
        raw_predicted_relations += len(predicted.get("relations", []))
        located_by_id: dict[Any, tuple[Hashable, ...] | None] = {}
        for entity_index, entity in enumerate(predicted.get("entities", [])):
            span, status = resolve_entity_span(entity, job)
            predicted_resolution[status] += 1
            if span is None:
                located_by_id[entity.get("id")] = None
                # Unresolved predictions still count as distinct false positives.
                predicted_entities_by_document[document_id].add(
                    (
                        document_id,
                        "unresolved-prediction",
                        job_id,
                        entity_index,
                        str(entity.get("type")),
                    )
                )
                continue
            key = resolved_entity_key(document_id, span, entity.get("type"))
            located_by_id[entity.get("id")] = key
            predicted_entities_by_document[document_id].add(key)

        for relation_index, relation in enumerate(predicted.get("relations", [])):
            source_key = located_by_id.get(relation.get("source_id"))
            target_key = located_by_id.get(relation.get("target_id"))
            if source_key is None or target_key is None:
                relation_key = (
                    document_id,
                    "unresolved-prediction-relation",
                    job_id,
                    relation_index,
                )
            else:
                relation_key = (
                    document_id,
                    source_key,
                    str(relation.get("type")),
                    target_key,
                )
            predicted_relations_by_document[document_id].add(relation_key)
            predicted_claim_relations_by_document[document_id].add(
                (*relation_key, str(relation.get("claim_status")))
            )

    document_ids = sorted(language_by_document)
    per_document: dict[str, dict[str, Any]] = {}
    by_language_items: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for document_id in document_ids:
        gold_entities = gold_entities_by_document[document_id]
        predicted_entities = predicted_entities_by_document[document_id]
        gold_relations = gold_relations_by_document[document_id]
        predicted_relations = predicted_relations_by_document[document_id]
        gold_claim_relations = gold_claim_relations_by_document[document_id]
        predicted_claim_relations = predicted_claim_relations_by_document[document_id]
        item = {
            "document_id": document_id,
            "language": language_by_document[document_id],
            "parent_records": parent_records_by_document[document_id],
            "entity": score(
                len(gold_entities),
                len(predicted_entities),
                len(gold_entities & predicted_entities),
            ),
            "relation": score(
                len(gold_relations),
                len(predicted_relations),
                len(gold_relations & predicted_relations),
            ),
            "relation_with_claim_status": score(
                len(gold_claim_relations),
                len(predicted_claim_relations),
                len(gold_claim_relations & predicted_claim_relations),
            ),
        }
        per_document[document_id] = item
        by_language_items[item["language"]].append(item)

    items = list(per_document.values())
    result = {
        "metric": "strict-global-character-span-document-deduplicated",
        "selection_split": "validation"
        if not args.allow_non_validation
        else "explicit-non-validation-opt-in",
        "formal_test_read": bool(args.allow_non_validation),
        "unit": "document",
        "documents": len(document_ids),
        "parent_records": len(source_index),
        "windows": len(requested),
        "prediction_rows": len(set(requested) & set(predictions)),
        "overall": {field: aggregate(items, field) for field in FIELDS},
        "macro_by_document": {field: macro(items, field) for field in FIELDS},
        "resolution": {
            "gold_parent_entities": dict(sorted(gold_resolution.items())),
            "predicted_window_entities": dict(sorted(predicted_resolution.items())),
        },
        "deduplication": {
            "gold_parent_entity_objects": sum(
                len(parent.get("entities", [])) for parent in source_gold.values()
            ),
            "gold_unique_entity_objects": sum(
                len(values) for values in gold_entities_by_document.values()
            ),
            "gold_window_entity_occurrences": gold_window_entity_occurrences,
            "gold_parent_relation_objects": sum(
                len(parent.get("relations", [])) for parent in source_gold.values()
            ),
            "gold_unique_relation_objects": sum(
                len(values) for values in gold_relations_by_document.values()
            ),
            "gold_window_relation_occurrences": gold_window_relation_occurrences,
            "predicted_window_entity_occurrences": raw_predicted_entities,
            "predicted_unique_entity_objects": sum(
                len(values) for values in predicted_entities_by_document.values()
            ),
            "predicted_window_relation_occurrences": raw_predicted_relations,
            "predicted_unique_relation_objects": sum(
                len(values) for values in predicted_relations_by_document.values()
            ),
        },
        "by_language": {},
        "per_document": per_document,
    }
    for language, language_items in sorted(by_language_items.items()):
        result["by_language"][language] = {
            "documents": len(language_items),
            "parent_records": sum(
                int(item["parent_records"]) for item in language_items
            ),
            **{field: aggregate(language_items, field) for field in FIELDS},
            "macro_by_document": {
                field: macro(language_items, field) for field in FIELDS
            },
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "per_document"},
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
