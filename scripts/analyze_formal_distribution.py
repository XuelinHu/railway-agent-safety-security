#!/usr/bin/env python3
"""Analyze formal-split sparsity, dispersion, long tails, and KG coverage."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def canonical_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip().casefold()
    outer_markup = {
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
    changed = True
    while changed and len(text) >= 2:
        changed = False
        closing = outer_markup.get(text[0])
        if closing and text.endswith(closing):
            text = text[1:-1].strip()
            changed = True
    return text


def quantile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def describe(values: list[int | float]) -> dict[str, int | float]:
    numeric = [float(value) for value in values]
    if not numeric:
        return {"count": 0}
    mean = statistics.fmean(numeric)
    std = statistics.pstdev(numeric)
    return {
        "count": len(numeric),
        "mean": round(mean, 4),
        "median": round(statistics.median(numeric), 4),
        "std": round(std, 4),
        "cv": round(std / mean, 4) if mean else 0.0,
        "min": round(min(numeric), 4),
        "p90": round(quantile(numeric, 0.9), 4),
        "p95": round(quantile(numeric, 0.95), 4),
        "max": round(max(numeric), 4),
        "iqr": round(quantile(numeric, 0.75) - quantile(numeric, 0.25), 4),
    }


def distribution_stats(counter: Counter[str]) -> dict[str, Any]:
    total = sum(counter.values())
    probabilities = [count / total for count in counter.values()] if total else []
    entropy = -sum(probability * math.log2(probability) for probability in probabilities if probability)
    max_entropy = math.log2(len(counter)) if len(counter) > 1 else 0.0
    return {
        "total": total,
        "labels": len(counter),
        "singleton_labels": sum(count == 1 for count in counter.values()),
        "singleton_share": round(sum(count == 1 for count in counter.values()) / len(counter), 4) if counter else 0.0,
        "top5_share": round(sum(count for _, count in counter.most_common(5)) / total, 4) if total else 0.0,
        "entropy": round(entropy, 4),
        "normalized_entropy": round(entropy / max_entropy, 4) if max_entropy else 0.0,
        "counts": dict(counter.most_common()),
    }


def key_for_entity(entity: dict[str, Any], language: str) -> str:
    return f"{language}|{entity.get('type', '')}|{canonical_text(entity.get('text', ''))}"


def split_rows(root: Path) -> dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]:
    result: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for split in ("train", "validation", "test"):
        annotations = load_jsonl(root / f"{split}.jsonl")
        indexes = load_jsonl(root / f"{split}_index.jsonl")
        if len(annotations) != len(indexes):
            raise ValueError(f"{split} annotation/index count mismatch")
        result[split] = list(zip(annotations, indexes))
    return result


def analyze_split(rows: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    entity_types: Counter[str] = Counter()
    relation_types: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    document_entities: Counter[str] = Counter()
    document_relations: Counter[str] = Counter()
    document_blocks: Counter[str] = Counter()
    entity_counts: list[int] = []
    relation_counts: list[int] = []
    entity_per_relation_values: list[float] = []
    isolated_entity_count = 0
    total_entities = 0
    total_relations = 0
    candidate_pairs = 0
    for annotation, _index in rows:
        document_id = annotation["document_id"]
        language = annotation.get("language", "unknown")
        languages[language] += 1
        entities = annotation.get("entities", [])
        relations = annotation.get("relations", [])
        entity_types.update(str(entity.get("type", "")) for entity in entities)
        relation_types.update(str(relation.get("type", "")) for relation in relations)
        entity_counts.append(len(entities))
        relation_counts.append(len(relations))
        document_entities[document_id] += len(entities)
        document_relations[document_id] += len(relations)
        document_blocks[document_id] += 1
        total_entities += len(entities)
        total_relations += len(relations)
        candidate_pairs += len(entities) * max(len(entities) - 1, 0)
        degree = Counter()
        for relation in relations:
            degree[relation.get("source_id")] += 1
            degree[relation.get("target_id")] += 1
        isolated_entity_count += sum(entity.get("id") not in degree for entity in entities)
    entity_per_relation_values = [
        document_entities[document_id] / document_relations[document_id]
        for document_id in document_entities
        if document_relations[document_id]
    ]
    return {
        "records": len(rows),
        "documents": len(document_blocks),
        "languages_by_record": dict(languages),
        "entity_count": describe(entity_counts),
        "relation_count": describe(relation_counts),
        "entities_per_document": describe(list(document_entities.values())),
        "relations_per_document": describe(list(document_relations.values())),
        "entity_per_relation_by_document": describe(entity_per_relation_values),
        "isolated_entity_share": round(isolated_entity_count / total_entities, 4) if total_entities else 0.0,
        "relation_to_entity_ratio": round(total_relations / total_entities, 4) if total_entities else 0.0,
        "ordered_pair_candidates": candidate_pairs,
        "observed_relation_pair_share": round(total_relations / candidate_pairs, 4) if candidate_pairs else 0.0,
        "unobserved_pair_share": round(1 - total_relations / candidate_pairs, 4) if candidate_pairs else 0.0,
        "entity_types": distribution_stats(entity_types),
        "relation_types": distribution_stats(relation_types),
    }


def kg_coverage(kg_root: Path, split_rows_by_name: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]) -> dict[str, Any]:
    concepts = load_jsonl(kg_root / "knowledge_graph" / "concepts.jsonl")
    relations = load_jsonl(kg_root / "knowledge_graph" / "relations.jsonl")
    concept_ids = {concept.get("concept_id") for concept in concepts}
    connected_ids = {
        concept_id
        for relation in relations
        for concept_id in (relation.get("source_concept_id"), relation.get("target_concept_id"))
        if concept_id
    }
    train_keys = {
        f"{concept.get('language', '')}|{concept.get('type', '')}|{canonical_text(concept.get('canonical_name', ''))}"
        for concept in concepts
    }
    result: dict[str, Any] = {
        "train_concepts": len(concepts),
        "train_relations": len(relations),
        "concepts_with_multiple_mentions": sum(concept.get("mention_count", 0) > 1 for concept in concepts),
        "isolated_concept_share": round(len(concept_ids - connected_ids) / len(concept_ids), 4) if concept_ids else 0.0,
        "relation_to_concept_ratio": round(len(relations) / len(concepts), 4) if concepts else 0.0,
    }
    for split in ("validation", "test"):
        total = 0
        hits = 0
        by_language: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for annotation, _index in split_rows_by_name[split]:
            language = annotation.get("language", "unknown")
            for entity in annotation.get("entities", []):
                total += 1
                by_language[language][0] += 1
                if key_for_entity(entity, language) in train_keys:
                    hits += 1
                    by_language[language][1] += 1
        result[split] = {
            "entity_mentions": total,
            "exact_typed_canonical_hits": hits,
            "coverage": round(hits / total, 4) if total else 0.0,
            "by_language": {
                language: {
                    "mentions": values[0],
                    "hits": values[1],
                    "coverage": round(values[1] / values[0], 4) if values[0] else 0.0,
                }
                for language, values in sorted(by_language.items())
            },
        }
    return result


def run(args: argparse.Namespace) -> int:
    split_rows_by_name = split_rows(args.formal_root)
    result = {
        "formal_root": str(args.formal_root),
        "kg_root": str(args.kg_root),
        "splits": {split: analyze_split(rows) for split, rows in split_rows_by_name.items()},
        "kg_coverage": kg_coverage(args.kg_root, split_rows_by_name),
        "interpretation_thresholds": {
            "high_cv": "CV >= 1 indicates strong relative dispersion and possible long-document/outlier effects",
            "low_kg_coverage": "exact typed canonical train KG coverage below 0.5 indicates retrieval cannot cover most evaluation mentions without aliases or semantic retrieval",
            "high_isolated_entity_share": "isolated entities indicate relation supervision is sparse or entity candidates are not coupled to edges",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, default=Path("data/processed/reviewed/formal_split"))
    parser.add_argument("--kg-root", type=Path, default=Path("data/processed/experiments/formal_v2"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/experiments/formal_distribution.json"))
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
