#!/usr/bin/env python3
"""Build leakage-safe, evidence-gated KG-v2 prompts for fixed experiment windows."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def strip_outer_markup(value: Any) -> str:
    text = normalize_text(value)
    pairs = {"《": "》", "（": "）", "(": ")", "【": "】", "[": "]", "“": "”", "「": "」", "『": "』"}
    while len(text) >= 2 and text[0] in pairs and text.endswith(pairs[text[0]]):
        text = text[1:-1].strip()
    return text


def content_length(value: str) -> int:
    return len(re.sub(r"[^\w\u3400-\u9fff]+", "", value, flags=re.UNICODE))


def segment_texts(job: dict[str, Any]) -> list[str]:
    return [normalize_text(segment.get("text")) for segment in job.get("segments", []) if normalize_text(segment.get("text"))]


def build_anchor_catalog(concepts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for concept in concepts:
        language = str(concept.get("language", "unknown"))
        name = strip_outer_markup(concept.get("canonical_name"))
        entity_type = str(concept.get("type", ""))
        if not name or not entity_type:
            continue
        key = (language, name)
        entry = grouped.setdefault(
            key,
            {
                "language": language,
                "name": name,
                "type_counts": Counter(),
                "type_documents": defaultdict(set),
                "type_concepts": defaultdict(set),
            },
        )
        weight = max(int(concept.get("mention_count", 1) or 1), 1)
        entry["type_counts"][entity_type] += weight
        entry["type_documents"][entity_type].update(concept.get("source_documents", []))
        entry["type_concepts"][entity_type].add(concept.get("concept_id"))

    catalog = []
    for entry in grouped.values():
        entity_type, support = entry["type_counts"].most_common(1)[0]
        total = sum(entry["type_counts"].values())
        catalog.append(
            {
                "language": entry["language"],
                "name": entry["name"],
                "type": entity_type,
                "purity": support / total,
                "support": support,
                "source_documents": entry["type_documents"][entity_type],
                "concept_ids": entry["type_concepts"][entity_type],
            }
        )
    return catalog


def select_exact_anchors(
    job: dict[str, Any],
    catalog: list[dict[str, Any]],
    limit: int,
    min_purity: float,
    min_en_chars: int,
    min_zh_chars: int,
    per_type_limit: int,
) -> list[dict[str, Any]]:
    texts = segment_texts(job)
    language = str(job.get("language", "unknown"))
    document_id = str(job.get("document_id", ""))
    matches = []
    for candidate in catalog:
        if candidate["language"] not in {"unknown", language}:
            continue
        minimum = min_zh_chars if language == "zh" else min_en_chars
        if content_length(candidate["name"]) < minimum or candidate["purity"] < min_purity:
            continue
        if not (set(candidate["source_documents"]) - {document_id}):
            continue
        if not any(candidate["name"] in text for text in texts):
            continue
        matches.append(candidate)
    matches.sort(key=lambda row: (-content_length(row["name"]), -row["purity"], -row["support"], row["type"], row["name"]))
    selected = []
    type_counts: Counter[str] = Counter()
    for candidate in matches:
        if type_counts[candidate["type"]] >= per_type_limit:
            continue
        selected.append(
            {
                "text": candidate["name"],
                "type": candidate["type"],
                "purity": round(candidate["purity"], 4),
                "support": candidate["support"],
            }
        )
        type_counts[candidate["type"]] += 1
        if len(selected) >= limit:
            break
    return selected


def concept_map(concepts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(concept.get("concept_id")): concept for concept in concepts if concept.get("concept_id")}


def build_edge_catalog(
    relations: list[dict[str, Any]], concepts_by_id: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    for relation in relations:
        source = concepts_by_id.get(str(relation.get("source_concept_id")))
        target = concepts_by_id.get(str(relation.get("target_concept_id")))
        if not source or not target:
            continue
        language = str(source.get("language", "unknown"))
        if target.get("language", language) != language:
            continue
        source_name = strip_outer_markup(source.get("canonical_name"))
        target_name = strip_outer_markup(target.get("canonical_name"))
        if not source_name or not target_name or source_name == target_name:
            continue
        key = (
            language,
            source_name,
            str(source.get("type", "")),
            str(relation.get("type", "")),
            target_name,
            str(target.get("type", "")),
        )
        entry = grouped.setdefault(
            key,
            {
                "language": language,
                "source": source_name,
                "source_type": str(source.get("type", "")),
                "relation": str(relation.get("type", "")),
                "target": target_name,
                "target_type": str(target.get("type", "")),
                "documents": set(),
                "support": 0,
            },
        )
        entry["documents"].add(str(relation.get("document_id", "")))
        entry["support"] += 1
    return list(grouped.values())


def select_edge_priors(job: dict[str, Any], catalog: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    texts = segment_texts(job)
    language = str(job.get("language", "unknown"))
    document_id = str(job.get("document_id", ""))
    matches = []
    for edge in catalog:
        if edge["language"] not in {"unknown", language}:
            continue
        if not (set(edge["documents"]) - {document_id}):
            continue
        if not any(edge["source"] in text and edge["target"] in text for text in texts):
            continue
        matches.append(edge)
    matches.sort(
        key=lambda row: (
            -row["support"],
            -(content_length(row["source"]) + content_length(row["target"])),
            row["relation"],
        )
    )
    return [
        {
            "source": row["source"],
            "source_type": row["source_type"],
            "relation": row["relation"],
            "target": row["target"],
            "target_type": row["target_type"],
            "support": row["support"],
        }
        for row in matches[:limit]
    ]


def evidence_items(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def build_relation_example_catalog(
    relations: list[dict[str, Any]], concepts_by_id: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for relation in relations:
        source = concepts_by_id.get(str(relation.get("source_concept_id")))
        target = concepts_by_id.get(str(relation.get("target_concept_id")))
        if not source or not target:
            continue
        language = str(source.get("language", "unknown"))
        if target.get("language", language) != language:
            continue
        items = evidence_items(relation.get("evidence"))
        quote = normalize_text(items[0].get("text")) if items else ""
        if content_length(quote) < 8:
            continue
        key = (
            language,
            str(source.get("type", "")),
            str(relation.get("type", "")),
            str(target.get("type", "")),
            quote,
        )
        entry = grouped.setdefault(
            key,
            {
                "language": language,
                "source_type": str(source.get("type", "")),
                "relation": str(relation.get("type", "")),
                "target_type": str(target.get("type", "")),
                "quote": quote,
                "documents": set(),
                "support": 0,
            },
        )
        entry["documents"].add(str(relation.get("document_id", "")))
        entry["support"] += 1
    return list(grouped.values())


def retrieve_semantic_patterns(
    jobs: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    model_path: Path | None,
    device: str,
    batch_size: int,
    threshold: float,
    limit: int,
) -> dict[str, list[dict[str, Any]]]:
    result = {str(job["job_id"]): [] for job in jobs}
    if model_path is None or not examples or not jobs or limit <= 0:
        return result
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(model_path), device=device, local_files_only=True)
    example_embeddings = model.encode(
        [example["quote"] for example in examples],
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    queries = []
    query_jobs = []
    for job_index, job in enumerate(jobs):
        for text in segment_texts(job):
            if content_length(text) < 8:
                continue
            queries.append(text)
            query_jobs.append(job_index)
    if not queries:
        return result
    query_embeddings = model.encode(
        queries,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    query_indexes_by_job: dict[int, list[int]] = defaultdict(list)
    for query_index, job_index in enumerate(query_jobs):
        query_indexes_by_job[job_index].append(query_index)
    example_indexes_by_language: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        example_indexes_by_language[example["language"]].append(index)

    for job_index, job in enumerate(jobs):
        query_indexes = query_indexes_by_job.get(job_index, [])
        candidate_indexes = example_indexes_by_language.get(str(job.get("language", "unknown")), [])
        if not query_indexes or not candidate_indexes:
            continue
        scores = query_embeddings[query_indexes] @ example_embeddings[candidate_indexes].T
        best_scores = scores.max(axis=0)
        order = best_scores.argsort()[::-1]
        selected = []
        used_relation_types = set()
        document_id = str(job.get("document_id", ""))
        for local_index in order:
            score = float(best_scores[local_index])
            if score < threshold:
                break
            example = examples[candidate_indexes[int(local_index)]]
            if not (set(example["documents"]) - {document_id}):
                continue
            if example["relation"] in used_relation_types:
                continue
            selected.append(
                {
                    "source_type": example["source_type"],
                    "relation": example["relation"],
                    "target_type": example["target_type"],
                    "quote": example["quote"],
                    "score": round(score, 4),
                    "support": example["support"],
                }
            )
            used_relation_types.add(example["relation"])
            if len(selected) >= limit:
                break
        result[str(job["job_id"])] = selected
    return result


def clip(value: str, limit: int = 180) -> str:
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def render_context(
    anchors: list[dict[str, Any]], edges: list[dict[str, Any]], patterns: list[dict[str, Any]]
) -> str:
    lines = [
        "KG_RULES: KG_V2_EVIDENCE_GATED",
        "The supplied source text always overrides graph memory.",
        "ENTITY_ANCHORS are type hypotheses only: use one only when its text occurs in the supplied segments, and copy the actual source span exactly.",
        "EDGE_PRIORS are usable only when both endpoints and one supporting local quote occur in the supplied segments.",
        "SEMANTIC_RELATION_PATTERNS are training examples, never facts or entity candidates for this document.",
        "Do not emit a relation from graph similarity alone. Missing graph context is unknown, not negative evidence.",
        "ENTITY_ANCHORS:",
    ]
    lines.extend(
        f'- "{row["text"]}" => {row["type"]} (type_purity={row["purity"]:.2f}, support={row["support"]})'
        for row in anchors
    )
    if not anchors:
        lines.append("- none")
    lines.append("EDGE_PRIORS:")
    lines.extend(
        f'- "{row["source"]}" [{row["source_type"]}] --{row["relation"]}--> "{row["target"]}" [{row["target_type"]}] (support={row["support"]})'
        for row in edges
    )
    if not edges:
        lines.append("- none")
    lines.append("SEMANTIC_RELATION_PATTERNS:")
    lines.extend(
        f'- {row["source_type"]} --{row["relation"]}--> {row["target_type"]} (similarity={row["score"]:.2f}): "{clip(row["quote"])}"'
        for row in patterns
    )
    if not patterns:
        lines.append("- none")
    return "\n".join(lines)


def numeric_summary(values: list[float | int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "mean": 0, "median": 0, "max": 0}
    return {
        "min": min(values),
        "mean": round(mean(values), 4),
        "median": round(median(values), 4),
        "max": max(values),
    }


def run(args: argparse.Namespace) -> int:
    jobs = load_jsonl(args.jobs)
    concepts = load_jsonl(args.concepts)
    relations = load_jsonl(args.relations)
    anchors = build_anchor_catalog(concepts)
    concepts_by_id = concept_map(concepts)
    edges = build_edge_catalog(relations, concepts_by_id)
    relation_examples = build_relation_example_catalog(relations, concepts_by_id)
    semantic_by_job = retrieve_semantic_patterns(
        jobs,
        relation_examples,
        args.semantic_model,
        args.device,
        args.batch_size,
        args.semantic_threshold,
        args.semantic_limit,
    )

    output = []
    anchor_counts = []
    edge_counts = []
    pattern_counts = []
    selected_scores = []
    for job in jobs:
        selected_anchors = select_exact_anchors(
            job,
            anchors,
            args.anchor_limit,
            args.min_type_purity,
            args.min_en_chars,
            args.min_zh_chars,
            args.anchor_per_type,
        )
        selected_edges = select_edge_priors(job, edges, args.edge_limit)
        selected_patterns = semantic_by_job[str(job["job_id"])]
        base_instruction = str(job.get("system_instruction", "")).split("\n\nKG_RULES:", 1)[0].rstrip()
        context = render_context(selected_anchors, selected_edges, selected_patterns)
        metadata = {
            "version": "kg-v2-evidence-gated-1.0",
            "train_graph_only": True,
            "leave_current_document_out": True,
            "anchors": selected_anchors,
            "edge_priors": selected_edges,
            "semantic_relation_patterns": selected_patterns,
        }
        output.append(
            {
                **job,
                "system_instruction": f"{base_instruction}\n\n{context}",
                "experiment_mode": "kg_v2_evidence_gated",
                "kg_v2_context": metadata,
            }
        )
        anchor_counts.append(len(selected_anchors))
        edge_counts.append(len(selected_edges))
        pattern_counts.append(len(selected_patterns))
        selected_scores.extend(row["score"] for row in selected_patterns)

    write_jsonl(args.output, output)
    audit = {
        "version": "kg-v2-evidence-gated-1.0",
        "jobs": len(output),
        "train_concepts": len(concepts),
        "train_relations": len(relations),
        "anchor_catalog": len(anchors),
        "relation_example_catalog": len(relation_examples),
        "settings": {
            "anchor_limit": args.anchor_limit,
            "anchor_per_type": args.anchor_per_type,
            "edge_limit": args.edge_limit,
            "min_type_purity": args.min_type_purity,
            "semantic_model": str(args.semantic_model) if args.semantic_model else None,
            "semantic_threshold": args.semantic_threshold,
            "semantic_limit": args.semantic_limit,
        },
        "anchors_per_job": numeric_summary(anchor_counts),
        "edge_priors_per_job": numeric_summary(edge_counts),
        "semantic_patterns_per_job": numeric_summary(pattern_counts),
        "selected_semantic_scores": numeric_summary(selected_scores),
        "jobs_with_anchors": sum(value > 0 for value in anchor_counts),
        "jobs_with_edge_priors": sum(value > 0 for value in edge_counts),
        "jobs_with_semantic_patterns": sum(value > 0 for value in pattern_counts),
        "output": str(args.output),
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True, help="Fixed baseline window jobs")
    parser.add_argument("--concepts", type=Path, required=True, help="Training-only KG concepts.jsonl")
    parser.add_argument("--relations", type=Path, required=True, help="Training-only KG relations.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--semantic-model", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--semantic-threshold", type=float, default=0.72)
    parser.add_argument("--semantic-limit", type=int, default=4)
    parser.add_argument("--anchor-limit", type=int, default=12)
    parser.add_argument("--anchor-per-type", type=int, default=2)
    parser.add_argument("--edge-limit", type=int, default=6)
    parser.add_argument("--min-type-purity", type=float, default=0.8)
    parser.add_argument("--min-en-chars", type=int, default=4)
    parser.add_argument("--min-zh-chars", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
