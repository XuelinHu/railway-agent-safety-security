#!/usr/bin/env python3
"""Probe semantic top-k retrieval coverage of the training-only safety KG."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def canonical(value: Any) -> str:
    text = " ".join(str(value or "").split()).casefold()
    pairs = {"《": "》", "（": "）", "(": ")", "【": "】", "[": "]", "“": "”", "「": "」", "『": "』"}
    while len(text) >= 2 and text[0] in pairs and text.endswith(pairs[text[0]]):
        text = text[1:-1].strip()
    return text


def run(args: argparse.Namespace) -> int:
    from sentence_transformers import SentenceTransformer

    concepts = load_jsonl(args.kg_root / "knowledge_graph" / "concepts.jsonl")
    concept_keys = {
        (concept.get("language", "unknown"), concept.get("type", ""), canonical(concept.get("canonical_name"))): index
        for index, concept in enumerate(concepts)
    }
    concept_texts = [
        f"{concept.get('language', 'unknown')} {concept.get('type', '')}: {canonical(concept.get('canonical_name'))}"
        for concept in concepts
    ]
    model = SentenceTransformer(
        str(args.model),
        device=args.device,
        local_files_only=True,
    )
    concept_embeddings = model.encode(
        concept_texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    result: dict[str, Any] = {"model": str(args.model), "concepts": len(concepts), "splits": {}}
    examples: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        annotations = load_jsonl(args.formal_root / f"{split}.jsonl")
        indexes = load_jsonl(args.formal_root / f"{split}_index.jsonl")
        queries: list[dict[str, Any]] = []
        for index in indexes:
            annotation = annotations[index["record_index"]]
            for entity in annotation.get("entities", []):
                queries.append(
                    {
                        "language": annotation.get("language", "unknown"),
                        "type": entity.get("type", ""),
                        "text": entity.get("text", ""),
                        "key": (annotation.get("language", "unknown"), entity.get("type", ""), canonical(entity.get("text"))),
                    }
                )
        query_texts = [f"{query['language']} {query['type']}: {canonical(query['text'])}" for query in queries]
        query_embeddings = model.encode(
            query_texts,
            batch_size=args.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        scores = query_embeddings @ concept_embeddings.T
        by_language: dict[str, dict[str, int]] = defaultdict(lambda: {"queries": 0, "exact_in_kg": 0})
        topk_counts = {str(k): 0 for k in args.topk}
        exact_count = 0
        for query, row in zip(queries, scores):
            language = query["language"]
            by_language[language]["queries"] += 1
            exact_index = concept_keys.get(query["key"])
            if exact_index is not None:
                exact_count += 1
                by_language[language]["exact_in_kg"] += 1
            order = row.argsort()[::-1]
            for k in args.topk:
                if exact_index is not None and exact_index in order[:k]:
                    topk_counts[str(k)] += 1
            if len(examples) < args.max_examples and exact_index is None:
                top = order[: args.example_topk]
                examples.append(
                    {
                        "split": split,
                        "language": language,
                        "type": query["type"],
                        "query": query["text"],
                        "nearest": [
                            {"name": concepts[index].get("canonical_name"), "type": concepts[index].get("type"), "score": round(float(row[index]), 4)}
                            for index in top
                        ],
                    }
                )
        result["splits"][split] = {
            "queries": len(queries),
            "exact_in_kg": exact_count,
            "exact_coverage": round(exact_count / len(queries), 4) if queries else 0.0,
            "topk_recall_of_exact_targets": {
                str(k): round(topk_counts[str(k)] / exact_count, 4) if exact_count else 0.0 for k in args.topk
            },
            "by_language": dict(sorted(by_language.items())),
        }
    result["unseen_query_examples"] = examples
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["splits"], ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, default=Path("data/processed/reviewed/formal_split"))
    parser.add_argument("--kg-root", type=Path, default=Path("data/processed/experiments/formal_v2"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--topk", type=int, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument("--example-topk", type=int, default=5)
    parser.add_argument("--max-examples", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
