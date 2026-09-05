#!/usr/bin/env python3
"""Build split-specific teacher jobs for baseline and KG-constrained experiments."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def train_concepts(mentions_path: Path) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    documents: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for mention in load_jsonl(mentions_path):
        if mention.get("split") == "train":
            key = (
                mention.get("language", "unknown"),
                mention.get("type", "UNKNOWN"),
                mention.get("canonical_name")
                or mention.get("normalized_name")
                or mention.get("text", ""),
            )
            counts[key] += 1
            documents[key].add(mention.get("document_id", ""))
    return [
        {
            "language": key[0],
            "type": key[1],
            "name": key[2],
            "count": count,
            "source_documents": documents[key],
        }
        for key, count in counts.items()
        if key[1]
    ]


def retrieve_concept_context(
    job: dict[str, Any], concepts: list[dict[str, Any]], limit: int, balanced: bool = False
) -> str:
    text = "\n".join(segment.get("text", "") for segment in job.get("segments", [])).casefold()
    matches = []
    for concept in concepts:
        if concept.get("language") not in {None, "unknown", job.get("language", "unknown")}:
            continue
        # During training, leave the current document out so the KG prompt does
        # not disclose labels taken from the same document.
        other_documents = concept["source_documents"] - {job["document_id"]}
        if not other_documents or concept["name"].casefold() not in text:
            continue
        matches.append(concept)
    matches.sort(key=lambda item: (-item["count"], -len(item["name"]), item["type"], item["name"]))
    if balanced:
        by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in matches:
            by_type[item["type"]].append(item)
        values = []
        type_names = sorted(by_type)
        index = 0
        while len(values) < limit and type_names:
            selected_any = False
            for entity_type in type_names:
                bucket = by_type[entity_type]
                if index < len(bucket):
                    item = bucket[index]
                    values.append(f"{item['type']}: {item['name']}")
                    selected_any = True
                    if len(values) >= limit:
                        break
            if not selected_any:
                break
            index += 1
    else:
        values = [f"{item['type']}: {item['name']}" for item in matches[:limit]]
    return "; ".join(values) if values else "No exact train-KG concept match."


def constraint_context(ontology: dict[str, Any], concepts: str) -> str:
    signatures = ontology.get("allowed_relation_signatures", {})
    lines = []
    for relation, signature in signatures.items():
        lines.append(
            f"{relation}: source={','.join(signature.get('source', []))}; "
            f"target={','.join(signature.get('target', []))}"
        )
    return (
        "\n\nKG_RULES: exact source spans only; omit unsupported entities/relations; "
        "omit relations with an illegal endpoint type pair.\nLEGAL:\n"
        + "\n".join(lines)
        + "\nHINTS (train KG, verify in text):\n"
        + concepts
    )


def run(args: argparse.Namespace) -> int:
    jobs = load_jsonl(args.jobs)
    manifest = load_jsonl(args.manifest)
    split_by_document = {row["document_id"]: row["split"] for row in manifest}
    selected = [job for job in jobs if split_by_document.get(job["document_id"]) == args.split]
    if args.mode == "kg_constrained":
        ontology = yaml.safe_load(args.ontology.read_text(encoding="utf-8"))
        concepts = train_concepts(args.mentions)
        for job in selected:
            context = retrieve_concept_context(job, concepts, args.concept_limit, args.balanced_concepts)
            job["system_instruction"] += constraint_context(ontology, context)
            job["experiment_mode"] = args.mode
    else:
        for job in selected:
            job["experiment_mode"] = args.mode
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for job in selected:
            stream.write(json.dumps(job, ensure_ascii=False) + "\n")
    print(json.dumps({"split": args.split, "mode": args.mode, "jobs": len(selected), "output": str(args.output)}, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, default=Path("data/processed/preannotation/jobs.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/reviewed/split_manifest.jsonl"))
    parser.add_argument("--mentions", type=Path, default=Path("data/processed/reviewed/knowledge_graph/mentions.jsonl"))
    parser.add_argument("--ontology", type=Path, default=Path("configs/risk_ontology.yaml"))
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--mode", choices=["baseline", "kg_constrained"], default="baseline")
    parser.add_argument("--concept-limit", type=int, default=20)
    parser.add_argument("--balanced-concepts", action="store_true", help="Round-robin exact KG hints across entity types")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
