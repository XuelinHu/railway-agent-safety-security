#!/usr/bin/env python3
"""Build split-specific teacher jobs for baseline and KG-constrained experiments."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def train_concept_context(mentions_path: Path, limit: int) -> str:
    counts: Counter[tuple[str, str]] = Counter()
    for mention in load_jsonl(mentions_path):
        if mention.get("split") == "train":
            counts[(mention.get("type", "UNKNOWN"), mention.get("normalized_name") or mention.get("text", ""))] += 1
    values = [f"{entity_type}: {name}" for (entity_type, name), _ in counts.most_common(limit) if name]
    return "; ".join(values)


def constraint_context(ontology: dict[str, Any], concepts: str) -> str:
    signatures = ontology.get("allowed_relation_signatures", {})
    lines = []
    for relation, signature in signatures.items():
        lines.append(
            f"{relation}: source={','.join(signature.get('source', []))}; "
            f"target={','.join(signature.get('target', []))}"
        )
    return (
        "\n\nKNOWLEDGE-GRAPH CONSTRAINTS: Use the following ontology signatures before proposing a relation. "
        "If a source/target type pair is illegal, omit that relation. Keep entity text as exact source spans. "
        "Known training concepts are retrieval hints only; do not copy a concept unless the supplied text supports it.\n"
        "LEGAL RELATION SIGNATURES:\n"
        + "\n".join(lines)
        + "\nTRAINING CONCEPT HINTS:\n"
        + concepts
    )


def run(args: argparse.Namespace) -> int:
    jobs = load_jsonl(args.jobs)
    manifest = load_jsonl(args.manifest)
    split_by_document = {row["document_id"]: row["split"] for row in manifest}
    selected = [job for job in jobs if split_by_document.get(job["document_id"]) == args.split]
    if args.mode == "kg_constrained":
        ontology = yaml.safe_load(args.ontology.read_text(encoding="utf-8"))
        concepts = train_concept_context(args.mentions, args.concept_limit)
        suffix = constraint_context(ontology, concepts)
        for job in selected:
            job["system_instruction"] += suffix
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
    parser.add_argument("--concept-limit", type=int, default=120)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
