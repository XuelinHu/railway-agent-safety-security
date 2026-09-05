#!/usr/bin/env python3
"""Build the formal train-only KG and audit split leakage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from promote_reviewed_annotations import build_graph, load_jsonl


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> int:
    formal_root = args.formal_root
    jobs = {row["job_id"]: row for row in load_jsonl(formal_root / "jobs.jsonl")}
    split_indexes = {
        split: load_jsonl(formal_root / f"{split}_index.jsonl")
        for split in ("train", "validation", "test")
    }
    split_annotations = {
        split: load_jsonl(formal_root / f"{split}.jsonl")
        for split in split_indexes
    }
    split_by_job = {
        row["job_id"]: split
        for split, rows in split_indexes.items()
        for row in rows
    }
    if set(split_by_job) != set(jobs):
        raise ValueError("formal split indexes do not exactly match formal jobs")
    for split in split_indexes:
        if len(split_indexes[split]) != len(split_annotations[split]):
            raise ValueError(f"formal {split} annotations and index counts differ")

    train_items = []
    for index, annotation in zip(split_indexes["train"], split_annotations["train"]):
        job_id = index["job_id"]
        if annotation["document_id"] != jobs[job_id]["document_id"]:
            raise ValueError(f"formal train document mismatch for {job_id}")
        item = json.loads(json.dumps(annotation, ensure_ascii=False))
        item["_job_id"] = job_id
        train_items.append(item)

    train_documents = {item["document_id"] for item in train_items}
    non_train_documents = {
        jobs[job_id]["document_id"]
        for job_id, split in split_by_job.items()
        if split != "train"
    }
    overlap = train_documents & non_train_documents
    if overlap:
        raise ValueError(f"document leakage across formal splits: {sorted(overlap)}")

    output = args.output
    graph_stats = build_graph(
        train_items,
        jobs,
        {document_id: "train" for document_id in train_documents},
        output / "knowledge_graph",
    )
    concepts = load_jsonl(output / "knowledge_graph" / "concepts.jsonl")
    mentions = load_jsonl(output / "knowledge_graph" / "mentions.jsonl")
    relations = load_jsonl(output / "knowledge_graph" / "relations.jsonl")
    leakage = {
        "train_documents": len(train_documents),
        "non_train_documents": len(non_train_documents),
        "document_overlap": sorted(overlap),
        "concepts_with_non_train_sources": sum(
            bool(set(concept.get("source_documents", [])) - train_documents)
            for concept in concepts
        ),
        "mentions_with_non_train_sources": sum(
            mention.get("document_id") not in train_documents for mention in mentions
        ),
        "relations_with_non_train_sources": sum(
            relation.get("document_id") not in train_documents for relation in relations
        ),
    }
    if any(leakage[key] for key in ("document_overlap", "concepts_with_non_train_sources", "mentions_with_non_train_sources", "relations_with_non_train_sources")):
        raise ValueError(f"formal train-only KG leakage detected: {leakage}")
    summary = {
        "source": str(formal_root),
        "train_jobs": len(train_items),
        "train_documents": len(train_documents),
        "graph": graph_stats,
        "leakage_audit": leakage,
        "split_job_counts": {split: len(rows) for split, rows in split_indexes.items()},
        "split_document_counts": {
            split: len({row["document_id"] for row in rows})
            for split, rows in split_indexes.items()
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, default=Path("data/processed/reviewed/formal_split"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/experiments/formal"))
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
