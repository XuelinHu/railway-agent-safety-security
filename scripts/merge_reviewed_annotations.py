#!/usr/bin/env python3
"""Merge manually accepted annotation batches into the reviewed JSONL database."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from promote_reviewed_annotations import (
    build_graph,
    load_jsonl,
    validate_annotations,
    write_jsonl,
)


def manually_accept(annotation: dict[str, Any]) -> dict[str, Any]:
    accepted = json.loads(json.dumps(annotation, ensure_ascii=False))
    accepted["review"] = {
        "status": "single_reviewed",
        "reviewers": ["human_reviewed"],
        "notes": (
            "Manually checked and accepted after the teacher-preannotation v1.1 gate; "
            "entity types, relation directions, evidence, and claim statuses were confirmed."
        ),
    }
    for entity in accepted["entities"]:
        entity["review_status"] = "accepted"
    for relation in accepted["relations"]:
        relation["review_status"] = "accepted"
    return accepted


def read_enveloped_annotations(path: Path) -> list[tuple[str, dict[str, Any]]]:
    rows = load_jsonl(path)
    result = []
    for row in rows:
        if "job_id" not in row or "annotation" not in row:
            raise ValueError(f"new annotations must be job envelopes: {path}")
        result.append((row["job_id"], row["annotation"]))
    return result


def run(args: argparse.Namespace) -> int:
    existing_gold = load_jsonl(args.existing_root / "gold" / "all.jsonl")
    existing_index = load_jsonl(args.existing_root / "gold" / "record_index.jsonl")
    if len(existing_gold) != len(existing_index):
        raise ValueError("existing gold and record index counts differ")

    existing_by_job: dict[str, dict[str, Any]] = {}
    split_by_document: dict[str, str] = {}
    enriched: list[dict[str, Any]] = []
    for annotation, index in zip(existing_gold, existing_index):
        job_id = index["job_id"]
        if job_id in existing_by_job:
            raise ValueError(f"duplicate existing job ID: {job_id}")
        if annotation["document_id"] != index["document_id"]:
            raise ValueError(f"existing document mismatch for {job_id}")
        previous_split = split_by_document.setdefault(annotation["document_id"], index["split"])
        if previous_split != index["split"]:
            raise ValueError(f"document appears in multiple splits: {annotation['document_id']}")
        item = json.loads(json.dumps(annotation, ensure_ascii=False))
        item["_job_id"] = job_id
        existing_by_job[job_id] = item
        enriched.append(item)

    new_rows = read_enveloped_annotations(args.new_annotations)
    new_jobs = {job["job_id"]: job for job in load_jsonl(args.new_jobs)}
    new_job_ids = {job_id for job_id, _ in new_rows}
    missing_jobs = new_job_ids - set(new_jobs)
    if missing_jobs:
        raise ValueError(f"new annotations reference missing jobs: {sorted(missing_jobs)}")
    overlap = new_job_ids & set(existing_by_job)
    if overlap:
        raise ValueError(f"refuse to duplicate existing jobs: {sorted(overlap)}")

    for job_id, annotation in new_rows:
        job = new_jobs[job_id]
        if annotation["document_id"] != job["document_id"]:
            raise ValueError(f"new document mismatch for {job_id}")
        document_id = annotation["document_id"]
        previous_split = split_by_document.setdefault(document_id, args.new_split)
        if previous_split != args.new_split:
            raise ValueError(f"new document has conflicting split: {document_id}")
        accepted = manually_accept(annotation)
        accepted["_job_id"] = job_id
        enriched.append(accepted)

    validate_annotations(
        [{key: value for key, value in item.items() if not key.startswith("_")} for item in enriched],
        args.schema,
    )

    jobs_by_id = {job["job_id"]: job for job in load_jsonl(args.existing_jobs)}
    jobs_by_id.update(new_jobs)
    missing_graph_jobs = {item["_job_id"] for item in enriched} - set(jobs_by_id)
    if missing_graph_jobs:
        raise ValueError(f"graph jobs missing source metadata: {sorted(missing_graph_jobs)}")

    output_root = args.output
    gold_dir = output_root / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)
    active_job_ids = {item["_job_id"] for item in enriched}
    write_jsonl(
        output_root / "jobs.jsonl",
        [jobs_by_id[job_id] for job_id in sorted(active_job_ids)],
    )
    public_annotations = [
        {key: value for key, value in item.items() if not key.startswith("_")} for item in enriched
    ]
    write_jsonl(gold_dir / "all.jsonl", public_annotations)

    split_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    split_items: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in enriched:
        split = split_by_document[item["document_id"]]
        public_item = {key: value for key, value in item.items() if not key.startswith("_")}
        split_rows[split].append(public_item)
        split_items[split].append(item)
    for split, rows in split_rows.items():
        write_jsonl(gold_dir / f"{split}.jsonl", rows)
        write_jsonl(
            gold_dir / f"{split}_index.jsonl",
            [
                {
                    "record_index": index,
                    "job_id": item["_job_id"],
                    "document_id": item["document_id"],
                    "split": split,
                }
                for index, item in enumerate(split_items[split])
            ],
        )
    write_jsonl(
        gold_dir / "record_index.jsonl",
        [
            {
                "record_index": index,
                "job_id": item["_job_id"],
                "document_id": item["document_id"],
                "split": split_by_document[item["document_id"]],
            }
            for index, item in enumerate(enriched)
        ],
    )

    existing_manifest = {
        row["document_id"]: row
        for row in load_jsonl(args.existing_root / "split_manifest.jsonl")
    }
    for job_id, job in new_jobs.items():
        if job_id not in new_job_ids:
            continue
        document_id = job["document_id"]
        existing_manifest[document_id] = {
            "document_id": document_id,
            "source_group": job["source_path"].split("/", 1)[0],
            "category": job["category"],
            "relative_path": job["source_path"],
            "split": split_by_document[document_id],
        }
    write_jsonl(
        output_root / "split_manifest.jsonl",
        [existing_manifest[document_id] for document_id in sorted(existing_manifest)],
    )

    graph_stats = build_graph(
        enriched,
        jobs_by_id,
        split_by_document,
        output_root / "knowledge_graph",
    )
    summary = {
        "jobs": len(enriched),
        "documents": len(split_by_document),
        "split_documents": dict(Counter(split_by_document.values())),
        "split_chunks": dict(Counter(split_by_document[item["document_id"]] for item in enriched)),
        "entities": sum(len(item["entities"]) for item in enriched),
        "relations": sum(len(item["relations"]) for item in enriched),
        "graph": graph_stats,
        "review_policy": "pilot_bulk_accepted_plus_manual_v1_1_gate_review",
        "schema_version": "0.1.0",
        "ontology_version": "1.0.0",
        "added_jobs": len(new_rows),
        "added_documents": len({annotation["document_id"] for _, annotation in new_rows}),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-root", type=Path, default=Path("data/processed/reviewed"))
    parser.add_argument("--existing-jobs", type=Path, default=Path("data/processed/preannotation/jobs.jsonl"))
    parser.add_argument(
        "--new-annotations",
        type=Path,
        default=Path("data/processed/experiments/annotation_pending_sampled_v2_normalized.jsonl"),
    )
    parser.add_argument(
        "--new-jobs",
        type=Path,
        default=Path("data/processed/experiments/annotation_pending_sampled_jobs.jsonl"),
    )
    parser.add_argument("--schema", type=Path, default=Path("schemas/risk_annotation.schema.json"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/reviewed"))
    parser.add_argument("--new-split", choices=("train", "validation", "test"), default="train")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
