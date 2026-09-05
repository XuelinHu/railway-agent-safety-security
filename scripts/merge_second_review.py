#!/usr/bin/env python3
"""Promote a completed formal test second review into the reviewed database."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def accepted_double_review(annotation: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(annotation, ensure_ascii=False))
    result["review"] = {
        "status": "double_reviewed",
        "reviewers": ["human_reviewed", "second_reviewer"],
        "notes": "All entities and relations were accepted during the independent formal test second review.",
    }
    for entity in result.get("entities", []):
        entity["review_status"] = "accepted"
    for relation in result.get("relations", []):
        relation["review_status"] = "accepted"
    return result


def public(annotation: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in annotation.items() if not key.startswith("_")}


def write_gold(gold_root: Path, enriched: list[dict[str, Any]], split_by_job: dict[str, str]) -> None:
    gold_dir = gold_root / "gold"
    write_jsonl(gold_dir / "all.jsonl", [public(row) for row in enriched])
    for split in ("train", "validation", "test"):
        split_rows = [row for row in enriched if split_by_job[row["_job_id"]] == split]
        write_jsonl(gold_dir / f"{split}.jsonl", [public(row) for row in split_rows])
        write_jsonl(
            gold_dir / f"{split}_index.jsonl",
            [
                {
                    "record_index": index,
                    "job_id": row["_job_id"],
                    "document_id": row["document_id"],
                    "split": split,
                }
                for index, row in enumerate(split_rows)
            ],
        )
    write_jsonl(
        gold_dir / "record_index.jsonl",
        [
            {
                "record_index": index,
                "job_id": row["_job_id"],
                "document_id": row["document_id"],
                "split": split_by_job[row["_job_id"]],
            }
            for index, row in enumerate(enriched)
        ],
    )


def run(args: argparse.Namespace) -> int:
    queue = load_jsonl(args.queue)
    queue_by_job = {row["job_id"]: row for row in queue}
    if len(queue_by_job) != len(queue):
        raise ValueError("second-review queue contains duplicate job IDs")
    if not queue:
        raise ValueError("second-review queue is empty")
    if any(row.get("review_meta", {}).get("second_review_status") != "pending_second_review" for row in queue):
        raise ValueError("queue is not in the expected pending second-review state")

    index = load_jsonl(args.gold_root / "gold" / "record_index.jsonl")
    gold = load_jsonl(args.gold_root / "gold" / "all.jsonl")
    if len(index) != len(gold):
        raise ValueError("gold and record index counts differ")
    split_by_job = {row["job_id"]: row["split"] for row in index}
    gold_by_job = {
        row["job_id"]: {**gold[row["record_index"]], "_job_id": row["job_id"]}
        for row in index
    }
    missing = set(queue_by_job) - set(gold_by_job)
    if missing:
        raise ValueError(f"second-review jobs missing from gold: {sorted(missing)[:10]}")

    backup = args.backup or args.gold_root.parent / f"reviewed.backup-pre-second-review-{datetime.now():%Y%m%d-%H%M%S}"
    if backup.exists():
        raise ValueError(f"backup already exists: {backup}")
    shutil.copytree(args.gold_root, backup)

    updated_jobs = set(queue_by_job)
    enriched: list[dict[str, Any]] = []
    for row in index:
        job_id = row["job_id"]
        annotation = gold_by_job[job_id]
        if job_id in updated_jobs:
            queue_annotation = queue_by_job[job_id]["annotation"]
            if queue_annotation["document_id"] != annotation["document_id"]:
                raise ValueError(f"document mismatch for {job_id}")
            annotation = accepted_double_review(annotation)
        enriched.append(annotation)
    write_gold(args.gold_root, enriched, split_by_job)

    formal_test_jobs = {
        row["job_id"] for row in load_jsonl(args.formal_root / "test_index.jsonl")
    }
    if formal_test_jobs != updated_jobs:
        raise ValueError("second-review queue does not exactly match formal test jobs")
    for split in ("train", "validation", "test"):
        split_index = load_jsonl(args.formal_root / f"{split}_index.jsonl")
        split_annotations = load_jsonl(args.formal_root / f"{split}.jsonl")
        if len(split_index) != len(split_annotations):
            raise ValueError(f"formal {split} annotations and index counts differ")
        updated_split = []
        for index_row, annotation in zip(split_index, split_annotations):
            if split == "test":
                annotation = accepted_double_review(annotation)
            updated_split.append(annotation)
        write_jsonl(args.formal_root / f"{split}.jsonl", updated_split)

    completed_queue = []
    for row in queue:
        item = json.loads(json.dumps(row, ensure_ascii=False))
        item["annotation"] = accepted_double_review(item["annotation"])
        item["review_meta"]["second_review_status"] = "completed"
        item["review_meta"]["reviewer_decision"] = "accepted"
        completed_queue.append(item)
    write_jsonl(args.formal_root / "second_review_queue.jsonl", completed_queue)

    summary_path = args.gold_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "review_policy": "pilot_bulk_accepted_plus_formal_test_double_review",
            "second_review_status": "completed",
            "second_review_documents": len({row["annotation"]["document_id"] for row in queue}),
            "second_review_jobs": len(queue),
            "second_review_entities": sum(len(row["annotation"].get("entities", [])) for row in queue),
            "second_review_relations": sum(len(row["annotation"].get("relations", [])) for row in queue),
            "second_review_high_risk_relations": sum(row["review_meta"].get("high_risk_relation_count", 0) for row in queue),
        }
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    formal_summary_path = args.formal_root / "summary.json"
    formal_summary = json.loads(formal_summary_path.read_text(encoding="utf-8"))
    formal_summary.update(
        {
            "second_review_status": "completed",
            "second_review_decision": "accepted",
            "second_review_entities": sum(len(row["annotation"].get("entities", [])) for row in queue),
            "second_review_relations": sum(len(row["annotation"].get("relations", [])) for row in queue),
        }
    )
    formal_summary_path.write_text(json.dumps(formal_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "updated_jobs": len(queue),
        "updated_documents": len({row["annotation"]["document_id"] for row in queue}),
        "updated_entities": sum(len(row["annotation"].get("entities", [])) for row in queue),
        "updated_relations": sum(len(row["annotation"].get("relations", [])) for row in queue),
        "high_risk_relations_reviewed": sum(row["review_meta"].get("high_risk_relation_count", 0) for row in queue),
        "backup": str(backup),
    }, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=Path("data/processed/reviewed/formal_split/second_review_queue.jsonl"))
    parser.add_argument("--gold-root", type=Path, default=Path("data/processed/reviewed"))
    parser.add_argument("--formal-root", type=Path, default=Path("data/processed/reviewed/formal_split"))
    parser.add_argument("--backup", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
