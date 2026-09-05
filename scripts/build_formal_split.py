#!/usr/bin/env python3
"""Build a cluster-isolated 100/20/30 split and a second-review queue."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_rank(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_clusters(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {row["document_id"]: row["cluster_id"] for row in csv.DictReader(stream)}


def allocate_split(groups: dict[str, list[dict[str, Any]]], targets: dict[str, int]) -> dict[str, str]:
    """Greedily place whole clusters while balancing source, language, and category."""
    total = sum(targets.values())
    dimensions = ("source_group", "language", "category")
    global_counts = {dimension: Counter(row[dimension] for rows in groups.values() for row in rows) for dimension in dimensions}
    desired = {
        split: {
            dimension: {
                value: targets[split] * count / total
                for value, count in global_counts[dimension].items()
            }
            for dimension in dimensions
        }
        for split in targets
    }
    assigned_counts = {split: Counter() for split in targets}
    assignment: dict[str, str] = {}
    ordered = sorted(groups.items(), key=lambda item: (-len(item[1]), stable_rank(item[0])))
    for cluster_id, rows in ordered:
        size = len(rows)
        assigned_documents = lambda split: sum(
            len(groups[cluster]) for cluster, value in assignment.items() if value == split
        )
        options = [split for split in targets if assigned_documents(split) + size <= targets[split]]
        if not options:
            raise ValueError(f"cannot fit cluster {cluster_id} into requested split sizes")

        def score(split: str) -> tuple[float, float, str]:
            total_deficit = targets[split] - assigned_documents(split)
            dimension_deficit = 0.0
            for dimension in dimensions:
                for value in set(row[dimension] for row in rows):
                    current = assigned_counts[split][f"{dimension}:{value}"]
                    target = desired[split][dimension].get(value, 0.0)
                    dimension_deficit += max(target - current, 0.0)
            return (dimension_deficit, float(total_deficit), stable_rank(f"{cluster_id}|{split}"))

        selected = max(options, key=score)
        assignment[cluster_id] = selected
        for row in rows:
            for dimension in dimensions:
                assigned_counts[selected][f"{dimension}:{row[dimension]}"] += 1
    return assignment


def relation_risk(audit_rows: list[dict[str, Any]], relation: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    audit = next((row for row in audit_rows if row.get("relation_id") == relation.get("id")), None)
    if audit and not audit.get("accepted"):
        reasons.extend(audit.get("reasons", []))
    if relation.get("claim_status") in {"inferred", "uncertain"}:
        reasons.append(f"claim_status:{relation['claim_status']}")
    if relation.get("confidence", 1) < 0.8:
        reasons.append("low_confidence")
    return list(dict.fromkeys(reasons))


def prepare_second_review(annotation: dict[str, Any]) -> dict[str, Any]:
    """Reset item-level decisions while keeping the first-review provenance."""
    second = json.loads(json.dumps(annotation, ensure_ascii=False))
    second["review"] = {
        "status": "unreviewed",
        "reviewers": [],
        "notes": "Formal test split second review; first-review decisions remain in the source gold annotation.",
    }
    for entity in second.get("entities", []):
        entity["review_status"] = "pending"
    for relation in second.get("relations", []):
        relation["review_status"] = "pending"
    return second


def run(args: argparse.Namespace) -> int:
    annotations = load_jsonl(args.annotations)
    index = load_jsonl(args.index)
    jobs = load_jsonl(args.jobs)
    clusters = load_clusters(args.clusters)
    audit_rows = load_jsonl(args.relation_audit) if args.relation_audit.exists() else []
    jobs_by_id = {row["job_id"]: row for row in jobs}
    index_by_job = {row["job_id"]: row for row in index}
    annotation_by_job = {
        row["job_id"]: annotations[row["record_index"]]
        for row in index
    }
    if set(index_by_job) != set(annotation_by_job):
        raise ValueError("annotation and index job IDs do not match")

    document_rows: dict[str, dict[str, Any]] = {}
    for job_id, row in index_by_job.items():
        document_id = row["document_id"]
        job = jobs_by_id[job_id]
        document_rows.setdefault(
            document_id,
            {
                "document_id": document_id,
                "source_group": job["source_path"].split("/", 1)[0],
                "language": job["language"],
                "category": job["category"],
                "cluster_id": clusters.get(document_id, f"singleton_{document_id}"),
            },
        )
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in document_rows.values():
        groups[row["cluster_id"]].append(row)
    cluster_docs = {cluster_id: {row["document_id"] for row in rows} for cluster_id, rows in groups.items()}
    if any(len(documents & set(document_rows)) != len(documents) for documents in cluster_docs.values()):
        raise ValueError("cluster contains documents outside the reviewed manifest")
    targets = {"train": args.train, "validation": args.validation, "test": args.test}
    if sum(targets.values()) != len(document_rows):
        raise ValueError(f"split targets {targets} do not sum to {len(document_rows)} documents")
    cluster_splits = allocate_split(groups, targets)
    split_by_document = {
        document_id: cluster_splits[cluster_id]
        for cluster_id, documents in cluster_docs.items()
        for document_id in documents
    }
    if len(set(split_by_document)) != len(document_rows):
        raise ValueError("document split assignment is incomplete")
    cross_split_clusters = sum(
        len({split_by_document[document_id] for document_id in documents}) > 1
        for documents in cluster_docs.values()
    )

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest = [{**row, "split": split_by_document[row["document_id"]]} for row in sorted(document_rows.values(), key=lambda row: row["document_id"])]
    write_jsonl(output / "split_manifest.jsonl", manifest)
    split_items: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job_id, row in index_by_job.items():
        annotation = annotation_by_job[job_id]
        item = json.loads(json.dumps(annotation, ensure_ascii=False))
        item["_job_id"] = job_id
        split_items[split_by_document[row["document_id"]]].append(item)
    for split, items in split_items.items():
        write_jsonl(output / f"{split}.jsonl", [{k: v for k, v in item.items() if not k.startswith("_")} for item in items])
        write_jsonl(
            output / f"{split}_index.jsonl",
            [
                {"record_index": position, "job_id": item["_job_id"], "document_id": item["document_id"], "split": split}
                for position, item in enumerate(items)
            ],
        )

    test_documents = {row["document_id"] for row in manifest if row["split"] == "test"}
    audit_by_job: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in audit_rows:
        audit_by_job[row.get("job_id", "")].append(row)
    second_review: list[dict[str, Any]] = []
    for item in sorted(split_items["test"], key=lambda row: row["_job_id"]):
        annotation = {k: v for k, v in item.items() if not k.startswith("_")}
        second_annotation = prepare_second_review(annotation)
        risky = []
        for relation in second_annotation["relations"]:
            reasons = relation_risk(audit_by_job[item["_job_id"]], relation)
            if reasons:
                risky.append({"relation_id": relation["id"], "reasons": reasons})
        second_review.append(
            {
                "job_id": item["_job_id"],
                "annotation": second_annotation,
                "review_meta": {
                    "formal_split": "test",
                    "cluster_id": clusters.get(annotation["document_id"], f"singleton_{annotation['document_id']}"),
                    "second_review_status": "pending_second_review",
                    "first_review_status": annotation.get("review", {}).get("status", "unknown"),
                    "high_risk_relation_count": len(risky),
                    "high_risk_relations": risky,
                },
            }
        )
    write_jsonl(output / "second_review_queue.jsonl", second_review)
    write_jsonl(output / "jobs.jsonl", [jobs_by_id[item["_job_id"]] for item in sorted(sum(split_items.values(), []), key=lambda row: row["_job_id"])])
    summary = {
        "documents": len(document_rows),
        "records": len(annotations),
        "split_documents": dict(Counter(split_by_document.values())),
        "split_records": {split: len(items) for split, items in split_items.items()},
        "cluster_count": len(groups),
        "cross_split_clusters": cross_split_clusters,
        "second_review_documents": len(test_documents),
        "second_review_records": len(second_review),
        "high_risk_relations_in_second_review": sum(row["review_meta"]["high_risk_relation_count"] for row in second_review),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path("data/processed/reviewed/gold/all.jsonl"))
    parser.add_argument("--index", type=Path, default=Path("data/processed/reviewed/gold/record_index.jsonl"))
    parser.add_argument("--jobs", type=Path, default=Path("data/processed/reviewed/jobs.jsonl"))
    parser.add_argument("--clusters", type=Path, default=Path("data/catalog/near_duplicate_clusters.csv"))
    parser.add_argument("--relation-audit", type=Path, default=Path("data/processed/experiments/annotation_pending_terra_all_relation_verification.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/reviewed/formal_split"))
    parser.add_argument("--train", type=int, default=100)
    parser.add_argument("--validation", type=int, default=20)
    parser.add_argument("--test", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
