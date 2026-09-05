#!/usr/bin/env python3
"""Build a review queue by attaching normalization and relation-audit metadata."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run(args: argparse.Namespace) -> int:
    annotations = load_jsonl(args.annotations)
    normalize_errors = load_jsonl(args.normalize_errors) if args.normalize_errors.exists() else []
    relation_audit = load_jsonl(args.relation_audit) if args.relation_audit.exists() else []
    errors_by_job: defaultdict[str | None, list[str]] = defaultdict(list)
    for item in normalize_errors:
        errors_by_job[item.get("job_id")].extend(item.get("candidate_errors", []))
    audit_by_job: defaultdict[str | None, list[dict[str, Any]]] = defaultdict(list)
    for item in relation_audit:
        audit_by_job[item.get("job_id")].append(item)

    rows: list[dict[str, Any]] = []
    for row in annotations:
        annotation = row.get("annotation", row)
        job_id = row.get("job_id")
        errors = errors_by_job.get(job_id, [])
        decisions = audit_by_job.get(job_id, [])
        rejected = [item for item in decisions if not item.get("accepted")]
        priority = "high" if errors or rejected else "normal"
        rows.append(
            {
                "job_id": job_id,
                "annotation": annotation,
                "review_meta": {
                    "priority": priority,
                    "normalization_error_count": len(errors),
                    "normalization_errors": errors,
                    "relation_input_count": len(decisions),
                    "relation_rejected_count": len(rejected),
                    "relation_rejections": [
                        {
                            "relation_id": item.get("relation_id"),
                            "reasons": item.get("reasons", []),
                        }
                        for item in rejected
                    ],
                    "review_status": "pending_manual_review",
                },
            }
        )

    rows.sort(key=lambda row: (row["review_meta"]["priority"] != "high", row.get("job_id") or ""))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "records": len(rows),
        "high_priority": sum(row["review_meta"]["priority"] == "high" for row in rows),
        "normal_priority": sum(row["review_meta"]["priority"] == "normal" for row in rows),
        "output": str(args.output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--normalize-errors", type=Path, required=True)
    parser.add_argument("--relation-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
