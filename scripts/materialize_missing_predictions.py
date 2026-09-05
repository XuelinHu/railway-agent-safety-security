#!/usr/bin/env python3
"""Create an ordered prediction file, filling terminal failures with empty rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def empty_prediction(job: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "job_id": job["job_id"],
        "annotation": {
            "schema_version": "0.1.0",
            "document_id": job["document_id"],
            "language": job.get("language", "unknown"),
            "entities": [],
            "relations": [],
            "review": {
                "status": "unreviewed",
                "reviewers": [],
                "notes": reason,
            },
        },
    }


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.resolve() == args.predictions.resolve():
        raise ValueError("--output must differ from --predictions")
    jobs = load_jsonl(args.jobs)
    expected = [str(job["job_id"]) for job in jobs]
    if len(expected) != len(set(expected)):
        raise ValueError("jobs contain duplicate job_id values")

    predictions: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(args.predictions):
        job_id = str(row.get("job_id", ""))
        if not job_id:
            raise ValueError("prediction row is missing job_id")
        if job_id in predictions:
            raise ValueError(f"duplicate prediction job_id {job_id!r}")
        if not isinstance(row.get("annotation", row), dict):
            raise ValueError(f"prediction {job_id!r} has no annotation object")
        predictions[job_id] = row
    unknown = sorted(set(predictions) - set(expected))
    if unknown:
        raise ValueError(f"predictions contain unknown jobs: {unknown[:3]}")

    job_by_id = {str(job["job_id"]): job for job in jobs}
    missing = [job_id for job_id in expected if job_id not in predictions]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for job_id in expected:
            row = predictions.get(job_id) or empty_prediction(
                job_by_id[job_id], args.reason
            )
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(args.output)

    summary = {
        "status": "complete",
        "jobs": len(expected),
        "successful_prediction_rows": len(predictions),
        "failures_materialized_as_empty": len(missing),
        "missing_job_ids": missing,
        "jobs_path": str(args.jobs),
        "source_predictions": str(args.predictions),
        "output": str(args.output),
        "gold_read": False,
    }
    if args.summary:
        write_json_atomic(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument(
        "--reason",
        default="empty validation prediction materialized from terminal inference failure",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
