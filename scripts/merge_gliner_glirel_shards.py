#!/usr/bin/env python3
"""Merge disjoint GLiNER + GLiREL shards in public split job order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def merge(
    jobs_path: Path,
    shards: list[Path],
    output: Path,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    if jobs_path.name not in {"validation_baseline_jobs.jsonl", "test_baseline_jobs.jsonl"}:
        raise ValueError("--jobs must point to validation_baseline_jobs.jsonl or test_baseline_jobs.jsonl")
    split = jobs_path.name.removesuffix("_baseline_jobs.jsonl")
    if not shards:
        raise ValueError("at least one shard is required")
    resolved_output = output.resolve()
    if resolved_output in {path.resolve() for path in shards}:
        raise ValueError("merged output must differ from every shard path")

    jobs = load_jsonl(jobs_path)
    order = [row["job_id"] for row in jobs]
    if len(order) != len(set(order)):
        raise ValueError(f"{split} jobs contain duplicate job IDs")
    job_by_id = {row["job_id"]: row for row in jobs}
    predictions: dict[str, dict[str, Any]] = {}
    source_by_job: dict[str, str] = {}
    for shard in shards:
        for row in load_jsonl(shard):
            if not isinstance(row, dict) or not isinstance(row.get("job_id"), str):
                raise ValueError(f"{shard}: every row must contain a string job_id")
            job_id = row["job_id"]
            if job_id not in job_by_id:
                raise ValueError(f"{shard}: job {job_id!r} is outside the {split} input")
            if job_id in predictions:
                raise ValueError(
                    f"duplicate prediction {job_id!r} in {source_by_job[job_id]} and {shard}"
                )
            annotation = row.get("annotation")
            if not isinstance(annotation, dict):
                raise ValueError(f"{shard}: job {job_id!r} has no wrapped annotation")
            if annotation.get("document_id") != job_by_id[job_id].get("document_id"):
                raise ValueError(f"{shard}: job {job_id!r} has the wrong document_id")
            predictions[job_id] = row
            source_by_job[job_id] = str(shard)

    missing = [job_id for job_id in order if job_id not in predictions]
    if missing and not allow_incomplete:
        raise ValueError(
            f"refusing incomplete merge: {len(missing)} {split} jobs are missing; "
            f"first missing={missing[:3]}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for job_id in order:
            if job_id in predictions:
                stream.write(json.dumps(predictions[job_id], ensure_ascii=False) + "\n")
    temporary.replace(output)
    result = {
        "jobs_expected": len(order),
        "predictions": len(predictions),
        "missing": len(missing),
        "complete": not missing,
        "shards": [str(path) for path in shards],
        "output": str(output),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    raise SystemExit(
        0
        if merge(
            arguments.jobs,
            arguments.shards,
            arguments.output,
            arguments.allow_incomplete,
        )
        else 1
    )
