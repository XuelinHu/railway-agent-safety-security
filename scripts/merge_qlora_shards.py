#!/usr/bin/env python3
"""Validate and merge sharded QLoRA inference outputs in source-job order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TERMINAL_STATUSES = {"success", "failed"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise ValueError(f"required shard is missing: {path}") from error
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(row)
    return rows


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def merge(
    jobs_path: Path,
    output_path: Path,
    log_path: Path,
    workers: int,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")

    jobs = load_jsonl(jobs_path)
    job_order: list[str] = []
    for row in jobs:
        job_id = row.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise ValueError(f"{jobs_path}: every job must contain a non-empty string job_id")
        job_order.append(job_id)
    if len(job_order) != len(set(job_order)):
        raise ValueError(f"{jobs_path}: duplicate job IDs are not allowed")
    expected = set(job_order)
    shard_count = min(workers, len(job_order))

    predictions: dict[str, dict[str, Any]] = {}
    terminal_logs: dict[str, dict[str, Any]] = {}
    for worker in range(shard_count):
        part_output = Path(f"{output_path}.part{worker}")
        part_log = Path(f"{log_path}.part{worker}")
        for row in load_jsonl(part_output):
            job_id = row.get("job_id")
            if not isinstance(job_id, str) or not job_id:
                raise ValueError(f"{part_output}: every prediction needs a string job_id")
            if job_id not in expected:
                raise ValueError(f"{part_output}: unknown job_id {job_id!r}")
            if job_id in predictions:
                raise ValueError(f"duplicate prediction for job_id {job_id!r}")
            predictions[job_id] = row
        for row in load_jsonl(part_log):
            job_id = row.get("job_id")
            if not isinstance(job_id, str) or not job_id:
                raise ValueError(f"{part_log}: every log row needs a string job_id")
            if job_id not in expected:
                raise ValueError(f"{part_log}: unknown job_id {job_id!r}")
            terminal_logs[job_id] = row

    missing_logs = expected - terminal_logs.keys()
    if missing_logs:
        raise ValueError(
            f"terminal logs are missing for {len(missing_logs)} jobs; "
            f"first missing={sorted(missing_logs)[:3]}"
        )
    invalid_statuses = {
        job_id: row.get("status")
        for job_id, row in terminal_logs.items()
        if row.get("status") not in TERMINAL_STATUSES
    }
    if invalid_statuses:
        raise ValueError(f"invalid terminal statuses: {list(invalid_statuses.items())[:3]}")

    successful = {
        job_id for job_id, row in terminal_logs.items() if row["status"] == "success"
    }
    prediction_ids = set(predictions)
    missing_predictions = successful - prediction_ids
    unexpected_predictions = prediction_ids - successful
    if missing_predictions or unexpected_predictions:
        raise ValueError(
            "prediction/log mismatch: "
            f"missing successful predictions={sorted(missing_predictions)[:3]}, "
            f"predictions without final success={sorted(unexpected_predictions)[:3]}"
        )

    ordered_predictions = [predictions[job_id] for job_id in job_order if job_id in predictions]
    ordered_logs = [terminal_logs[job_id] for job_id in job_order]
    atomic_write_jsonl(output_path, ordered_predictions)
    atomic_write_jsonl(log_path, ordered_logs)
    result = {
        "jobs": len(job_order),
        "predictions": len(ordered_predictions),
        "terminal_logs": len(ordered_logs),
        "terminal_failures": len(job_order) - len(ordered_predictions),
        "workers": workers,
        "shards": shard_count,
    }
    print(json.dumps(result, ensure_ascii=False))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    try:
        merge(arguments.jobs, arguments.output, arguments.log, arguments.workers)
    except ValueError as error:
        raise SystemExit(str(error)) from error
