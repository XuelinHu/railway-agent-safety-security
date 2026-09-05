#!/usr/bin/env python3
"""Reuse paired window boundaries while rebinding an experiment instruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run(args: argparse.Namespace) -> int:
    windows = load_jsonl(args.windows)
    source = {row["job_id"]: row for row in load_jsonl(args.source_jobs)}
    output = []
    for window in windows:
        parent = window.get("parent_job_id", window["job_id"])
        if parent not in source:
            raise ValueError(f"source job not found for window {window['job_id']}: {parent}")
        row = {**window, "system_instruction": source[parent]["system_instruction"]}
        output.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in output:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"windows": len(output), "output": str(args.output)}, ensure_ascii=False))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--source-jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
