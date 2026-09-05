#!/usr/bin/env python3
"""Rebase the frozen v1 selection matrix onto independent v2 asset/run paths."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rebase(value: Any) -> Any:
    if isinstance(value, str):
        return (
            value.replace("lr_v1_", "lr_v2_")
            .replace("low_resource_v1", "low_resource_v2")
            .replace(
                "configs/low_resource_protocol_v1.yaml",
                "configs/low_resource_protocol_v2.yaml",
            )
            .replace(
                "scripts/build_low_resource_assets.py",
                "scripts/build_low_resource_v2_assets.py",
            )
        )
    if isinstance(value, list):
        return [rebase(item) for item in value]
    if isinstance(value, dict):
        return {key: rebase(item) for key, item in value.items()}
    return value


def run(args: argparse.Namespace) -> int:
    training = [rebase(row) for row in load_jsonl(args.source_matrix)]
    derived = [rebase(row) for row in load_jsonl(args.source_derived_matrix)]
    if len(training) != 36 or len(derived) != 12:
        raise ValueError("source matrices are not the frozen 36+12 design")
    if len({row["run_id"] for row in training}) != len(training):
        raise ValueError("v2 run IDs are not unique")
    if any("low_resource_v1" in row["output_directory"] for row in training):
        raise ValueError("v2 matrix retained a v1 output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    run_path = args.output / "run_matrix.jsonl"
    derived_path = args.output / "derived_matrix.jsonl"
    write_jsonl(run_path, training)
    write_jsonl(derived_path, derived)
    summary = {
        "protocol_id": "low-resource-manifests-v2",
        "amended_from": "low-resource-manifests-v1",
        "formal_test_read": False,
        "selection_changed": False,
        "training_runs": len(training),
        "derived_groups": len(derived),
        "source_matrix": {
            "path": str(args.source_matrix),
            "sha256": sha256_file(args.source_matrix),
        },
        "source_derived_matrix": {
            "path": str(args.source_derived_matrix),
            "sha256": sha256_file(args.source_derived_matrix),
        },
        "run_matrix": {"path": str(run_path), "sha256": sha256_file(run_path)},
        "derived_matrix": {
            "path": str(derived_path),
            "sha256": sha256_file(derived_path),
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-matrix",
        type=Path,
        default=Path(
            "data/processed/experiments/formal/low_resource_manifests_v1/run_matrix.jsonl"
        ),
    )
    parser.add_argument(
        "--source-derived-matrix",
        type=Path,
        default=Path(
            "data/processed/experiments/formal/low_resource_manifests_v1/derived_matrix.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/experiments/formal/low_resource_manifests_v2"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
