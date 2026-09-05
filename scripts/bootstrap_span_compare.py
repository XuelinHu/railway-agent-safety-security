#!/usr/bin/env python3
"""Document-level paired inference for two span-aware validation metrics."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from bootstrap_compare import add_counts, compare_field


FIELDS = ("entity", "relation", "relation_with_claim_status")


def load_metrics(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("metric") not in {
        "strict-global-character-span-one-to-one",
        "strict-global-character-span-document-deduplicated",
    }:
        raise ValueError(f"not a span-aware metric artifact: {path}")
    if result.get("selection_split") != "validation" or result.get("formal_test_read"):
        raise ValueError(f"refusing non-validation metric artifact: {path}")
    return result


def document_units(
    left: dict[str, Any], right: dict[str, Any]
) -> list[dict[str, Any]]:
    left_jobs = left.get("per_document", left.get("per_job", {}))
    right_jobs = right.get("per_document", right.get("per_job", {}))
    if set(left_jobs) != set(right_jobs):
        raise ValueError("span metric evaluation-unit IDs differ")
    units: dict[str, dict[str, Any]] = {}
    for unit_id in sorted(left_jobs):
        left_item = left_jobs[unit_id]
        right_item = right_jobs[unit_id]
        if left_item.get("document_id") != right_item.get("document_id"):
            raise ValueError(f"document mismatch for {unit_id}")
        document_id = str(left_item.get("document_id"))
        unit = units.setdefault(
            document_id,
            {
                "document_id": document_id,
                "left": {
                    field: {"gold": 0, "predicted": 0, "correct": 0}
                    for field in FIELDS
                },
                "right": {
                    field: {"gold": 0, "predicted": 0, "correct": 0}
                    for field in FIELDS
                },
            },
        )
        for system, item in (("left", left_item), ("right", right_item)):
            for field in FIELDS:
                add_counts(unit[system][field], item[field])
    return [units[key] for key in sorted(units)]


def run(args: argparse.Namespace) -> int:
    left = load_metrics(args.left)
    right = load_metrics(args.right)
    units = document_units(left, right)
    rng = random.Random(args.seed)
    result = {
        "metric": "strict-global-character-span-one-to-one",
        "selection_split": "validation",
        "formal_test_read": False,
        "unit": "document",
        "documents": len(units),
        "iterations": args.iterations,
        "seed": args.seed,
        "systems": {"left": args.left.name, "right": args.right.name},
        "interpretation": "right_minus_left",
        "fields": {
            field: compare_field(
                [unit["left"] for unit in units],
                [unit["right"] for unit in units],
                field,
                args.iterations,
                rng,
            )
            for field in FIELDS
        },
    }
    # compare_field retains its historical baseline/kg labels; expose aliases
    # so consumers do not mistake a system name for an experimental role.
    for comparison in result["fields"].values():
        comparison["left"] = comparison.pop("baseline")
        comparison["right"] = comparison.pop("kg")
        comparison["right_minus_left"] = comparison.pop("kg_minus_baseline")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260830)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
