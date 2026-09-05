#!/usr/bin/env python3
"""Materialize v2 low-resource assets from frozen semantic train/validation windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import build_low_resource_assets as v1


def run(args: argparse.Namespace) -> int:
    for label, path in (
        ("validation gold", args.validation_gold),
        ("validation index", args.validation_index),
        ("train window summary", args.train_window_summary),
        ("validation window summary", args.validation_window_summary),
    ):
        v1.require_no_test_path(path, label)
    base_args = argparse.Namespace(
        document_manifest=args.document_manifest,
        baseline_jobs=args.baseline_jobs,
        gold=args.gold,
        validation_baseline_jobs=args.validation_baseline_jobs,
        index=args.index,
        concepts=args.concepts,
        mentions=args.mentions,
        relations=args.relations,
        ontology=args.ontology,
        output=args.output,
        v1_concept_limit=args.v1_concept_limit,
        v1_balanced_concepts=args.v1_balanced_concepts,
    )
    v1.run(base_args)

    validation_gold = v1.load_jsonl(args.validation_gold)
    validation_index = v1.load_jsonl(args.validation_index)
    validation_jobs = v1.load_jsonl(args.validation_baseline_jobs)
    if len(validation_gold) != len(validation_index):
        raise ValueError("validation gold/index counts differ")
    if {str(row["job_id"]) for row in validation_index} != {
        str(row["job_id"]) for row in validation_jobs
    }:
        raise ValueError("validation gold/index/job IDs differ")
    if any(row.get("split") != "validation" for row in validation_index):
        raise ValueError("validation index contains a non-validation row")

    validation_root = args.output / "validation"
    validation_gold_path = validation_root / "gold.jsonl"
    validation_index_path = validation_root / "index.jsonl"
    v1.write_jsonl(validation_gold_path, validation_gold)
    v1.write_jsonl(validation_index_path, validation_index)

    manifest_path = args.output / "asset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol_id"] = "low-resource-assets-v2"
    manifest["semantic_window_sources"] = {
        "train": {
            "path": str(args.train_window_summary),
            "sha256": v1.sha256_file(args.train_window_summary),
        },
        "validation": {
            "path": str(args.validation_window_summary),
            "sha256": v1.sha256_file(args.validation_window_summary),
        },
    }
    manifest["validation_gold"] = {
        "path": str(validation_gold_path),
        "sha256": v1.sha256_file(validation_gold_path),
    }
    manifest["validation_index"] = {
        "path": str(validation_index_path),
        "sha256": v1.sha256_file(validation_index_path),
    }
    output_paths = [
        path
        for path in sorted(args.output.rglob("*"))
        if path.is_file() and path != manifest_path
    ]
    manifest["outputs"] = {
        str(path.relative_to(args.output)): v1.sha256_file(path)
        for path in output_paths
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "protocol_id": manifest["protocol_id"],
                "output": str(args.output),
                "train_windows": manifest["window_jobs"],
                "validation_windows": len(validation_jobs),
                "formal_test_read": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-manifest", type=Path, required=True)
    parser.add_argument(
        "--baseline-jobs",
        type=Path,
        default=Path(
            "data/processed/experiments/formal/windowed_train_v6/baseline_jobs.jsonl"
        ),
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("data/processed/experiments/formal/windowed_train_v6/gold.jsonl"),
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("data/processed/experiments/formal/windowed_train_v6/index.jsonl"),
    )
    parser.add_argument(
        "--validation-baseline-jobs",
        type=Path,
        default=Path(
            "data/processed/experiments/formal/windowed_validation_v6/baseline_jobs.jsonl"
        ),
    )
    parser.add_argument(
        "--validation-gold",
        type=Path,
        default=Path(
            "data/processed/experiments/formal/windowed_validation_v6/gold.jsonl"
        ),
    )
    parser.add_argument(
        "--validation-index",
        type=Path,
        default=Path(
            "data/processed/experiments/formal/windowed_validation_v6/index.jsonl"
        ),
    )
    parser.add_argument(
        "--train-window-summary",
        type=Path,
        default=Path("data/processed/experiments/formal/windowed_train_v6/summary.json"),
    )
    parser.add_argument(
        "--validation-window-summary",
        type=Path,
        default=Path(
            "data/processed/experiments/formal/windowed_validation_v6/summary.json"
        ),
    )
    parser.add_argument(
        "--concepts",
        type=Path,
        default=Path(
            "data/processed/experiments/formal/knowledge_graph/concepts.jsonl"
        ),
    )
    parser.add_argument(
        "--mentions",
        type=Path,
        default=Path(
            "data/processed/experiments/formal/knowledge_graph/mentions.jsonl"
        ),
    )
    parser.add_argument(
        "--relations",
        type=Path,
        default=Path(
            "data/processed/experiments/formal/knowledge_graph/relations.jsonl"
        ),
    )
    parser.add_argument(
        "--ontology", type=Path, default=Path("configs/risk_ontology.yaml")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--v1-concept-limit", type=int, default=20)
    parser.add_argument(
        "--v1-balanced-concepts",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
