#!/usr/bin/env python3
"""Materialize train-only assets for one frozen low-resource document manifest.

This script never opens formal validation/test data.  It filters the canonical
training windows and training KG to the selected documents, rebuilds concept
support counts, and regenerates V1 prompts with leave-one-document-out hints.
V2 prompts are then built from the emitted subset KG with build_kg_v2_jobs.py.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml

from build_experiment_jobs import constraint_context, retrieve_concept_context
from build_low_resource_manifests import require_train_only_path, sha256_file


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def require_no_test_path(path: Path, label: str) -> None:
    if any(part.casefold() == "test" or "test_" in part.casefold() or "_test" in part.casefold() for part in path.parts):
        raise ValueError(f"{label} must not reference formal test: {path}")


def subset_concepts(
    concepts: list[dict[str, Any]], mentions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    source = {str(row["concept_id"]): row for row in concepts}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for mention in mentions:
        grouped[str(mention["concept_id"])].append(mention)
    output = []
    for concept_id, rows in sorted(grouped.items()):
        if concept_id not in source:
            raise ValueError(f"mention references missing concept: {concept_id}")
        original = source[concept_id]
        output.append(
            {
                **original,
                "mention_count": len(rows),
                "source_documents": sorted(
                    {str(row["document_id"]) for row in rows}
                ),
            }
        )
    return output


def v1_concepts(mentions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for mention in mentions:
        key = (
            str(mention.get("language", "unknown")),
            str(mention.get("type", "UNKNOWN")),
            str(
                mention.get("canonical_name")
                or mention.get("normalized_name")
                or mention.get("text", "")
            ),
        )
        entry = grouped.setdefault(
            key,
            {
                "language": key[0],
                "type": key[1],
                "name": key[2],
                "count": 0,
                "source_documents": set(),
            },
        )
        entry["count"] += 1
        entry["source_documents"].add(str(mention.get("document_id", "")))
    return list(grouped.values())


def build_v1_jobs(
    baseline_jobs: list[dict[str, Any]],
    mentions: list[dict[str, Any]],
    ontology: dict[str, Any],
    concept_limit: int,
    balanced: bool,
) -> list[dict[str, Any]]:
    concepts = v1_concepts(mentions)
    output = []
    for job in baseline_jobs:
        context = retrieve_concept_context(job, concepts, concept_limit, balanced)
        base_instruction = str(job.get("system_instruction", "")).split(
            "\n\nKG_RULES:", 1
        )[0]
        output.append(
            {
                **job,
                "system_instruction": base_instruction
                + constraint_context(ontology, context),
                "experiment_mode": "kg_constrained",
                "low_resource_provenance": {
                    "train_subset_only": True,
                    "leave_current_document_out": True,
                    "concept_limit": concept_limit,
                    "balanced_concepts": balanced,
                },
            }
        )
    return output


def run(args: argparse.Namespace) -> int:
    for label, path in (
        ("document manifest", args.document_manifest),
        ("training windows", args.baseline_jobs),
        ("training gold", args.gold),
        ("training index", args.index),
        ("training KG concepts", args.concepts),
        ("training KG mentions", args.mentions),
        ("training KG relations", args.relations),
    ):
        require_train_only_path(path, label)
    require_no_test_path(args.validation_baseline_jobs, "validation jobs")

    manifest = load_jsonl(args.document_manifest)
    selected_documents = {str(row["document_id"]) for row in manifest}
    if not selected_documents:
        raise ValueError("document manifest is empty")
    if any(row.get("split") != "train" for row in manifest):
        raise ValueError("document manifest contains a non-train row")

    all_jobs = load_jsonl(args.baseline_jobs)
    validation_jobs = load_jsonl(args.validation_baseline_jobs)
    all_gold = load_jsonl(args.gold)
    all_index = load_jsonl(args.index)
    if len(all_gold) != len(all_index):
        raise ValueError("canonical training gold/index counts differ")
    selected_jobs = [
        row for row in all_jobs if str(row.get("document_id")) in selected_documents
    ]
    selected_job_ids = {str(row["job_id"]) for row in selected_jobs}
    validation_documents = {
        str(row.get("document_id")) for row in validation_jobs
    }
    if selected_documents & validation_documents:
        raise ValueError("selected training documents overlap validation jobs")
    selected_gold = []
    selected_index = []
    for row in all_index:
        if str(row.get("job_id")) not in selected_job_ids:
            continue
        original_position = int(row["record_index"])
        selected_gold.append(all_gold[original_position])
        selected_index.append(
            {
                **row,
                "record_index": len(selected_index),
                "source_record_index": original_position,
                "split": "train",
            }
        )
    if {str(row["job_id"]) for row in selected_index} != selected_job_ids:
        raise ValueError("selected training jobs are not aligned with gold/index")

    mentions = [
        row
        for row in load_jsonl(args.mentions)
        if str(row.get("document_id")) in selected_documents
    ]
    if any(row.get("split") != "train" for row in mentions):
        raise ValueError("subset KG mentions contain non-train provenance")
    concepts = subset_concepts(load_jsonl(args.concepts), mentions)
    concept_ids = {str(row["concept_id"]) for row in concepts}
    relations = [
        row
        for row in load_jsonl(args.relations)
        if str(row.get("document_id")) in selected_documents
    ]
    if any(row.get("split") != "train" for row in relations):
        raise ValueError("subset KG relations contain non-train provenance")
    if any(
        str(row.get("source_concept_id")) not in concept_ids
        or str(row.get("target_concept_id")) not in concept_ids
        for row in relations
    ):
        raise ValueError("subset relation endpoint is absent from subset concepts")

    ontology = yaml.safe_load(args.ontology.read_text(encoding="utf-8"))
    v1_jobs = build_v1_jobs(
        selected_jobs,
        mentions,
        ontology,
        args.v1_concept_limit,
        args.v1_balanced_concepts,
    )
    validation_v1_jobs = build_v1_jobs(
        validation_jobs,
        mentions,
        ontology,
        args.v1_concept_limit,
        args.v1_balanced_concepts,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    graph_root = args.output / "knowledge_graph"
    validation_root = args.output / "validation"
    write_jsonl(args.output / "baseline_jobs.jsonl", selected_jobs)
    write_jsonl(args.output / "gold.jsonl", selected_gold)
    write_jsonl(args.output / "index.jsonl", selected_index)
    write_jsonl(args.output / "kg_v1_jobs.jsonl", v1_jobs)
    write_jsonl(validation_root / "baseline_jobs.jsonl", validation_jobs)
    write_jsonl(validation_root / "kg_v1_jobs.jsonl", validation_v1_jobs)
    write_jsonl(graph_root / "concepts.jsonl", concepts)
    write_jsonl(graph_root / "mentions.jsonl", mentions)
    write_jsonl(graph_root / "relations.jsonl", relations)

    selected_with_windows = {str(row["document_id"]) for row in selected_jobs}
    output_files = [
        args.output / "baseline_jobs.jsonl",
        args.output / "gold.jsonl",
        args.output / "index.jsonl",
        args.output / "kg_v1_jobs.jsonl",
        graph_root / "concepts.jsonl",
        graph_root / "mentions.jsonl",
        graph_root / "relations.jsonl",
        validation_root / "baseline_jobs.jsonl",
        validation_root / "kg_v1_jobs.jsonl",
    ]
    output_files.extend(
        path
        for path in (
            args.output / "kg_v2_jobs.jsonl",
            args.output / "kg_v2_audit.json",
            validation_root / "kg_v2_jobs.jsonl",
            validation_root / "kg_v2_audit.json",
        )
        if path.is_file()
    )
    summary = {
        "protocol_id": "low-resource-assets-v1",
        "formal_test_read": False,
        "validation_source_read": True,
        "validation_used_for_selection": False,
        "document_manifest": {
            "path": str(args.document_manifest),
            "sha256": sha256_file(args.document_manifest),
        },
        "selected_documents": len(selected_documents),
        "effective_documents_with_windows": len(selected_with_windows),
        "documents_without_windows": sorted(
            selected_documents - selected_with_windows
        ),
        "window_jobs": len(selected_jobs),
        "gold_windows": len(selected_gold),
        "validation_jobs": len(validation_jobs),
        "validation_documents": len(validation_documents),
        "subset_graph": {
            "concepts": len(concepts),
            "mentions": len(mentions),
            "relations": len(relations),
            "source_documents": len(
                {str(row["document_id"]) for row in mentions}
            ),
        },
        "v1": {
            "concept_limit": args.v1_concept_limit,
            "balanced_concepts": args.v1_balanced_concepts,
            "leave_current_document_out": True,
        },
        "v2_next_commands": {
            split: (
                "python scripts/build_kg_v2_jobs.py "
                f"--jobs {jobs_path} "
                f"--concepts {graph_root / 'concepts.jsonl'} "
                f"--relations {graph_root / 'relations.jsonl'} "
                f"--output {output_path} "
                f"--audit {audit_path} "
                "--semantic-model /ds2/xuelin/cache/huggingface/hub/"
                "models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181 "
                "--device cuda --batch-size 16 --semantic-threshold 0.72 "
                "--semantic-limit 4 --anchor-limit 12 --anchor-per-type 2 "
                "--edge-limit 6 --min-type-purity 0.8 --min-en-chars 4 --min-zh-chars 2"
            )
            for split, jobs_path, output_path, audit_path in (
                (
                    "train",
                    args.output / "baseline_jobs.jsonl",
                    args.output / "kg_v2_jobs.jsonl",
                    args.output / "kg_v2_audit.json",
                ),
                (
                    "validation",
                    validation_root / "baseline_jobs.jsonl",
                    validation_root / "kg_v2_jobs.jsonl",
                    validation_root / "kg_v2_audit.json",
                ),
            )
        },
        "outputs": {
            str(path.relative_to(args.output)): sha256_file(path)
            for path in output_files
        },
    }
    (args.output / "asset_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-manifest", type=Path, required=True)
    parser.add_argument(
        "--baseline-jobs",
        type=Path,
        default=Path(
            "data/processed/experiments/formal/windowed_train_v2/baseline_jobs.jsonl"
        ),
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("data/processed/experiments/formal/windowed_train_v2/gold.jsonl"),
    )
    parser.add_argument(
        "--validation-baseline-jobs",
        type=Path,
        default=Path(
            "data/processed/experiments/formal/windowed_validation_v2/baseline_jobs.jsonl"
        ),
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("data/processed/experiments/formal/windowed_train_v2/index.jsonl"),
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
        "--v1-balanced-concepts", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
