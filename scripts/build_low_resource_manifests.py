#!/usr/bin/env python3
"""Build deterministic, nested, train-only low-resource document manifests.

The selector reads the formal *training* annotations/index and a train-only
window-job inventory.  The corpus duplicate-cluster catalog is used only to
keep selected training documents together; validation/test split artifacts are
neither inputs nor discovered by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_BUDGETS = (10, 25, 50, 100)
DEFAULT_SEEDS = (20260830, 20260831, 20260901)
TRAINABLE_SYSTEMS = ("baseline", "kg_v1", "kg_v2")
DERIVED_SYSTEMS = ("kg_v2_verified", "kg_v3_raw", "kg_v3_final")
FORBIDDEN_SPLIT_TOKEN = re.compile(r"(^|[_.-])(validation|test)([_.-]|$)")


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
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rank(value: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}|{value}".encode("utf-8")).hexdigest()


def require_train_only_path(path: Path, label: str) -> None:
    """Reject accidental formal validation/test inputs before opening them."""
    if any(FORBIDDEN_SPLIT_TOKEN.search(part.casefold()) for part in path.parts):
        raise ValueError(f"{label} must be train-only; refusing path: {path}")


def load_cluster_metadata(
    path: Path, train_documents: set[str]
) -> tuple[dict[str, dict[str, str]], int]:
    selected: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            document_id = str(row.get("document_id", ""))
            if document_id in train_documents:
                selected[document_id] = {
                    "cluster_id": str(row.get("cluster_id") or f"singleton_{document_id}"),
                    "source_group": str(row.get("source_group") or "unknown"),
                    "category": str(row.get("category") or "unknown"),
                }
    return selected, len(train_documents - set(selected))


def build_inventory(
    annotations: list[dict[str, Any]],
    index: list[dict[str, Any]],
    train_jobs: list[dict[str, Any]],
    cluster_metadata: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    if len(annotations) != len(index):
        raise ValueError("train annotations and train index counts differ")
    if any(row.get("split") != "train" for row in index):
        raise ValueError("train index contains a non-train split row")
    if any(
        annotation.get("document_id") != row.get("document_id")
        for annotation, row in zip(annotations, index)
    ):
        raise ValueError("train annotation/index document alignment failed")

    train_documents = {str(row["document_id"]) for row in index}
    job_documents = {str(row.get("document_id")) for row in train_jobs}
    if job_documents - train_documents:
        raise ValueError("train-job inventory contains documents outside formal train")

    by_document: dict[str, dict[str, Any]] = {
        document_id: {
            "document_id": document_id,
            "languages": set(),
            "record_count": 0,
            "window_job_count": 0,
            "entity_count": 0,
            "relation_count": 0,
            "entity_types": set(),
            "relation_types": set(),
        }
        for document_id in train_documents
    }
    for annotation in annotations:
        item = by_document[str(annotation["document_id"])]
        item["languages"].add(str(annotation.get("language", "unknown")))
        item["record_count"] += 1
        item["entity_count"] += len(annotation.get("entities", []))
        item["relation_count"] += len(annotation.get("relations", []))
        item["entity_types"].update(
            str(entity.get("type", "unknown"))
            for entity in annotation.get("entities", [])
        )
        item["relation_types"].update(
            str(relation.get("type", "unknown"))
            for relation in annotation.get("relations", [])
        )
    job_metadata: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"languages": set(), "categories": set(), "source_groups": set()}
    )
    for job in train_jobs:
        document_id = str(job["document_id"])
        by_document[document_id]["window_job_count"] += 1
        job_metadata[document_id]["languages"].add(
            str(job.get("language", "unknown"))
        )
        job_metadata[document_id]["categories"].add(
            str(job.get("category", "unknown"))
        )
        source_path = str(job.get("source_path", ""))
        job_metadata[document_id]["source_groups"].add(
            source_path.split("/", 1)[0] if source_path else "unknown"
        )

    inventory = []
    for document_id in sorted(train_documents):
        item = by_document[document_id]
        metadata = cluster_metadata.get(
            document_id,
            {
                "cluster_id": f"singleton_{document_id}",
                "source_group": "unknown",
                "category": "unknown",
            },
        )
        languages = item["languages"] | job_metadata[document_id]["languages"]
        categories = job_metadata[document_id]["categories"] - {"unknown"}
        source_groups = job_metadata[document_id]["source_groups"] - {"unknown"}
        inventory.append(
            {
                "document_id": document_id,
                "cluster_id": metadata["cluster_id"],
                "language": sorted(languages)[0] if len(languages) == 1 else "mixed",
                "source_group": (
                    sorted(source_groups)[0]
                    if len(source_groups) == 1
                    else metadata["source_group"]
                ),
                "category": (
                    sorted(categories)[0]
                    if len(categories) == 1
                    else metadata["category"]
                ),
                "record_count": item["record_count"],
                "window_job_count": item["window_job_count"],
                "entity_count": item["entity_count"],
                "relation_count": item["relation_count"],
                "entity_types": sorted(item["entity_types"]),
                "relation_types": sorted(item["relation_types"]),
            }
        )
    return inventory


def cluster_groups(inventory: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in inventory:
        grouped[str(row["cluster_id"])].append(row)
    return [
        sorted(rows, key=lambda row: row["document_id"])
        for _, rows in sorted(grouped.items())
    ]


def subset_sum_possible(sizes: list[int], target: int) -> bool:
    possible = {0}
    for size in sizes:
        possible |= {value + size for value in tuple(possible) if value + size <= target}
    return target in possible


def selection_score(
    selected: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    inventory: list[dict[str, Any]],
    salt: str,
) -> tuple[float, int, str]:
    proposed = selected + candidate
    global_language = Counter(row["language"] for row in inventory)
    global_domain = Counter(
        f"{row['source_group']}:{row['category']}" for row in inventory
    )
    proposed_language = Counter(row["language"] for row in proposed)
    proposed_domain = Counter(
        f"{row['source_group']}:{row['category']}" for row in proposed
    )
    size = len(proposed)
    total = len(inventory)

    def imbalance(global_counts: Counter[str], selected_counts: Counter[str]) -> float:
        return sum(
            abs(selected_counts[key] - size * count / total)
            for key, count in global_counts.items()
        )

    entity_coverage = len(
        {entity_type for row in proposed for entity_type in row["entity_types"]}
    )
    relation_coverage = len(
        {relation_type for row in proposed for relation_type in row["relation_types"]}
    )
    coverage_reward = 0.02 * (entity_coverage + relation_coverage)
    zero_window_documents = sum(row["window_job_count"] == 0 for row in proposed)
    cluster_id = str(candidate[0]["cluster_id"])
    return (
        imbalance(global_language, proposed_language)
        + 0.35 * imbalance(global_domain, proposed_domain)
        - coverage_reward,
        zero_window_documents,
        stable_rank(cluster_id, salt),
    )


def select_nested_groups(
    inventory: list[dict[str, Any]], budgets: Iterable[int], salt: str
) -> dict[int, list[dict[str, Any]]]:
    budgets = tuple(sorted(dict.fromkeys(int(value) for value in budgets)))
    if not budgets or budgets[0] <= 0 or budgets[-1] > len(inventory):
        raise ValueError("budgets must be positive and no larger than inventory")
    remaining = cluster_groups(inventory)
    if not subset_sum_possible([len(group) for group in remaining], budgets[-1]):
        raise ValueError("duplicate-cluster sizes cannot satisfy the largest budget")
    selected_groups: list[list[dict[str, Any]]] = []
    snapshots: dict[int, list[dict[str, Any]]] = {}
    selected_size = 0
    for budget in budgets:
        if budget < selected_size:
            raise ValueError("budgets must be nested in ascending order")
        while selected_size < budget:
            needed = budget - selected_size
            viable = []
            for group in remaining:
                size = len(group)
                if size > needed:
                    continue
                other_sizes = [len(other) for other in remaining if other is not group]
                if subset_sum_possible(other_sizes, needed - size):
                    viable.append(group)
            if not viable:
                raise ValueError(f"cannot satisfy exact nested budget {budget}")
            flat_selected = [row for group in selected_groups for row in group]
            chosen = min(
                viable,
                key=lambda group: selection_score(
                    flat_selected, group, inventory, f"{salt}|{budget}"
                ),
            )
            selected_groups.append(chosen)
            remaining.remove(chosen)
            selected_size += len(chosen)
        snapshots[budget] = [
            row for group in selected_groups for row in group
        ]
    return snapshots


def coverage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "documents": len(rows),
        "clusters": len({row["cluster_id"] for row in rows}),
        "records": sum(row["record_count"] for row in rows),
        "window_jobs": sum(row["window_job_count"] for row in rows),
        "documents_without_window_jobs": sorted(
            row["document_id"] for row in rows if row["window_job_count"] == 0
        ),
        "languages": dict(sorted(Counter(row["language"] for row in rows).items())),
        "source_groups": dict(
            sorted(Counter(row["source_group"] for row in rows).items())
        ),
        "categories": dict(
            sorted(Counter(row["category"] for row in rows).items())
        ),
        "entity_types": sorted(
            {entity_type for row in rows for entity_type in row["entity_types"]}
        ),
        "relation_types": sorted(
            {relation_type for row in rows for relation_type in row["relation_types"]}
        ),
        "entities": sum(row["entity_count"] for row in rows),
        "relations": sum(row["relation_count"] for row in rows),
    }


def cluster_boundary_violations(
    selected_documents: set[str], inventory: list[dict[str, Any]]
) -> list[str]:
    groups = cluster_groups(inventory)
    return sorted(
        str(group[0]["cluster_id"])
        for group in groups
        if 0 < len({row["document_id"] for row in group} & selected_documents) < len(group)
    )


def build_run_matrix(
    budgets: Iterable[int], seeds: Iterable[int], manifest_info: dict[int, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    training_rows = []
    derived_rows = []
    execution_order = [100, 10, 25, 50]
    ordered_budgets = [budget for budget in execution_order if budget in budgets]
    ordered_budgets.extend(sorted(set(budgets) - set(ordered_budgets)))
    for budget in ordered_budgets:
        budget_tag = f"d{budget:03d}"
        for seed in seeds:
            seed_tag = f"seed{seed}"
            for system in TRAINABLE_SYSTEMS:
                run_id = f"lr_v1_{budget_tag}_{seed_tag}_{system}"
                assets = (
                    "data/processed/experiments/formal/low_resource_v1/"
                    f"{budget_tag}/assets"
                )
                job_name = {
                    "baseline": "baseline_jobs.jsonl",
                    "kg_v1": "kg_v1_jobs.jsonl",
                    "kg_v2": "kg_v2_jobs.jsonl",
                }[system]
                root = (
                    "data/processed/experiments/formal/low_resource_v1/"
                    f"{budget_tag}/{seed_tag}/{system}"
                )
                training_rows.append(
                    {
                        "run_id": run_id,
                        "budget_documents": budget,
                        "seed": seed,
                        "system": system,
                        "trainable": True,
                        "document_manifest": manifest_info[budget]["path"],
                        "document_manifest_sha256": manifest_info[budget]["sha256"],
                        "protocol_config": "configs/low_resource_protocol_v1.yaml",
                        "asset_builder": "scripts/build_low_resource_assets.py",
                        "training_inputs": {
                            "gold": f"{assets}/gold.jsonl",
                            "index": f"{assets}/index.jsonl",
                            "jobs": f"{assets}/{job_name}",
                        },
                        "validation_jobs": f"{assets}/validation/{job_name}",
                        "trainer": "scripts/train_qlora.py",
                        "inference": "scripts/run_qlora_inference.py",
                        "strict_evaluator": "scripts/evaluate_span_aware.py",
                        "evidence_evaluator": "scripts/evaluate_evidence_graph.py",
                        "output_directory": root,
                        "status": "planned",
                    }
                )
            derived_rows.append(
                {
                    "derivation_id": f"lr_v1_{budget_tag}_{seed_tag}_derived",
                    "budget_documents": budget,
                    "seed": seed,
                    "source_systems": ["kg_v1", "kg_v2"],
                    "derived_systems": list(DERIVED_SYSTEMS),
                    "protocol_config": "configs/low_resource_protocol_v1.yaml",
                    "verifier": "scripts/verify_relations.py",
                    "fusion": "scripts/fuse_kg_v1_v2_predictions.py",
                    "frozen_runner": "scripts/run_frozen_kg_v3.py",
                    "strict_evaluator": "scripts/evaluate_span_aware.py",
                    "evidence_evaluator": "scripts/evaluate_evidence_graph.py",
                    "output_directory": (
                        "data/processed/experiments/formal/low_resource_v1/"
                        f"{budget_tag}/{seed_tag}/derived"
                    ),
                    "status": "planned",
                }
            )
    return training_rows, derived_rows


def run(args: argparse.Namespace) -> int:
    require_train_only_path(args.train_annotations, "train annotations")
    require_train_only_path(args.train_index, "train index")
    require_train_only_path(args.train_jobs, "train jobs")
    annotations = load_jsonl(args.train_annotations)
    index = load_jsonl(args.train_index)
    train_jobs = load_jsonl(args.train_jobs)
    train_documents = {str(row["document_id"]) for row in index}
    cluster_metadata, missing_cluster_rows = load_cluster_metadata(
        args.clusters, train_documents
    )
    inventory = build_inventory(
        annotations, index, train_jobs, cluster_metadata
    )
    budgets = tuple(args.budgets)
    seeds = tuple(args.seeds)
    selections = select_nested_groups(inventory, budgets, args.selection_salt)

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_info: dict[int, dict[str, Any]] = {}
    budget_summaries: dict[str, Any] = {}
    previous: set[str] = set()
    for budget in sorted(budgets):
        rows = selections[budget]
        selected = {row["document_id"] for row in rows}
        if not previous <= selected:
            raise ValueError(f"budget {budget} is not nested")
        violations = cluster_boundary_violations(selected, inventory)
        if violations:
            raise ValueError(f"cluster boundary violation(s): {violations}")
        ordered_rows = [
            {
                **row,
                "budget_documents": budget,
                "selection_rank": rank,
                "split": "train",
            }
            for rank, row in enumerate(rows, 1)
        ]
        path = args.output / f"train_documents_{budget:03d}.jsonl"
        write_jsonl(path, ordered_rows)
        relative_path = str(path)
        manifest_info[budget] = {
            "path": relative_path,
            "sha256": sha256_file(path),
        }
        budget_summaries[str(budget)] = {
            **coverage_summary(rows),
            "manifest": relative_path,
            "manifest_sha256": manifest_info[budget]["sha256"],
            "cluster_boundary_violations": violations,
            "nested_from_previous": previous <= selected,
        }
        previous = selected

    training_rows, derived_rows = build_run_matrix(budgets, seeds, manifest_info)
    run_matrix_path = args.output / "run_matrix.jsonl"
    derived_matrix_path = args.output / "derived_matrix.jsonl"
    write_jsonl(run_matrix_path, training_rows)
    write_jsonl(derived_matrix_path, derived_rows)
    summary = {
        "protocol_id": "low-resource-manifests-v1",
        "selection_policy": "deterministic-nested-cluster-closed-stratified-train-only",
        "selection_salt": args.selection_salt,
        "budgets": list(budgets),
        "seeds": list(seeds),
        "trainable_systems": list(TRAINABLE_SYSTEMS),
        "derived_systems": list(DERIVED_SYSTEMS),
        "training_runs": len(training_rows),
        "derived_groups": len(derived_rows),
        "formal_test_read": False,
        "validation_used_for_selection": False,
        "source_files": {
            "train_annotations": {
                "path": str(args.train_annotations),
                "sha256": sha256_file(args.train_annotations),
            },
            "train_index": {
                "path": str(args.train_index),
                "sha256": sha256_file(args.train_index),
            },
            "train_jobs": {
                "path": str(args.train_jobs),
                "sha256": sha256_file(args.train_jobs),
            },
            "duplicate_cluster_catalog": {
                "path": str(args.clusters),
                "sha256": sha256_file(args.clusters),
                "usage": "metadata lookup restricted to known train document IDs",
            },
        },
        "inventory": coverage_summary(inventory),
        "missing_cluster_rows": missing_cluster_rows,
        "budget_summaries": budget_summaries,
        "run_matrix": {
            "path": str(run_matrix_path),
            "sha256": sha256_file(run_matrix_path),
        },
        "derived_matrix": {
            "path": str(derived_matrix_path),
            "sha256": sha256_file(derived_matrix_path),
        },
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-annotations",
        type=Path,
        default=Path("data/processed/reviewed/formal_split/train.jsonl"),
    )
    parser.add_argument(
        "--train-index",
        type=Path,
        default=Path("data/processed/reviewed/formal_split/train_index.jsonl"),
    )
    parser.add_argument(
        "--train-jobs",
        type=Path,
        default=Path(
            "data/processed/experiments/formal/windowed_train_v2/baseline_jobs.jsonl"
        ),
    )
    parser.add_argument(
        "--clusters",
        type=Path,
        default=Path("data/catalog/near_duplicate_clusters.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/experiments/formal/low_resource_manifests_v1"
        ),
    )
    parser.add_argument("--budgets", nargs="+", type=int, default=list(DEFAULT_BUDGETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--selection-salt", default="low-resource-v1-20260830")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
