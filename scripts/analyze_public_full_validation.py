#!/usr/bin/env python3
"""Build validation-only public benchmark evidence and paired statistics.

This script is intentionally post-processing only. It never opens a test split,
never invokes model inference, and writes all derived artifacts beneath the
requested analysis directory.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
import sys
import time
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import bootstrap_compare  # noqa: E402
import evaluate_annotations  # noqa: E402
import evaluate_evidence_graph  # noqa: E402
import evaluate_span_aware  # noqa: E402


DATASETS = ("conll04", "scierc", "ade")
METHODS = ("baseline", "kg")
PHASES = ("raw", "expanded", "verified")
EXPECTED_MISSING = 13
DEFAULT_SEED = 20260904
SYSTEMS = {
    ("baseline", "raw"): {
        "id": "baseline_raw",
        "label": "Baseline raw",
    },
    ("kg", "raw"): {
        "id": "kg_raw",
        "label": "KG raw",
    },
    ("baseline", "expanded"): {
        "id": "baseline_evidence",
        "label": "Baseline+Evidence",
    },
    ("kg", "expanded"): {
        "id": "kg_evidence",
        "label": "KG+Evidence",
    },
    ("baseline", "verified"): {
        "id": "baseline_evidence_verifier",
        "label": "Baseline+Verifier",
    },
    ("kg", "verified"): {
        "id": "kg_evidence_verifier_evge_like",
        "label": "KG+Evidence+Verifier (EVGE-like)",
    },
}
CONTRASTS = {
    "baseline_raw_vs_kg_raw": {
        "left": ("baseline", "raw"),
        "right": ("kg", "raw"),
    },
    "baseline_verifier_vs_kg_evidence_verifier": {
        "left": ("baseline", "verified"),
        "right": ("kg", "verified"),
    },
}
TERMINOLOGY_NOTE = (
    "KG+Evidence+Verifier (EVGE-like) denotes the current KG-prompted output "
    "after deterministic evidence expansion and relation verification. It is "
    "not an EAE/HRGE fusion PGE implementation."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_prediction(root: Path, dataset: str, method: str, phase: str) -> Path:
    suffix = {
        "raw": "validation.jsonl",
        "expanded": "validation_expanded.jsonl",
        "verified": "validation_verified.jsonl",
    }[phase]
    return root / f"{dataset}_{method}_{suffix}"


def assert_validation_only(paths: Iterable[Path]) -> None:
    for path in paths:
        lowered_parts = {part.casefold() for part in path.parts}
        if "test" in lowered_parts or "test_gold.jsonl" in path.name.casefold():
            raise ValueError(f"test-split path is forbidden: {path}")


def index_prediction_rows(
    rows: list[dict[str, Any]], expected_ids: set[str], source: Path
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        job_id = str(row.get("job_id", ""))
        if not job_id:
            raise ValueError(f"prediction without job_id in {source}")
        if job_id in indexed:
            raise ValueError(f"duplicate prediction job_id {job_id} in {source}")
        indexed[job_id] = row
    unknown = sorted(set(indexed) - expected_ids)
    if unknown:
        raise ValueError(f"unknown prediction job IDs in {source}: {unknown[:3]}")
    return indexed


def empty_prediction(job: dict[str, Any], phase: str) -> dict[str, Any]:
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
                "notes": (
                    "Empty validation prediction materialized for statistical "
                    f"evaluation after terminal inference failure; phase={phase}."
                ),
            },
        },
    }


def materialize_complete(
    source: Path, jobs_path: Path, output: Path, phase: str
) -> dict[str, Any]:
    jobs = read_jsonl(jobs_path)
    job_ids = [str(job["job_id"]) for job in jobs]
    if len(job_ids) != len(set(job_ids)):
        raise ValueError(f"duplicate job IDs in {jobs_path}")
    predictions = index_prediction_rows(read_jsonl(source), set(job_ids), source)
    missing = [job_id for job_id in job_ids if job_id not in predictions]
    job_by_id = {str(job["job_id"]): job for job in jobs}
    completed = [
        predictions.get(job_id, empty_prediction(job_by_id[job_id], phase))
        for job_id in job_ids
    ]
    write_jsonl(output, completed)
    return {
        "source": str(source),
        "output": str(output),
        "jobs": len(jobs),
        "source_prediction_rows": len(predictions),
        "materialized_empty_rows": len(missing),
        "missing_job_ids": missing,
    }


def run_annotation_evaluation(
    gold: Path,
    gold_index: Path,
    predictions: Path,
    jobs: Path,
    output: Path,
) -> dict[str, Any]:
    arguments = Namespace(
        gold=gold,
        gold_index=gold_index,
        predictions=predictions,
        pred_index=None,
        jobs=jobs,
        limit=None,
        offset=0,
        include_missing_as_empty=True,
        output=output,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        evaluate_annotations.run(arguments)
    return json.loads(output.read_text(encoding="utf-8"))


def run_graph_evaluation(
    gold: Path,
    gold_index: Path,
    predictions: Path,
    jobs: Path,
    ontology: Path,
    output: Path,
) -> dict[str, Any]:
    arguments = Namespace(
        gold=gold,
        gold_index=gold_index,
        predictions=predictions,
        jobs=jobs,
        ontology=ontology,
        output=output,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        evaluate_evidence_graph.run(arguments)
    return json.loads(output.read_text(encoding="utf-8"))


def build_span_index(source: Path, output: Path) -> list[dict[str, Any]]:
    adapted: list[dict[str, Any]] = []
    for row in read_jsonl(source):
        job_id = str(row["job_id"])
        adapted.append(
            {
                **row,
                "parent_job_id": job_id,
                "split": "validation",
            }
        )
    write_jsonl(output, adapted)
    return adapted


def run_span_evaluation(
    gold: Path,
    adapted_index: Path,
    predictions: Path,
    jobs: Path,
    output: Path,
) -> dict[str, Any]:
    arguments = Namespace(
        source_gold=gold,
        source_gold_index=adapted_index,
        gold=gold,
        gold_index=adapted_index,
        predictions=predictions,
        jobs=jobs,
        output=output,
        allow_non_validation=False,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        evaluate_span_aware.run(arguments)
    return json.loads(output.read_text(encoding="utf-8"))


def f1_array(counts: np.ndarray) -> np.ndarray:
    denominator = counts[..., 0] + counts[..., 1]
    return np.divide(
        2.0 * counts[..., 2],
        denominator,
        out=np.zeros_like(denominator, dtype=np.float64),
        where=denominator != 0,
    )


def percentile_interval(values: np.ndarray) -> dict[str, float]:
    lower, upper = np.quantile(values, (0.025, 0.975))
    return {"lower": round(float(lower), 4), "upper": round(float(upper), 4)}


def field_arrays(
    units: list[dict[str, Any]], field: str
) -> tuple[np.ndarray, np.ndarray]:
    keys = ("gold", "predicted", "correct")
    left = np.asarray(
        [[unit["baseline"][field][key] for key in keys] for unit in units],
        dtype=np.int64,
    )
    right = np.asarray(
        [[unit["kg"][field][key] for key in keys] for unit in units],
        dtype=np.int64,
    )
    return left, right


def compare_field_vectorized(
    units: list[dict[str, Any]],
    field: str,
    iterations: int,
    rng: np.random.Generator,
    batch_size: int = 512,
) -> dict[str, Any]:
    left, right = field_arrays(units, field)
    n = len(units)
    if n == 0:
        raise ValueError("paired comparison requires at least one document")

    left_total = left.sum(axis=0)
    right_total = right.sum(axis=0)
    left_document_f1 = f1_array(left)
    right_document_f1 = f1_array(right)
    left_pooled = float(f1_array(left_total))
    right_pooled = float(f1_array(right_total))
    left_macro = float(left_document_f1.mean())
    right_macro = float(right_document_f1.mean())

    samples = {
        name: np.empty(iterations, dtype=np.float64)
        for name in (
            "left_pooled",
            "right_pooled",
            "left_macro",
            "right_macro",
            "difference_pooled",
            "difference_macro",
            "permutation_pooled",
            "permutation_macro",
        )
    }
    combined_total = left_total + right_total
    delta = right - left
    cursor = 0
    while cursor < iterations:
        batch = min(batch_size, iterations - cursor)
        selected = rng.integers(0, n, size=(batch, n))
        sampled_left = left[selected].sum(axis=1)
        sampled_right = right[selected].sum(axis=1)
        sampled_left_pooled = f1_array(sampled_left)
        sampled_right_pooled = f1_array(sampled_right)
        sampled_left_macro = left_document_f1[selected].mean(axis=1)
        sampled_right_macro = right_document_f1[selected].mean(axis=1)

        swap = rng.integers(0, 2, size=(batch, n), dtype=np.int8).astype(bool)
        permuted_left = left_total + np.where(
            swap[..., None], delta[None, ...], 0
        ).sum(axis=1)
        permuted_right = combined_total - permuted_left
        permuted_left_macro = np.where(
            swap, right_document_f1, left_document_f1
        ).mean(axis=1)
        permuted_right_macro = np.where(
            swap, left_document_f1, right_document_f1
        ).mean(axis=1)

        target = slice(cursor, cursor + batch)
        samples["left_pooled"][target] = sampled_left_pooled
        samples["right_pooled"][target] = sampled_right_pooled
        samples["left_macro"][target] = sampled_left_macro
        samples["right_macro"][target] = sampled_right_macro
        samples["difference_pooled"][target] = (
            sampled_right_pooled - sampled_left_pooled
        )
        samples["difference_macro"][target] = (
            sampled_right_macro - sampled_left_macro
        )
        samples["permutation_pooled"][target] = (
            f1_array(permuted_right) - f1_array(permuted_left)
        )
        samples["permutation_macro"][target] = (
            permuted_right_macro - permuted_left_macro
        )
        cursor += batch

    pooled_difference = right_pooled - left_pooled
    macro_difference = right_macro - left_macro

    def p_value(values: np.ndarray, observed: float) -> float:
        extreme = int(np.count_nonzero(np.abs(values) >= abs(observed)))
        return round((extreme + 1) / (iterations + 1), 5)

    def observed(counts: np.ndarray, pooled_f1: float, macro_f1: float) -> dict[str, Any]:
        return {
            "pooled_f1": round(pooled_f1, 4),
            "macro_f1": round(macro_f1, 4),
            "gold": int(counts[0]),
            "predicted": int(counts[1]),
            "correct": int(counts[2]),
        }

    return {
        "left": {
            **observed(left_total, left_pooled, left_macro),
            "pooled_f1_ci95": percentile_interval(samples["left_pooled"]),
            "macro_f1_ci95": percentile_interval(samples["left_macro"]),
        },
        "right": {
            **observed(right_total, right_pooled, right_macro),
            "pooled_f1_ci95": percentile_interval(samples["right_pooled"]),
            "macro_f1_ci95": percentile_interval(samples["right_macro"]),
        },
        "right_minus_left": {
            "pooled_f1": round(pooled_difference, 4),
            "macro_f1": round(macro_difference, 4),
            "pooled_f1_ci95": percentile_interval(samples["difference_pooled"]),
            "macro_f1_ci95": percentile_interval(samples["difference_macro"]),
            "paired_permutation_p_pooled_f1": p_value(
                samples["permutation_pooled"], pooled_difference
            ),
            "paired_permutation_p_macro_f1": p_value(
                samples["permutation_macro"], macro_difference
            ),
        },
    }


def compare_units(
    units: list[dict[str, Any]],
    comparison_id: str,
    dataset: str,
    left_system: dict[str, str],
    right_system: dict[str, str],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    return {
        "comparison_id": comparison_id,
        "dataset": dataset,
        "split": "validation",
        "unit": "document",
        "documents": len(units),
        "iterations": iterations,
        "seed": seed,
        "left_system": left_system,
        "right_system": right_system,
        "difference_direction": "right_minus_left",
        "tests": {
            "confidence_interval": "paired document bootstrap percentile CI",
            "significance": "two-sided paired document permutation with plus-one correction",
        },
        "matching": {
            "entity": "whitespace-normalized case-folded exact text plus entity type",
            "relation": "normalized endpoint texts, relation type, and direction",
            "within_job_duplicate_handling": "set",
            "note": (
                "These paired comparisons follow bootstrap_compare.py matching. "
                "Character-span one-to-one strict metrics are reported separately."
            ),
        },
        "fields": {
            field: compare_field_vectorized(units, field, iterations, rng)
            for field in ("entity", "relation")
        },
        "terminology_note": TERMINOLOGY_NOTE,
    }


def build_units(
    gold: Path,
    gold_index: Path,
    left: Path,
    right: Path,
    jobs: Path,
) -> list[dict[str, Any]]:
    return bootstrap_compare.build_units(
        read_jsonl(gold),
        read_jsonl(left),
        read_jsonl(right),
        read_jsonl(jobs),
        read_jsonl(gold_index),
    )


def graph_summary(result: dict[str, Any]) -> dict[str, Any]:
    overall = result["overall"]
    return {
        "jobs": result["jobs"],
        "entity_evidence_coverage": overall["entity_evidence_coverage"],
        "entity_evidence_correctness": overall["entity_evidence_correctness"],
        "relation_evidence_coverage": overall["relation_evidence_coverage"],
        "relation_evidence_correctness": overall["relation_evidence_correctness"],
        "unsupported_claim_rate": overall["unsupported_claim_rate"],
        "invalid_relation_rate": overall["invalid_relation_rate"],
        "node_f1": pooled_graph_f1(result, "node"),
        "edge_f1": pooled_graph_f1(result, "edge"),
        "node_jaccard_macro_by_job": overall["node_jaccard"],
        "edge_jaccard_macro_by_job": overall["edge_jaccard"],
        "causal_edge": overall["causal_edge"],
        "counts": overall["counts"],
    }


def span_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "metric": result["metric"],
        "selection_split": result["selection_split"],
        "formal_test_read": result["formal_test_read"],
        "jobs": result["jobs"],
        "documents": result["documents"],
        "generation_success_rate": result["generation_success_rate"],
        "entity": result["overall"]["entity"],
        "relation": result["overall"]["relation"],
        "relation_with_claim_status": result["overall"][
            "relation_with_claim_status"
        ],
        "macro_by_job": result["macro_by_job"],
        "resolution": result["resolution"],
    }


def pooled_graph_f1(result: dict[str, Any], field: str) -> float:
    totals = {key: 0 for key in ("gold", "predicted", "correct")}
    for item in result["per_job"].values():
        for key in totals:
            totals[key] += int(item["graph"][field][key])
    denominator = totals["gold"] + totals["predicted"]
    return round(2 * totals["correct"] / denominator, 4) if denominator else 0.0


def add_tsv_row(
    rows: list[dict[str, Any]],
    *,
    record_type: str,
    dataset: str,
    system_or_contrast: str,
    phase: str,
    field: str,
    metric: str,
    value: Any,
    ci95: dict[str, Any] | None = None,
    p_value: Any = "",
    jobs_or_documents: Any = "",
    iterations: Any = "",
) -> None:
    rows.append(
        {
            "record_type": record_type,
            "dataset": dataset,
            "system_or_contrast": system_or_contrast,
            "phase": phase,
            "field": field,
            "metric": metric,
            "value": value,
            "ci95_lower": ci95["lower"] if ci95 else "",
            "ci95_upper": ci95["upper"] if ci95 else "",
            "paired_permutation_p": p_value,
            "jobs_or_documents": jobs_or_documents,
            "iterations": iterations,
        }
    )


def build_summary_tsv(
    systems: dict[str, Any], comparisons: dict[str, Any]
) -> str:
    rows: list[dict[str, Any]] = []
    graph_fields = (
        "entity_evidence_coverage",
        "entity_evidence_correctness",
        "relation_evidence_coverage",
        "relation_evidence_correctness",
        "unsupported_claim_rate",
        "invalid_relation_rate",
        "node_f1",
        "edge_f1",
        "node_jaccard_macro_by_job",
        "edge_jaccard_macro_by_job",
    )
    for dataset in DATASETS:
        for method in METHODS:
            for phase in PHASES:
                record = systems[dataset][method][phase]
                graph = record["evidence_graph"]
                for metric in graph_fields:
                    add_tsv_row(
                        rows,
                        record_type="system_metric",
                        dataset=dataset,
                        system_or_contrast=record["system"]["label"],
                        phase=phase,
                        field="evidence_graph",
                        metric=metric,
                        value=graph[metric],
                        jobs_or_documents=graph["jobs"],
                    )
                if phase == "expanded":
                    metrics = record["annotation_evidence_metrics"]
                    for field in ("entity_strict", "relation_strict"):
                        add_tsv_row(
                            rows,
                            record_type="system_metric",
                            dataset=dataset,
                            system_or_contrast=record["system"]["label"],
                            phase=phase,
                            field=field,
                            metric="pooled_f1",
                            value=metrics[field]["f1"],
                            jobs_or_documents=metrics["jobs_evaluated"],
                        )
                if phase in {"expanded", "verified"}:
                    span = record["span_strict_metrics"]
                    for field in ("entity", "relation", "relation_with_claim_status"):
                        add_tsv_row(
                            rows,
                            record_type="system_metric",
                            dataset=dataset,
                            system_or_contrast=record["system"]["label"],
                            phase=phase,
                            field=field,
                            metric="character_span_strict_pooled_f1",
                            value=span[field]["f1"],
                            jobs_or_documents=span["jobs"],
                        )

    for comparison_id, cohorts in comparisons.items():
        for dataset, result in cohorts.items():
            for field, values in result["fields"].items():
                for summary_name in ("pooled_f1", "macro_f1"):
                    p_key = f"paired_permutation_p_{summary_name}"
                    add_tsv_row(
                        rows,
                        record_type="paired_comparison",
                        dataset=dataset,
                        system_or_contrast=comparison_id,
                        phase="raw" if "_raw_" in comparison_id else "verified",
                        field=field,
                        metric=f"right_minus_left_{summary_name}",
                        value=values["right_minus_left"][summary_name],
                        ci95=values["right_minus_left"][f"{summary_name}_ci95"],
                        p_value=values["right_minus_left"][p_key],
                        jobs_or_documents=result["documents"],
                        iterations=result["iterations"],
                    )

    fields = list(rows[0])
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, dialect="excel-tab", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/processed/public_benchmarks_full"),
    )
    parser.add_argument(
        "--run-root", type=Path, default=Path("outputs/public_full_stage1")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/public_full_stage1/validation_analysis"),
    )
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--expected-missing", type=int, default=EXPECTED_MISSING
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    started_at = utc_now()
    started_clock = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output / "status.json"
    base_status = {
        "schema_version": "1.0",
        "service": "public-validation-analysis.service",
        "mode": "cpu_postprocessing_only",
        "split": "validation",
        "formal_test_read": False,
        "gpu_used": False,
        "started_at": started_at,
        "pid": os.getpid(),
    }
    write_json(status_path, {**base_status, "status": "running"})

    try:
        input_paths: list[Path] = []
        materialization: dict[str, Any] = {}
        systems: dict[str, Any] = {}
        complete_paths: dict[tuple[str, str, str], Path] = {}
        missing_records: list[dict[str, str]] = []

        for dataset in DATASETS:
            dataset_root = args.data_root / dataset
            gold = dataset_root / "validation_gold.jsonl"
            gold_index = dataset_root / "validation_index.jsonl"
            ontology = dataset_root / "ontology.yaml"
            adapted_span_index = (
                args.output / "adapted_indexes" / f"{dataset}_validation.jsonl"
            )
            input_paths.extend((gold, gold_index, ontology))
            adapted_rows = build_span_index(gold_index, adapted_span_index)
            if not adapted_rows or any(
                row.get("parent_job_id") != row.get("job_id")
                or row.get("split") != "validation"
                for row in adapted_rows
            ):
                raise ValueError(f"invalid adapted span index: {adapted_span_index}")
            materialization[dataset] = {}
            systems[dataset] = {}
            for method in METHODS:
                jobs = dataset_root / f"validation_{method}_jobs.jsonl"
                input_paths.append(jobs)
                materialization[dataset][method] = {}
                systems[dataset][method] = {}
                raw_missing: list[str] | None = None
                for phase in PHASES:
                    source = source_prediction(args.run_root, dataset, method, phase)
                    completed = (
                        args.output
                        / "completed_predictions"
                        / f"{dataset}_{method}_{phase}.jsonl"
                    )
                    input_paths.append(source)
                    assert_validation_only((gold, gold_index, jobs, ontology, source))
                    audit = materialize_complete(source, jobs, completed, phase)
                    materialization[dataset][method][phase] = audit
                    complete_paths[(dataset, method, phase)] = completed
                    missing = audit["missing_job_ids"]
                    if raw_missing is None:
                        raw_missing = missing
                        for job_id in missing:
                            missing_records.append(
                                {
                                    "dataset": dataset,
                                    "method": method,
                                    "job_id": job_id,
                                }
                            )
                    elif missing != raw_missing:
                        raise ValueError(
                            f"missing IDs changed across phases for {dataset}/{method}: "
                            f"raw={raw_missing}, {phase}={missing}"
                        )

                    graph_path = (
                        args.output
                        / "evidence_graph"
                        / f"{dataset}_{method}_{phase}.json"
                    )
                    graph = run_graph_evaluation(
                        gold, gold_index, completed, jobs, ontology, graph_path
                    )
                    phase_record: dict[str, Any] = {
                        "system": SYSTEMS[(method, phase)],
                        "source_predictions": str(source),
                        "completed_predictions": str(completed),
                        "source_prediction_rows": audit["source_prediction_rows"],
                        "materialized_empty_rows": audit["materialized_empty_rows"],
                        "evidence_graph_metrics_path": str(graph_path),
                        "evidence_graph": graph_summary(graph),
                    }
                    if phase == "expanded":
                        evidence_metrics_path = (
                            args.output
                            / "evidence_metrics"
                            / f"{dataset}_{method}_expanded.json"
                        )
                        evidence_metrics = run_annotation_evaluation(
                            gold,
                            gold_index,
                            source,
                            jobs,
                            evidence_metrics_path,
                        )
                        phase_record.update(
                            {
                                "annotation_evidence_metrics_path": str(
                                    evidence_metrics_path
                                ),
                                "annotation_evidence_metrics": evidence_metrics,
                            }
                        )
                    if phase in {"expanded", "verified"}:
                        span_metrics_path = (
                            args.output
                            / "span_metrics"
                            / f"{dataset}_{method}_{phase}.json"
                        )
                        span_metrics = run_span_evaluation(
                            gold,
                            adapted_span_index,
                            source,
                            jobs,
                            span_metrics_path,
                        )
                        if span_metrics.get("formal_test_read") is not False:
                            raise ValueError(
                                f"span evaluator reported non-validation access: {span_metrics_path}"
                            )
                        phase_record.update(
                            {
                                "span_strict_metrics_path": str(span_metrics_path),
                                "span_strict_metrics": span_summary(span_metrics),
                            }
                        )
                    systems[dataset][method][phase] = phase_record

        unique_missing = len(missing_records)
        if unique_missing != args.expected_missing:
            raise ValueError(
                f"expected {args.expected_missing} method-job generation failures; "
                f"observed {unique_missing}"
            )

        write_json(
            args.output / "missing_predictions.json",
            {
                "split": "validation",
                "unique_method_job_failures": unique_missing,
                "rows_materialized_across_three_phases": unique_missing * len(PHASES),
                "records": missing_records,
            },
        )
        missing_tsv = io.StringIO()
        missing_writer = csv.DictWriter(
            missing_tsv,
            fieldnames=("dataset", "method", "job_id"),
            dialect="excel-tab",
            lineterminator="\n",
        )
        missing_writer.writeheader()
        missing_writer.writerows(missing_records)
        atomic_text(args.output / "missing_predictions.tsv", missing_tsv.getvalue())

        comparisons: dict[str, Any] = {}
        for contrast_index, (comparison_id, contrast) in enumerate(CONTRASTS.items()):
            method_left, phase_left = contrast["left"]
            method_right, phase_right = contrast["right"]
            comparison_units: dict[str, list[dict[str, Any]]] = {}
            for dataset in DATASETS:
                dataset_root = args.data_root / dataset
                units = build_units(
                    dataset_root / "validation_gold.jsonl",
                    dataset_root / "validation_index.jsonl",
                    complete_paths[(dataset, method_left, phase_left)],
                    complete_paths[(dataset, method_right, phase_right)],
                    dataset_root / "validation_baseline_jobs.jsonl",
                )
                comparison_units[dataset] = units

            all_units = [
                unit
                for dataset in DATASETS
                for unit in comparison_units[dataset]
            ]
            comparison_units["all_datasets"] = all_units
            comparisons[comparison_id] = {}
            for cohort_index, (dataset, units) in enumerate(comparison_units.items()):
                seed = args.seed + contrast_index * 100 + cohort_index
                result = compare_units(
                    units,
                    comparison_id,
                    dataset,
                    SYSTEMS[contrast["left"]],
                    SYSTEMS[contrast["right"]],
                    args.iterations,
                    seed,
                )
                comparison_path = (
                    args.output / "comparisons" / f"{dataset}_{comparison_id}.json"
                )
                write_json(comparison_path, result)
                result["artifact"] = str(comparison_path)
                comparisons[comparison_id][dataset] = result

        input_manifest = {
            "split": "validation",
            "formal_test_read": False,
            "files": [
                {
                    "path": str(path),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in dict.fromkeys(input_paths)
            ],
        }
        write_json(args.output / "input_manifest.json", input_manifest)

        comparison_results = [
            result for cohorts in comparisons.values() for result in cohorts.values()
        ]
        span_results = [
            systems[dataset][method][phase]["span_strict_metrics"]
            for dataset in DATASETS
            for method in METHODS
            for phase in ("expanded", "verified")
        ]
        quality_checks = [
            {
                "id": "validation_only_inputs",
                "status": "passed",
                "observed": len(input_manifest["files"]),
                "expected": "all declared inputs belong to validation or shared ontology",
            },
            {
                "id": "method_job_generation_failures",
                "status": "passed",
                "observed": unique_missing,
                "expected": args.expected_missing,
            },
            {
                "id": "completed_prediction_files",
                "status": "passed",
                "observed": len(complete_paths),
                "expected": len(DATASETS) * len(METHODS) * len(PHASES),
            },
            {
                "id": "adapted_span_indexes",
                "status": "passed",
                "observed": len(DATASETS),
                "expected": len(DATASETS),
                "condition": "parent_job_id=job_id and split=validation for every row",
            },
            {
                "id": "span_strict_results",
                "status": "passed",
                "observed": len(span_results),
                "expected": len(DATASETS) * len(METHODS) * 2,
                "condition": (
                    "metric=strict-global-character-span-one-to-one and "
                    "formal_test_read=false"
                ),
            },
            {
                "id": "paired_comparisons",
                "status": "passed",
                "observed": len(comparison_results),
                "expected": len(CONTRASTS) * (len(DATASETS) + 1),
                "iterations_each": args.iterations,
            },
            {
                "id": "gpu_isolation",
                "status": "passed",
                "observed": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "expected": "empty CUDA_VISIBLE_DEVICES",
            },
        ]
        if any(
            result["metric"] != "strict-global-character-span-one-to-one"
            or result["formal_test_read"] is not False
            for result in span_results
        ):
            raise ValueError("span strict metric identity or validation guard failed")
        if any(
            result["iterations"] != args.iterations
            for result in comparison_results
        ):
            raise ValueError("paired comparison iteration count mismatch")
        if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
            raise ValueError("CUDA_VISIBLE_DEVICES must be empty for CPU-only analysis")
        checks_document = {
            "status": "passed",
            "split": "validation",
            "formal_test_read": False,
            "gpu_used": False,
            "checks": quality_checks,
        }
        write_json(args.output / "checks.json", checks_document)

        finished_at = utc_now()
        summary = {
            "schema_version": "1.0",
            "scope": {
                "split": "validation",
                "datasets": list(DATASETS),
                "formal_test_read": False,
                "gpu_used": False,
                "postprocessing": "CPU only",
            },
            "terminology": {
                "kg_verified_label": "KG+Evidence+Verifier (EVGE-like)",
                "note": TERMINOLOGY_NOTE,
            },
            "statistical_protocol": {
                "unit": "document",
                "iterations": args.iterations,
                "base_seed": args.seed,
                "bootstrap": "paired document resampling with replacement; percentile 95% CI",
                "permutation": "paired document label swap; two-sided plus-one corrected p-value",
                "multiplicity_adjustment": "none; dataset/field results are reported separately",
                "matching": (
                    "bootstrap_compare.py normalized exact-text/type entity matching "
                    "and normalized directed endpoint/type relation matching; this is "
                    "distinct from the separately reported character-span one-to-one metric"
                ),
            },
            "span_strict_protocol": {
                "metric": "strict-global-character-span-one-to-one",
                "phases": ["expanded", "verified"],
                "adapted_index": (
                    "Each validation index row adds parent_job_id=job_id and "
                    "split=validation. The same validation gold/index pair is "
                    "used as source gold and evaluation gold."
                ),
                "formal_test_read": False,
            },
            "generation_failures": {
                "unique_method_job_failures": unique_missing,
                "materialized_rows_per_phase": unique_missing,
                "rows_materialized_across_three_phases": unique_missing * len(PHASES),
                "manifest_json": str(args.output / "missing_predictions.json"),
                "manifest_tsv": str(args.output / "missing_predictions.tsv"),
            },
            "materialization": materialization,
            "systems": systems,
            "comparisons": comparisons,
            "input_manifest": str(args.output / "input_manifest.json"),
            "quality_checks": {
                "status": "passed",
                "artifact": str(args.output / "checks.json"),
                "checks": quality_checks,
            },
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_seconds": round(time.monotonic() - started_clock, 3),
        }
        write_json(args.output / "summary.json", summary)
        atomic_text(
            args.output / "summary.tsv", build_summary_tsv(systems, comparisons)
        )
        status = {
            **base_status,
            "status": "complete",
            "finished_at": finished_at,
            "duration_seconds": summary["duration_seconds"],
            "datasets": list(DATASETS),
            "method_job_failures_materialized": unique_missing,
            "materialized_rows_across_three_phases": unique_missing * len(PHASES),
            "paired_iterations": args.iterations,
            "comparison_count": sum(len(value) for value in comparisons.values()),
            "artifacts": {
                "summary_json": str(args.output / "summary.json"),
                "summary_tsv": str(args.output / "summary.tsv"),
                "missing_predictions_json": str(
                    args.output / "missing_predictions.json"
                ),
                "input_manifest_json": str(args.output / "input_manifest.json"),
                "checks_json": str(args.output / "checks.json"),
            },
            "terminology_note": TERMINOLOGY_NOTE,
        }
        write_json(status_path, status)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "output": str(args.output),
                    "method_job_failures_materialized": unique_missing,
                    "iterations": args.iterations,
                    "duration_seconds": summary["duration_seconds"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0
    except Exception as error:
        write_json(
            status_path,
            {
                **base_status,
                "status": "failed",
                "finished_at": utc_now(),
                "duration_seconds": round(time.monotonic() - started_clock, 3),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
