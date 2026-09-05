#!/usr/bin/env python3
"""Prepare leakage-safe public EAE and HRGE jobs on CPU.

Only ``train`` and ``validation`` source jobs are opened.  The converter first
materializes a physically train-only graph, EAE is rebuilt with balanced exact
anchors, and HRGE is built with the manuscript's frozen 12/6/4 context caps and
BGE-M3 thresholds.  The resulting jobs are inputs to two separately trained
generators; PGE can only be composed after both prediction streams exist.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_experiment_jobs as eae_builder  # noqa: E402
import build_kg_v2_jobs as hrge_builder  # noqa: E402
from convert_public_train_graph import (  # noqa: E402
    convert_paths,
    load_jsonl,
    sha256,
    write_json_atomic,
    write_jsonl_atomic,
)


DATASETS = ("conll04", "scierc", "ade")
PREPARED_SPLITS = ("train", "validation")
DEFAULT_BGE_M3 = Path(
    "/ds2/xuelin/cache/huggingface/hub/"
    "models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"
)
SETTINGS = {
    "eae_concept_limit": 20,
    "eae_balanced_concepts": True,
    "hrge_anchor_limit": 12,
    "hrge_anchor_per_type": 2,
    "hrge_edge_limit": 6,
    "hrge_min_type_purity": 0.8,
    "hrge_min_en_chars": 4,
    "hrge_min_zh_chars": 2,
    "hrge_semantic_threshold": 0.72,
    "hrge_semantic_limit": 4,
    "exact_train_validation_text_quarantine": True,
}


def stable_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def model_identity(model_path: Path | None) -> dict[str, Any] | None:
    if model_path is None:
        return None
    required = (
        "config.json",
        "modules.json",
        "pytorch_model.bin",
        "sentencepiece.bpe.model",
        "tokenizer.json",
    )
    missing = [name for name in required if not (model_path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"semantic model snapshot is incomplete at {model_path}: missing {missing}"
        )
    files = {}
    for name in required:
        path = model_path / name
        resolved = path.resolve()
        files[name] = {
            "blob": resolved.name,
            "size": resolved.stat().st_size,
        }
    return {"path": str(model_path), "revision": model_path.name, "files": files}


def source_inputs(dataset_root: Path) -> dict[str, Path]:
    paths = {
        "train_jobs": dataset_root / "train_baseline_jobs.jsonl",
        "validation_jobs": dataset_root / "validation_baseline_jobs.jsonl",
        "mentions": dataset_root / "knowledge_graph" / "mentions.jsonl",
        "training_edges": dataset_root / "knowledge_graph" / "training_edges.jsonl",
        "split_manifest": dataset_root / "split_manifest.jsonl",
        "ontology": dataset_root / "ontology.yaml",
        "eae_builder": SCRIPT_DIR / "build_experiment_jobs.py",
        "hrge_builder": SCRIPT_DIR / "build_kg_v2_jobs.py",
        "graph_converter": SCRIPT_DIR / "convert_public_train_graph.py",
        "preparation_runner": Path(__file__).resolve(),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required public preparation inputs are missing: {missing}")
    if any("test" in name for name in ("train_jobs", "validation_jobs")):
        raise AssertionError("internal split input labels unexpectedly include test")
    return paths


def source_fingerprint(
    dataset: str,
    inputs: dict[str, Path],
    semantic_model: Path | None,
    batch_size: int,
) -> tuple[str, dict[str, Any]]:
    identities = {
        name: {"path": str(path), "sha256": sha256(path)}
        for name, path in sorted(inputs.items())
    }
    payload = {
        "version": "public-eae-hrge-cpu-v1",
        "dataset": dataset,
        "prepared_splits": PREPARED_SPLITS,
        "settings": SETTINGS,
        "batch_size": batch_size,
        "semantic_model": model_identity(semantic_model),
        "inputs": identities,
    }
    return stable_digest(payload), payload


def existing_outputs_are_valid(manifest_path: Path, fingerprint: str) -> bool:
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if manifest.get("fingerprint") != fingerprint:
        return False
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        return False
    for item in outputs.values():
        if not isinstance(item, dict):
            return False
        path = Path(str(item.get("path", "")))
        expected = str(item.get("sha256", ""))
        if not path.is_file() or not expected or sha256(path) != expected:
            return False
    return True


def validated_source_jobs(
    path: Path,
    split: str,
    assignments: dict[str, str],
) -> list[dict[str, Any]]:
    if split not in PREPARED_SPLITS:
        raise ValueError(f"refusing unapproved preparation split {split!r}")
    rows = load_jsonl(path)
    seen = set()
    for row in rows:
        job_id = str(row.get("job_id", ""))
        document_id = str(row.get("document_id", ""))
        if not job_id or job_id in seen:
            raise ValueError(f"{path} contains a missing or duplicate job_id: {job_id!r}")
        seen.add(job_id)
        if assignments.get(document_id) != split:
            raise ValueError(
                f"{path} contains {document_id!r}, which is not assigned to {split}"
            )
        source_path = str(row.get("source_path", ""))
        if f":{split}:" not in source_path:
            raise ValueError(f"{path} job {job_id} has inconsistent source_path {source_path!r}")
    return rows


def normalized_job_text(job: dict[str, Any]) -> str:
    text = "\n".join(str(segment.get("text", "")) for segment in job.get("segments", []))
    normalized = re.sub(r"\s+", " ", text).strip().casefold()
    if not normalized:
        raise ValueError(f"job {job.get('job_id')!r} has no source text")
    return normalized


def exact_validation_overlap_quarantine(
    split_jobs: dict[str, list[dict[str, Any]]]
) -> dict[str, list[str]]:
    training_documents_by_text: dict[str, set[str]] = {}
    for job in split_jobs["train"]:
        training_documents_by_text.setdefault(normalized_job_text(job), set()).add(
            str(job["document_id"])
        )
    return {
        str(job["job_id"]): sorted(training_documents_by_text[normalized_job_text(job)])
        for job in split_jobs["validation"]
        if normalized_job_text(job) in training_documents_by_text
    }


def tag_eae_jobs(
    path: Path,
    expected_ids: set[str],
    baseline_by_id: dict[str, dict[str, Any]],
    quarantine: dict[str, list[str]],
) -> None:
    rows = load_jsonl(path)
    actual_ids = {str(row.get("job_id", "")) for row in rows}
    if actual_ids != expected_ids or len(rows) != len(expected_ids):
        raise ValueError(f"EAE builder output at {path} does not exactly match source jobs")
    for row in rows:
        job_id = str(row["job_id"])
        row["experiment_mode"] = "eae_exact_anchor"
        row["method_name"] = "EAE"
        row["train_graph_only"] = True
        if job_id in quarantine:
            row["system_instruction"] = baseline_by_id[job_id]["system_instruction"]
            row["graph_context_quarantine"] = {
                "reason": "exact_source_text_occurs_in_training_split",
                "excluded_training_documents": quarantine[job_id],
            }
    write_jsonl_atomic(path, rows)


def quarantine_hrge_jobs(
    path: Path,
    quarantine: dict[str, list[str]],
) -> dict[str, Any]:
    rows = load_jsonl(path)
    for row in rows:
        job_id = str(row.get("job_id", ""))
        if job_id not in quarantine:
            continue
        base_instruction = str(row.get("system_instruction", "")).split(
            "\n\nKG_RULES:", 1
        )[0].rstrip()
        row["system_instruction"] = (
            f"{base_instruction}\n\n{hrge_builder.render_context([], [], [])}"
        )
        context = row.get("kg_v2_context", {})
        context["anchors"] = []
        context["edge_priors"] = []
        context["semantic_relation_patterns"] = []
        context["exact_source_overlap_quarantine"] = {
            "reason": "exact_source_text_occurs_in_training_split",
            "excluded_training_documents": quarantine[job_id],
        }
        row["kg_v2_context"] = context
    write_jsonl_atomic(path, rows)
    return {
        "policy": "clear_all_graph_context_for_exact_train_validation_source_overlap",
        "quarantined_validation_jobs": len(quarantine),
        "validation_job_ids": sorted(quarantine),
        "excluded_training_documents_by_validation_job": dict(sorted(quarantine.items())),
        "remaining_limitation": (
            "The benchmark generator still trains on the official training split; "
            "content-identical validation rows therefore require a sensitivity analysis."
        ),
    }


def split_hrge_jobs(
    combined_path: Path,
    split_jobs: dict[str, list[dict[str, Any]]],
    output_paths: dict[str, Path],
) -> None:
    rows = load_jsonl(combined_path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        job_id = str(row.get("job_id", ""))
        if not job_id or job_id in by_id:
            raise ValueError(f"HRGE builder emitted a missing or duplicate job_id: {job_id!r}")
        context = row.get("kg_v2_context", {})
        if (
            context.get("train_graph_only") is not True
            or context.get("leave_current_document_out") is not True
        ):
            raise ValueError(f"HRGE builder omitted leakage guards for {job_id}")
        row["method_name"] = "HRGE"
        by_id[job_id] = row

    expected_all = {
        str(row["job_id"])
        for split in PREPARED_SPLITS
        for row in split_jobs[split]
    }
    if set(by_id) != expected_all:
        raise ValueError("HRGE combined output does not exactly match train+validation source jobs")
    for split in PREPARED_SPLITS:
        ordered = [by_id[str(source["job_id"])] for source in split_jobs[split]]
        write_jsonl_atomic(output_paths[split], ordered)


def configure_cpu(threads: int) -> None:
    if threads < 1:
        raise ValueError("threads must be positive")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    try:
        import torch

        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(max(1, min(4, threads)))
        except RuntimeError:
            pass
    except ImportError:
        if threads:
            raise RuntimeError("PyTorch is required for semantic CPU preparation")


def prepare_dataset(
    dataset: str,
    source_root: Path,
    output_root: Path,
    semantic_model: Path | None,
    batch_size: int,
    force: bool = False,
) -> dict[str, Any]:
    if dataset not in DATASETS:
        raise ValueError(f"unsupported public dataset {dataset!r}")
    dataset_root = source_root / dataset
    target = output_root / dataset
    graph_dir = target / "knowledge_graph"
    jobs_dir = target / "jobs"
    audits_dir = target / "audits"
    manifest_path = target / "preparation_manifest.json"
    inputs = source_inputs(dataset_root)
    fingerprint, fingerprint_payload = source_fingerprint(
        dataset, inputs, semantic_model, batch_size
    )
    if not force and existing_outputs_are_valid(manifest_path, fingerprint):
        return {
            "dataset": dataset,
            "status": "skipped_unchanged",
            "fingerprint": fingerprint,
            "manifest": str(manifest_path),
        }

    assignments = {
        str(row["document_id"]): str(row["split"])
        for row in load_jsonl(inputs["split_manifest"])
    }
    split_jobs = {
        split: validated_source_jobs(inputs[f"{split}_jobs"], split, assignments)
        for split in PREPARED_SPLITS
    }
    if set(row["document_id"] for row in split_jobs["train"]) & set(
        row["document_id"] for row in split_jobs["validation"]
    ):
        raise ValueError(f"{dataset} train and validation jobs overlap by document")
    quarantine = exact_validation_overlap_quarantine(split_jobs)

    conversion_audit = convert_paths(
        dataset,
        inputs["mentions"],
        inputs["training_edges"],
        inputs["split_manifest"],
        graph_dir,
        "en",
    )

    eae_paths = {}
    for split in PREPARED_SPLITS:
        output = jobs_dir / f"{split}_eae_jobs.jsonl"
        eae_builder.run(
            SimpleNamespace(
                jobs=inputs[f"{split}_jobs"],
                manifest=inputs["split_manifest"],
                mentions=graph_dir / "mentions.jsonl",
                ontology=inputs["ontology"],
                split=split,
                mode="kg_constrained",
                concept_limit=SETTINGS["eae_concept_limit"],
                balanced_concepts=SETTINGS["eae_balanced_concepts"],
                output=output,
            )
        )
        tag_eae_jobs(
            output,
            {str(row["job_id"]) for row in split_jobs[split]},
            {str(row["job_id"]): row for row in split_jobs[split]},
            quarantine,
        )
        eae_paths[split] = output

    combined_source = jobs_dir / "train_validation_source_jobs.jsonl"
    combined_hrge = jobs_dir / "train_validation_hrge_jobs.jsonl"
    hrge_audit = audits_dir / "train_validation_hrge_context.json"
    write_jsonl_atomic(
        combined_source,
        [row for split in PREPARED_SPLITS for row in split_jobs[split]],
    )
    hrge_builder.run(
        SimpleNamespace(
            jobs=combined_source,
            concepts=graph_dir / "concepts.jsonl",
            relations=graph_dir / "relations.jsonl",
            output=combined_hrge,
            audit=hrge_audit,
            semantic_model=semantic_model,
            device="cpu",
            batch_size=batch_size,
            semantic_threshold=SETTINGS["hrge_semantic_threshold"],
            semantic_limit=SETTINGS["hrge_semantic_limit"],
            anchor_limit=SETTINGS["hrge_anchor_limit"],
            anchor_per_type=SETTINGS["hrge_anchor_per_type"],
            edge_limit=SETTINGS["hrge_edge_limit"],
            min_type_purity=SETTINGS["hrge_min_type_purity"],
            min_en_chars=SETTINGS["hrge_min_en_chars"],
            min_zh_chars=SETTINGS["hrge_min_zh_chars"],
        )
    )
    overlap_audit = quarantine_hrge_jobs(combined_hrge, quarantine)
    overlap_audit_path = audits_dir / "exact_source_overlap_quarantine.json"
    write_json_atomic(overlap_audit_path, overlap_audit)
    hrge_paths = {
        split: jobs_dir / f"{split}_hrge_jobs.jsonl" for split in PREPARED_SPLITS
    }
    split_hrge_jobs(combined_hrge, split_jobs, hrge_paths)

    output_paths = {
        "train_only_concepts": graph_dir / "concepts.jsonl",
        "train_only_mentions": graph_dir / "mentions.jsonl",
        "train_only_relations": graph_dir / "relations.jsonl",
        "graph_conversion_audit": graph_dir / "conversion_audit.json",
        "train_eae_jobs": eae_paths["train"],
        "validation_eae_jobs": eae_paths["validation"],
        "train_hrge_jobs": hrge_paths["train"],
        "validation_hrge_jobs": hrge_paths["validation"],
        "train_validation_source_jobs": combined_source,
        "train_validation_hrge_jobs": combined_hrge,
        "hrge_builder_audit_before_overlap_quarantine": hrge_audit,
        "exact_source_overlap_quarantine_audit": overlap_audit_path,
    }
    manifest = {
        "version": "public-eae-hrge-cpu-v1",
        "dataset": dataset,
        "status": "prepared_train_and_validation",
        "fingerprint": fingerprint,
        "prepared_splits": list(PREPARED_SPLITS),
        "test_job_file_read": False,
        "test_gold_read": False,
        "validation_gold_read": False,
        "semantic_device": "cpu",
        "settings": SETTINGS,
        "fingerprint_inputs": fingerprint_payload,
        "graph_audit": conversion_audit,
        "exact_source_overlap_audit": overlap_audit,
        "job_counts": {split: len(rows) for split, rows in split_jobs.items()},
        "outputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in output_paths.items()
        },
        "pge_contract": {
            "ready_for_generator_training": True,
            "pge_predictions_available": False,
            "required_prediction_streams": ["EAE", "HRGE"],
            "post_prediction_steps": [
                "expand compact EAE and HRGE predictions against their own jobs",
                "verify HRGE relations with verify_relations.py and local co-occurrence",
                "apply fuse_kg_v1_v2_predictions.py with relation_mode=verified",
            ],
            "entity_gate_any_of": [
                "EAE/HRGE exact normalized text-and-type agreement",
                "source-gated HRGE anchor type match",
                "verified HRGE relation endpoint",
            ],
        },
    }
    write_json_atomic(manifest_path, manifest)
    gc.collect()
    return {
        "dataset": dataset,
        "status": manifest["status"],
        "fingerprint": fingerprint,
        "manifest": str(manifest_path),
        "job_counts": manifest["job_counts"],
        "graph_counts": {
            key: conversion_audit[key] for key in ("concepts", "mentions", "relations")
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=list(DATASETS)
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/processed/public_benchmarks_full"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/public_benchmarks_hrge_v1"),
    )
    parser.add_argument("--semantic-model", type=Path, default=DEFAULT_BGE_M3)
    parser.add_argument(
        "--disable-semantic",
        action="store_true",
        help="Testing only: prepare exact/edge HRGE context without semantic patterns.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    semantic_model = None if args.disable_semantic else args.semantic_model
    configure_cpu(args.threads)
    summaries = []
    for dataset in args.datasets:
        summary = prepare_dataset(
            dataset,
            args.source_root,
            args.output_root,
            semantic_model,
            args.batch_size,
            args.force,
        )
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    print(
        json.dumps(
            {"status": "complete", "semantic_device": "cpu", "datasets": summaries},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
