#!/usr/bin/env python3
"""Derive formal GLiNER entity-only validation artifacts without inference.

The source artifacts are the already completed GLiNER + GLiREL validation
predictions.  This script copies the GLiNER entities byte-for-byte at the JSON
value level, unconditionally replaces every relation list with an empty list,
and evaluates only against the public validation gold files.  It neither loads
models nor accepts test-split inputs.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("conll04", "scierc", "ade")
METHOD = "GLiNER entity-only"
SOURCE_METHOD = "GLiNER + GLiREL zero-shot validation baseline"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_jsonl(path: Path, text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(row)
    return rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return parse_jsonl(path, path.read_text(encoding="utf-8"))


def read_jsonl_snapshot(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Read one immutable byte snapshot and return its parsed rows and hash."""

    payload = path.read_bytes()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path}: input is not UTF-8") from error
    return parse_jsonl(path, text), sha256_bytes(payload)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def require_validation_file(path: Path, expected_name: str, role: str) -> None:
    """Reject alternate split names and symlink redirection before any read."""

    if path.name != expected_name:
        raise ValueError(f"{role} must be named {expected_name!r}, got {path.name!r}")
    if path.is_symlink():
        raise ValueError(f"{role} must not be a symlink: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{role} does not exist: {path}")


def require_anchored_validation_file(
    path: Path,
    root: Path,
    relative_parent: Path,
    expected_name: str,
    role: str,
) -> None:
    """Require a regular file whose physical parent stays under its declared root."""

    require_validation_file(path, expected_name, role)
    root_real = root.resolve(strict=True)
    expected_parent = root_real / relative_parent
    # Comparing the resolved parent to the non-resolved child of root_real
    # deliberately rejects a symlinked dataset/source subdirectory.
    if path.parent.resolve(strict=True) != expected_parent:
        raise ValueError(f"{role} parent escapes its expected root: {path}")
    if path.resolve(strict=True) != expected_parent / expected_name:
        raise ValueError(f"{role} realpath is not the expected anchored file: {path}")


def validate_jobs(dataset: str, rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        raise ValueError(f"{dataset}: validation jobs are empty")
    expected_prefix = f"{dataset}_validation_"
    job_ids: list[str] = []
    for position, row in enumerate(rows):
        job_id = row.get("job_id")
        document_id = row.get("document_id")
        if not isinstance(job_id, str) or not job_id.startswith(expected_prefix):
            raise ValueError(f"{dataset}: job row {position} is not a validation job")
        if not isinstance(document_id, str) or not document_id.startswith(expected_prefix):
            raise ValueError(f"{dataset}: job {job_id!r} is not a validation document")
        if not job_id.startswith(f"{document_id}_C"):
            raise ValueError(f"{dataset}: job {job_id!r} does not belong to {document_id!r}")
        if row.get("category") not in (None, dataset):
            raise ValueError(f"{dataset}: job {job_id!r} has the wrong category")
        job_ids.append(job_id)
    if len(job_ids) != len(set(job_ids)):
        raise ValueError(f"{dataset}: validation jobs contain duplicate job IDs")
    return job_ids


def validate_frozen_validation_alignment(
    dataset: str,
    jobs: list[dict[str, Any]],
    gold_index: list[dict[str, Any]],
    gold: list[dict[str, Any]],
) -> dict[str, Any]:
    """Prove that jobs, index rows, gold rows, and documents are one-to-one.

    This preflight is intentionally stricter than the legacy evaluator.  In
    particular, the index must cover the frozen job order and the complete
    contiguous gold file, so an unindexed extra row (including a test canary)
    cannot silently enter an evaluation input.
    """

    job_ids = validate_jobs(dataset, jobs)
    job_documents = [row["document_id"] for row in jobs]
    if len(job_documents) != len(set(job_documents)):
        raise ValueError(f"{dataset}: validation jobs contain duplicate documents")
    if len(gold_index) != len(job_ids):
        raise ValueError(
            f"{dataset}: validation index/job row count mismatch "
            f"({len(gold_index)} != {len(job_ids)})"
        )
    if len(gold) != len(job_ids):
        raise ValueError(
            f"{dataset}: validation gold/job row count mismatch "
            f"({len(gold)} != {len(job_ids)}); refusing extra or missing gold rows"
        )

    indexed_jobs: list[str] = []
    record_indices: list[int] = []
    for position, item in enumerate(gold_index):
        job_id = item.get("job_id")
        record_index = item.get("record_index")
        if not isinstance(job_id, str):
            raise ValueError(f"{dataset}: validation index row {position} has no job_id")
        if not isinstance(record_index, int) or isinstance(record_index, bool):
            raise ValueError(f"{dataset}: validation index row {position} has invalid record_index")
        indexed_jobs.append(job_id)
        record_indices.append(record_index)
        if item.get("document_id") not in (None, jobs[position]["document_id"]):
            raise ValueError(f"{dataset}: validation index row {position} has wrong document_id")
    if len(indexed_jobs) != len(set(indexed_jobs)):
        raise ValueError(f"{dataset}: validation index contains duplicate job IDs")
    if len(record_indices) != len(set(record_indices)):
        raise ValueError(f"{dataset}: validation index contains duplicate gold record indices")
    if indexed_jobs != job_ids:
        raise ValueError(f"{dataset}: validation index is not in frozen validation job order")
    if record_indices != list(range(len(gold))):
        raise ValueError(f"{dataset}: validation index does not exactly cover the frozen gold file")

    expected_prefix = f"{dataset}_validation_"
    gold_documents: list[str] = []
    for position, annotation in enumerate(gold):
        document_id = annotation.get("document_id")
        if not isinstance(document_id, str) or not document_id.startswith(expected_prefix):
            raise ValueError(
                f"{dataset}: gold row {position} is not a frozen validation document"
            )
        gold_documents.append(document_id)
    if len(gold_documents) != len(set(gold_documents)):
        raise ValueError(f"{dataset}: validation gold contains duplicate documents")
    for position, document_id in enumerate(gold_documents):
        expected_document = jobs[position]["document_id"]
        if document_id != expected_document:
            raise ValueError(
                f"{dataset}: gold/index/job document mismatch at row {position}: "
                f"{document_id!r} != {expected_document!r}"
            )
    if set(gold_documents) != set(job_documents):
        raise ValueError(f"{dataset}: validation gold and jobs have different document sets")

    return {
        "status": "proved-one-to-one",
        "jobs": len(job_ids),
        "documents": len(job_documents),
        "gold_rows": len(gold),
        "index_rows": len(gold_index),
        "frozen_order_verified": True,
        "unique_job_ids": True,
        "unique_record_indices": True,
        "unique_documents": True,
        "extra_gold_rows": 0,
        "job_ids_sha256": canonical_sha256(job_ids),
        "document_ids_sha256": canonical_sha256(job_documents),
    }


def project_predictions(
    dataset: str,
    jobs: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Project a complete validation artifact onto its entity-only component."""

    job_ids = validate_jobs(dataset, jobs)
    job_by_id = {row["job_id"]: row for row in jobs}
    source_by_id: dict[str, dict[str, Any]] = {}
    source_order: list[str] = []
    source_entity_payload: list[dict[str, Any]] = []
    relations_discarded = 0

    for position, row in enumerate(source_rows):
        job_id = row.get("job_id")
        if not isinstance(job_id, str):
            raise ValueError(f"{dataset}: source row {position} has no string job_id")
        if job_id in source_by_id:
            raise ValueError(f"{dataset}: duplicate source prediction {job_id!r}")
        if job_id not in job_by_id:
            raise ValueError(f"{dataset}: source prediction {job_id!r} is outside validation jobs")
        annotation = row.get("annotation")
        if not isinstance(annotation, dict):
            raise ValueError(f"{dataset}: source prediction {job_id!r} has no annotation")
        if annotation.get("document_id") != job_by_id[job_id].get("document_id"):
            raise ValueError(f"{dataset}: source prediction {job_id!r} has the wrong document_id")
        entities = annotation.get("entities")
        relations = annotation.get("relations")
        if not isinstance(entities, list) or not all(isinstance(item, dict) for item in entities):
            raise ValueError(f"{dataset}: source prediction {job_id!r} has invalid entities")
        if not isinstance(relations, list) or not all(isinstance(item, dict) for item in relations):
            raise ValueError(f"{dataset}: source prediction {job_id!r} has invalid relations")
        source_by_id[job_id] = row
        source_order.append(job_id)
        source_entity_payload.append({"job_id": job_id, "entities": entities})
        relations_discarded += len(relations)

    missing = [job_id for job_id in job_ids if job_id not in source_by_id]
    if missing:
        raise ValueError(
            f"{dataset}: refusing incomplete source; {len(missing)} validation jobs are missing, "
            f"first={missing[:3]}"
        )
    if source_order != job_ids:
        raise ValueError(f"{dataset}: source predictions are not in frozen validation job order")

    projected: list[dict[str, Any]] = []
    projected_entity_payload: list[dict[str, Any]] = []
    for job_id in job_ids:
        source_annotation = source_by_id[job_id]["annotation"]
        annotation = copy.deepcopy(source_annotation)
        annotation["relations"] = []
        annotation["review"] = {
            "status": "unreviewed",
            "reviewers": [],
            "notes": (
                "Formal GLiNER entity-only validation projection; entities are unchanged from "
                "the frozen source artifact and all GLiREL relations are explicitly discarded."
            ),
        }
        projected.append({"job_id": job_id, "annotation": annotation})
        projected_entity_payload.append({"job_id": job_id, "entities": annotation["entities"]})

    source_entity_hash = canonical_sha256(source_entity_payload)
    projected_entity_hash = canonical_sha256(projected_entity_payload)
    if source_entity_hash != projected_entity_hash:
        raise AssertionError("entity-only projection changed the source entities")
    if any(row["annotation"]["relations"] for row in projected):
        raise AssertionError("entity-only projection retained relation predictions")

    return projected, {
        "jobs": len(projected),
        "entities_retained": sum(len(row["annotation"]["entities"]) for row in projected),
        "relations_in_source": relations_discarded,
        "relations_discarded": relations_discarded,
        "relations_in_output": 0,
        "source_entity_payload_sha256": source_entity_hash,
        "output_entity_payload_sha256": projected_entity_hash,
        "entities_unchanged": True,
        "relations_explicitly_discarded": True,
    }


def load_span_evaluator() -> Any:
    path = Path(__file__).with_name("evaluate_public_validation_spans.py")
    spec = importlib.util.spec_from_file_location("gliner_entity_only_span_evaluator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load strict span evaluator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate_strict_spans(
    gold: Path,
    gold_index: Path,
    predictions: Path,
    jobs: Path,
    output: Path,
) -> dict[str, Any]:
    evaluator = load_span_evaluator()
    temporary = output.with_name(f".{output.name}.evaluating")
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result = evaluator.run(
                SimpleNamespace(
                    gold=gold,
                    gold_index=gold_index,
                    predictions=predictions,
                    jobs=jobs,
                    output=temporary,
                )
            )
        if result != 0:
            raise RuntimeError(f"strict span evaluator returned {result}")
        metrics = json.loads(temporary.read_text(encoding="utf-8"))
        if metrics.get("metric") != "strict-source-character-span":
            raise ValueError("strict span evaluator produced an unexpected metric")
        if metrics.get("selection_split") != "validation":
            raise ValueError("strict span evaluator did not report validation selection_split")
        os.replace(temporary, output)
        return metrics
    finally:
        temporary.unlink(missing_ok=True)


def file_record(path: Path, digest: str | None = None) -> dict[str, str]:
    resolved = path.resolve(strict=digest is None)
    try:
        display = str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        display = str(resolved)
    return {
        "path": display,
        "sha256": digest if digest is not None else sha256_file(resolved),
    }


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def valid_entity_only_metrics(metrics: dict[str, Any], jobs: int) -> bool:
    if (
        metrics.get("metric") != "strict-source-character-span"
        or metrics.get("selection_split") != "validation"
        or metrics.get("jobs_gold") != jobs
        or metrics.get("jobs_predicted") != jobs
        or metrics.get("jobs_evaluated") != jobs
        or metrics.get("jobs_missing_predictions") != 0
        or metrics.get("generation_success_rate") != 1.0
    ):
        return False
    entity = metrics.get("entity_strict")
    relation = metrics.get("relation_strict")
    return (
        isinstance(entity, dict)
        and isinstance(relation, dict)
        and relation.get("predicted") == 0
        and isinstance(metrics.get("per_job"), dict)
        and len(metrics["per_job"]) == jobs
    )


def current_derivation(
    dataset: str,
    output: Path,
    metrics_output: Path,
    lineage_output: Path,
    projected: list[dict[str, Any]],
    projection: dict[str, Any],
    alignment: dict[str, Any],
    input_records: dict[str, dict[str, str]],
) -> dict[str, Any] | None:
    """Return a verified same-input derivation, otherwise fail closed."""

    artifacts = (output, metrics_output, lineage_output)
    if not all(path.is_file() and not path.is_symlink() for path in artifacts):
        return None
    try:
        predictions = read_jsonl(output)
        metrics = load_json_object(metrics_output)
        lineage = load_json_object(lineage_output)
    except (OSError, ValueError):
        return None
    if predictions != projected or not valid_entity_only_metrics(metrics, projection["jobs"]):
        return None
    expected_outputs = {
        "predictions": file_record(output),
        "strict_character_span_metrics": file_record(metrics_output),
    }
    if (
        lineage.get("status") != "complete"
        or lineage.get("method") != METHOD
        or lineage.get("scope") != "entity-only"
        or lineage.get("dataset") != dataset
        or lineage.get("split") != "validation"
        or lineage.get("inputs") != input_records
        or lineage.get("outputs") != expected_outputs
        or lineage.get("headline_metric") != {"entity_strict": metrics["entity_strict"]}
        or lineage.get("relation_metrics_interpretation")
        != "not-applicable-entity-only-output"
    ):
        return None
    derivation = lineage.get("derivation")
    if not isinstance(derivation, dict):
        return None
    required_derivation = {
        "operation": "copy-entities-drop-relations",
        "source_method": SOURCE_METHOD,
        "entity_inference_reused": True,
        "entity_inference_rerun": False,
        "gpu_used": False,
        "gold_used_for_selection": False,
        "gold_used_for_evaluation_only": True,
        "test_split_access": "forbidden-and-not-read",
        **projection,
        "validation_alignment": alignment,
    }
    if derivation != required_derivation:
        return None
    result = copy.deepcopy(lineage)
    result["outputs"]["lineage"] = file_record(lineage_output)
    return result


def derive_dataset(
    dataset: str,
    data_root: Path,
    source_root: Path,
    output_root: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    if dataset not in DATASETS:
        raise ValueError(f"unsupported dataset {dataset!r}")
    dataset_root = data_root / dataset
    source = source_root / f"{dataset}_validation.jsonl"
    jobs = dataset_root / "validation_baseline_jobs.jsonl"
    gold = dataset_root / "validation_gold.jsonl"
    gold_index = dataset_root / "validation_index.jsonl"
    output = output_root / f"{dataset}_validation.jsonl"
    metrics_output = output_root / f"{dataset}_validation.character_span_metrics.json"
    lineage_output = output_root / f"{dataset}_validation.lineage.json"

    require_anchored_validation_file(
        source,
        source_root,
        Path(),
        f"{dataset}_validation.jsonl",
        "source predictions",
    )
    require_anchored_validation_file(
        jobs,
        data_root,
        Path(dataset),
        "validation_baseline_jobs.jsonl",
        "validation jobs",
    )
    require_anchored_validation_file(
        gold,
        data_root,
        Path(dataset),
        "validation_gold.jsonl",
        "validation gold",
    )
    require_anchored_validation_file(
        gold_index,
        data_root,
        Path(dataset),
        "validation_index.jsonl",
        "validation gold index",
    )
    if output.resolve(strict=False) == source.resolve(strict=True):
        raise ValueError("entity-only output must be independent from the source prediction artifact")

    # Gold is consumed only as a byte snapshot after the physical paths have
    # been anchored.  Its complete one-to-one relationship with the frozen
    # jobs/index is proved before any prediction or metric output is written.
    job_rows, jobs_hash = read_jsonl_snapshot(jobs)
    gold_index_rows, gold_index_hash = read_jsonl_snapshot(gold_index)
    gold_rows, gold_hash = read_jsonl_snapshot(gold)
    alignment = validate_frozen_validation_alignment(
        dataset, job_rows, gold_index_rows, gold_rows
    )
    source_rows, source_hash = read_jsonl_snapshot(source)
    projected, projection = project_predictions(dataset, job_rows, source_rows)
    input_records = {
        "source_predictions": file_record(source, source_hash),
        "validation_jobs": file_record(jobs, jobs_hash),
        "validation_gold": file_record(gold, gold_hash),
        "validation_gold_index": file_record(gold_index, gold_index_hash),
    }

    existing = [path for path in (output, metrics_output, lineage_output) if path.exists()]
    if existing:
        current = current_derivation(
            dataset,
            output,
            metrics_output,
            lineage_output,
            projected,
            projection,
            alignment,
            input_records,
        )
        if current is not None:
            return current
        if not overwrite:
            raise FileExistsError(
                "existing entity-only artifacts are partial, stale, or invalid; "
                f"rerun with --overwrite to repair: {existing}"
            )

    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_root, prefix=f".{dataset}.entity-only.staging."
    ) as staging_directory:
        staging = Path(staging_directory)
        staged_jobs = staging / "validation_baseline_jobs.jsonl"
        staged_gold = staging / "validation_gold.jsonl"
        staged_gold_index = staging / "validation_index.jsonl"
        staged_output = staging / output.name
        staged_metrics = staging / metrics_output.name
        staged_lineage = staging / lineage_output.name
        atomic_write_jsonl(staged_jobs, job_rows)
        atomic_write_jsonl(staged_gold, gold_rows)
        atomic_write_jsonl(staged_gold_index, gold_index_rows)
        atomic_write_jsonl(staged_output, projected)
        metrics = evaluate_strict_spans(
            staged_gold,
            staged_gold_index,
            staged_output,
            staged_jobs,
            staged_metrics,
        )

        if not valid_entity_only_metrics(metrics, projection["jobs"]):
            raise ValueError(f"{dataset}: strict metrics do not cover the complete validation split")
        if metrics["relation_strict"]["predicted"] != 0:
            raise AssertionError(
                f"{dataset}: strict metrics found relation predictions in entity-only output"
            )

        lineage: dict[str, Any] = {
            "status": "complete",
            "method": METHOD,
            "scope": "entity-only",
            "dataset": dataset,
            "split": "validation",
            "created_at": utc_now(),
            "derivation": {
                "operation": "copy-entities-drop-relations",
                "source_method": SOURCE_METHOD,
                "entity_inference_reused": True,
                "entity_inference_rerun": False,
                "gpu_used": False,
                "gold_used_for_selection": False,
                "gold_used_for_evaluation_only": True,
                "test_split_access": "forbidden-and-not-read",
                **projection,
                "validation_alignment": alignment,
            },
            "inputs": input_records,
            "outputs": {
                "predictions": file_record(output, sha256_file(staged_output)),
                "strict_character_span_metrics": file_record(
                    metrics_output, sha256_file(staged_metrics)
                ),
            },
            "headline_metric": {"entity_strict": metrics["entity_strict"]},
            "relation_metrics_interpretation": "not-applicable-entity-only-output",
        }
        atomic_write_json(staged_lineage, lineage)

        # Commit the lineage last.  A crash can leave replaceable data files,
        # but can never leave a complete lineage pointing at mixed versions.
        os.replace(staged_output, output)
        os.replace(staged_metrics, metrics_output)
        os.replace(staged_lineage, lineage_output)

    lineage["outputs"]["lineage"] = file_record(lineage_output)
    return lineage


def derive_all(
    data_root: Path,
    source_root: Path,
    output_root: Path,
    datasets: tuple[str, ...] = DATASETS,
    overwrite: bool = False,
) -> dict[str, Any]:
    unknown = sorted(set(datasets) - set(DATASETS))
    if unknown:
        raise ValueError(f"unsupported datasets: {unknown}")
    if not datasets:
        raise ValueError("at least one dataset is required")
    if len(datasets) != len(set(datasets)):
        raise ValueError("datasets must be unique")

    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "status.json"
    status: dict[str, Any] = {
        "status": "running",
        "method": METHOD,
        "scope": "entity-only",
        "split": "validation",
        "datasets_requested": list(datasets),
        "datasets": {},
        "execution": {
            "postprocessing_only": True,
            "gpu_used": False,
            "inference_rerun": False,
            "test_split_access": "forbidden-and-not-read",
        },
        "started_at": utc_now(),
        "updated_at": utc_now(),
    }
    atomic_write_json(status_path, status)
    try:
        for dataset in datasets:
            lineage = derive_dataset(
                dataset, data_root, source_root, output_root, overwrite=overwrite
            )
            status["datasets"][dataset] = {
                "status": "complete",
                "jobs": lineage["derivation"]["jobs"],
                "entities": lineage["derivation"]["entities_retained"],
                "relations": 0,
                "entity_strict": lineage["headline_metric"]["entity_strict"],
                "predictions": lineage["outputs"]["predictions"]["path"],
                "strict_character_span_metrics": lineage["outputs"][
                    "strict_character_span_metrics"
                ]["path"],
                "lineage": str(output_root / f"{dataset}_validation.lineage.json"),
                "lineage_sha256": lineage["outputs"]["lineage"]["sha256"],
                "validation_alignment": lineage["derivation"]["validation_alignment"],
            }
            status["updated_at"] = utc_now()
            atomic_write_json(status_path, status)
    except Exception as error:
        status["status"] = "failed"
        status["error_type"] = type(error).__name__
        status["error"] = str(error)
        status["updated_at"] = utc_now()
        atomic_write_json(status_path, status)
        raise

    status["status"] = "complete"
    status["finished_at"] = utc_now()
    status["updated_at"] = status["finished_at"]
    atomic_write_json(status_path, status)
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/processed/public_benchmarks_full"),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("outputs/public_horizontal_validation/gliner_glirel"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/public_horizontal_validation/gliner_entity_only"),
    )
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = derive_all(
        args.data_root,
        args.source_root,
        args.output_root,
        tuple(args.datasets),
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
