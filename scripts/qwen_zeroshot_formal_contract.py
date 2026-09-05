#!/usr/bin/env python3
"""Validate and report the Qwen3-4B zero-shot formal-test run contract.

The formal runner uses this module for four deliberately small operations:

* verify that the audited release gate is complete;
* validate the immutable public test inputs before inference;
* create the explicit test-span index required by the span evaluator; and
* validate/report resumable output coverage without trusting file existence.

No command in this module discovers test data.  All paths are derived from the
explicit data/run roots and the three frozen dataset identifiers below.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATASETS = ("conll04", "scierc", "ade")
EXPECTED_JOBS = {"conll04": 288, "scierc": 551, "ade": 427}
SCHEMA_VERSION = "qwen3-4b-zero-shot-formal-test-v1"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
TERMINAL_STATUSES = {"success", "failed"}
SCRIPT_DIR = Path(__file__).resolve().parent
PREPARATION_SCHEMA_VERSION = "public-formal-test-inputs-v2"


class ContractError(ValueError):
    """Raised when an input or output violates the frozen run contract."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ContractError(f"required JSON file is missing: {path}") from error
    except json.JSONDecodeError as error:
        raise ContractError(f"{path}: invalid JSON: {error.msg}") from error
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected one JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise ContractError(f"required JSONL file is missing: {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ContractError(
                f"{path}:{line_number}: invalid JSON: {error.msg}"
            ) from error
        if not isinstance(row, dict):
            raise ContractError(f"{path}:{line_number}: expected one JSON object")
        rows.append(row)
    return rows


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
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
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def artifact_paths(data_root: Path, run_root: Path, dataset: str) -> dict[str, Path]:
    base = data_root / dataset
    prefix = run_root / f"{dataset}_test"
    return {
        "jobs": base / "test_baseline_jobs.jsonl",
        "gold": base / "test_gold.jsonl",
        "gold_index": base / "test_index.jsonl",
        "ontology": base / "ontology.yaml",
        "partial_predictions": prefix.with_name(f"{prefix.name}_partial.jsonl"),
        "terminal_log": prefix.with_suffix(".log"),
        "materialization": prefix.with_name(f"{prefix.name}_materialization.json"),
        "complete_predictions": prefix.with_name(f"{prefix.name}_complete.jsonl"),
        "expanded_predictions": prefix.with_name(f"{prefix.name}_expanded.jsonl"),
        "expansion_errors": prefix.with_name(f"{prefix.name}_expand_errors.jsonl"),
        "verified_predictions": prefix.with_name(f"{prefix.name}_verified.jsonl"),
        "verification_audit": prefix.with_name(
            f"{prefix.name}_verification_audit.jsonl"
        ),
        "span_index": prefix.with_name(f"{prefix.name}_span_index.jsonl"),
        "normalized_text_metrics": prefix.with_name(
            f"{prefix.name}_normalized_text_metrics.json"
        ),
        "character_span_metrics": prefix.with_name(
            f"{prefix.name}_character_span_metrics.json"
        ),
    }


def require_job_ids(
    rows: list[dict[str, Any]], path: Path, *, annotation: bool = False
) -> list[str]:
    identifiers: list[str] = []
    for position, row in enumerate(rows, 1):
        job_id = row.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise ContractError(f"{path}:{position}: missing non-empty job_id")
        if annotation and not isinstance(row.get("annotation", row), dict):
            raise ContractError(f"{path}:{position}: missing annotation object")
        identifiers.append(job_id)
    if len(identifiers) != len(set(identifiers)):
        raise ContractError(f"{path}: duplicate job_id values")
    return identifiers


def validate_test_inputs(
    data_root: Path, dataset: str, expected: int | None = None
) -> dict[str, Any]:
    if dataset not in EXPECTED_JOBS:
        raise ContractError(f"unknown formal dataset: {dataset}")
    expected = EXPECTED_JOBS[dataset] if expected is None else expected
    if expected != EXPECTED_JOBS[dataset]:
        raise ContractError(
            f"{dataset}: expected-count override {expected} differs from frozen "
            f"count {EXPECTED_JOBS[dataset]}"
        )
    paths = artifact_paths(data_root, Path("."), dataset)
    jobs = load_jsonl(paths["jobs"])
    gold = load_jsonl(paths["gold"])
    index = load_jsonl(paths["gold_index"])
    job_ids = require_job_ids(jobs, paths["jobs"])
    if len(job_ids) != expected:
        raise ContractError(
            f"{dataset}: expected {expected} test jobs, found {len(job_ids)}"
        )
    if len(gold) != expected or len(index) != expected:
        raise ContractError(
            f"{dataset}: jobs/gold/index counts differ from {expected}: "
            f"jobs={len(jobs)} gold={len(gold)} index={len(index)}"
        )

    forbidden = {
        "answer",
        "answers",
        "gold",
        "gold_annotation",
        "label",
        "labels",
        "target",
        "target_annotation",
    }
    prefix = f"{dataset}_test_"
    source_prefix = f"public:{dataset}:test:"
    for position, (job, index_row, gold_row) in enumerate(zip(jobs, index, gold)):
        job_id = job_ids[position]
        document_id = job.get("document_id")
        if not job_id.startswith(prefix):
            raise ContractError(f"{dataset}: non-test job_id {job_id!r}")
        if not isinstance(document_id, str) or not document_id.startswith(prefix):
            raise ContractError(f"{dataset}: invalid test document_id for {job_id}")
        if job.get("category") != dataset:
            raise ContractError(f"{dataset}: category mismatch for {job_id}")
        if job.get("source_path", "").startswith(source_prefix) is False:
            raise ContractError(f"{dataset}: source split mismatch for {job_id}")
        if job.get("experiment_mode") != "baseline":
            raise ContractError(f"{dataset}: non-baseline job {job_id}")
        leaked = sorted(forbidden & set(job))
        if leaked:
            raise ContractError(f"{dataset}: forbidden fields in {job_id}: {leaked}")
        segments = job.get("segments")
        if not isinstance(segments, list) or not segments:
            raise ContractError(f"{dataset}: no source segments for {job_id}")
        segment_ids: list[str] = []
        for segment in segments:
            if not isinstance(segment, dict):
                raise ContractError(f"{dataset}: invalid segment in {job_id}")
            segment_id = segment.get("segment_id")
            start, end, text = (
                segment.get("start"),
                segment.get("end"),
                segment.get("text"),
            )
            if not isinstance(segment_id, str) or not segment_id:
                raise ContractError(f"{dataset}: invalid segment ID in {job_id}")
            if (
                not isinstance(start, int)
                or not isinstance(end, int)
                or not isinstance(text, str)
                or start < 0
                or end != start + len(text)
            ):
                raise ContractError(f"{dataset}: invalid segment offsets in {job_id}")
            segment_ids.append(segment_id)
        if len(segment_ids) != len(set(segment_ids)):
            raise ContractError(f"{dataset}: duplicate segment IDs in {job_id}")
        if index_row.get("job_id") != job_id or index_row.get("record_index") != position:
            raise ContractError(f"{dataset}: gold index mismatch at row {position}")
        if gold_row.get("document_id") != document_id:
            raise ContractError(f"{dataset}: gold document mismatch for {job_id}")

    if not paths["ontology"].is_file() or paths["ontology"].stat().st_size == 0:
        raise ContractError(f"{dataset}: missing or empty ontology: {paths['ontology']}")
    return {
        "dataset": dataset,
        "split": "test",
        "formal_test_read": True,
        "expected_jobs": expected,
        "jobs": len(jobs),
        "gold_rows": len(gold),
        "index_rows": len(index),
        "input_sha256": {
            name: sha256(paths[name])
            for name in ("jobs", "gold", "gold_index", "ontology")
        },
    }


def write_span_index(source: Path, output: Path) -> dict[str, Any]:
    rows = load_jsonl(source)
    identifiers = require_job_ids(rows, source)
    enriched: list[dict[str, Any]] = []
    for position, row in enumerate(rows):
        if row.get("record_index") != position:
            raise ContractError(f"{source}: non-canonical record_index at row {position}")
        enriched.append({**row, "parent_job_id": identifiers[position], "split": "test"})
    atomic_write_jsonl(output, enriched)
    return {"status": "complete", "jobs": len(enriched), "output": str(output)}


def canonical_validate_promotion(path: Path) -> dict[str, Any]:
    """Delegate promotion authenticity to the formal preparation boundary."""
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    try:
        from prepare_public_test_inputs import validate_promotion

        return validate_promotion(path)
    except (ImportError, ValueError, OSError) as error:
        raise ContractError(f"canonical promotion verification failed: {error}") from error


def canonical_validate_prepared_release(
    promotion: Path, prepared_root: Path
) -> dict[str, Any]:
    """Run the preparation boundary's full all-output release verifier."""

    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    try:
        from prepare_public_test_inputs import validate_prepared_release

        return validate_prepared_release(
            promotion_path=promotion, output_root=prepared_root
        )
    except (ImportError, ValueError, OSError) as error:
        raise ContractError(
            f"canonical prepared-release verification failed: {error}"
        ) from error


def validate_preparation_manifest(
    path: Path,
    dataset: str,
    promotion_path: Path,
    promotion: dict[str, Any],
) -> dict[str, Any]:
    """Validate the immutable v2 manifest without opening its test-job outputs."""
    manifest = load_json(path)
    if not (
        manifest.get("schema_version") == PREPARATION_SCHEMA_VERSION
        and manifest.get("status") == "prepared_test"
        and manifest.get("dataset") == dataset
        and manifest.get("prepared_split") == "test"
        and manifest.get("test_baseline_jobs_read") is True
        and manifest.get("test_gold_read") is False
        and manifest.get("validation_gold_read") is False
        and manifest.get("job_count") == EXPECTED_JOBS[dataset]
    ):
        raise ContractError(f"{dataset}: prepared-test manifest header is not canonical")
    fingerprint_inputs = manifest.get("fingerprint_inputs")
    fingerprint = manifest.get("fingerprint")
    if (
        not isinstance(fingerprint_inputs, dict)
        or not isinstance(fingerprint, str)
        or stable_digest(fingerprint_inputs) != fingerprint
    ):
        raise ContractError(f"{dataset}: prepared-test manifest fingerprint is invalid")
    promotion_identity = {
        "path": str(promotion_path),
        "bytes": promotion_path.stat().st_size,
        "sha256": sha256(promotion_path),
    }
    if (
        manifest.get("promotion") != promotion_identity
        or fingerprint_inputs.get("promotion") != promotion_identity
        or manifest.get("promotion_attestation_sha256")
        != promotion.get("attestation_sha256")
        or fingerprint_inputs.get("promotion_attestation_sha256")
        != promotion.get("attestation_sha256")
        or fingerprint_inputs.get("schema_version") != PREPARATION_SCHEMA_VERSION
        or fingerprint_inputs.get("dataset") != dataset
    ):
        raise ContractError(f"{dataset}: prepared-test promotion binding is invalid")
    test_identity = fingerprint_inputs.get("test_baseline_jobs")
    if not (
        isinstance(test_identity, dict)
        and set(test_identity) == {"path", "bytes", "sha256"}
        and Path(str(test_identity.get("path", ""))).name
        == "test_baseline_jobs.jsonl"
        and isinstance(test_identity.get("bytes"), int)
        and test_identity["bytes"] > 0
        and isinstance(test_identity.get("sha256"), str)
        and len(test_identity["sha256"]) == 64
    ):
        raise ContractError(f"{dataset}: prepared-test source identity is malformed")
    test_schema = manifest.get("test_schema")
    if not isinstance(test_schema, dict) or test_schema.get(
        "expected_jobs"
    ) != EXPECTED_JOBS[dataset]:
        raise ContractError(f"{dataset}: prepared-test job-count schema is invalid")
    outputs = manifest.get("outputs")
    expected_outputs = {
        "test_eae_jobs",
        "test_hrge_jobs",
        "hrge_context_audit",
        "train_test_exact_text_quarantine_audit",
    }
    if not isinstance(outputs, dict) or set(outputs) != expected_outputs:
        raise ContractError(f"{dataset}: prepared-test output registry is not closed")
    for name, identity in outputs.items():
        if not (
            isinstance(identity, dict)
            and set(identity) == {"path", "bytes", "sha256"}
            and isinstance(identity.get("path"), str)
            and isinstance(identity.get("bytes"), int)
            and identity["bytes"] >= 0
            and isinstance(identity.get("sha256"), str)
            and len(identity["sha256"]) == 64
        ):
            raise ContractError(f"{dataset}: malformed prepared output identity {name}")
        try:
            Path(identity["path"]).resolve(strict=False).relative_to(
                path.parent.resolve(strict=False)
            )
        except ValueError as error:
            raise ContractError(
                f"{dataset}: prepared output escapes dataset root: {name}"
            ) from error
    return {
        "path": str(path),
        "sha256": sha256(path),
        "fingerprint": fingerprint,
        "job_count": manifest["job_count"],
    }


def verify_release(
    release_status: Path, promotion: Path, prepared_root: Path
) -> dict[str, Any]:
    release = load_json(release_status)
    expected_release_keys = {
        "status", "stage", "promotion_status", "gate_review_status",
        "datasets", "canonical_prepared_release", "updated_at",
    }
    if not isinstance(release, dict) or set(release) != expected_release_keys:
        raise ContractError("formal release status schema is not closed")
    if release.get("status") != "complete" or release.get("stage") != "complete":
        raise ContractError("formal release status is not complete")
    if release.get("promotion_status") != "promoted":
        raise ContractError("formal release does not identify a promoted validation gate")
    if release.get("gate_review_status") != "passed":
        raise ContractError("formal release does not retain the independent review gate")
    # Recompute both promotion authenticity and every generated preparation
    # output.  The older manifest-header-only check allowed a writer/verifier
    # schema mismatch to be released and is intentionally not a release gate.
    promotion_value = canonical_validate_promotion(promotion)
    canonical_prepared = canonical_validate_prepared_release(
        promotion, prepared_root
    )
    if (
        canonical_prepared.get("status") != "verified_release"
        or canonical_prepared.get("schema_version")
        != "public-formal-test-release-v2"
        or canonical_prepared.get("promotion", {}).get("sha256")
        != sha256(promotion)
        or canonical_prepared.get("promotion_attestation_sha256")
        != promotion_value.get("attestation_sha256")
        or not isinstance(canonical_prepared.get("release_sha256"), str)
        or len(canonical_prepared["release_sha256"]) != 64
    ):
        raise ContractError("canonical prepared-release fingerprint is malformed")

    attestation_path = release_status.parent / "prepared_release_attestation.json"
    attestation = load_json(attestation_path)
    if attestation != canonical_prepared:
        raise ContractError(
            "persisted prepared-release attestation differs from canonical verification"
        )
    attestation_record = {
        "path": str(attestation_path),
        "bytes": attestation_path.stat().st_size,
        "sha256": sha256(attestation_path),
    }
    expected_release_record = {
        **attestation_record,
        "status": "verified_release",
        "schema_version": "public-formal-test-release-v2",
        "release_sha256": canonical_prepared["release_sha256"],
    }
    if release.get("canonical_prepared_release") != expected_release_record:
        raise ContractError(
            "formal release status is not bound to the canonical prepared release"
        )

    manifests: dict[str, dict[str, Any]] = {}
    release_datasets = release.get("datasets")
    if not isinstance(release_datasets, dict) or set(release_datasets) != set(DATASETS):
        raise ContractError("formal release has no dataset preparation records")
    for dataset in DATASETS:
        release_row = release_datasets.get(dataset)
        if (
            not isinstance(release_row, dict)
            or set(release_row) != {"preparation_status", "manifest"}
            or release_row.get("preparation_status") != "prepared_test"
        ):
            raise ContractError(f"{dataset}: release preparation is not complete")
        manifest_path = prepared_root / dataset / "preparation_manifest.json"
        recorded_manifest = release_row.get("manifest")
        if (
            not isinstance(recorded_manifest, str)
            or Path(recorded_manifest).resolve(strict=False)
            != manifest_path.resolve(strict=False)
        ):
            raise ContractError(f"{dataset}: release points to the wrong preparation manifest")
        canonical_dataset = canonical_prepared.get("datasets", {}).get(dataset)
        if not isinstance(canonical_dataset, dict):
            raise ContractError(f"{dataset}: missing canonical preparation verification")
        canonical_manifest = canonical_dataset.get("manifest")
        if not (
            isinstance(canonical_manifest, dict)
            and Path(str(canonical_manifest.get("path", ""))).resolve(strict=False)
            == manifest_path.resolve(strict=False)
            and canonical_manifest.get("sha256") == sha256(manifest_path)
        ):
            raise ContractError(
                f"{dataset}: canonical preparation manifest identity changed"
            )
        manifests[dataset] = canonical_dataset
    result = {
        "status": "complete",
        "release_status": {"path": str(release_status), "sha256": sha256(release_status)},
        "promotion": {
            "path": str(promotion),
            "sha256": sha256(promotion),
            "schema_version": promotion_value.get("schema_version"),
            "attestation_sha256": promotion_value.get("attestation_sha256"),
        },
        "preparation_manifests": manifests,
        "prepared_release_attestation": attestation_record,
        "prepared_release_sha256": canonical_prepared["release_sha256"],
    }
    result["canonical_fingerprint"] = stable_digest(result)
    return result


def _exact_prediction_ids(path: Path, expected_ids: list[str]) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    identifiers = require_job_ids(rows, path, annotation=True)
    if identifiers != expected_ids:
        raise ContractError(f"{path}: prediction IDs/order do not exactly match test jobs")
    return rows


def validate_dataset(
    data_root: Path, run_root: Path, dataset: str, expected: int | None = None
) -> dict[str, Any]:
    inputs = validate_test_inputs(data_root, dataset, expected)
    expected = inputs["expected_jobs"]
    paths = artifact_paths(data_root, run_root, dataset)
    job_ids = require_job_ids(load_jsonl(paths["jobs"]), paths["jobs"])
    expected_set = set(job_ids)

    partial_rows = load_jsonl(paths["partial_predictions"])
    partial_ids = require_job_ids(
        partial_rows, paths["partial_predictions"], annotation=True
    )
    if not set(partial_ids) <= expected_set:
        raise ContractError(f"{dataset}: partial predictions contain unknown jobs")

    log_rows = load_jsonl(paths["terminal_log"])
    terminal_ids = require_job_ids(log_rows, paths["terminal_log"])
    if set(terminal_ids) != expected_set or len(terminal_ids) != expected:
        raise ContractError(f"{dataset}: terminal log does not cover every test job")
    invalid_statuses = {
        row["job_id"]: row.get("status")
        for row in log_rows
        if row.get("status") not in TERMINAL_STATUSES
    }
    if invalid_statuses:
        raise ContractError(f"{dataset}: invalid terminal statuses: {invalid_statuses}")
    successful = {
        row["job_id"] for row in log_rows if row.get("status") == "success"
    }
    if set(partial_ids) != successful:
        raise ContractError(f"{dataset}: successful log rows and partial predictions differ")

    materialization = load_json(paths["materialization"])
    missing = [job_id for job_id in job_ids if job_id not in successful]
    if not (
        materialization.get("status") == "complete"
        and materialization.get("jobs") == expected
        and materialization.get("successful_prediction_rows") == len(successful)
        and materialization.get("failures_materialized_as_empty") == len(missing)
        and materialization.get("missing_job_ids") == missing
        and materialization.get("gold_read") is False
    ):
        raise ContractError(f"{dataset}: materialization accounting is inconsistent")

    _exact_prediction_ids(paths["complete_predictions"], job_ids)
    _exact_prediction_ids(paths["expanded_predictions"], job_ids)
    _exact_prediction_ids(paths["verified_predictions"], job_ids)
    expansion_errors = load_jsonl(paths["expansion_errors"])
    if any(row.get("job_id") not in expected_set for row in expansion_errors):
        raise ContractError(f"{dataset}: expansion audit contains unknown jobs")
    verification_audit = load_jsonl(paths["verification_audit"])
    if any(row.get("job_id") not in expected_set for row in verification_audit):
        raise ContractError(f"{dataset}: verification audit contains unknown jobs")

    span_rows = load_jsonl(paths["span_index"])
    span_ids = require_job_ids(span_rows, paths["span_index"])
    if span_ids != job_ids or any(
        row.get("parent_job_id") != row.get("job_id") or row.get("split") != "test"
        for row in span_rows
    ):
        raise ContractError(f"{dataset}: formal span index is not test-bound")

    normalized = load_json(paths["normalized_text_metrics"])
    if not (
        normalized.get("jobs_gold") == expected
        and normalized.get("jobs_predicted") == expected
        and normalized.get("jobs_evaluated") == expected
        and normalized.get("jobs_missing_predictions") == 0
        and normalized.get("generation_success_rate") == 1.0
    ):
        raise ContractError(f"{dataset}: normalized-text metric coverage is incomplete")

    spans = load_json(paths["character_span_metrics"])
    per_job = spans.get("per_job")
    if not (
        spans.get("metric") == "strict-global-character-span-one-to-one"
        and spans.get("selection_split") == "explicit-non-validation-opt-in"
        and spans.get("formal_test_read") is True
        and spans.get("jobs") == expected
        and spans.get("generation_success_rate") == 1.0
        and isinstance(per_job, dict)
        and set(per_job) == expected_set
    ):
        raise ContractError(f"{dataset}: character-span metric coverage is incomplete")

    return {
        "status": "complete",
        "dataset": dataset,
        "split": "test",
        "formal_test_read": True,
        "expected_jobs": expected,
        "successful_raw_rows": len(successful),
        "terminal_failures": len(missing),
        "materialized_failures": len(missing),
        "complete_rows": expected,
        "expanded_rows": expected,
        "verified_rows": expected,
        "normalized_text_metrics_complete": True,
        "character_span_metrics_complete": True,
    }


def artifact_record(path: Path, *, sealed: bool = False) -> dict[str, Any]:
    if sealed:
        return {"path": str(path), "access": "sealed_until_formal_release"}
    record: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        record.update({"bytes": path.stat().st_size, "sha256": sha256(path)})
    return record


def inspect_dataset(
    data_root: Path, run_root: Path, dataset: str, formal_test_read: bool
) -> dict[str, Any]:
    paths = artifact_paths(data_root, run_root, dataset)
    source_names = {"jobs", "gold", "gold_index", "ontology"}
    result: dict[str, Any] = {
        "expected_jobs": EXPECTED_JOBS[dataset],
        "split": "test",
        "artifacts": {
            name: artifact_record(
                path, sealed=(not formal_test_read and name in source_names)
            )
            for name, path in paths.items()
        },
        "completion_valid": False,
    }
    if not formal_test_read:
        result["formal_test_read"] = False
        return result
    result["formal_test_read"] = True
    try:
        result.update(validate_dataset(data_root, run_root, dataset))
        result["completion_valid"] = True
    except (ContractError, OSError) as error:
        result["validation_error"] = str(error)
        for label, name in (
            ("successful_raw_rows", "partial_predictions"),
            ("complete_rows", "complete_predictions"),
            ("expanded_rows", "expanded_predictions"),
            ("verified_rows", "verified_predictions"),
        ):
            path = paths[name]
            try:
                result[label] = len(load_jsonl(path)) if path.is_file() else 0
            except (ContractError, OSError):
                result[label] = None
        try:
            materialization = load_json(paths["materialization"])
            result["materialized_failures"] = materialization.get(
                "failures_materialized_as_empty"
            )
        except (ContractError, OSError):
            result["materialized_failures"] = None
    return result


def build_status(
    *,
    state: str,
    stage: str,
    active_dataset: str | None,
    formal_test_read: bool,
    data_root: Path,
    run_root: Path,
    release_status: Path,
    release_sha256: str | None,
    promotion: Path,
    prepared_root: Path,
    model_path: Path,
    gpu_lock: Path,
    error: str | None = None,
    release_fingerprint: str | None = None,
) -> dict[str, Any]:
    datasets = {
        dataset: inspect_dataset(data_root, run_root, dataset, formal_test_read)
        for dataset in DATASETS
    }
    release_record = artifact_record(release_status)
    release_record["captured_sha256"] = release_sha256
    release_record["captured_canonical_fingerprint"] = release_fingerprint
    if release_status.is_file():
        try:
            release_record["status"] = load_json(release_status).get("status")
        except ContractError as release_error:
            release_record["validation_error"] = str(release_error)

    promotion_record = artifact_record(promotion)
    if promotion.is_file():
        try:
            promotion_value = load_json(promotion)
            promotion_record.update(
                {
                    "schema_version": promotion_value.get("schema_version"),
                    "attestation_sha256": promotion_value.get("attestation_sha256"),
                }
            )
        except ContractError as promotion_error:
            promotion_record["validation_error"] = str(promotion_error)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": state,
        "stage": stage,
        "split": "test",
        "formal_test_read": formal_test_read,
        "active_dataset": active_dataset,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "release": release_record,
        "promotion": promotion_record,
        "test_preparation_root": str(prepared_root),
        "model": {
            "name": "Qwen/Qwen3-4B",
            "path": str(model_path),
            "revision": MODEL_REVISION,
            "adapter": None,
            "training_performed": False,
        },
        "inference": {
            "workers": 2,
            "resume": True,
            "compact_target": True,
            "max_input_tokens": 4096,
            "max_new_tokens": 1024,
            "max_seconds_per_job": 90,
            "retry_failed": False,
        },
        "gpu_lock": {
            "path": str(gpu_lock),
            "exclusive_flock_fd": 8,
            "scope": "per-dataset-inference-only",
            "released_before_cpu_postprocessing": True,
        },
        "expected_jobs": EXPECTED_JOBS,
        "expected_total_jobs": sum(EXPECTED_JOBS.values()),
        "terminal_failure_policy": "materialize exactly one empty annotation for each terminal inference failure",
        "datasets": datasets,
    }
    if error:
        payload["error"] = error

    if state == "complete":
        failures = [
            dataset
            for dataset, row in datasets.items()
            if row.get("completion_valid") is not True
        ]
        observed_release_sha = release_record.get("sha256")
        if not formal_test_read:
            raise ContractError("cannot complete a formal test without opening the test split")
        if failures:
            raise ContractError(f"cannot complete; invalid datasets: {failures}")
        if not release_sha256 or observed_release_sha != release_sha256:
            raise ContractError("cannot complete; formal release marker identity changed")
        if release_record.get("status") != "complete":
            raise ContractError("cannot complete; formal release is no longer complete")
        canonical_release = verify_release(release_status, promotion, prepared_root)
        if canonical_release["release_status"]["sha256"] != release_sha256:
            raise ContractError("cannot complete; canonical release identity changed")
        if (
            not release_fingerprint
            or canonical_release["canonical_fingerprint"] != release_fingerprint
        ):
            raise ContractError("cannot complete; release attestation fingerprint changed")
        payload["canonical_release_verification"] = canonical_release
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    release = subparsers.add_parser("verify-release")
    release.add_argument("--release-status", type=Path, required=True)
    release.add_argument("--promotion", type=Path, required=True)
    release.add_argument("--prepared-root", type=Path, required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--data-root", type=Path, required=True)
    preflight.add_argument("--dataset", choices=DATASETS, required=True)
    preflight.add_argument("--expected", type=int, required=True)

    span_index = subparsers.add_parser("span-index")
    span_index.add_argument("--source", type=Path, required=True)
    span_index.add_argument("--output", type=Path, required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--data-root", type=Path, required=True)
    validate.add_argument("--run-root", type=Path, required=True)
    validate.add_argument("--dataset", choices=DATASETS, required=True)
    validate.add_argument("--expected", type=int, required=True)
    validate.add_argument("--quiet", action="store_true")

    status = subparsers.add_parser("status")
    status.add_argument("--output", type=Path, required=True)
    status.add_argument("--state", required=True)
    status.add_argument("--stage", required=True)
    status.add_argument("--active-dataset")
    status.add_argument("--formal-test-read", choices=("true", "false"), required=True)
    status.add_argument("--data-root", type=Path, required=True)
    status.add_argument("--run-root", type=Path, required=True)
    status.add_argument("--release-status", type=Path, required=True)
    status.add_argument("--release-sha256")
    status.add_argument("--release-fingerprint")
    status.add_argument("--promotion", type=Path, required=True)
    status.add_argument("--prepared-root", type=Path, required=True)
    status.add_argument("--model-path", type=Path, required=True)
    status.add_argument("--gpu-lock", type=Path, required=True)
    status.add_argument("--error")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "verify-release":
            result = verify_release(args.release_status, args.promotion, args.prepared_root)
        elif args.command == "preflight":
            result = validate_test_inputs(args.data_root, args.dataset, args.expected)
        elif args.command == "span-index":
            result = write_span_index(args.source, args.output)
        elif args.command == "validate":
            result = validate_dataset(
                args.data_root, args.run_root, args.dataset, args.expected
            )
            if args.quiet:
                return 0
        elif args.command == "status":
            result = build_status(
                state=args.state,
                stage=args.stage,
                active_dataset=args.active_dataset,
                formal_test_read=args.formal_test_read == "true",
                data_root=args.data_root,
                run_root=args.run_root,
                release_status=args.release_status,
                release_sha256=args.release_sha256,
                promotion=args.promotion,
                prepared_root=args.prepared_root,
                model_path=args.model_path,
                gpu_lock=args.gpu_lock,
                error=args.error,
                release_fingerprint=args.release_fingerprint,
            )
            atomic_write_json(args.output, result)
        else:  # pragma: no cover - argparse guarantees a known command.
            raise AssertionError(args.command)
    except (ContractError, OSError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
