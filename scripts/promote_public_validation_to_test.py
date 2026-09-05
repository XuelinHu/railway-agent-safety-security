#!/usr/bin/env python3
"""Fail-closed promotion from the public validation audit to formal test.

This independent gate revalidates the closed 54-row/14-marker/3-comparison
audit envelope and PGE evidence safety.  It never opens a public-test input.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from audit_public_validation_results import MarkerSpec, audit_marker


DATASETS = ("conll04", "scierc", "ade")
EXPECTED_VALIDATION_JOBS = {"conll04": 231, "scierc": 275, "ade": 384}
MAX_RELATION_F1_DEGRADATION = 0.01
COMPARISON_ITERATIONS = 20_000
COMPARISON_SEED = 20_260_830
SCHEMA_VERSION = "public-formal-test-promotion-v2"
AUDIT_SCHEMA_VERSION = "public-validation-audit-v1"
POLICY_DOCUMENT = Path("paper/PUBLIC_FULL_RUN_PLAN.md")

SYSTEM_FAMILIES = {
    "baseline_raw": "public_stage1",
    "baseline_evidence": "public_stage1",
    "baseline_evidence_verifier": "public_stage1",
    "kg_raw": "public_stage1",
    "kg_evidence": "public_stage1",
    "kg_evidence_verifier_evge_like": "public_stage1",
    "soe": "public_pge_seed42",
    "eae": "public_pge_seed42",
    "hrge": "public_pge_seed42",
    "evge": "public_pge_seed42",
    "cfe": "public_pge_seed42",
    "pge": "public_pge_seed42",
    "qwen3_4b_zero_shot": "horizontal_baselines",
    "spert_fresh_seed42": "horizontal_baselines",
    "gliner_glirel_threshold_05": "horizontal_baselines",
    "gliner_glirel_canonical_t0": "horizontal_baselines",
    "gliner_glirel_train_calibrated": "horizontal_baselines",
    "gliner_entity_only": "horizontal_baselines",
}

MARKER_SPECS = {
    "stage1_analysis": (("complete",), "validation_formal"),
    "qwen_generation": (("validation_complete",), "qwen_generation"),
    "qwen_retry": (("complete", "complete_with_terminal_failures"), "qwen_retry"),
    "pge": (("complete",), "validation_formal_seed42"),
    "pge_manifest": (("complete",), "validation_formal_seed42"),
    "spert": (("complete",), "spert_validation_seed42"),
    "glirel_05": (("complete",), "prediction_complete"),
    "glirel_t0": (("complete",), "prediction_complete"),
    "glirel_calibrated": (("complete",), "train_calibrated_validation"),
    "gliner_entity": (("complete",), "gliner_entity_validation"),
    "post_pge": (("complete",), "post_pge_validation"),
    "glirel_calibration_conll04": (("complete",), "train_calibration"),
    "glirel_calibration_scierc": (("complete",), "train_calibration"),
    "glirel_calibration_ade": (("complete",), "train_calibration"),
}

TSV_COLUMNS = (
    "system_id", "label", "family", "role", "dataset", "audit_status",
    "metric_protocol", "jobs_expected", "prediction_rows", "terminal_failures",
    "accounted_jobs", "entity_precision", "entity_recall", "entity_f1",
    "relation_precision", "relation_recall", "relation_f1", "seed",
    "relation_baseline_valid", "parameters_json", "prediction_sha256",
    "metric_sha256", "note", "error",
)


def stable_digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _has_test_token(path: Path) -> bool:
    return any(
        "test" in [token for token in re.split(r"[._-]+", part.casefold()) if token]
        for part in path.parts
    )


def assert_validation_input(path: Path) -> Path:
    """Reject test namespaces, symlinks, and non-regular promotion inputs."""

    candidate = path if path.is_absolute() else Path.cwd() / path
    if _has_test_token(candidate):
        raise ValueError(f"promotion must not open a test-named path: {path}")
    for component in (candidate, *candidate.parents):
        try:
            if component.is_symlink():
                raise ValueError(f"symlinked promotion input is forbidden: {path}")
        except OSError as error:
            raise ValueError(f"cannot inspect promotion input {path}: {error}") from error
    resolved = candidate.resolve(strict=False)
    if _has_test_token(resolved):
        raise ValueError(f"promotion input resolves into a test namespace: {path}")
    try:
        mode = candidate.stat().st_mode
    except FileNotFoundError:
        raise FileNotFoundError(path) from None
    if not stat.S_ISREG(mode):
        raise ValueError(f"promotion input is not a regular file: {path}")
    return candidate


class InputTracker:
    """Read every artifact once and retain the exact bytes that were checked."""

    def __init__(self) -> None:
        self._payloads: dict[Path, bytes] = {}

    def bytes(self, path: Path) -> bytes:
        candidate = assert_validation_input(path)
        key = candidate.resolve(strict=True)
        if key not in self._payloads:
            self._payloads[key] = candidate.read_bytes()
        return self._payloads[key]

    def text(self, path: Path) -> str:
        try:
            return self.bytes(path).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"validation input is not UTF-8: {path}") from error

    def json(self, path: Path) -> dict[str, Any]:
        try:
            value = json.loads(self.text(path))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at {path}: {error}") from error
        if not isinstance(value, dict):
            raise ValueError(f"expected a JSON object at {path}")
        return value

    def identity(self, path: Path) -> dict[str, Any]:
        payload = self.bytes(path)
        return {
            "path": str(path),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }


def sha256(path: Path) -> str:
    return InputTracker().identity(path)["sha256"]


def file_identity(path: Path) -> dict[str, Any]:
    return InputTracker().identity(path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _as_float(value: Any, label: str, *, rate: bool = True) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} is not numeric: {value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not numeric: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite: {result}")
    if rate and not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} is outside [0, 1]: {result}")
    return result


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} is not a non-negative integer: {value!r}")
    return value


def validate_point_estimate_gate(soe: Any, pge: Any, dataset: str) -> float:
    """Return the finite PGE-SOE delta or reject the documented point gate."""

    soe_value = _as_float(soe, f"{dataset}/SOE relation F1")
    pge_value = _as_float(pge, f"{dataset}/PGE relation F1")
    delta = pge_value - soe_value
    if not math.isfinite(delta):
        raise ValueError(f"non-finite promotion delta for {dataset}")
    if delta < -MAX_RELATION_F1_DEGRADATION - 1e-12:
        raise ValueError(
            f"{dataset} PGE relation F1 degrades by {-delta:.6f}, "
            f"exceeding {MAX_RELATION_F1_DEGRADATION:.2f}"
        )
    return delta


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return (value if value.is_absolute() else Path.cwd() / value).resolve(strict=False)


def _load_input_manifest(
    path: Path, tracker: InputTracker
) -> tuple[dict[str, Any], dict[Path, dict[str, Any]]]:
    manifest = tracker.json(path)
    expected_header = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "complete",
        "selection_split": "validation",
        "formal_test_read": False,
        "test_namespace_status": "sealed_not_read",
        "registry_mode": "closed_explicit_paths_no_discovery",
    }
    for key, expected in expected_header.items():
        if manifest.get(key) != expected:
            raise ValueError(f"validation audit input manifest has invalid {key}")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("validation audit input manifest has no files")
    indexed: dict[Path, dict[str, Any]] = {}
    for index, record in enumerate(files):
        if not isinstance(record, dict) or set(record) != {
            "path", "sha256", "bytes", "roles"
        }:
            raise ValueError(f"input-manifest file record {index} has an invalid schema")
        raw_path = record.get("path")
        roles = record.get("roles")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("input manifest contains an empty path")
        if not isinstance(roles, list) or not roles or any(
            not isinstance(role, str) or not role for role in roles
        ):
            raise ValueError(f"input-manifest roles are invalid for {raw_path}")
        actual = tracker.identity(Path(raw_path))
        if record.get("sha256") != actual["sha256"] or record.get("bytes") != actual["bytes"]:
            raise ValueError(f"validation audit input changed after audit: {raw_path}")
        resolved = _resolve(raw_path)
        if resolved in indexed:
            raise ValueError(f"duplicate resolved input-manifest path: {raw_path}")
        indexed[resolved] = record
    return manifest, indexed


def _require_manifest_identity(
    path: str | Path,
    expected_sha256: str,
    manifest_files: dict[Path, dict[str, Any]],
    tracker: InputTracker,
    *,
    required_role: str | None = None,
) -> dict[str, Any]:
    record = manifest_files.get(_resolve(path))
    if record is None:
        raise ValueError(f"artifact is absent from audit input manifest: {path}")
    actual = tracker.identity(Path(path))
    if expected_sha256 != actual["sha256"] or record["sha256"] != actual["sha256"]:
        raise ValueError(f"validation artifact hash mismatch: {path}")
    if required_role is not None and required_role not in record["roles"]:
        raise ValueError(f"artifact lacks audit role {required_role}: {path}")
    return actual


def _validate_closed_audit_header(audit: dict[str, Any]) -> None:
    for key, expected in {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "complete",
        "selection_split": "validation",
        "formal_test_read": False,
        "test_namespace_status": "sealed_not_read",
    }.items():
        if audit.get(key) != expected:
            raise ValueError(f"validation audit has invalid {key}")
    if audit.get("test_artifacts_opened") != []:
        raise ValueError("validation audit must explicitly record no opened test artifacts")
    if audit.get("failures") != []:
        raise ValueError("validation audit contains failures or omits the failure list")
    if audit.get("datasets") != list(DATASETS):
        raise ValueError("validation audit dataset registry is not closed")
    registry = audit.get("registry")
    if not isinstance(registry, dict) or registry.get("mode") != "closed_explicit_paths_no_discovery":
        raise ValueError("validation audit registry mode is not closed")
    if registry.get("system_dataset_rows") != 54 or registry.get("post_pge_comparisons") != 3:
        raise ValueError("validation audit registry dimensions are not 54/3")
    if registry.get("systems") != sorted(SYSTEM_FAMILIES):
        raise ValueError("validation audit system registry is incomplete")
    counts = audit.get("counts")
    expected_keys = {
        "passed_system_dataset_rows", "failed_system_dataset_rows",
        "passed_markers", "failed_markers", "passed_comparisons",
        "failed_comparisons", "errors", "warnings",
    }
    if not isinstance(counts, dict) or set(counts) != expected_keys:
        raise ValueError("validation audit count schema is incomplete")
    required = {
        "passed_system_dataset_rows": 54,
        "failed_system_dataset_rows": 0,
        "passed_markers": 14,
        "failed_markers": 0,
        "passed_comparisons": 3,
        "failed_comparisons": 0,
        "errors": 0,
    }
    for key, expected in required.items():
        if counts.get(key) != expected:
            raise ValueError(f"validation audit closed-world count mismatch: {key}")
    _nonnegative_int(counts.get("warnings"), "validation audit warnings")


def _validate_markers(
    audit: dict[str, Any], manifest_files: dict[Path, dict[str, Any]], tracker: InputTracker
) -> list[dict[str, Any]]:
    markers = audit.get("markers")
    if not isinstance(markers, list) or len(markers) != 14:
        raise ValueError("validation audit must contain exactly 14 markers")
    indexed: dict[str, dict[str, Any]] = {}
    for row in markers:
        if not isinstance(row, dict) or set(row) != {
            "marker_id", "path", "status", "contract", "audit_status"
        }:
            raise ValueError("validation marker schema is invalid")
        marker_id = str(row.get("marker_id", ""))
        if marker_id not in MARKER_SPECS or marker_id in indexed:
            raise ValueError(f"unexpected or duplicate validation marker: {marker_id}")
        terminal_states, contract = MARKER_SPECS[marker_id]
        if row.get("audit_status") != "passed" or row.get("contract") != contract:
            raise ValueError(f"validation marker did not pass: {marker_id}")
        path = Path(str(row.get("path", "")))
        record = manifest_files.get(_resolve(path))
        if record is None or f"marker:{marker_id}" not in record["roles"]:
            raise ValueError(f"marker is not audit-manifest anchored: {marker_id}")
        recomputed = audit_marker(
            tracker.json(path), MarkerSpec(marker_id, path, terminal_states, contract)
        )
        if recomputed["status"] != row["status"] or recomputed["audit_status"] != "passed":
            raise ValueError(f"marker revalidation disagrees with audit: {marker_id}")
        indexed[marker_id] = row
    if set(indexed) != set(MARKER_SPECS):
        raise ValueError("validation marker coverage is incomplete")
    return [indexed[name] for name in MARKER_SPECS]


def _validate_results(
    audit: dict[str, Any], manifest_files: dict[Path, dict[str, Any]], tracker: InputTracker
) -> dict[tuple[str, str], dict[str, Any]]:
    results = audit.get("results")
    if not isinstance(results, list) or len(results) != 54:
        raise ValueError("validation audit must contain exactly 54 result rows")
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in results:
        if not isinstance(row, dict):
            raise ValueError("validation result row is not an object")
        system, dataset = row.get("system_id"), row.get("dataset")
        key = (str(system), str(dataset))
        if system not in SYSTEM_FAMILIES or dataset not in DATASETS or key in indexed:
            raise ValueError(f"unexpected or duplicate validation result row: {key}")
        if (
            row.get("family") != SYSTEM_FAMILIES[str(system)]
            or row.get("audit_status") != "passed"
            or row.get("selection_split") != "validation"
            or row.get("formal_test_read") is not False
        ):
            raise ValueError(f"validation result did not pass: {key}")
        prediction, metric = row.get("prediction"), row.get("metric")
        if not isinstance(prediction, dict) or not isinstance(metric, dict):
            raise ValueError(f"result lacks prediction/metric provenance: {key}")
        jobs = EXPECTED_VALIDATION_JOBS[str(dataset)]
        accounting = row.get("terminal_failure_accounting")
        if not isinstance(accounting, dict) or "terminal_failures" not in accounting:
            raise ValueError(f"terminal-failure accounting is absent: {key}")
        terminal_failures = _nonnegative_int(
            accounting["terminal_failures"], f"{key} terminal_failures"
        )
        rows = prediction.get("rows")
        required_coverage = {
            "unique_job_ids": rows, "expected_jobs": jobs,
            "accounted_jobs": jobs, "complete_coverage": True,
        }
        expected_row_accounting = (
            rows + terminal_failures == jobs
            if accounting.get("policy") == "missing_predictions_scored_as_empty"
            else rows == jobs
        )
        if (
            isinstance(rows, bool)
            or not isinstance(rows, int)
            or rows < 0
            or not expected_row_accounting
            or any(prediction.get(name) != value for name, value in required_coverage.items())
        ):
            raise ValueError(f"validation result coverage is incomplete: {key}")
        if metric.get("jobs") != jobs:
            raise ValueError(f"validation metric denominator is incomplete: {key}")
        for name, artifact in (("prediction", prediction), ("metric", metric)):
            if not isinstance(artifact.get("path"), str) or not isinstance(
                artifact.get("sha256"), str
            ):
                raise ValueError(f"{name} identity is incomplete: {key}")
            _require_manifest_identity(
                artifact["path"], artifact["sha256"], manifest_files, tracker,
                required_role=f"{name}:{system}:{dataset}",
            )
        headline = metric.get("headline")
        if not isinstance(headline, dict) or set(headline) != {
            "entity", "relation", "relation_with_claim_status"
        }:
            raise ValueError(f"metric headline is incomplete: {key}")
        for field, values in headline.items():
            if not isinstance(values, dict):
                raise ValueError(f"metric headline is malformed: {key}/{field}")
            for name in ("precision", "recall", "f1"):
                _as_float(values.get(name), f"{key} {field}.{name}")
            for name in ("gold", "predicted", "correct"):
                _nonnegative_int(values.get(name), f"{key} {field}.{name}")
            if values["correct"] > min(values["gold"], values["predicted"]):
                raise ValueError(f"metric counts are impossible: {key}/{field}")
        lineage = row.get("lineage")
        if not isinstance(lineage, list) or not lineage:
            raise ValueError(f"result lineage is empty: {key}")
        for item in lineage:
            if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
                raise ValueError(f"result lineage schema is invalid: {key}")
            _require_manifest_identity(
                item["path"], item["sha256"], manifest_files, tracker,
                required_role=f"lineage:{system}:{dataset}",
            )
        indexed[key] = row
    expected = {(system, dataset) for system in SYSTEM_FAMILIES for dataset in DATASETS}
    if set(indexed) != expected:
        raise ValueError("validation result matrix is not exactly 18x3")
    return indexed


def _load_tsv_metrics(
    path: Path, tracker: InputTracker, results: dict[tuple[str, str], dict[str, Any]]
) -> dict[str, dict[str, float]]:
    with io.StringIO(tracker.text(path), newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if tuple(reader.fieldnames or ()) != TSV_COLUMNS:
            raise ValueError("validation TSV is not the frozen 24-column schema")
        rows = list(reader)
    if len(rows) != 54:
        raise ValueError("validation TSV must contain exactly 54 rows")
    indexed: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (str(row.get("system_id")), str(row.get("dataset")))
        if key not in results or key in indexed:
            raise ValueError(f"unexpected or duplicate validation TSV row: {key}")
        source = results[key]
        prediction, metric = source["prediction"], source["metric"]
        try:
            jobs_expected = int(row.get("jobs_expected", ""))
            prediction_rows = int(row.get("prediction_rows", ""))
            accounted_jobs = int(row.get("accounted_jobs", ""))
        except ValueError as error:
            raise ValueError(f"validation TSV count is malformed: {key}") from error
        if (
            row.get("audit_status") != "passed"
            or row.get("family") != source["family"]
            or jobs_expected != prediction["expected_jobs"]
            or prediction_rows != prediction["rows"]
            or accounted_jobs != prediction["accounted_jobs"]
            or row.get("prediction_sha256") != prediction["sha256"]
            or row.get("metric_sha256") != metric["sha256"]
            or row.get("error") != ""
        ):
            raise ValueError(f"validation TSV disagrees with JSON: {key}")
        relation_f1 = _as_float(row.get("relation_f1"), f"TSV {key} relation_f1")
        if abs(relation_f1 - float(metric["headline"]["relation"]["f1"])) > 1e-12:
            raise ValueError(f"validation TSV relation F1 disagrees with JSON: {key}")
        indexed[key] = row
    if set(indexed) != set(results):
        raise ValueError("validation TSV coverage is incomplete")
    return {
        dataset: {
            system: _as_float(
                indexed[(system, dataset)]["relation_f1"],
                f"TSV {dataset}/{system} relation_f1",
            )
            for system in ("soe", "pge")
        }
        for dataset in DATASETS
    }


def _validate_comparisons(
    audit: dict[str, Any],
    results: dict[tuple[str, str], dict[str, Any]],
    manifest_files: dict[Path, dict[str, Any]],
    tracker: InputTracker,
) -> dict[str, dict[str, Any]]:
    rows = audit.get("comparisons")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("validation audit must contain exactly three comparisons")
    selected: dict[str, dict[str, Any]] = {}
    for summary_row in rows:
        if not isinstance(summary_row, dict):
            raise ValueError("comparison row is not an object")
        dataset = summary_row.get("dataset")
        if dataset not in DATASETS or dataset in selected:
            raise ValueError(f"unexpected or duplicate comparison: {dataset}")
        if (
            summary_row.get("comparison") != "SOE_vs_PGE"
            or summary_row.get("audit_status") != "passed"
            or summary_row.get("iterations") != COMPARISON_ITERATIONS
            or summary_row.get("seed") != COMPARISON_SEED
        ):
            raise ValueError(f"comparison contract failed: {dataset}")
        path = Path(str(summary_row.get("path", "")))
        identity = _require_manifest_identity(
            path, str(summary_row.get("sha256", "")), manifest_files, tracker,
            required_role=f"comparison:soe_vs_pge:{dataset}",
        )
        value = tracker.json(path)
        if (
            value.get("schema_version") != "public-post-pge-soe-vs-pge-v1"
            or value.get("dataset") != dataset
            or value.get("selection_split") != "validation"
            or value.get("formal_test_read") is not False
            or value.get("iterations") != COMPARISON_ITERATIONS
            or value.get("seed") != COMPARISON_SEED
            or value.get("documents") != EXPECTED_VALIDATION_JOBS[str(dataset)]
            or set(value.get("fields", {}))
            != {"entity", "relation", "relation_with_claim_status"}
        ):
            raise ValueError(f"comparison file violates frozen contract: {dataset}")
        if value["fields"] != summary_row.get("fields"):
            raise ValueError(f"comparison fields differ from summary: {dataset}")
        for system in ("soe", "pge"):
            artifact = value.get("source_artifacts", {}).get(system)
            metric = results[(system, str(dataset))]["metric"]
            if (
                not isinstance(artifact, dict)
                or _resolve(str(artifact.get("path", ""))) != _resolve(metric["path"])
                or artifact.get("sha256") != metric["sha256"]
            ):
                raise ValueError(f"comparison source mismatch: {dataset}/{system}")
        try:
            relation = value["fields"]["relation"]
            soe = _as_float(relation["left"]["pooled_f1"], f"{dataset}/SOE F1")
            pge = _as_float(relation["right"]["pooled_f1"], f"{dataset}/PGE F1")
            delta = _as_float(
                relation["right_minus_left"]["pooled_f1"],
                f"{dataset} PGE-SOE delta", rate=False,
            )
            ci = relation["right_minus_left"]["pooled_f1_ci95"]
            lower = _as_float(ci["lower"], f"{dataset} CI lower", rate=False)
            upper = _as_float(ci["upper"], f"{dataset} CI upper", rate=False)
        except (KeyError, TypeError) as error:
            raise ValueError(f"comparison relation field is malformed: {dataset}") from error
        if abs(delta - (pge - soe)) > 1e-4 or lower > upper:
            raise ValueError(f"comparison relation delta/CI is inconsistent: {dataset}")
        selected[str(dataset)] = {
            "soe": soe,
            "pge": pge,
            "delta": pge - soe,
            "ci95": {"lower": lower, "upper": upper},
            "comparison": identity,
        }
    if set(selected) != set(DATASETS):
        raise ValueError("comparison dataset coverage is incomplete")
    return selected


def _pge_manifest_identity(
    results: dict[tuple[str, str], dict[str, Any]], tracker: InputTracker
) -> tuple[Path, dict[str, Any]]:
    identities = []
    for dataset in DATASETS:
        matches = [
            item for item in results[("pge", dataset)]["lineage"]
            if Path(str(item.get("path", ""))).name == "run_manifest.json"
        ]
        if len(matches) != 1:
            raise ValueError(f"PGE lineage lacks one run manifest for {dataset}")
        identities.append(matches[0])
    first = identities[0]
    if any(
        _resolve(item["path"]) != _resolve(first["path"])
        or item["sha256"] != first["sha256"]
        for item in identities[1:]
    ):
        raise ValueError("PGE rows do not share one run manifest")
    path = Path(first["path"])
    actual = tracker.identity(path)
    if actual["sha256"] != first["sha256"]:
        raise ValueError("PGE run-manifest hash differs from audited lineage")
    return path, actual


def _validate_pge_safety(
    results: dict[tuple[str, str], dict[str, Any]], tracker: InputTracker
) -> dict[str, Any]:
    run_path, run_identity = _pge_manifest_identity(results, tracker)
    run_manifest = tracker.json(run_path)
    if (
        run_manifest.get("status") != "complete"
        or run_manifest.get("selection_split") != "validation"
        or run_manifest.get("formal_test_read") is not False
        or run_manifest.get("seed") != 42
    ):
        raise ValueError("PGE run manifest is not complete seed42 validation")
    files = run_manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("PGE run manifest has no file hash registry")
    evidence_rows: dict[str, Any] = {}
    for dataset in DATASETS:
        evidence_path = run_path.parent / dataset / "metrics" / "pge_evidence.json"
        matching = [
            (raw_path, digest) for raw_path, digest in files.items()
            if _resolve(raw_path) == _resolve(evidence_path)
        ]
        if len(matching) != 1:
            raise ValueError(f"PGE run manifest does not uniquely anchor {dataset} evidence")
        raw_path, expected_hash = matching[0]
        identity = tracker.identity(Path(raw_path))
        if expected_hash != identity["sha256"]:
            raise ValueError(f"PGE evidence hash mismatch for {dataset}")
        evidence = tracker.json(Path(raw_path))
        overall, per_job = evidence.get("overall"), evidence.get("per_job")
        jobs = EXPECTED_VALIDATION_JOBS[dataset]
        if (
            evidence.get("jobs") != jobs
            or not isinstance(overall, dict)
            or overall.get("jobs") != jobs
            or not isinstance(per_job, dict)
            or len(per_job) != jobs
        ):
            raise ValueError(f"PGE evidence denominator is incomplete for {dataset}")
        counts = overall.get("counts")
        if not isinstance(counts, dict):
            raise ValueError(f"PGE evidence micro counts are absent for {dataset}")
        entities = _nonnegative_int(counts.get("entity_count"), f"{dataset} entity_count")
        relations = _nonnegative_int(counts.get("relation_count"), f"{dataset} relation_count")
        if entities == 0 or relations == 0:
            raise ValueError(f"PGE evidence micro denominator is empty for {dataset}")
        expected_counts = {
            "entity_evidence_present_count": entities,
            "entity_evidence_valid_count": entities,
            "relation_evidence_present_count": relations,
            "relation_evidence_valid_count": relations,
            "unsupported_claim_count": 0,
            "invalid_relation_count": 0,
        }
        for name, expected in expected_counts.items():
            if counts.get(name) != expected:
                raise ValueError(f"PGE evidence safety count failed: {dataset}/{name}")
        expected_rates = {
            "entity_evidence_coverage": 1.0,
            "entity_evidence_correctness": 1.0,
            "relation_evidence_coverage": 1.0,
            "relation_evidence_correctness": 1.0,
            "unsupported_claim_rate": 0.0,
            "invalid_relation_rate": 0.0,
        }
        for name, expected in expected_rates.items():
            if _as_float(overall.get(name), f"{dataset} {name}") != expected:
                raise ValueError(f"PGE evidence safety rate failed: {dataset}/{name}")
        evidence_rows[dataset] = {
            "artifact": identity,
            "jobs": jobs,
            "micro_counts": {
                "entity_count": entities,
                "relation_count": relations,
                **expected_counts,
            },
            "micro_rates": expected_rates,
            "status": "passed",
        }
    return {"run_manifest": run_identity, "datasets": evidence_rows}


def build_promotion(summary_json: Path, summary_tsv: Path) -> dict[str, Any]:
    """Revalidate the closed audit envelope and construct its attestation."""

    tracker = InputTracker()
    audit = tracker.json(summary_json)
    _validate_closed_audit_header(audit)
    input_manifest_path = summary_json.parent / "input_manifest.json"
    _, manifest_files = _load_input_manifest(input_manifest_path, tracker)
    markers = _validate_markers(audit, manifest_files, tracker)
    results = _validate_results(audit, manifest_files, tracker)
    tsv_metrics = _load_tsv_metrics(summary_tsv, tracker, results)
    comparisons = _validate_comparisons(audit, results, manifest_files, tracker)
    pge_safety = _validate_pge_safety(results, tracker)

    decisions: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        soe, pge = tsv_metrics[dataset]["soe"], tsv_metrics[dataset]["pge"]
        comparison = comparisons[dataset]
        if abs(soe - comparison["soe"]) > 1e-12 or abs(pge - comparison["pge"]) > 1e-12:
            raise ValueError(f"JSON/TSV relation F1 mismatch for {dataset}")
        delta = validate_point_estimate_gate(soe, pge, dataset)
        decisions[dataset] = {
            "status": "passed",
            "soe_relation_f1": soe,
            "pge_relation_f1": pge,
            "pge_minus_soe_relation_f1": round(delta, 10),
            "maximum_allowed_degradation": MAX_RELATION_F1_DEGRADATION,
            "comparison": comparison["comparison"],
            "exploratory_paired_bootstrap_ci95": comparison["ci95"],
            "ci_is_gating": False,
            "pge_evidence_safety": pge_safety["datasets"][dataset],
        }

    attestation = {
        "selection_split": "validation",
        "previous_test_namespace_status": "sealed_not_read",
        "formal_test_read_before_promotion": False,
        "test_artifacts_opened_before_promotion": [],
        "promoted_systems": ["soe", "pge"],
        "gate": {
            "comparison": "SOE_vs_PGE",
            "metric": "strict pooled relation F1",
            "decision_rule": "pragmatic_point_estimate_severe_degradation_gate",
            "maximum_allowed_degradation": MAX_RELATION_F1_DEGRADATION,
            "margin_basis": "documented pre-test <=0.01 relation-F1 severe-degradation rule",
            "preregistered_non_inferiority_claim": False,
            "confidence_intervals": "exploratory_non_gating",
            "comparison_iterations": COMPARISON_ITERATIONS,
            "comparison_seed": COMPARISON_SEED,
            "required_system_dataset_rows": 54,
            "required_markers": 14,
            "required_comparisons": 3,
            "audit_errors": 0,
            "pge_evidence_safety_required": True,
        },
        "datasets": decisions,
        "validation_audit_counts": audit["counts"],
        "validated_marker_ids": [row["marker_id"] for row in markers],
        "pge_run_manifest": pge_safety["run_manifest"],
        "inputs": {
            "summary_json": tracker.identity(summary_json),
            "summary_tsv": tracker.identity(summary_tsv),
            "input_manifest_json": tracker.identity(input_manifest_path),
            "promotion_policy_document": tracker.identity(POLICY_DOCUMENT),
        },
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "promoted",
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "attestation_sha256": stable_digest(attestation),
        **attestation,
    }


def comparable_promotion(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "promoted_at"}


def promote(summary_json: Path, summary_tsv: Path, output: Path) -> dict[str, Any]:
    payload = build_promotion(summary_json, summary_tsv)
    write_json_atomic(output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-json", type=Path,
        default=Path("outputs/public_validation_audit/summary.json"),
    )
    parser.add_argument(
        "--summary-tsv", type=Path,
        default=Path("outputs/public_validation_audit/summary.tsv"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/public_formal_matrix/promotion.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = promote(args.summary_json, args.summary_tsv, args.output)
    print(json.dumps({
        "status": payload["status"],
        "output": str(args.output),
        "attestation_sha256": payload["attestation_sha256"],
        "datasets": payload["datasets"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
