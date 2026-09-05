#!/usr/bin/env python3
"""Audit and summarize every registered public *validation* result.

The input registry in this module is deliberately closed: no directory walk,
glob, split discovery, or user-supplied prediction filename is used.  Every
data artifact opened by the worker is either a validation job/prediction/
metric file or an explicitly registered provenance marker.  Test-named paths
and symlinks are rejected before their contents are read.
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
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DATASETS = ("conll04", "scierc", "ade")
EXPECTED_VALIDATION_JOBS = {"conll04": 231, "scierc": 275, "ade": 384}
SCHEMA_VERSION = "public-validation-audit-v1"
ALLOWED_SPAN_METRICS = {
    "strict-global-character-span-one-to-one",
    "strict-global-character-span-document-deduplicated",
    "strict-source-character-span",
}
TERMINAL_QWEN_STATES = {"complete", "complete_with_terminal_failures"}
TEST_ACCESS_SEAL = "sealed_not_read"


class AuditError(RuntimeError):
    """An input violates the frozen validation-only audit contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _has_test_filename(path: Path) -> bool:
    tokens = [token for token in re.split(r"[._-]+", path.name.casefold()) if token]
    return "test" in tokens


def assert_sealed_path(path: Path, *, output: bool = False) -> Path:
    """Reject a test namespace or symlink without opening its target file."""

    candidate = path if path.is_absolute() else Path.cwd() / path
    if any(part.casefold() == "test" for part in candidate.parts):
        raise AuditError(f"test namespace is sealed: {path}")
    if _has_test_filename(candidate):
        raise AuditError(f"test-named {'output' if output else 'input'} is forbidden: {path}")

    # A validation-looking symlink must not redirect the auditor into another
    # namespace.  lstat/is_symlink do not read the target contents.
    for component in (candidate, *candidate.parents):
        try:
            if component.is_symlink():
                raise AuditError(f"symlinked audit path is forbidden: {path}")
        except OSError as error:
            raise AuditError(f"cannot inspect audit path {path}: {error}") from error

    resolved = candidate.resolve(strict=False)
    if any(part.casefold() == "test" for part in resolved.parts) or _has_test_filename(
        resolved
    ):
        raise AuditError(f"resolved path enters sealed test namespace: {path}")
    return candidate


def atomic_bytes(path: Path, payload: bytes) -> None:
    target = assert_sealed_path(path, output=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=target.parent, prefix=f".{target.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )


@dataclass
class InputRecord:
    path: str
    sha256: str
    bytes: int
    roles: set[str] = field(default_factory=set)


class InputTracker:
    """Read registered files once and retain an independently computed hash."""

    def __init__(self) -> None:
        self._payloads: dict[Path, bytes] = {}
        self._records: dict[Path, InputRecord] = {}

    def bytes(self, path: Path, role: str) -> bytes:
        candidate = assert_sealed_path(path)
        key = candidate.resolve(strict=False)
        if key not in self._payloads:
            try:
                mode = candidate.stat().st_mode
            except FileNotFoundError as error:
                raise AuditError(f"missing registered input: {display_path(path)}") from error
            except OSError as error:
                raise AuditError(f"cannot stat registered input {path}: {error}") from error
            if not stat.S_ISREG(mode):
                raise AuditError(f"registered input is not a regular file: {path}")
            try:
                payload = candidate.read_bytes()
            except OSError as error:
                raise AuditError(f"cannot read registered input {path}: {error}") from error
            self._payloads[key] = payload
            self._records[key] = InputRecord(
                path=display_path(candidate),
                sha256=hashlib.sha256(payload).hexdigest(),
                bytes=len(payload),
            )
        self._records[key].roles.add(role)
        return self._payloads[key]

    def json(self, path: Path, role: str) -> dict[str, Any]:
        payload = self.bytes(path, role)
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AuditError(f"invalid JSON in {display_path(path)}: {error}") from error
        if not isinstance(value, dict):
            raise AuditError(f"expected JSON object in {display_path(path)}")
        return value

    def jsonl(self, path: Path, role: str) -> list[dict[str, Any]]:
        payload = self.bytes(path, role)
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise AuditError(f"invalid UTF-8 in {display_path(path)}") from error
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise AuditError(
                    f"invalid JSONL in {display_path(path)}:{line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise AuditError(
                    f"expected object in {display_path(path)}:{line_number}"
                )
            rows.append(row)
        return rows

    def hash(self, path: Path, role: str) -> str:
        self.bytes(path, role)
        return self._records[assert_sealed_path(path).resolve(strict=False)].sha256

    def manifest(self) -> list[dict[str, Any]]:
        return [
            {
                "path": record.path,
                "sha256": record.sha256,
                "bytes": record.bytes,
                "roles": sorted(record.roles),
            }
            for _, record in sorted(
                self._records.items(), key=lambda item: item[1].path
            )
        ]


@dataclass(frozen=True)
class Roots:
    data: Path
    stage1_run: Path
    stage1_analysis: Path
    pge: Path
    qwen: Path
    spert: Path
    glirel: Path
    glirel_t0: Path
    glirel_calibrated: Path
    glirel_calibration: Path
    gliner_entity: Path
    post_pge: Path
    pge_config: Path


@dataclass(frozen=True)
class MarkerSpec:
    marker_id: str
    path: Path
    terminal_states: tuple[str, ...]
    contract: str


@dataclass(frozen=True)
class SystemSpec:
    system_id: str
    label: str
    family: str
    role: str
    marker_id: str
    seed: int | str
    parameters: dict[str, Any]
    prediction: Path
    metric: Path
    metric_kind: str
    lineage: tuple[Path, ...]
    terminal_mode: str = "none"
    relation_baseline_valid: bool | None = True
    note: str = ""


def build_markers(roots: Roots) -> tuple[MarkerSpec, ...]:
    return (
        MarkerSpec(
            "stage1_analysis",
            roots.stage1_analysis / "status.json",
            ("complete",),
            "validation_formal",
        ),
        MarkerSpec(
            "qwen_generation",
            roots.qwen / "status.json",
            ("validation_complete",),
            "qwen_generation",
        ),
        MarkerSpec(
            "qwen_retry",
            roots.qwen / "retry_status.json",
            tuple(sorted(TERMINAL_QWEN_STATES)),
            "qwen_retry",
        ),
        MarkerSpec(
            "pge",
            roots.pge / "status.json",
            ("complete",),
            "validation_formal_seed42",
        ),
        MarkerSpec(
            "pge_manifest",
            roots.pge / "run_manifest.json",
            ("complete",),
            "validation_formal_seed42",
        ),
        MarkerSpec(
            "spert",
            roots.spert / "status.json",
            ("complete",),
            "spert_validation_seed42",
        ),
        MarkerSpec(
            "glirel_05",
            roots.glirel / "status.json",
            ("complete",),
            "prediction_complete",
        ),
        MarkerSpec(
            "glirel_t0",
            roots.glirel_t0 / "status.json",
            ("complete",),
            "prediction_complete",
        ),
        MarkerSpec(
            "glirel_calibrated",
            roots.glirel_calibrated / "status.json",
            ("complete",),
            "train_calibrated_validation",
        ),
        MarkerSpec(
            "gliner_entity",
            roots.gliner_entity / "status.json",
            ("complete",),
            "gliner_entity_validation",
        ),
        MarkerSpec(
            "post_pge",
            roots.post_pge / "status.json",
            ("complete",),
            "post_pge_validation",
        ),
        *tuple(
            MarkerSpec(
                f"glirel_calibration_{dataset}",
                roots.glirel_calibration / dataset / "status.json",
                ("complete",),
                "train_calibration",
            )
            for dataset in DATASETS
        ),
    )


def build_systems(roots: Roots) -> tuple[SystemSpec, ...]:
    systems: list[SystemSpec] = []
    stage1_variants = (
        ("baseline_raw", "Baseline raw", "baseline", "raw", "legacy_annotation"),
        (
            "baseline_evidence",
            "Baseline + evidence",
            "baseline",
            "expanded",
            "strict_span",
        ),
        (
            "baseline_evidence_verifier",
            "Baseline + evidence + verifier",
            "baseline",
            "verified",
            "strict_span",
        ),
        ("kg_raw", "KG raw", "kg", "raw", "legacy_annotation"),
        ("kg_evidence", "KG + evidence", "kg", "expanded", "strict_span"),
        (
            "kg_evidence_verifier_evge_like",
            "KG + evidence + verifier (EVGE-like)",
            "kg",
            "verified",
            "strict_span",
        ),
    )
    for dataset in DATASETS:
        for system_id, label, method, phase, metric_kind in stage1_variants:
            metric = (
                roots.stage1_run / f"{dataset}_{method}_validation_raw_metrics.json"
                if phase == "raw"
                else roots.stage1_analysis
                / "span_metrics"
                / f"{dataset}_{method}_{phase}.json"
            )
            systems.append(
                SystemSpec(
                    system_id=system_id,
                    label=label,
                    family="public_stage1",
                    role="formal_internal_baseline",
                    marker_id="stage1_analysis",
                    seed=42,
                    parameters={
                        "method": method,
                        "phase": phase,
                        "base_model_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
                    },
                    prediction=roots.stage1_analysis
                    / "completed_predictions"
                    / f"{dataset}_{method}_{phase}.jsonl",
                    metric=metric,
                    metric_kind=metric_kind,
                    lineage=(
                        roots.stage1_analysis / "summary.json",
                        roots.stage1_analysis / "missing_predictions.json",
                        roots.stage1_run
                        / f"{dataset}_{method}"
                        / "training_metrics.json",
                    ),
                    terminal_mode=f"stage1:{method}:{phase}",
                    note=(
                        "Legacy normalized-text metric; validation scope is inherited "
                        "from the audited stage1 validation-analysis envelope."
                        if phase == "raw"
                        else ""
                    ),
                )
            )

    pge_labels = {
        "soe": "SOE",
        "eae": "EAE",
        "hrge": "HRGE",
        "evge": "EVGE",
        "cfe": "CFE",
        "pge": "PGE",
    }
    for dataset in DATASETS:
        for system_id, label in pge_labels.items():
            if system_id == "soe":
                prediction = (
                    roots.stage1_analysis
                    / "completed_predictions"
                    / f"{dataset}_baseline_expanded.jsonl"
                )
                terminal_mode = "stage1:baseline:expanded"
                extra_lineage = (
                    roots.stage1_run / f"{dataset}_baseline" / "training_metrics.json",
                )
            else:
                suffix = "expanded" if system_id in {"eae", "hrge"} else None
                prediction = roots.pge / dataset / (
                    f"{system_id}_validation_{suffix}.jsonl"
                    if suffix
                    else f"{system_id}_validation.jsonl"
                )
                terminal_mode = (
                    f"pge_branch:{system_id}"
                    if system_id in {"eae", "hrge"}
                    else (
                        "pge_derived:hrge"
                        if system_id == "evge"
                        else "pge_derived:eae,hrge"
                    )
                )
                extra_lineage = (
                    (roots.pge / dataset / f"{system_id}_adapter" / "training_metrics.json",)
                    if system_id in {"eae", "hrge"}
                    else ()
                )
            systems.append(
                SystemSpec(
                    system_id=system_id,
                    label=label,
                    family="public_pge_seed42",
                    role="formal_internal_system",
                    marker_id="pge",
                    seed=42,
                    parameters={"system": system_id, "config": display_path(roots.pge_config)},
                    prediction=prediction,
                    metric=roots.pge / dataset / "metrics" / f"{system_id}_span.json",
                    metric_kind="strict_span",
                    lineage=(roots.pge / "run_manifest.json", roots.pge_config, *extra_lineage),
                    terminal_mode=terminal_mode,
                )
            )

    for dataset in DATASETS:
        systems.extend(
            (
                SystemSpec(
                    system_id="qwen3_4b_zero_shot",
                    label="Qwen3-4B zero-shot",
                    family="horizontal_baselines",
                    role="formal_zero_shot_baseline",
                    marker_id="qwen_retry",
                    seed="not_applicable_zero_shot_decoding",
                    parameters={
                        "model": "Qwen3-4B",
                        "retry_workers": 2,
                        "retry_max_new_tokens": 1024,
                        "retry_max_seconds_per_job": 90,
                    },
                    prediction=roots.qwen / f"{dataset}_validation.jsonl",
                    metric=roots.qwen
                    / f"{dataset}_validation_character_span_metrics.json",
                    metric_kind="strict_source_span",
                    lineage=(roots.qwen / "status.json", roots.qwen / "retry_status.json"),
                    terminal_mode="qwen_logs",
                ),
                SystemSpec(
                    system_id="spert_fresh_seed42",
                    label="SpERT fresh",
                    family="horizontal_baselines",
                    role="formal_trained_baseline",
                    marker_id="spert",
                    seed=42,
                    parameters={
                        "seed": 42,
                        "epochs": 20,
                        "relation_filter_threshold": 0.4,
                    },
                    prediction=roots.spert
                    / dataset
                    / "seed42"
                    / "validation_predictions.jsonl",
                    metric=roots.spert
                    / dataset
                    / "seed42"
                    / "validation_character_span_metrics.json",
                    metric_kind="strict_source_span",
                    lineage=(roots.spert / "status.json", roots.spert / "preflight.json"),
                ),
                SystemSpec(
                    system_id="gliner_glirel_threshold_05",
                    label="GLiNER + GLiREL (threshold=0.5)",
                    family="horizontal_baselines",
                    role="uncalibrated_diagnostic",
                    marker_id="glirel_05",
                    seed="not_applicable_deterministic_inference",
                    parameters={
                        "entity_threshold": 0.5,
                        "relation_threshold": 0.5,
                        "relation_label_mode": "canonical",
                        "dtype": "float16",
                    },
                    prediction=roots.glirel / f"{dataset}_validation.jsonl",
                    metric=roots.glirel
                    / f"{dataset}_validation.character_span_metrics.json",
                    metric_kind="strict_source_span",
                    lineage=(roots.glirel / "status.json",),
                    relation_baseline_valid=False,
                    note=(
                        "Uncalibrated diagnostic only: relation F1=0 at threshold 0.5; "
                        "do not report as the effective GLiREL relation baseline."
                    ),
                ),
                SystemSpec(
                    system_id="gliner_glirel_canonical_t0",
                    label="GLiNER + GLiREL canonical threshold=0",
                    family="horizontal_baselines",
                    role="threshold_sensitivity",
                    marker_id="glirel_t0",
                    seed="not_applicable_deterministic_inference",
                    parameters={
                        "entity_threshold": 0.5,
                        "relation_threshold": 0.0,
                        "relation_label_mode": "canonical",
                        "dtype": "float16",
                    },
                    prediction=roots.glirel_t0 / f"{dataset}_validation.jsonl",
                    metric=roots.glirel_t0
                    / f"{dataset}_validation.character_span_metrics.json",
                    metric_kind="strict_source_span",
                    lineage=(
                        roots.glirel_t0 / "status.json",
                        roots.glirel_t0 / "compatibility_canary.json",
                    ),
                    relation_baseline_valid=False,
                    note="Canonical-label threshold sensitivity arm; no validation tuning.",
                ),
                SystemSpec(
                    system_id="gliner_glirel_train_calibrated",
                    label="GLiNER + GLiREL train-calibrated",
                    family="horizontal_baselines",
                    role="formal_train_calibrated_baseline",
                    marker_id="glirel_calibrated",
                    seed="not_applicable_deterministic_inference",
                    parameters={
                        "entity_threshold": 0.5,
                        "selection_split": "train_inner_calibration",
                        "validation_gold_used_for_selection": False,
                    },
                    prediction=roots.glirel_calibrated
                    / f"{dataset}_validation.jsonl",
                    metric=roots.glirel_calibrated
                    / f"{dataset}_validation.character_span_metrics.json",
                    metric_kind="strict_source_span",
                    lineage=(
                        roots.glirel_calibrated / "status.json",
                        roots.glirel_calibrated / "protocol.json",
                        roots.glirel_calibrated / f"{dataset}_lineage.json",
                        roots.glirel_calibration / dataset / "calibration.json",
                    ),
                ),
                SystemSpec(
                    system_id="gliner_entity_only",
                    label="GLiNER entity-only",
                    family="horizontal_baselines",
                    role="formal_entity_only_baseline",
                    marker_id="gliner_entity",
                    seed="not_applicable_deterministic_projection",
                    parameters={
                        "operation": "copy_entities_drop_relations",
                        "entity_inference_reused": True,
                    },
                    prediction=roots.gliner_entity / f"{dataset}_validation.jsonl",
                    metric=roots.gliner_entity
                    / f"{dataset}_validation.character_span_metrics.json",
                    metric_kind="strict_source_span",
                    lineage=(
                        roots.gliner_entity / "status.json",
                        roots.gliner_entity / f"{dataset}_validation.lineage.json",
                    ),
                    relation_baseline_valid=None,
                    note="Entity-only baseline; relation metrics are not applicable.",
                ),
            )
        )
    return tuple(systems)


def _require(value: bool, message: str) -> None:
    if not value:
        raise AuditError(message)


def audit_marker(value: dict[str, Any], spec: MarkerSpec) -> dict[str, Any]:
    status = value.get("status")
    _require(
        status in spec.terminal_states,
        f"marker {spec.marker_id} is not terminal: {status!r}; expected {spec.terminal_states}",
    )
    contract = spec.contract
    if contract in {"validation_formal", "validation_formal_seed42", "post_pge_validation"}:
        split = value.get("selection_split", value.get("split"))
        _require(split == "validation", f"{spec.marker_id} selection_split is not validation")
        _require(
            value.get("formal_test_read") is False,
            f"{spec.marker_id} does not declare formal_test_read=false",
        )
    if contract == "validation_formal_seed42":
        _require(value.get("seed") == 42, f"{spec.marker_id} seed is not 42")
    elif contract == "spert_validation_seed42":
        _require(value.get("split") == "validation", "SpERT split is not validation")
        _require(value.get("seed") == 42, "SpERT seed is not 42")
        _require(
            value.get("test_split_access") == "forbidden-and-not-read",
            "SpERT test-split access is not sealed",
        )
    elif contract == "gliner_entity_validation":
        _require(value.get("split") == "validation", "GLiNER entity split is not validation")
        _require(
            value.get("execution", {}).get("test_split_access")
            == "forbidden-and-not-read",
            "GLiNER entity-only marker does not seal test access",
        )
    elif contract == "train_calibrated_validation":
        _require(
            value.get("selection_split") == "train_inner_calibration",
            "calibrated GLiREL selection is not train-only",
        )
        _require(
            value.get("validation_gold_used_for_selection") is False,
            "calibrated GLiREL used validation gold for selection",
        )
        _require(
            value.get("test_gold_used_for_selection") is False,
            "calibrated GLiREL used test gold for selection",
        )
    elif contract == "qwen_retry":
        _require(
            value.get("terminal_failure_evaluation")
            == "missing predictions are scored as empty annotations",
            "Qwen terminal-failure denominator policy is missing",
        )
        datasets = value.get("datasets")
        _require(isinstance(datasets, dict), "Qwen retry dataset accounting is missing")
        _require(
            value.get("remaining_failed_jobs")
            == sum(
                item.get("remaining_failed_jobs", -1)
                for item in datasets.values()
                if isinstance(item, dict)
            ),
            "Qwen aggregate terminal-failure count mismatch",
        )
    elif contract == "qwen_generation":
        _require(isinstance(value.get("datasets"), dict), "Qwen dataset status is missing")
    elif contract == "prediction_complete":
        datasets = value.get("datasets")
        _require(isinstance(datasets, dict), f"{spec.marker_id} dataset states are missing")
        for dataset in DATASETS:
            entry = datasets.get(dataset, {})
            _require(
                entry.get("predictions_complete") is True,
                f"{spec.marker_id}/{dataset} predictions are incomplete",
            )
            _require(
                entry.get("character_span_metrics_ready") is True,
                f"{spec.marker_id}/{dataset} strict metric is incomplete",
            )
    elif contract == "train_calibration":
        _require(value.get("dataset") in DATASETS, "calibration dataset is invalid")
        _require(
            isinstance(value.get("selected_configuration"), dict),
            "calibration selection is missing",
        )
    elif contract == "post_pge_validation":
        _require(value.get("completed_datasets") == 3, "post-PGE comparisons are incomplete")
    return {
        "marker_id": spec.marker_id,
        "path": display_path(spec.path),
        "status": status,
        "contract": contract,
        "audit_status": "passed",
    }


def index_rows(
    rows: Iterable[dict[str, Any]], path: Path, *, require_annotation: bool
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(rows, 1):
        job_id = row.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise AuditError(f"missing job_id in {display_path(path)}:{line_number}")
        if "_validation_" not in job_id:
            raise AuditError(f"non-validation job ID in {display_path(path)}: {job_id}")
        if job_id in indexed:
            raise AuditError(f"duplicate job_id in {display_path(path)}: {job_id}")
        if require_annotation and not isinstance(row.get("annotation"), dict):
            raise AuditError(f"prediction has no annotation object: {job_id}")
        indexed[job_id] = row
    return indexed


def validate_validation_jobs(
    rows: Iterable[dict[str, Any]], dataset: str, path: Path
) -> dict[str, dict[str, Any]]:
    indexed = index_rows(rows, path, require_annotation=False)
    forbidden = {"annotation", "annotations", "entities", "relations", "gold"}
    for job_id, row in indexed.items():
        document_id = row.get("document_id")
        source_path = row.get("source_path")
        _require(
            isinstance(document_id, str)
            and document_id.startswith(f"{dataset}_validation_"),
            f"non-validation document ID in {display_path(path)}: {document_id!r}",
        )
        _require(
            job_id.startswith(f"{document_id}_C"),
            f"job/document identity mismatch in {display_path(path)}: {job_id}",
        )
        _require(row.get("category") == dataset, f"job category mismatch: {job_id}")
        _require(
            isinstance(source_path, str)
            and f":{dataset}:validation:" in source_path,
            f"job source is not the validation split: {job_id}",
        )
        leaked = forbidden & set(row)
        _require(not leaked, f"job contains forbidden gold/prediction fields: {job_id}")
    return indexed


def require_full_validation_job_count(
    indexed: dict[str, dict[str, Any]], dataset: str, path: Path
) -> None:
    """Fail closed if a self-consistent but truncated public split is supplied."""

    expected = EXPECTED_VALIDATION_JOBS[dataset]
    _require(
        len(indexed) == expected,
        f"full validation job count mismatch for {dataset} in {display_path(path)}: "
        f"{len(indexed)} != {expected}",
    )


def validate_finite_metrics(value: Any, path: str = "metric") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            validate_finite_metrics(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            validate_finite_metrics(item, f"{path}[{index}]")
    elif isinstance(value, bool):
        return
    elif isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise AuditError(f"non-finite numeric metric: {path}")
        leaf = path.rsplit(".", 1)[-1]
        if (
            leaf in {"precision", "recall", "f1", "generation_success_rate"}
            or leaf.endswith("_rate")
            or leaf.endswith("_coverage")
            or leaf.endswith("_correctness")
        ) and not 0.0 <= value <= 1.0:
            raise AuditError(f"rate metric outside [0,1]: {path}={value}")
        count_names = {
            "gold",
            "predicted",
            "correct",
            "jobs",
            "documents",
            "jobs_gold",
            "jobs_predicted",
            "jobs_evaluated",
            "jobs_missing_predictions",
            "resolved",
            "unresolved_gold_entities",
            "unresolved_predicted_entities",
        }
        if leaf in count_names and (not isinstance(value, int) or value < 0):
            raise AuditError(f"count metric is not a non-negative integer: {path}={value}")


def normalized_headline(metric: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if isinstance(metric.get("overall"), dict):
        source = metric["overall"]
        entity = source.get("entity")
        relation = source.get("relation")
        relation_status = source.get("relation_with_claim_status")
    else:
        entity = metric.get("entity_strict")
        relation = metric.get("relation_strict")
        relation_status = metric.get("relation_with_claim_status")
    for name, value in (
        ("entity", entity),
        ("relation", relation),
        ("relation_with_claim_status", relation_status),
    ):
        if not isinstance(value, dict):
            raise AuditError(f"metric headline {name} is missing")
        for key in ("precision", "recall", "f1", "gold", "predicted", "correct"):
            if key not in value:
                raise AuditError(f"metric headline {name}.{key} is missing")
        for key in ("precision", "recall", "f1"):
            item = value[key]
            if (
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(item)
                or not 0.0 <= item <= 1.0
            ):
                raise AuditError(f"metric headline {name}.{key} is invalid")
        for key in ("gold", "predicted", "correct"):
            item = value[key]
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise AuditError(f"metric headline {name}.{key} is invalid")
        if value["correct"] > min(value["gold"], value["predicted"]):
            raise AuditError(f"metric headline {name} has impossible counts")
    return {
        "entity": entity,
        "relation": relation,
        "relation_with_claim_status": relation_status,
    }


def validate_source_artifact_record(
    record: Any, expected_path: Path, expected_sha256: str
) -> None:
    """Compare provenance paths canonically without opening the recorded path."""

    _require(isinstance(record, dict), "comparison source record is missing")
    raw_path = record.get("path")
    _require(isinstance(raw_path, str) and raw_path, "comparison source path is missing")
    recorded = assert_sealed_path(Path(raw_path))
    expected = assert_sealed_path(expected_path)
    _require(
        recorded.resolve(strict=False) == expected.resolve(strict=False),
        "comparison source path mismatch",
    )
    _require(
        record.get("sha256") == expected_sha256,
        "comparison source SHA-256 mismatch",
    )


def audit_metric(
    metric: dict[str, Any],
    kind: str,
    expected_ids: set[str],
    predicted_rows: int,
    terminal_failures: int,
) -> tuple[str, dict[str, dict[str, Any]]]:
    expected = len(expected_ids)
    validate_finite_metrics(metric)
    if kind == "legacy_annotation":
        protocol = "legacy-normalized-text-exact"
        _require("metric" not in metric, "legacy metric unexpectedly changed schema")
        _require(metric.get("jobs_gold") == expected, "legacy metric gold jobs mismatch")
        _require(metric.get("jobs_evaluated") == expected, "legacy metric denominator mismatch")
        _require(
            metric.get("jobs_predicted") + metric.get("jobs_missing_predictions")
            == expected,
            "legacy metric success/failure accounting mismatch",
        )
    else:
        protocol = metric.get("metric")
        _require(protocol in ALLOWED_SPAN_METRICS, f"unsupported strict metric: {protocol}")
        _require(
            metric.get("selection_split") == "validation",
            "metric selection_split is not validation",
        )
        if "formal_test_read" in metric:
            _require(metric.get("formal_test_read") is False, "metric read formal test data")
        if protocol.startswith("strict-global-character-span"):
            _require(metric.get("jobs") == expected, "strict metric job count mismatch")
        else:
            _require(metric.get("jobs_gold") == expected, "strict metric gold jobs mismatch")
            _require(metric.get("jobs_evaluated") == expected, "strict metric denominator mismatch")
            _require(
                metric.get("jobs_predicted") == predicted_rows,
                "strict metric prediction-row count mismatch",
            )
            _require(
                metric.get("jobs_missing_predictions") == terminal_failures,
                "strict metric terminal-failure count mismatch",
            )
    per_job = metric.get("per_job")
    if isinstance(per_job, dict):
        _require(set(per_job) == expected_ids, "metric per_job IDs do not cover validation jobs")
    elif kind != "legacy_annotation":
        raise AuditError("strict metric per_job mapping is missing")
    return str(protocol), normalized_headline(metric)


def audit_generation_success_rate(
    family: str,
    metric: dict[str, Any],
    expected_jobs: int,
    lineage_terminal_failures: int,
) -> None:
    """Check legacy stage1 rates without conflating lineage with row coverage.

    PGE evaluates fully materialized artifacts, so its metric-level generation
    rate can correctly be 1.0 even when an upstream generator failure is kept
    separately in ``terminal_failure_accounting``.
    """

    if family != "public_stage1":
        return
    observed = metric.get("generation_success_rate")
    if not isinstance(observed, (int, float)):
        return
    expected = (expected_jobs - lineage_terminal_failures) / expected_jobs
    _require(
        abs(float(observed) - expected) <= 0.00011,
        "strict metric generation success rate mismatches terminal failures",
    )


class Auditor:
    def __init__(self, roots: Roots, output_root: Path) -> None:
        self.roots = roots
        self.output_root = output_root
        self.tracker = InputTracker()
        self.marker_specs = build_markers(roots)
        self.system_specs = build_systems(roots)
        self.markers: dict[str, dict[str, Any]] = {}
        self.marker_results: list[dict[str, Any]] = []
        self.failures: list[dict[str, str]] = []
        self.warnings: list[dict[str, str]] = []
        self.jobs: dict[str, dict[str, dict[str, Any]]] = {}
        self.stage1_summary: dict[str, Any] = {}
        self.stage1_missing: dict[str, Any] = {}
        self.pge_manifest: dict[str, Any] = {}

    def fail(self, scope: str, error: Exception | str) -> None:
        self.failures.append({"scope": scope, "error": str(error)})

    def load_core_inputs(self) -> None:
        for spec in self.marker_specs:
            try:
                value = self.tracker.json(spec.path, f"marker:{spec.marker_id}")
                result = audit_marker(value, spec)
                self.markers[spec.marker_id] = value
                self.marker_results.append(result)
            except Exception as error:
                self.fail(f"marker:{spec.marker_id}", error)
                self.marker_results.append(
                    {
                        "marker_id": spec.marker_id,
                        "path": display_path(spec.path),
                        "status": "invalid_or_missing",
                        "contract": spec.contract,
                        "audit_status": "failed",
                        "error": str(error),
                    }
                )

        try:
            self.stage1_summary = self.tracker.json(
                self.roots.stage1_analysis / "summary.json", "stage1:lineage"
            )
            scope = self.stage1_summary.get("scope", {})
            _require(scope.get("split") == "validation", "stage1 summary is not validation")
            _require(scope.get("formal_test_read") is False, "stage1 summary read formal test")
        except Exception as error:
            self.fail("stage1:summary", error)
        try:
            self.stage1_missing = self.tracker.json(
                self.roots.stage1_analysis / "missing_predictions.json",
                "stage1:terminal_failures",
            )
            _require(
                self.stage1_missing.get("split") == "validation",
                "stage1 missing-prediction manifest is not validation",
            )
        except Exception as error:
            self.fail("stage1:missing_predictions", error)
        try:
            self.pge_manifest = self.tracker.json(
                self.roots.pge / "run_manifest.json", "pge:lineage"
            )
        except Exception as error:
            self.fail("pge:run_manifest", error)

        for dataset in DATASETS:
            path = self.roots.data / dataset / "validation_baseline_jobs.jsonl"
            try:
                rows = self.tracker.jsonl(path, f"jobs:{dataset}")
                indexed = validate_validation_jobs(rows, dataset, path)
                require_full_validation_job_count(indexed, dataset, path)
                self.jobs[dataset] = indexed
            except Exception as error:
                self.fail(f"jobs:{dataset}", error)

    def stage1_terminal_ids(self, dataset: str, method: str) -> set[str]:
        records = self.stage1_missing.get("records")
        if not isinstance(records, list):
            raise AuditError("stage1 terminal-failure records are missing")
        ids = {
            str(row.get("job_id"))
            for row in records
            if isinstance(row, dict)
            and row.get("dataset") == dataset
            and row.get("method") == method
        }
        if "None" in ids:
            raise AuditError("stage1 terminal-failure record has no job_id")
        return ids

    def pge_branch_failures(self, dataset: str, branch: str) -> tuple[set[str], dict[str, Any]]:
        path = self.roots.pge / dataset / f"{branch}_validation_materialization.json"
        value = self.tracker.json(path, f"terminal_failures:{dataset}:{branch}")
        _require(value.get("status") == "complete", f"{dataset}/{branch} materialization incomplete")
        _require(value.get("gold_read") is False, f"{dataset}/{branch} materialization read gold")
        _require(value.get("jobs") == len(self.jobs[dataset]), "PGE materialization jobs mismatch")
        ids = value.get("missing_job_ids")
        _require(isinstance(ids, list), "PGE materialization missing IDs absent")
        _require(
            value.get("failures_materialized_as_empty") == len(ids),
            "PGE materialization failure count mismatch",
        )
        return {str(item) for item in ids}, value

    def qwen_outcomes(self, dataset: str) -> dict[str, str]:
        outcomes: dict[str, str] = {}
        for worker in (0, 1):
            path = self.roots.qwen / f"{dataset}_validation.log.part{worker}"
            rows = self.tracker.jsonl(path, f"qwen_terminal_log:{dataset}:part{worker}")
            for row in rows:
                job_id = row.get("job_id")
                status_value = row.get("status")
                _require(isinstance(job_id, str) and job_id, "Qwen terminal log lacks job_id")
                _require(
                    status_value in {"success", "failed"},
                    f"Qwen terminal log has invalid state for {job_id}: {status_value}",
                )
                outcomes[job_id] = str(status_value)
        _require(
            set(outcomes) == set(self.jobs[dataset]),
            f"Qwen terminal outcomes do not cover {dataset} validation jobs",
        )
        return outcomes

    def terminal_accounting(
        self, spec: SystemSpec, dataset: str, prediction_ids: set[str]
    ) -> tuple[set[str], dict[str, Any]]:
        mode = spec.terminal_mode
        expected_ids = set(self.jobs[dataset])
        if mode == "none":
            return set(), {"policy": "all_predictions_present", "terminal_failures": 0}
        if mode.startswith("stage1:"):
            _, method, phase = mode.split(":", 2)
            ids = self.stage1_terminal_ids(dataset, method)
            summary_entry = (
                self.stage1_summary.get("materialization", {})
                .get(dataset, {})
                .get(method, {})
                .get(phase, {})
            )
            _require(summary_entry.get("jobs") == len(expected_ids), "stage1 jobs mismatch")
            _require(
                set(summary_entry.get("missing_job_ids", [])) == ids,
                "stage1 terminal ID lineage mismatch",
            )
            _require(
                summary_entry.get("materialized_empty_rows") == len(ids),
                "stage1 materialized-empty count mismatch",
            )
            return ids, {
                "policy": "terminal_failures_materialized_as_empty",
                "terminal_failures": len(ids),
                "terminal_job_ids": sorted(ids),
            }
        if mode == "qwen_logs":
            outcomes = self.qwen_outcomes(dataset)
            failed = {job_id for job_id, state in outcomes.items() if state == "failed"}
            successful = {job_id for job_id, state in outcomes.items() if state == "success"}
            _require(successful == prediction_ids, "Qwen successes do not equal prediction IDs")
            marker = self.markers.get("qwen_retry", {})
            dataset_marker = marker.get("datasets", {}).get(dataset, {})
            generation_marker = (
                self.markers.get("qwen_generation", {}).get("datasets", {}).get(dataset, {})
            )
            _require(
                generation_marker.get("expected_jobs") == len(expected_ids),
                "Qwen original marker denominator mismatch",
            )
            _require(
                generation_marker.get("successful_predictions", 0)
                + generation_marker.get("terminal_failures", 0)
                == len(expected_ids),
                "Qwen original marker outcome count mismatch",
            )
            _require(
                dataset_marker.get("terminal_jobs") == len(expected_ids),
                "Qwen marker denominator mismatch",
            )
            _require(
                dataset_marker.get("remaining_failed_jobs") == len(failed),
                "Qwen marker failure count mismatch",
            )
            return failed, {
                "policy": "missing_predictions_scored_as_empty",
                "terminal_failures": len(failed),
                "terminal_job_ids": sorted(failed),
                "terminal_outcomes": len(outcomes),
            }
        if mode.startswith("pge_branch:"):
            branch = mode.split(":", 1)[1]
            ids, _ = self.pge_branch_failures(dataset, branch)
            return ids, {
                "policy": "terminal_failures_materialized_as_empty",
                "terminal_failures": len(ids),
                "terminal_job_ids": sorted(ids),
            }
        if mode.startswith("pge_derived:"):
            branches = mode.split(":", 1)[1].split(",")
            upstream: dict[str, Any] = {}
            for branch in branches:
                ids, _ = self.pge_branch_failures(dataset, branch)
                upstream[branch] = {
                    "failures_materialized_as_empty": len(ids),
                    "terminal_job_ids": sorted(ids),
                }
            return set(), {
                "policy": "derived_from_full_materialized_inputs",
                "terminal_failures": 0,
                "upstream_generator_failures": upstream,
            }
        raise AuditError(f"unknown terminal accounting mode: {mode}")

    def validate_lineage(self, spec: SystemSpec, dataset: str) -> list[dict[str, str]]:
        _require(spec.seed is not None, "seed policy is absent")
        _require(bool(spec.parameters), "parameter provenance is absent")
        _require(bool(spec.lineage), "lineage artifacts are absent")
        result = []
        for path in spec.lineage:
            digest = self.tracker.hash(path, f"lineage:{spec.system_id}:{dataset}")
            if path.suffix == ".json":
                value = self.tracker.json(path, f"lineage:{spec.system_id}:{dataset}")
                _require(value, f"empty lineage JSON: {display_path(path)}")
                if path.name == "training_metrics.json" and spec.seed == 42:
                    _require(value.get("seed") == 42, "training seed lineage mismatch")
                if path.name == "calibration.json":
                    _require(
                        value.get("split") == "train_inner_calibration",
                        "GLiREL calibration is not train-inner-only",
                    )
                    _require(value.get("validation_gold_read") is False, "calibration read validation gold")
                    _require(value.get("test_gold_read") is False, "calibration read test gold")
            result.append({"path": display_path(path), "sha256": digest})
        return result

    def audit_system(self, spec: SystemSpec, dataset: str) -> dict[str, Any]:
        expected = self.jobs[dataset]
        expected_ids = set(expected)
        marker_result = next(
            (row for row in self.marker_results if row["marker_id"] == spec.marker_id),
            None,
        )
        _require(
            marker_result is not None and marker_result.get("audit_status") == "passed",
            f"required marker failed: {spec.marker_id}",
        )
        prediction_rows = self.tracker.jsonl(
            spec.prediction, f"prediction:{spec.system_id}:{dataset}"
        )
        predictions = index_rows(prediction_rows, spec.prediction, require_annotation=True)
        unknown = set(predictions) - expected_ids
        _require(not unknown, f"unknown prediction IDs: {sorted(unknown)[:3]}")
        for job_id, row in predictions.items():
            _require(
                row["annotation"].get("document_id")
                == expected[job_id].get("document_id"),
                f"prediction document ID mismatch: {job_id}",
            )
        terminal_ids, failure_accounting = self.terminal_accounting(
            spec, dataset, set(predictions)
        )
        if spec.terminal_mode == "qwen_logs":
            _require(
                set(predictions) | terminal_ids == expected_ids,
                "prediction plus terminal-failure IDs do not cover validation jobs",
            )
            _require(not (set(predictions) & terminal_ids), "Qwen outcome ID overlap")
        else:
            _require(set(predictions) == expected_ids, "prediction IDs do not cover validation jobs")
            _require(
                len(prediction_rows) == len(expected),
                "prediction row count differs from validation jobs",
            )
            if failure_accounting["policy"] == "terminal_failures_materialized_as_empty":
                for job_id in terminal_ids:
                    annotation = predictions[job_id]["annotation"]
                    _require(
                        annotation.get("entities") == [] and annotation.get("relations") == [],
                        f"terminal prediction was not materialized as empty: {job_id}",
                    )

        metric = self.tracker.json(spec.metric, f"metric:{spec.system_id}:{dataset}")
        protocol, headline = audit_metric(
            metric,
            spec.metric_kind,
            expected_ids,
            len(prediction_rows),
            len(terminal_ids) if spec.terminal_mode == "qwen_logs" else 0,
        )
        if spec.metric_kind == "legacy_annotation":
            expected_missing = len(terminal_ids)
            _require(
                metric.get("jobs_missing_predictions") == expected_missing,
                "legacy metric terminal-failure count mismatch",
            )
        elif metric.get("metric", "").startswith("strict-global-character-span"):
            audit_generation_success_rate(
                spec.family, metric, len(expected_ids), len(terminal_ids)
            )

        lineage = self.validate_lineage(spec, dataset)
        if spec.role == "uncalibrated_diagnostic":
            self.warnings.append(
                {
                    "scope": f"{spec.system_id}:{dataset}",
                    "warning": spec.note,
                }
            )
        return {
            "system_id": spec.system_id,
            "label": spec.label,
            "family": spec.family,
            "role": spec.role,
            "dataset": dataset,
            "audit_status": "passed",
            "selection_split": "validation",
            "formal_test_read": False,
            "seed": spec.seed,
            "parameters": spec.parameters,
            "lineage": lineage,
            "marker_id": spec.marker_id,
            "marker_status": marker_result["status"],
            "prediction": {
                "path": display_path(spec.prediction),
                "sha256": self.tracker.hash(
                    spec.prediction, f"prediction:{spec.system_id}:{dataset}"
                ),
                "rows": len(prediction_rows),
                "unique_job_ids": len(predictions),
                "expected_jobs": len(expected_ids),
                "accounted_jobs": len(set(predictions) | terminal_ids),
                "complete_coverage": (set(predictions) | terminal_ids) == expected_ids,
            },
            "terminal_failure_accounting": failure_accounting,
            "metric": {
                "path": display_path(spec.metric),
                "sha256": self.tracker.hash(
                    spec.metric, f"metric:{spec.system_id}:{dataset}"
                ),
                "protocol": protocol,
                "jobs": len(expected_ids),
                "headline": headline,
            },
            "relation_baseline_valid": spec.relation_baseline_valid,
            "note": spec.note,
        }

    def audit_comparisons(self) -> list[dict[str, Any]]:
        results = []
        marker = next(
            (row for row in self.marker_results if row["marker_id"] == "post_pge"), None
        )
        for dataset in DATASETS:
            try:
                _require(
                    marker is not None and marker.get("audit_status") == "passed",
                    "post-PGE marker failed",
                )
                path = self.roots.post_pge / "comparisons" / f"{dataset}_soe_vs_pge.json"
                value = self.tracker.json(path, f"comparison:soe_vs_pge:{dataset}")
                _require(
                    value.get("schema_version") == "public-post-pge-soe-vs-pge-v1",
                    "post-PGE comparison schema mismatch",
                )
                _require(value.get("dataset") == dataset, "post-PGE dataset mismatch")
                _require(value.get("selection_split") == "validation", "comparison is not validation")
                _require(value.get("formal_test_read") is False, "comparison read formal test")
                _require(value.get("iterations") == 20_000, "comparison iterations mismatch")
                _require(value.get("seed") == 20_260_830, "comparison bootstrap seed mismatch")
                _require(value.get("documents") == len(self.jobs[dataset]), "comparison jobs mismatch")
                _require(
                    set(value.get("fields", {}))
                    == {"entity", "relation", "relation_with_claim_status"},
                    "comparison fields mismatch",
                )
                validate_finite_metrics(value)
                for system in ("soe", "pge"):
                    source_path = self.roots.pge / dataset / "metrics" / f"{system}_span.json"
                    validate_source_artifact_record(
                        value.get("source_artifacts", {}).get(system),
                        source_path,
                        self.tracker.hash(
                            source_path, f"comparison_source:{dataset}:{system}"
                        ),
                    )
                results.append(
                    {
                        "dataset": dataset,
                        "comparison": "SOE_vs_PGE",
                        "audit_status": "passed",
                        "path": display_path(path),
                        "sha256": self.tracker.hash(path, f"comparison:soe_vs_pge:{dataset}"),
                        "iterations": value["iterations"],
                        "seed": value["seed"],
                        "fields": value["fields"],
                    }
                )
            except Exception as error:
                self.fail(f"comparison:soe_vs_pge:{dataset}", error)
                results.append(
                    {
                        "dataset": dataset,
                        "comparison": "SOE_vs_PGE",
                        "audit_status": "failed",
                        "error": str(error),
                    }
                )
        return results

    def run(self) -> tuple[dict[str, Any], dict[str, Any]]:
        self.load_core_inputs()
        results: list[dict[str, Any]] = []
        for spec in self.system_specs:
            matching_datasets = [
                dataset
                for dataset in DATASETS
                if spec.prediction.name.startswith(f"{dataset}_")
                or dataset in spec.prediction.parts
            ]
            if len(matching_datasets) != 1:
                raise AuditError(
                    f"registered prediction path does not identify one dataset: {spec.prediction}"
                )
            dataset = matching_datasets[0]
            try:
                if dataset not in self.jobs:
                    raise AuditError(f"validation jobs unavailable for {dataset}")
                results.append(self.audit_system(spec, dataset))
            except Exception as error:
                self.fail(f"system:{spec.system_id}:{dataset}", error)
                results.append(
                    {
                        "system_id": spec.system_id,
                        "label": spec.label,
                        "family": spec.family,
                        "role": spec.role,
                        "dataset": dataset,
                        "audit_status": "failed",
                        "selection_split": "validation",
                        "formal_test_read": False,
                        "seed": spec.seed,
                        "parameters": spec.parameters,
                        "relation_baseline_valid": spec.relation_baseline_valid,
                        "note": spec.note,
                        "error": str(error),
                    }
                )
        comparisons = self.audit_comparisons()
        status = "complete" if not self.failures else "failed_audit"
        summary = {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "selection_split": "validation",
            "formal_test_read": False,
            "test_namespace_status": TEST_ACCESS_SEAL,
            "test_artifacts_opened": [],
            "generated_at": utc_now(),
            "datasets": list(DATASETS),
            "registry": {
                "mode": "closed_explicit_paths_no_discovery",
                "systems": sorted({spec.system_id for spec in self.system_specs}),
                "system_dataset_rows": len(self.system_specs),
                "post_pge_comparisons": len(comparisons),
            },
            "counts": {
                "passed_system_dataset_rows": sum(
                    row["audit_status"] == "passed" for row in results
                ),
                "failed_system_dataset_rows": sum(
                    row["audit_status"] == "failed" for row in results
                ),
                "passed_markers": sum(
                    row["audit_status"] == "passed" for row in self.marker_results
                ),
                "failed_markers": sum(
                    row["audit_status"] == "failed" for row in self.marker_results
                ),
                "passed_comparisons": sum(
                    row["audit_status"] == "passed" for row in comparisons
                ),
                "failed_comparisons": sum(
                    row["audit_status"] == "failed" for row in comparisons
                ),
                "errors": len(self.failures),
                "warnings": len(self.warnings),
            },
            "markers": self.marker_results,
            "results": results,
            "comparisons": comparisons,
            "failures": self.failures,
            "warnings": self.warnings,
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "selection_split": "validation",
            "formal_test_read": False,
            "test_namespace_status": TEST_ACCESS_SEAL,
            "registry_mode": "closed_explicit_paths_no_discovery",
            "files": self.tracker.manifest(),
        }
        return summary, manifest


def summary_tsv(summary: dict[str, Any]) -> str:
    columns = (
        "system_id",
        "label",
        "family",
        "role",
        "dataset",
        "audit_status",
        "metric_protocol",
        "jobs_expected",
        "prediction_rows",
        "terminal_failures",
        "accounted_jobs",
        "entity_precision",
        "entity_recall",
        "entity_f1",
        "relation_precision",
        "relation_recall",
        "relation_f1",
        "seed",
        "relation_baseline_valid",
        "parameters_json",
        "prediction_sha256",
        "metric_sha256",
        "note",
        "error",
    )
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in summary["results"]:
        metric = row.get("metric", {})
        headline = metric.get("headline", {})
        entity = headline.get("entity", {})
        relation = headline.get("relation", {})
        prediction = row.get("prediction", {})
        failures = row.get("terminal_failure_accounting", {})
        writer.writerow(
            {
                "system_id": row["system_id"],
                "label": row["label"],
                "family": row["family"],
                "role": row["role"],
                "dataset": row["dataset"],
                "audit_status": row["audit_status"],
                "metric_protocol": metric.get("protocol", ""),
                "jobs_expected": prediction.get("expected_jobs", ""),
                "prediction_rows": prediction.get("rows", ""),
                "terminal_failures": failures.get("terminal_failures", ""),
                "accounted_jobs": prediction.get("accounted_jobs", ""),
                "entity_precision": entity.get("precision", ""),
                "entity_recall": entity.get("recall", ""),
                "entity_f1": entity.get("f1", ""),
                "relation_precision": relation.get("precision", ""),
                "relation_recall": relation.get("recall", ""),
                "relation_f1": relation.get("f1", ""),
                "seed": row.get("seed", ""),
                "relation_baseline_valid": row.get("relation_baseline_valid", ""),
                "parameters_json": json.dumps(
                    row.get("parameters", {}), ensure_ascii=False, sort_keys=True
                ),
                "prediction_sha256": prediction.get("sha256", ""),
                "metric_sha256": metric.get("sha256", ""),
                "note": row.get("note", ""),
                "error": row.get("error", ""),
            }
        )
    return stream.getvalue()


def roots_from_args(args: argparse.Namespace) -> Roots:
    return Roots(
        data=args.data_root,
        stage1_run=args.stage1_run_root,
        stage1_analysis=args.stage1_analysis_root,
        pge=args.pge_root,
        qwen=args.qwen_root,
        spert=args.spert_root,
        glirel=args.glirel_root,
        glirel_t0=args.glirel_t0_root,
        glirel_calibrated=args.glirel_calibrated_root,
        glirel_calibration=args.glirel_calibration_root,
        gliner_entity=args.gliner_entity_root,
        post_pge=args.post_pge_root,
        pge_config=args.pge_config,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/public_benchmarks_full"))
    parser.add_argument("--stage1-run-root", type=Path, default=Path("outputs/public_full_stage1"))
    parser.add_argument(
        "--stage1-analysis-root",
        type=Path,
        default=Path("outputs/public_full_stage1/validation_analysis"),
    )
    parser.add_argument("--pge-root", type=Path, default=Path("outputs/public_pge_validation_seed42"))
    parser.add_argument(
        "--qwen-root",
        type=Path,
        default=Path("outputs/public_horizontal_validation/qwen3_4b_zero_shot"),
    )
    parser.add_argument(
        "--spert-root",
        type=Path,
        default=Path("outputs/public_horizontal_validation/spert_fresh"),
    )
    parser.add_argument(
        "--glirel-root",
        type=Path,
        default=Path("outputs/public_horizontal_validation/gliner_glirel"),
    )
    parser.add_argument(
        "--glirel-t0-root",
        type=Path,
        default=Path("outputs/public_horizontal_validation/gliner_glirel_t0"),
    )
    parser.add_argument(
        "--glirel-calibrated-root",
        type=Path,
        default=Path("outputs/public_horizontal_validation/gliner_glirel_calibrated"),
    )
    parser.add_argument(
        "--glirel-calibration-root",
        type=Path,
        default=Path("outputs/public_horizontal_validation/glirel_train_calibration"),
    )
    parser.add_argument(
        "--gliner-entity-root",
        type=Path,
        default=Path("outputs/public_horizontal_validation/gliner_entity_only"),
    )
    parser.add_argument(
        "--post-pge-root",
        type=Path,
        default=Path("outputs/public_post_pge_validation_seed42"),
    )
    parser.add_argument("--pge-config", type=Path, default=Path("configs/public_pge_seed42.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/public_validation_audit"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = assert_sealed_path(args.output_root, output=True)
    auditor = Auditor(roots_from_args(args), output_root)
    summary, manifest = auditor.run()
    atomic_json(output_root / "summary.json", summary)
    atomic_bytes(output_root / "summary.tsv", summary_tsv(summary).encode("utf-8"))
    atomic_json(output_root / "input_manifest.json", manifest)
    status = {
        "schema_version": SCHEMA_VERSION,
        "status": summary["status"],
        "selection_split": "validation",
        "formal_test_read": False,
        "test_namespace_status": TEST_ACCESS_SEAL,
        "test_artifacts_opened": [],
        "registry_mode": "closed_explicit_paths_no_discovery",
        "counts": summary["counts"],
        "artifacts": {
            "summary_json": display_path(output_root / "summary.json"),
            "summary_tsv": display_path(output_root / "summary.tsv"),
            "input_manifest_json": display_path(output_root / "input_manifest.json"),
        },
        "finished_at": utc_now(),
    }
    atomic_json(output_root / "status.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
