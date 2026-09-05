#!/usr/bin/env python3
"""Validate resumable GLiNER + GLiREL public formal-test artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from qwen_zeroshot_formal_contract import (  # noqa: E402
    ContractError,
    DATASETS,
    EXPECTED_JOBS,
    artifact_record,
    atomic_write_json,
    load_json,
    load_jsonl,
    require_job_ids,
    sha256,
    validate_test_inputs,
    verify_release as qwen_verify_release,
)


SCHEMA_VERSION = "gliner-glirel-calibrated-formal-test-v1"
PROTOCOL_SCHEMA = "gliner-glirel-calibrated-validation-v1"


def canonical_validate_prepared_release(
    promotion: Path, prepared_root: Path
) -> dict[str, Any]:
    """Independently invoke the preparation boundary's full verifier."""

    try:
        from prepare_public_test_inputs import validate_prepared_release

        return validate_prepared_release(
            promotion_path=promotion, output_root=prepared_root
        )
    except (ImportError, ValueError, OSError) as error:
        raise ContractError(
            f"GLiNER/GLiREL canonical prepared-release verification failed: {error}"
        ) from error


def verify_release(
    release_status: Path, promotion: Path, prepared_root: Path
) -> dict[str, Any]:
    """Require both the shared consumer gate and a direct canonical replay."""

    result = qwen_verify_release(release_status, promotion, prepared_root)
    canonical = canonical_validate_prepared_release(promotion, prepared_root)
    if (
        canonical.get("status") != "verified_release"
        or canonical.get("release_sha256")
        != result.get("prepared_release_sha256")
    ):
        raise ContractError(
            "GLiNER/GLiREL canonical prepared-release verification changed"
        )
    return result


def paths_for(data_root: Path, run_root: Path, dataset: str) -> dict[str, Path]:
    base = data_root / dataset
    prefix = run_root / f"{dataset}_test"
    return {
        "jobs": base / "test_baseline_jobs.jsonl",
        "gold": base / "test_gold.jsonl",
        "gold_index": base / "test_index.jsonl",
        "predictions": prefix.with_suffix(".jsonl"),
        "normalized_text_metrics": prefix.with_name(
            f"{prefix.name}.normalized_text_metrics.json"
        ),
        "character_span_metrics": prefix.with_name(
            f"{prefix.name}.character_span_metrics.json"
        ),
    }


def validate_protocol(path: Path) -> dict[str, Any]:
    protocol = load_json(path)
    if (
        protocol.get("schema_version") != PROTOCOL_SCHEMA
        or protocol.get("selection_split") != "train_inner_calibration"
        or protocol.get("validation_gold_used_for_selection") is not False
        or protocol.get("test_gold_used_for_selection") is not False
        or protocol.get("entity_threshold") != 0.5
        or protocol.get("top_k_after_signature_filter") != 1
    ):
        raise ContractError("GLiNER/GLiREL calibrated protocol header is not frozen")
    datasets = protocol.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != set(DATASETS):
        raise ContractError("GLiNER/GLiREL protocol must cover exactly three datasets")
    configurations: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        row = datasets[dataset]
        if not isinstance(row, dict):
            raise ContractError(f"{dataset}: invalid calibrated protocol row")
        label_mode = row.get("label_mode")
        relation_threshold = row.get("relation_threshold")
        if label_mode not in {"canonical", "naturalized"}:
            raise ContractError(f"{dataset}: invalid relation label mode")
        if (
            isinstance(relation_threshold, bool)
            or not isinstance(relation_threshold, (int, float))
            or not math.isfinite(float(relation_threshold))
            or not 0.0 <= float(relation_threshold) <= 1.0
            or row.get("entity_threshold") != 0.5
            or row.get("selection_split") != "train_inner_calibration"
        ):
            raise ContractError(f"{dataset}: invalid calibrated thresholds")
        calibration_path = Path(str(row.get("calibration", "")))
        expected_hash = row.get("calibration_sha256")
        if (
            not calibration_path.is_file()
            or not isinstance(expected_hash, str)
            or sha256(calibration_path) != expected_hash
        ):
            raise ContractError(f"{dataset}: calibration artifact identity changed")
        calibration = load_json(calibration_path)
        selected = calibration.get("selected_configuration")
        if not (
            calibration.get("schema_version") == "glirel-train-calibration-result-v1"
            and calibration.get("status") == "complete"
            and calibration.get("dataset") == dataset
            and calibration.get("split") == "train_inner_calibration"
            and calibration.get("validation_gold_read") is False
            and calibration.get("test_gold_read") is False
            and isinstance(selected, dict)
            and selected.get("label_mode") == label_mode
            and selected.get("threshold") == relation_threshold
            and selected.get("f1") == row.get("gold_ner_oracle_calibration_f1")
        ):
            raise ContractError(f"{dataset}: protocol differs from train calibration")
        configurations[dataset] = {
            "entity_threshold": 0.5,
            "relation_threshold": float(relation_threshold),
            "relation_label_mode": label_mode,
            "calibration": {
                "path": str(calibration_path),
                "sha256": expected_hash,
            },
        }
    return {
        "status": "valid",
        "path": str(path),
        "sha256": sha256(path),
        "selection_split": "train_inner_calibration",
        "test_gold_used_for_selection": False,
        "datasets": configurations,
    }


def validate_dataset(
    data_root: Path, run_root: Path, dataset: str, expected: int
) -> dict[str, Any]:
    validate_test_inputs(data_root, dataset, expected)
    paths = paths_for(data_root, run_root, dataset)
    jobs = load_jsonl(paths["jobs"])
    job_ids = require_job_ids(jobs, paths["jobs"])
    job_by_id = {row["job_id"]: row for row in jobs}
    predictions = load_jsonl(paths["predictions"])
    prediction_ids = require_job_ids(
        predictions, paths["predictions"], annotation=True
    )
    if prediction_ids != job_ids:
        raise ContractError(
            f"{dataset}: predictions do not exactly cover test jobs in source order"
        )
    for row in predictions:
        annotation = row.get("annotation", row)
        if annotation.get("document_id") != job_by_id[row["job_id"]].get(
            "document_id"
        ):
            raise ContractError(f"{dataset}: prediction document identity mismatch")

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
        spans.get("metric") == "strict-source-character-span"
        and spans.get("selection_split") == "test"
        and spans.get("formal_test_read") is True
        and spans.get("jobs_gold") == expected
        and spans.get("jobs_predicted") == expected
        and spans.get("jobs_evaluated") == expected
        and spans.get("jobs_missing_predictions") == 0
        and spans.get("generation_success_rate") == 1.0
        and isinstance(per_job, dict)
        and set(per_job) == set(job_ids)
    ):
        raise ContractError(f"{dataset}: character-span metric coverage is incomplete")
    return {
        "status": "complete",
        "dataset": dataset,
        "split": "test",
        "formal_test_read": True,
        "expected_jobs": expected,
        "prediction_rows": len(predictions),
        "normalized_text_metrics_complete": True,
        "character_span_metrics_complete": True,
    }


def inspect_dataset(
    data_root: Path, run_root: Path, dataset: str, formal_test_read: bool
) -> dict[str, Any]:
    paths = paths_for(data_root, run_root, dataset)
    source_names = {"jobs", "gold", "gold_index"}
    result: dict[str, Any] = {
        "split": "test",
        "formal_test_read": formal_test_read,
        "expected_jobs": EXPECTED_JOBS[dataset],
        "completion_valid": False,
        "artifacts": {
            name: artifact_record(
                path, sealed=(not formal_test_read and name in source_names)
            )
            for name, path in paths.items()
        },
    }
    if not formal_test_read:
        return result
    try:
        result.update(
            validate_dataset(
                data_root, run_root, dataset, EXPECTED_JOBS[dataset]
            )
        )
        result["completion_valid"] = True
    except (ContractError, OSError) as error:
        result["validation_error"] = str(error)
        try:
            result["prediction_rows"] = len(load_jsonl(paths["predictions"]))
        except (ContractError, OSError):
            result["prediction_rows"] = None
    return result


def build_status(
    *,
    state: str,
    stage: str,
    active_dataset: str | None,
    formal_test_read: bool,
    workers: int,
    requested_workers: int,
    data_root: Path,
    run_root: Path,
    protocol_path: Path,
    protocol_sha256: str | None,
    release_status: Path,
    release_sha256: str | None,
    release_fingerprint: str | None,
    promotion: Path,
    prepared_root: Path,
    gpu_lock: Path,
    error: str | None = None,
) -> dict[str, Any]:
    datasets = {
        dataset: inspect_dataset(data_root, run_root, dataset, formal_test_read)
        for dataset in DATASETS
    }
    release_record = artifact_record(release_status)
    release_record.update(
        {
            "captured_sha256": release_sha256,
            "captured_canonical_fingerprint": release_fingerprint,
        }
    )
    protocol_record = artifact_record(protocol_path)
    protocol_record["captured_sha256"] = protocol_sha256
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": state,
        "stage": stage,
        "active_dataset": active_dataset,
        "split": "test",
        "formal_test_read": formal_test_read,
        "parallel_workers": workers,
        "requested_parallel_workers": requested_workers,
        "release": release_record,
        "promotion": artifact_record(promotion),
        "prepared_root": str(prepared_root),
        "protocol": protocol_record,
        "gpu_lock": {
            "path": str(gpu_lock),
            "exclusive_flock_fd": 8,
            "scope": "per-dataset-inference-only",
        },
        "expected_jobs": EXPECTED_JOBS,
        "expected_total_jobs": sum(EXPECTED_JOBS.values()),
        "datasets": datasets,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if error:
        payload["error"] = error
    if state == "complete":
        invalid = [
            dataset
            for dataset, row in datasets.items()
            if row.get("completion_valid") is not True
        ]
        if invalid:
            raise ContractError(f"cannot complete; invalid datasets: {invalid}")
        protocol = validate_protocol(protocol_path)
        if not protocol_sha256 or protocol["sha256"] != protocol_sha256:
            raise ContractError("cannot complete; calibrated protocol identity changed")
        canonical = verify_release(release_status, promotion, prepared_root)
        if not release_sha256 or canonical["release_status"]["sha256"] != release_sha256:
            raise ContractError("cannot complete; formal release marker changed")
        if (
            not release_fingerprint
            or canonical["canonical_fingerprint"] != release_fingerprint
        ):
            raise ContractError(
                "cannot complete; release/promotion/preparation fingerprint changed"
            )
        payload["canonical_release_verification"] = canonical
        payload["validated_protocol"] = protocol
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    release = subparsers.add_parser("verify-release")
    release.add_argument("--release-status", type=Path, required=True)
    release.add_argument("--promotion", type=Path, required=True)
    release.add_argument("--prepared-root", type=Path, required=True)
    protocol = subparsers.add_parser("validate-protocol")
    protocol.add_argument("--protocol", type=Path, required=True)
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
    status.add_argument("--workers", type=int, required=True)
    status.add_argument("--requested-workers", type=int, required=True)
    status.add_argument("--data-root", type=Path, required=True)
    status.add_argument("--run-root", type=Path, required=True)
    status.add_argument("--protocol", type=Path, required=True)
    status.add_argument("--protocol-sha256")
    status.add_argument("--release-status", type=Path, required=True)
    status.add_argument("--release-sha256")
    status.add_argument("--release-fingerprint")
    status.add_argument("--promotion", type=Path, required=True)
    status.add_argument("--prepared-root", type=Path, required=True)
    status.add_argument("--gpu-lock", type=Path, required=True)
    status.add_argument("--error")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "verify-release":
            result = verify_release(
                args.release_status, args.promotion, args.prepared_root
            )
        elif args.command == "validate-protocol":
            result = validate_protocol(args.protocol)
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
                workers=args.workers,
                requested_workers=args.requested_workers,
                data_root=args.data_root,
                run_root=args.run_root,
                protocol_path=args.protocol,
                protocol_sha256=args.protocol_sha256,
                release_status=args.release_status,
                release_sha256=args.release_sha256,
                release_fingerprint=args.release_fingerprint,
                promotion=args.promotion,
                prepared_root=args.prepared_root,
                gpu_lock=args.gpu_lock,
                error=args.error,
            )
            atomic_write_json(args.output, result)
        else:  # pragma: no cover
            raise AssertionError(args.command)
    except (ContractError, OSError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
