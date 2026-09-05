#!/usr/bin/env python3
"""Wait for PGE validation, then compare SOE and PGE on CPU only.

The worker intentionally has no dataset/test-file arguments.  Its only model
inputs are the already-produced strict validation metric artifacts below the
PGE run root.  This keeps the post-processing lane independent from formal
test data and makes it safe to queue before the PGE run has finished.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("conll04", "scierc", "ade")
EXPECTED_VALIDATION_JOBS = {"conll04": 231, "scierc": 275, "ade": 384}
FIELDS = ("entity", "relation", "relation_with_claim_status")
ITERATIONS = 20_000
BOOTSTRAP_SEED = 20_260_830
BOOTSTRAP_METRIC = "strict-global-character-span-one-to-one"
ALLOWED_METRICS = {
    BOOTSTRAP_METRIC,
}
SCHEMA_VERSION = "public-post-pge-soe-vs-pge-v1"
BOOTSTRAP_SCRIPT = REPOSITORY_ROOT / "scripts" / "bootstrap_span_compare.py"
BOOTSTRAP_DEPENDENCY = REPOSITORY_ROOT / "scripts" / "bootstrap_compare.py"
EXPECTED_SOURCE_NAMES = {"left": "soe_span.json", "right": "pge_span.json"}
EXPECTED_SYSTEM_ROLES = {"left": "SOE", "right": "PGE"}
OBSERVED_KEYS = {
    "pooled_f1",
    "macro_f1",
    "gold",
    "predicted",
    "correct",
    "pooled_f1_ci95",
    "macro_f1_ci95",
}
DIFFERENCE_KEYS = {
    "pooled_f1",
    "macro_f1",
    "pooled_f1_ci95",
    "macro_f1_ci95",
    "paired_permutation_p_pooled_f1",
    "paired_permutation_p_macro_f1",
}


class ProtocolError(RuntimeError):
    """Raised when an input could violate the frozen validation protocol."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise ProtocolError(f"invalid JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ProtocolError(f"expected a JSON object in {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    """Use stable repo-relative paths while retaining resolved identity checks."""

    resolved = path.resolve(strict=True)
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(resolved)


def validate_validation_artifact(value: dict[str, Any], path: Path) -> None:
    if value.get("selection_split") != "validation":
        raise ProtocolError(f"refusing non-validation artifact: {path}")
    if value.get("formal_test_read") is not False:
        raise ProtocolError(f"refusing artifact without formal_test_read=false: {path}")


def validate_upstream(status_path: Path, manifest_path: Path) -> dict[str, Any]:
    status = load_json(status_path)
    if status.get("status") != "complete":
        raise ProtocolError(f"PGE upstream is not complete: {status_path}")
    validate_validation_artifact(status, status_path)
    if status.get("seed") != 42:
        raise ProtocolError(f"expected frozen PGE seed 42 in {status_path}")

    manifest = load_json(manifest_path)
    if manifest.get("status") != "complete":
        raise ProtocolError(f"PGE run manifest is not complete: {manifest_path}")
    validate_validation_artifact(manifest, manifest_path)
    if manifest.get("seed") != 42:
        raise ProtocolError(f"expected frozen PGE seed 42 in {manifest_path}")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ProtocolError(f"PGE run manifest has no file hashes: {manifest_path}")
    for name, digest in files.items():
        if not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64:
            raise ProtocolError(f"PGE run manifest has an invalid file record: {manifest_path}")
    return manifest


def validate_span_metric(path: Path) -> dict[str, Any]:
    value = load_json(path)
    validate_validation_artifact(value, path)
    if value.get("metric") not in ALLOWED_METRICS:
        raise ProtocolError(f"not a supported strict span metric artifact: {path}")
    units = value.get("per_document", value.get("per_job"))
    if not isinstance(units, dict) or not units:
        raise ProtocolError(f"strict span metric has no evaluation units: {path}")
    return value


def metric_units(value: dict[str, Any], path: Path) -> tuple[str, dict[str, Any]]:
    present = [key for key in ("per_document", "per_job") if key in value]
    if len(present) != 1:
        raise ProtocolError(f"strict span metric must contain exactly one unit mapping: {path}")
    units = value[present[0]]
    if not isinstance(units, dict) or not units:
        raise ProtocolError(f"strict span metric has no evaluation units: {path}")
    return present[0], units


def validate_count_triplet(value: Any, path: Path, unit_id: str, field: str) -> None:
    if not isinstance(value, dict):
        raise ProtocolError(f"{path}: {unit_id}/{field} has no count object")
    counts: dict[str, int] = {}
    for key in ("gold", "predicted", "correct"):
        item = value.get(key)
        if not is_integer(item) or item < 0:
            raise ProtocolError(f"{path}: {unit_id}/{field}/{key} is not a non-negative integer")
        counts[key] = item
    if counts["correct"] > min(counts["gold"], counts["predicted"]):
        raise ProtocolError(f"{path}: {unit_id}/{field} has impossible counts")


def resolve_manifest_key(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    return candidate.resolve(strict=False)


def require_manifest_hash(manifest: dict[str, Any], path: Path) -> None:
    resolved = path.resolve(strict=True)
    matches = [
        digest
        for raw_path, digest in manifest["files"].items()
        if resolve_manifest_key(raw_path) == resolved
    ]
    if len(matches) != 1:
        raise ProtocolError(f"PGE manifest must contain exactly one hash for {path}")
    actual = sha256(resolved)
    if matches[0] != actual:
        raise ProtocolError(f"PGE manifest hash mismatch for {path}")


def validate_metric_pair(
    dataset: str,
    left_path: Path,
    right_path: Path,
    manifest: dict[str, Any],
    expected_jobs: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    if left_path.name != EXPECTED_SOURCE_NAMES["left"] or right_path.name != EXPECTED_SOURCE_NAMES["right"]:
        raise ProtocolError(f"{dataset}: SOE/PGE source metric filenames are not frozen")
    require_manifest_hash(manifest, left_path)
    require_manifest_hash(manifest, right_path)
    left = validate_span_metric(left_path)
    right = validate_span_metric(right_path)
    if left["metric"] != right["metric"]:
        raise ProtocolError(f"{dataset}: SOE/PGE metric types differ")
    expected_jobs = (
        EXPECTED_VALIDATION_JOBS[dataset] if expected_jobs is None else expected_jobs
    )
    if not is_integer(expected_jobs) or expected_jobs <= 0:
        raise ProtocolError(f"{dataset}: invalid frozen validation job count")
    left_kind, left_units = metric_units(left, left_path)
    right_kind, right_units = metric_units(right, right_path)
    if left_kind != right_kind:
        raise ProtocolError(f"{dataset}: SOE/PGE evaluation unit types differ")
    if set(left_units) != set(right_units):
        raise ProtocolError(f"{dataset}: SOE/PGE evaluation unit IDs differ")
    if len(left_units) != expected_jobs:
        raise ProtocolError(
            f"{dataset}: metric evaluation units are incomplete "
            f"({len(left_units)} != {expected_jobs})"
        )
    for role, metric in (("SOE", left), ("PGE", right)):
        if metric.get("jobs") != expected_jobs or metric.get("documents") != expected_jobs:
            raise ProtocolError(
                f"{dataset}: {role} metric jobs/documents do not match the frozen "
                f"validation count {expected_jobs}"
            )
    expected_prefix = f"{dataset}_validation_"
    documents: set[str] = set()
    for unit_id in sorted(left_units):
        if not isinstance(unit_id, str) or not unit_id.startswith(expected_prefix):
            raise ProtocolError(f"{dataset}: non-validation evaluation unit {unit_id!r}")
        left_item = left_units[unit_id]
        right_item = right_units[unit_id]
        if not isinstance(left_item, dict) or not isinstance(right_item, dict):
            raise ProtocolError(f"{dataset}: invalid evaluation unit {unit_id!r}")
        left_document = left_item.get("document_id")
        right_document = right_item.get("document_id")
        if (
            not isinstance(left_document, str)
            or not left_document.startswith(expected_prefix)
            or left_document != right_document
        ):
            raise ProtocolError(f"{dataset}: SOE/PGE document mismatch for {unit_id}")
        documents.add(left_document)
        for field in FIELDS:
            validate_count_triplet(left_item.get(field), left_path, unit_id, field)
            validate_count_triplet(right_item.get(field), right_path, unit_id, field)
            if left_item[field]["gold"] != right_item[field]["gold"]:
                raise ProtocolError(f"{dataset}: SOE/PGE gold counts differ for {unit_id}/{field}")
    if len(documents) != expected_jobs:
        raise ProtocolError(
            f"{dataset}: metric document set is incomplete "
            f"({len(documents)} != {expected_jobs})"
        )
    return left, right, len(documents)


def source_record(path: Path) -> dict[str, str]:
    return {"path": display_path(path), "sha256": sha256(path.resolve(strict=True))}


def implementation_records() -> dict[str, dict[str, str]]:
    return {
        "bootstrap_span_compare": source_record(BOOTSTRAP_SCRIPT),
        "bootstrap_compare": source_record(BOOTSTRAP_DEPENDENCY),
    }


def output_path(output_root: Path, dataset: str) -> Path:
    return output_root / "comparisons" / f"{dataset}_soe_vs_pge.json"


def is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_number(value: Any, lower: float, upper: float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and lower <= float(value) <= upper
    )


def valid_interval(value: Any, lower: float, upper: float) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"lower", "upper"}
        and is_number(value.get("lower"), lower, upper)
        and is_number(value.get("upper"), lower, upper)
        and float(value["lower"]) <= float(value["upper"])
    )


def valid_observed(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != OBSERVED_KEYS:
        return False
    if not all(is_number(value.get(key), 0.0, 1.0) for key in ("pooled_f1", "macro_f1")):
        return False
    if not all(is_integer(value.get(key)) and value[key] >= 0 for key in ("gold", "predicted", "correct")):
        return False
    if value["correct"] > min(value["gold"], value["predicted"]):
        return False
    return valid_interval(value.get("pooled_f1_ci95"), 0.0, 1.0) and valid_interval(
        value.get("macro_f1_ci95"), 0.0, 1.0
    )


def valid_difference(value: Any, left: dict[str, Any], right: dict[str, Any]) -> bool:
    if not isinstance(value, dict) or set(value) != DIFFERENCE_KEYS:
        return False
    if not all(is_number(value.get(key), -1.0, 1.0) for key in ("pooled_f1", "macro_f1")):
        return False
    if not valid_interval(value.get("pooled_f1_ci95"), -1.0, 1.0) or not valid_interval(
        value.get("macro_f1_ci95"), -1.0, 1.0
    ):
        return False
    if not all(
        is_number(value.get(key), 0.0, 1.0)
        for key in (
            "paired_permutation_p_pooled_f1",
            "paired_permutation_p_macro_f1",
        )
    ):
        return False
    # The serialized statistic is defined as PGE (right) minus SOE (left).
    # Checking the observed deltas makes a swapped or mislabeled result fail
    # closed even when all of its outer metadata looks plausible.
    for key in ("pooled_f1", "macro_f1"):
        expected = round(float(right[key]) - float(left[key]), 4)
        if not math.isclose(float(value[key]), expected, rel_tol=0.0, abs_tol=1e-12):
            return False
    return True


def valid_comparison_payload(
    value: dict[str, Any],
    dataset: str,
    left: dict[str, str],
    right: dict[str, str],
    upstream_manifest: dict[str, str],
    expected_documents: int,
) -> bool:
    if Path(left.get("path", "")).name != EXPECTED_SOURCE_NAMES["left"]:
        return False
    if Path(right.get("path", "")).name != EXPECTED_SOURCE_NAMES["right"]:
        return False
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("dataset") != dataset
        or value.get("selection_split") != "validation"
        or value.get("formal_test_read") is not False
        or value.get("iterations") != ITERATIONS
        or value.get("seed") != BOOTSTRAP_SEED
        or value.get("metric") != BOOTSTRAP_METRIC
        or value.get("unit") != "document"
        or not is_integer(value.get("documents"))
        or value["documents"] != expected_documents
        or value.get("comparison") != "SOE_vs_PGE"
        or value.get("interpretation") != "right_minus_left"
        or value.get("system_roles") != EXPECTED_SYSTEM_ROLES
        or value.get("systems") != EXPECTED_SOURCE_NAMES
        or value.get("source_artifacts") != {"soe": left, "pge": right}
        or value.get("upstream_manifest") != upstream_manifest
        or value.get("statistical_implementation") != implementation_records()
        or value.get("cpu_only") is not True
        or not isinstance(value.get("generated_at"), str)
        or not value["generated_at"]
    ):
        return False
    fields = value.get("fields")
    if not isinstance(fields, dict) or set(fields) != set(FIELDS):
        return False
    for field in FIELDS:
        comparison = fields.get(field)
        if not isinstance(comparison, dict) or set(comparison) != {
            "left",
            "right",
            "right_minus_left",
        }:
            return False
        left_value = comparison["left"]
        right_value = comparison["right"]
        if not valid_observed(left_value) or not valid_observed(right_value):
            return False
        if left_value["gold"] != right_value["gold"]:
            return False
        if not valid_difference(comparison["right_minus_left"], left_value, right_value):
            return False
    return True


def comparison_is_current(
    path: Path,
    dataset: str,
    left: dict[str, str],
    right: dict[str, str],
    upstream_manifest: dict[str, str],
    expected_documents: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        value = load_json(path)
    except (OSError, ProtocolError):
        return False
    return valid_comparison_payload(
        value, dataset, left, right, upstream_manifest, expected_documents
    )


class Worker:
    def __init__(self, args: argparse.Namespace) -> None:
        self.pge_root = args.pge_root.resolve()
        self.output_root = args.output_root.resolve()
        self.bootstrap_script = BOOTSTRAP_SCRIPT
        self.poll_seconds = args.poll_seconds
        self.expected_validation_jobs = dict(
            getattr(args, "expected_validation_jobs", EXPECTED_VALIDATION_JOBS)
        )
        self.upstream_status = self.pge_root / "status.json"
        self.upstream_manifest = self.pge_root / "run_manifest.json"
        self.status_path = self.output_root / "status.json"
        self.active_dataset: str | None = None
        self.stage = "initializing"
        self.error: str | None = None

    def dataset_statuses(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for dataset in DATASETS:
            left_path = self.pge_root / dataset / "metrics" / "soe_span.json"
            right_path = self.pge_root / dataset / "metrics" / "pge_span.json"
            comparison = output_path(self.output_root, dataset)
            complete = False
            if (
                left_path.is_file()
                and right_path.is_file()
                and self.upstream_manifest.is_file()
                and comparison.is_file()
            ):
                try:
                    manifest = validate_upstream(
                        self.upstream_status, self.upstream_manifest
                    )
                    _, _, documents = validate_metric_pair(
                        dataset,
                        left_path,
                        right_path,
                        manifest,
                        self.expected_validation_jobs[dataset],
                    )
                    complete = comparison_is_current(
                        comparison,
                        dataset,
                        source_record(left_path),
                        source_record(right_path),
                        source_record(self.upstream_manifest),
                        documents,
                    )
                except (OSError, ProtocolError):
                    complete = False
            result[dataset] = {
                "status": "complete" if complete else "pending",
                "soe_metric": str(left_path),
                "pge_metric": str(right_path),
                "comparison": str(comparison),
            }
        return result

    def upstream_state(self) -> str | None:
        if not self.upstream_status.is_file():
            return None
        try:
            return load_json(self.upstream_status).get("status")
        except ProtocolError:
            return "invalid"

    def write_status(self, state: str) -> None:
        datasets = self.dataset_statuses()
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": state,
            "selection_split": "validation",
            "formal_test_read": False,
            "test_artifacts_opened": [],
            "comparison": "SOE_vs_PGE",
            "iterations": ITERATIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "cpu_only": True,
            "active_dataset": self.active_dataset,
            "stage": self.stage,
            "waits_for": f"{self.upstream_status}:complete",
            "upstream_status": self.upstream_state(),
            "completed_datasets": sum(
                item["status"] == "complete" for item in datasets.values()
            ),
            "datasets": datasets,
            "updated_at": utc_now(),
        }
        if self.error is not None:
            payload["error"] = self.error
        atomic_json(self.status_path, payload)

    def wait_for_pge(self) -> None:
        self.stage = "waiting_for_pge_validation"
        self.write_status("waiting_for_pge")
        while True:
            if self.upstream_status.is_file():
                try:
                    marker = load_json(self.upstream_status)
                except ProtocolError:
                    marker = {}
                if marker.get("status") == "complete":
                    validate_upstream(self.upstream_status, self.upstream_manifest)
                    return
            time.sleep(self.poll_seconds)
            self.write_status("waiting_for_pge")

    def compare_dataset(self, dataset: str) -> None:
        left_path = self.pge_root / dataset / "metrics" / "soe_span.json"
        right_path = self.pge_root / dataset / "metrics" / "pge_span.json"
        manifest = validate_upstream(self.upstream_status, self.upstream_manifest)
        _, _, documents = validate_metric_pair(
            dataset,
            left_path,
            right_path,
            manifest,
            self.expected_validation_jobs[dataset],
        )
        left = source_record(left_path)
        right = source_record(right_path)
        upstream_manifest = source_record(self.upstream_manifest)
        target = output_path(self.output_root, dataset)
        if comparison_is_current(
            target, dataset, left, right, upstream_manifest, documents
        ):
            return

        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.bootstrap.tmp")
        try:
            environment = dict(os.environ)
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": "",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                    "TOKENIZERS_PARALLELISM": "false",
                }
            )
            subprocess.run(
                [
                    sys.executable,
                    str(self.bootstrap_script),
                    "--left",
                    str(left_path),
                    "--right",
                    str(right_path),
                    "--iterations",
                    str(ITERATIONS),
                    "--seed",
                    str(BOOTSTRAP_SEED),
                    "--output",
                    str(temporary),
                ],
                cwd=REPOSITORY_ROOT,
                env=environment,
                check=True,
            )
            result = load_json(temporary)
            validate_validation_artifact(result, temporary)
            if result.get("metric") not in ALLOWED_METRICS:
                raise ProtocolError("bootstrap produced a non-strict metric result")
            if result.get("iterations") != ITERATIONS or result.get("seed") != BOOTSTRAP_SEED:
                raise ProtocolError("bootstrap result does not match the frozen protocol")
            result.update(
                {
                    "schema_version": SCHEMA_VERSION,
                    "dataset": dataset,
                    "comparison": "SOE_vs_PGE",
                    "system_roles": EXPECTED_SYSTEM_ROLES,
                    "source_artifacts": {"soe": left, "pge": right},
                    "upstream_manifest": upstream_manifest,
                    "statistical_implementation": implementation_records(),
                    "cpu_only": True,
                    "generated_at": utc_now(),
                }
            )
            if not valid_comparison_payload(
                result, dataset, left, right, upstream_manifest, documents
            ):
                raise ProtocolError("bootstrap produced an incomplete or inconsistent result")
            atomic_json(target, result)
        finally:
            temporary.unlink(missing_ok=True)

    def run(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        lock_path = self.output_root / "runner.lock"
        with lock_path.open("a", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(f"post-PGE bootstrap worker is already active: {lock_path}")
                return

            try:
                self.wait_for_pge()
                self.stage = "input_preflight"
                self.write_status("running")
                manifest = validate_upstream(self.upstream_status, self.upstream_manifest)
                for dataset in DATASETS:
                    validate_metric_pair(
                        dataset,
                        self.pge_root / dataset / "metrics" / "soe_span.json",
                        self.pge_root / dataset / "metrics" / "pge_span.json",
                        manifest,
                        self.expected_validation_jobs[dataset],
                    )

                self.stage = "paired_bootstrap"
                for dataset in DATASETS:
                    self.active_dataset = dataset
                    self.write_status("running")
                    self.compare_dataset(dataset)

                # Never publish a complete aggregate marker based only on the
                # loop having returned. Re-hash every source, the PGE manifest,
                # and both statistics implementations and verify all three
                # result bodies one final time.
                validate_upstream(self.upstream_status, self.upstream_manifest)
                upstream_manifest = source_record(self.upstream_manifest)
                for dataset in DATASETS:
                    left_path = self.pge_root / dataset / "metrics" / "soe_span.json"
                    right_path = self.pge_root / dataset / "metrics" / "pge_span.json"
                    _, _, documents = validate_metric_pair(
                        dataset,
                        left_path,
                        right_path,
                        validate_upstream(
                            self.upstream_status, self.upstream_manifest
                        ),
                        self.expected_validation_jobs[dataset],
                    )
                    if not comparison_is_current(
                        output_path(self.output_root, dataset),
                        dataset,
                        source_record(left_path),
                        source_record(right_path),
                        upstream_manifest,
                        documents,
                    ):
                        raise ProtocolError(f"{dataset}: comparison failed final currentness audit")
                self.active_dataset = None
                self.stage = "complete"
                self.write_status("complete")
            except Exception as error:
                self.error = f"{type(error).__name__}: {error}"
                self.stage = "failed"
                self.write_status("failed")
                raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pge-root",
        type=Path,
        default=Path(
            os.environ.get(
                "PUBLIC_PGE_RUN_ROOT", "outputs/public_pge_validation_seed42"
            )
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            os.environ.get(
                "PUBLIC_POST_PGE_RUN_ROOT",
                "outputs/public_post_pge_validation_seed42",
            )
        ),
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=float(os.environ.get("PUBLIC_POST_PGE_POLL_SECONDS", "30")),
    )
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    return args


def main() -> int:
    worker = Worker(parse_args())
    worker.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
