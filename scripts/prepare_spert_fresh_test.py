#!/usr/bin/env python3
"""Materialize and verify SpERT public-test inputs after canonical promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import prepare_public_test_inputs as release  # noqa: E402
import prepare_spert_fresh_splits as training  # noqa: E402
import qwen_zeroshot_formal_contract as formal_release  # noqa: E402


DATASETS = training.DATASETS
EXPECTED_TEST_ROWS = release.EXPECTED_TEST_JOBS
NATIVE_NAMES = {
    "conll04": "conll04_test.json",
    "scierc": "scierc_test.json",
    "ade": "ade_test.json",
}
SCHEMA_VERSION = "spert-fresh-formal-test-input-v1"
PROMOTION_PATH = release.PROMOTION_PATH
RELEASE_STATUS_PATH = Path("outputs/public_formal_matrix/release_status.json")
PUBLIC_PREPARED_ROOT = Path("data/processed/public_benchmarks_hrge_test_v1")
NATIVE_ROOT = Path("data/external/spert")
TRAINING_ROOT = Path("data/processed/spert_fresh_train_v1")
REFERENCE_ROOT = release.SOURCE_ROOT
OUTPUT_ROOT = Path("data/processed/spert_fresh_test_v1")
SPERT_VALIDATION_ROOT = Path("outputs/public_horizontal_validation/spert_fresh")


def require_project_cwd() -> None:
    if Path.cwd().resolve(strict=True) != PROJECT_ROOT.resolve(strict=True):
        raise ValueError(
            f"SpERT formal-test preparation must run from the canonical project root: {PROJECT_ROOT}"
        )


def assert_safe_output_root(path: Path) -> Path:
    candidate = path if path.is_absolute() else PROJECT_ROOT / path
    for component in (candidate, *candidate.parents):
        if component.is_symlink():
            raise ValueError(f"symlinked SpERT formal output root is forbidden: {path}")
    if candidate.exists() and not candidate.is_dir():
        raise ValueError(f"SpERT formal output root is not a directory: {path}")
    return candidate


def stable_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return (value if value.is_absolute() else Path.cwd() / value).resolve(strict=False)


def assert_regular(path: Path) -> Path:
    candidate = path if path.is_absolute() else Path.cwd() / path
    for component in (candidate, *candidate.parents):
        if component.is_symlink():
            raise ValueError(f"symlinked SpERT formal input is forbidden: {path}")
    try:
        mode = candidate.stat().st_mode
    except FileNotFoundError:
        raise FileNotFoundError(path) from None
    if not stat.S_ISREG(mode):
        raise ValueError(f"SpERT formal input is not a regular file: {path}")
    return candidate


def read_regular_snapshot(path: Path) -> bytes:
    candidate = assert_regular(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as error:
        raise ValueError(f"cannot securely open SpERT formal input {path}: {error}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"SpERT formal input is not a regular file: {path}")
        resolved_fd = Path(f"/proc/self/fd/{descriptor}").resolve(strict=True)
        if resolved_fd != candidate.resolve(strict=True):
            raise ValueError(f"SpERT formal input changed during secure open: {path}")
        chunks = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        stable = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if stable != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError(f"SpERT formal input changed while being read: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


class Tracker:
    def __init__(self) -> None:
        self.payloads: dict[Path, bytes] = {}

    def bytes(self, path: Path) -> bytes:
        candidate = assert_regular(path)
        key = candidate.resolve(strict=True)
        if key not in self.payloads:
            self.payloads[key] = read_regular_snapshot(candidate)
        return self.payloads[key]

    def json(self, path: Path) -> Any:
        try:
            return json.loads(self.bytes(path).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid JSON at {path}: {error}") from error

    def jsonl(self, path: Path) -> list[dict[str, Any]]:
        rows = []
        try:
            text = self.bytes(path).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"invalid UTF-8 at {path}") from error
        for number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL row at {path}:{number}")
            rows.append(row)
        return rows

    def identity(self, path: Path) -> dict[str, Any]:
        payload = self.bytes(path)
        return {
            "path": str(path), "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }


def _identity_matches(record: Any, actual: dict[str, Any], *, bytes_optional: bool = False) -> bool:
    return (
        isinstance(record, dict)
        and isinstance(record.get("path"), str)
        and _resolve(record["path"]) == _resolve(actual["path"])
        and record.get("sha256") == actual["sha256"]
        and (bytes_optional or record.get("bytes") == actual["bytes"])
    )


def _validation_lineage(
    dataset: str, promotion: dict[str, Any], tracker: Tracker,
    training_root: Path, validation_root: Path,
) -> dict[str, Any]:
    summary_path = Path(promotion["inputs"]["summary_json"]["path"])
    summary = tracker.json(summary_path)
    matches = [
        row for row in summary.get("results", [])
        if row.get("system_id") == "spert_fresh_seed42" and row.get("dataset") == dataset
    ]
    if len(matches) != 1 or matches[0].get("audit_status") != "passed":
        raise ValueError(f"audited SpERT validation lineage is missing for {dataset}")
    lineage = matches[0].get("lineage")
    if not isinstance(lineage, list):
        raise ValueError(f"audited SpERT lineage is malformed for {dataset}")
    expected_lineage = {
        _resolve(validation_root / "status.json"),
        _resolve(validation_root / "preflight.json"),
    }
    if {_resolve(item.get("path", "")) for item in lineage} != expected_lineage:
        raise ValueError(f"audited SpERT lineage paths drifted for {dataset}")
    verified = {}
    for item in lineage:
        actual = tracker.identity(Path(item["path"]))
        if item.get("sha256") != actual["sha256"]:
            raise ValueError(f"audited SpERT lineage hash changed for {dataset}")
        verified[Path(item["path"]).name] = actual
    preflight = tracker.json(validation_root / "preflight.json")
    status = tracker.json(validation_root / "status.json")
    train_manifest_path = training_root / dataset / "manifest.json"
    train_manifest_identity = tracker.identity(train_manifest_path)
    if (
        preflight.get("status") != "ready"
        or preflight.get("test_split_access") != "forbidden-and-not-materialized"
        or preflight.get("datasets", {}).get(dataset, {}).get("manifest_sha256")
        != train_manifest_identity["sha256"]
        or status.get("status") != "complete"
        or status.get("split") != "validation"
        or status.get("seed") != 42
        or status.get("test_split_access") != "forbidden-and-not-read"
    ):
        raise ValueError(f"SpERT train/validation envelope is invalid for {dataset}")
    return {
        "audit_result": {
            "prediction": matches[0]["prediction"],
            "metric": matches[0]["metric"],
            "lineage": lineage,
        },
        "status": verified["status.json"],
        "preflight": verified["preflight.json"],
        "training_manifest": train_manifest_identity,
    }


def _validate_training_manifest(
    dataset: str, root: Path, tracker: Tracker
) -> tuple[dict[str, Any], dict[str, Path]]:
    target = root / dataset
    manifest_path = target / "manifest.json"
    manifest = tracker.json(manifest_path)
    if (
        manifest.get("dataset") != dataset
        or manifest.get("protocol") != training.PROTOCOL
        or manifest.get("test_split_access") != "forbidden-and-not-materialized"
        or set(manifest.get("rows", {})) != {"train", "validation"}
        or manifest["rows"].get("validation") != release.promoter.EXPECTED_VALIDATION_JOBS[dataset]
    ):
        raise ValueError(f"SpERT training manifest contract is invalid for {dataset}")
    paths = {
        "train": target / "train.json",
        "validation": target / "validation.json",
        "types": target / "types.json",
    }
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != {
        "train.json", "validation.json", "types.json"
    }:
        raise ValueError(f"SpERT training output registry is invalid for {dataset}")
    for key, path in paths.items():
        record = outputs[f"{key}.json"]
        actual = tracker.identity(path)
        if not _identity_matches(record, actual, bytes_optional=True):
            raise ValueError(f"SpERT training output hash mismatch for {dataset}/{key}")
    train_rows, validation_rows = tracker.json(paths["train"]), tracker.json(paths["validation"])
    types = tracker.json(paths["types"])
    training.validate_rows(dataset, "train", train_rows, types)
    training.validate_rows(dataset, "validation", validation_rows, types)
    return manifest, paths


def _verify_alignment(
    dataset: str,
    native_rows: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    gold: list[dict[str, Any]],
) -> None:
    release.validate_source_jobs(Path(f"{dataset}/test_baseline_jobs.jsonl"), jobs, dataset, "test")
    if len(native_rows) != EXPECTED_TEST_ROWS[dataset] or len(jobs) != len(native_rows):
        raise ValueError(f"SpERT formal-test count mismatch for {dataset}")
    prefix = f"{dataset}_test_"
    gold_by_id: dict[str, dict[str, Any]] = {}
    for annotation in gold:
        document_id = annotation.get("document_id")
        if not isinstance(document_id, str) or not document_id.startswith(prefix):
            raise ValueError(f"public test gold has invalid document ID for {dataset}")
        identifier = document_id[len(prefix):]
        if identifier in gold_by_id:
            raise ValueError(f"public test gold has duplicate ID for {dataset}: {identifier}")
        gold_by_id[identifier] = annotation
    if len(gold_by_id) != EXPECTED_TEST_ROWS[dataset]:
        raise ValueError(f"public test gold count mismatch for {dataset}")
    for native, job in zip(native_rows, jobs, strict=True):
        identifier = training.row_id(native)
        expected_document = f"{prefix}{identifier}"
        if job.get("document_id") != expected_document or job.get("job_id") != f"{expected_document}_C1":
            raise ValueError(f"SpERT native/job order or ID mismatch for {dataset}/{identifier}")
        text = " ".join(native["tokens"])
        if job["segments"][0].get("text") != text:
            raise ValueError(f"SpERT native/job text mismatch for {dataset}/{identifier}")
        reference = gold_by_id.get(identifier)
        if reference is None:
            raise ValueError(f"SpERT native row lacks public gold for {dataset}/{identifier}")
        native_fingerprint = training.native_annotation_fingerprint(native)
        reference_fingerprint = training.reference_annotation_fingerprint(reference)
        if (
            native_fingerprint["entities"] != reference_fingerprint["entities"]
            or native_fingerprint["relations"] != reference_fingerprint["relations"]
            or (
                reference_fingerprint["text"] is not None
                and native_fingerprint["text"] != reference_fingerprint["text"]
            )
        ):
            raise ValueError(f"SpERT native/public-gold mismatch for {dataset}/{identifier}")


def verified_inputs(
    dataset: str,
    promotion_path: Path,
    native_root: Path,
    training_root: Path,
    reference_root: Path,
    validation_root: Path,
    release_status_path: Path = RELEASE_STATUS_PATH,
    public_prepared_root: Path = PUBLIC_PREPARED_ROOT,
) -> dict[str, Any]:
    require_project_cwd()
    # This must occur before constructing or opening any SpERT/public test
    # path.  The canonical release verifier recomputes promotion-v2 and checks
    # every released public preparation manifest; a status-only marker is not
    # sufficient.
    try:
        release_value = formal_release.verify_release(
            release_status_path, promotion_path, public_prepared_root
        )
    except Exception as error:
        raise ValueError(f"canonical formal release is not complete: {error}") from error
    promotion = release.validate_promotion(promotion_path)
    tracker = Tracker()
    promotion_identity = tracker.identity(promotion_path)
    if tracker.json(promotion_path) != promotion:
        raise ValueError("promotion changed during SpERT test preparation")
    if (
        release_value.get("promotion", {}).get("sha256") != promotion_identity["sha256"]
        or release_value.get("promotion", {}).get("attestation_sha256")
        != promotion.get("attestation_sha256")
    ):
        raise ValueError("canonical release/promotion identity changed during SpERT preparation")
    lineage = _validation_lineage(
        dataset, promotion, tracker, training_root, validation_root
    )
    train_manifest, train_paths = _validate_training_manifest(dataset, training_root, tracker)
    if lineage["training_manifest"]["sha256"] != tracker.identity(
        training_root / dataset / "manifest.json"
    )["sha256"]:
        raise ValueError(f"SpERT training manifest lineage mismatch for {dataset}")

    native_path = native_root / dataset / NATIVE_NAMES[dataset]
    jobs_path = reference_root / dataset / "test_baseline_jobs.jsonl"
    gold_path = reference_root / dataset / "test_gold.jsonl"
    native_rows = tracker.json(native_path)
    types = tracker.json(train_paths["types"])
    jobs = tracker.jsonl(jobs_path)
    gold = tracker.jsonl(gold_path)
    training.validate_rows(dataset, "test", native_rows, types)
    _verify_alignment(dataset, native_rows, jobs, gold)
    train_rows = tracker.json(train_paths["train"])
    validation_rows = tracker.json(train_paths["validation"])
    test_ids = {training.row_id(row) for row in native_rows}
    if test_ids & {training.row_id(row) for row in train_rows}:
        raise ValueError(f"SpERT train/test orig_id overlap for {dataset}")
    if test_ids & {training.row_id(row) for row in validation_rows}:
        raise ValueError(f"SpERT validation/test orig_id overlap for {dataset}")
    inputs = {
        "promotion": promotion_identity,
        "native_test": tracker.identity(native_path),
        "test_jobs": tracker.identity(jobs_path),
        "test_gold_alignment_only": tracker.identity(gold_path),
        "training_manifest": tracker.identity(training_root / dataset / "manifest.json"),
        "training_types": tracker.identity(train_paths["types"]),
        "training_train": tracker.identity(train_paths["train"]),
        "training_validation": tracker.identity(train_paths["validation"]),
        "preparation_runner": tracker.identity(Path(__file__).resolve()),
    }
    fingerprint_payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "promotion_attestation_sha256": promotion["attestation_sha256"],
        "inputs": inputs,
        "expected_rows": EXPECTED_TEST_ROWS[dataset],
        "training_protocol": train_manifest["protocol"],
        "validation_lineage": lineage,
        "alignment": "native_test_equals_public_test_gold_and_job_order",
        "canonical_release_fingerprint": release_value["canonical_fingerprint"],
    }
    try:
        final_release_value = formal_release.verify_release(
            release_status_path, promotion_path, public_prepared_root
        )
    except Exception as error:
        raise ValueError(f"canonical formal release changed during preparation: {error}") from error
    if final_release_value != release_value:
        raise ValueError("canonical formal release identity changed during SpERT preparation")
    return {
        "promotion": promotion,
        "tracker": tracker,
        "native_rows": native_rows,
        "types": types,
        "inputs": inputs,
        "lineage": lineage,
        "release": {
            "release_status": release_value["release_status"],
            "promotion": release_value["promotion"],
            "canonical_fingerprint": release_value["canonical_fingerprint"],
        },
        "fingerprint_payload": fingerprint_payload,
        "fingerprint": stable_digest(fingerprint_payload),
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def staged_identity(staged: Path, final: Path) -> dict[str, Any]:
    payload = staged.read_bytes()
    return {"path": str(final), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def validate_manifest(
    dataset: str,
    manifest_path: Path,
    promotion_path: Path = PROMOTION_PATH,
    native_root: Path = NATIVE_ROOT,
    training_root: Path = TRAINING_ROOT,
    reference_root: Path = REFERENCE_ROOT,
    validation_root: Path = SPERT_VALIDATION_ROOT,
    release_status_path: Path = RELEASE_STATUS_PATH,
    public_prepared_root: Path = PUBLIC_PREPARED_ROOT,
) -> dict[str, Any]:
    context = verified_inputs(
        dataset, promotion_path, native_root, training_root, reference_root, validation_root,
        release_status_path, public_prepared_root,
    )
    tracker: Tracker = context["tracker"]
    manifest = tracker.json(manifest_path)
    target = manifest_path.parent
    expected = {
        "schema_version": SCHEMA_VERSION,
        "status": "prepared_test",
        "dataset": dataset,
        "split": "test",
        "formal_test_read": True,
        "test_gold_read_for_alignment": True,
        "test_gold_used_for_selection": False,
        "rows": EXPECTED_TEST_ROWS[dataset],
        "fingerprint": context["fingerprint"],
        "promotion_attestation_sha256": context["promotion"]["attestation_sha256"],
        "inputs": context["inputs"],
        "validation_lineage": context["lineage"],
        "canonical_release": context["release"],
        "orig_id_sha256": training.sha256_bytes(training.canonical_bytes(
            [training.row_id(row) for row in context["native_rows"]]
        )),
        "alignment": {
            "native_test_equals_public_test_gold": True,
            "native_test_equals_public_test_job_order_and_text": True,
            "train_validation_test_orig_ids_disjoint": True,
        },
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"SpERT test manifest has invalid {key}: {dataset}")
    output_paths = {"test.json": target / "test.json", "types.json": target / "types.json"}
    if set(manifest.get("outputs", {})) != set(output_paths):
        raise ValueError(f"SpERT test output registry is invalid for {dataset}")
    verified = {}
    for name, path in output_paths.items():
        actual = tracker.identity(path)
        if not _identity_matches(manifest["outputs"][name], actual):
            raise ValueError(f"SpERT test output hash mismatch for {dataset}/{name}")
        verified[name] = actual
    if tracker.json(output_paths["test.json"]) != context["native_rows"]:
        raise ValueError(f"SpERT materialized test rows differ from verified native rows: {dataset}")
    if tracker.json(output_paths["types.json"]) != context["types"]:
        raise ValueError(f"SpERT materialized types differ from training types: {dataset}")
    return {
        "dataset": dataset, "status": "verified_prepared_test",
        "manifest": tracker.identity(manifest_path), "outputs": verified,
        "fingerprint": context["fingerprint"],
    }


def prepare_dataset(
    dataset: str,
    promotion_path: Path = PROMOTION_PATH,
    native_root: Path = NATIVE_ROOT,
    training_root: Path = TRAINING_ROOT,
    reference_root: Path = REFERENCE_ROOT,
    output_root: Path = OUTPUT_ROOT,
    validation_root: Path = SPERT_VALIDATION_ROOT,
    release_status_path: Path = RELEASE_STATUS_PATH,
    public_prepared_root: Path = PUBLIC_PREPARED_ROOT,
) -> dict[str, Any]:
    # Inspect only output path metadata before the release gate; do not let a
    # pre-planted directory symlink redirect promoted test bytes elsewhere.
    assert_safe_output_root(output_root)
    context = verified_inputs(
        dataset, promotion_path, native_root, training_root, reference_root, validation_root,
        release_status_path, public_prepared_root,
    )
    target = output_root / dataset
    manifest_path = target / "manifest.json"
    if target.exists():
        try:
            return validate_manifest(
                dataset, manifest_path, promotion_path, native_root,
                training_root, reference_root, validation_root,
                release_status_path, public_prepared_root,
            )
        except (OSError, ValueError) as error:
            raise FileExistsError(f"refusing to overwrite invalid SpERT test input {target}: {error}") from error
    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{dataset}.spert-test.", dir=output_root))
    try:
        atomic_json(staging / "test.json", context["native_rows"])
        atomic_json(staging / "types.json", context["types"])
        outputs = {
            "test.json": staged_identity(staging / "test.json", target / "test.json"),
            "types.json": staged_identity(staging / "types.json", target / "types.json"),
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "prepared_test",
            "dataset": dataset,
            "split": "test",
            "formal_test_read": True,
            "test_gold_read_for_alignment": True,
            "test_gold_used_for_selection": False,
            "rows": EXPECTED_TEST_ROWS[dataset],
            "orig_id_sha256": training.sha256_bytes(training.canonical_bytes(
                [training.row_id(row) for row in context["native_rows"]]
            )),
            "fingerprint": context["fingerprint"],
            "promotion_attestation_sha256": context["promotion"]["attestation_sha256"],
            "inputs": context["inputs"],
            "validation_lineage": context["lineage"],
            "canonical_release": context["release"],
            "alignment": {
                "native_test_equals_public_test_gold": True,
                "native_test_equals_public_test_job_order_and_text": True,
                "train_validation_test_orig_ids_disjoint": True,
            },
            "outputs": outputs,
        }
        atomic_json(staging / "manifest.json", manifest)
        os.replace(staging, target)
        staging = Path()
        result = {
            "dataset": dataset, "status": "prepared_test", "rows": EXPECTED_TEST_ROWS[dataset],
            "manifest": str(manifest_path), "fingerprint": context["fingerprint"],
        }
        # Recompute the full release and input contract after publication so a
        # gate/source change during the atomic write cannot yield a reusable
        # prepared directory.
        validate_manifest(
            dataset, manifest_path, promotion_path, native_root,
            training_root, reference_root, validation_root,
            release_status_path, public_prepared_root,
        )
        return result
    finally:
        if staging != Path() and staging.exists():
            shutil.rmtree(staging)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.verify:
        rows = [validate_manifest(dataset, OUTPUT_ROOT / dataset / "manifest.json") for dataset in args.datasets]
        status = "verified_prepared_test"
    else:
        rows = [prepare_dataset(dataset) for dataset in args.datasets]
        status = "prepared_test"
    print(json.dumps({"status": status, "datasets": rows}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
