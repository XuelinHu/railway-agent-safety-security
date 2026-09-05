#!/usr/bin/env python3
"""Fail-closed contract for the frozen SpERT seed-42 formal-test run."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import convert_spert_test_predictions as converter  # noqa: E402
import evaluate_annotations as normalized_evaluator  # noqa: E402
import evaluate_public_validation_spans as span_evaluator  # noqa: E402
import prepare_spert_fresh_test as prepared  # noqa: E402
import qwen_zeroshot_formal_contract as release_contract  # noqa: E402


ROOT = SCRIPT_DIR.parent
DATASETS = ("conll04", "scierc", "ade")
EXPECTED_ROWS = {"conll04": 288, "scierc": 551, "ade": 427}
EXPECTED_VALIDATION_ROWS = {"conll04": 231, "scierc": 275, "ade": 384}
SCHEMA_VERSION = "spert-fresh-seed42-formal-test-v1"
SEED = 42
PROMOTION_PATH = Path("outputs/public_formal_matrix/promotion.json")
RELEASE_STATUS_PATH = Path("outputs/public_formal_matrix/release_status.json")
PUBLIC_PREPARED_ROOT = Path("data/processed/public_benchmarks_hrge_test_v1")
SPERT_PREPARED_ROOT = Path("data/processed/spert_fresh_test_v1")
DATA_ROOT = Path("data/processed/public_benchmarks_full")
VALIDATION_ROOT = Path("outputs/public_horizontal_validation/spert_fresh")
RUN_ROOT = Path("outputs/public_formal_matrix/horizontal/spert_fresh_seed42")
SPERT_REPO = Path("tools/external-baselines/spert")
SPERT_REPO_REVISION = "a53f468bebfa9de6d66456dcfbf4b62aef237bf7"
COMPAT_RUNNER = Path("scripts/run_spert_compat.py")
COMPAT_RUNNER_SHA256 = "4a8b976caf50e714b09025b2457b3f36ef29e89797098663eb587ea7d93528cc"
FORMAL_RUNNER = Path("scripts/run_spert_fresh_test.sh")


CHECKPOINT_SPECS: dict[str, dict[str, Any]] = {
    "conll04": {
        "pointer": "outputs/public_horizontal_validation/spert_fresh/conll04/seed42/final_model_path.txt",
        "pointer_sha256": "c5cd37704cb2cb0f0e73e5c8f4426d82d1fa3e9abc1d77a6d2e1892b3b4c3264",
        "checkpoint": "outputs/public_horizontal_validation/spert_fresh/conll04/seed42/models/spert_conll04_seed42/2026-09-04_20:17:52.339038/final_model",
        "train_args": "outputs/public_horizontal_validation/spert_fresh/conll04/seed42/train_logs/spert_conll04_seed42/2026-09-04_20:17:52.339038/args.json",
        "train_args_sha256": "1a55ceee033d85ea4b1130623056a0c595b676414d8fe6f6bdcc750bb13624fd",
        "extra_state": {"epoch": 20, "epoch_iteration": 0, "iteration": 9220, "updates_epoch": 461},
        "files": {
            "config.json": (634, "594ed2cbe6a78344de34dc827e91141265ee66367d90da7ed19a95cd6339a111"),
            "extra.state": (1313, "77cb571dec29914be000f2b0e8df055b9970f2e69ddf855ae834a279a39a8ea2"),
            "pytorch_model.bin": (433399507, "72bc32dae99ffac750ddfdf2ffed0be1260b8b3f6e5752f110602d93218712e5"),
            "special_tokens_map.json": (125, "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3"),
            "tokenizer_config.json": (1273, "470cff6e0353b08e2a6e9b4f61729ecdc47ccb3ced335fa5520e9ce334572d59"),
            "vocab.txt": (213450, "eeaa9875b23b04b4c54ef759d03db9d1ba1554838f8fb26c5d96fa551df93d02"),
        },
    },
    "scierc": {
        "pointer": "outputs/public_horizontal_validation/spert_fresh/scierc/seed42/final_model_path.txt",
        "pointer_sha256": "6d44e087ac3ae1016ccafb1c8cd6041973153a629214ee7d27154b9b308b5cde",
        "checkpoint": "outputs/public_horizontal_validation/spert_fresh/scierc/seed42/models/spert_scierc_seed42/2026-09-04_20:23:32.915750/final_model",
        "train_args": "outputs/public_horizontal_validation/spert_fresh/scierc/seed42/train_logs/spert_scierc_seed42/2026-09-04_20:23:32.915750/args.json",
        "train_args_sha256": "abaa7b595be8096bf41c6338ce7efe83425dda2f07a038dfb564b7eb75cf5b9e",
        "extra_state": {"epoch": 20, "epoch_iteration": 0, "iteration": 18600, "updates_epoch": 930},
        "files": {
            "config.json": (599, "44537d035ca96baf63e0a9e69427affc50d80f4060164577fe51b5aac2783ba1"),
            "extra.state": (1313, "92229852dcaafc1723c72a3de0aae9d9449d917d1ed414ce25b0c91806503e0c"),
            "pytorch_model.bin": (439863635, "780ab666c887b093e99281efb166ae5fd2aabbaaef234bf130ae4aedd2e82074"),
            "special_tokens_map.json": (125, "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3"),
            "tokenizer_config.json": (1301, "5e14ea3aa73d2606c5712f4d1358335c19e7393d8fd46b9c05d77c5b926dd1a0"),
            "vocab.txt": (227845, "f0a650346e51ede8996710f79ba65d83fdb8da05b159f17037b74ba4e3a36c6f"),
        },
    },
    "ade": {
        "pointer": "outputs/public_horizontal_validation/spert_fresh/ade/seed42/final_model_path.txt",
        "pointer_sha256": "8c1d1a52a6d41b4beff2796681e9d8534b896a6f094a0a97b28761e3aab3003f",
        "checkpoint": "outputs/public_horizontal_validation/spert_fresh/ade/seed42/models/spert_ade_seed42/2026-09-04_20:34:28.789653/final_model",
        "train_args": "outputs/public_horizontal_validation/spert_fresh/ade/seed42/train_logs/spert_ade_seed42/2026-09-04_20:34:28.789653/args.json",
        "train_args_sha256": "4833c50dd086ae64fdac390c365cf81e62f5c7b038c58689423d5111a4de95af",
        "extra_state": {"epoch": 20, "epoch_iteration": 0, "iteration": 34600, "updates_epoch": 1730},
        "files": {
            "config.json": (634, "594ed2cbe6a78344de34dc827e91141265ee66367d90da7ed19a95cd6339a111"),
            "extra.state": (1313, "82c2c08fa0c2c2e66d442095f5b6101a5e339fd97d116e8741058dbb13c0c348"),
            "pytorch_model.bin": (433349395, "3dbce3dd465050ba0c15a47bc0f66f76973adff0a4acf16c85e8bdc9d3b3ee8e"),
            "special_tokens_map.json": (125, "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3"),
            "tokenizer_config.json": (1273, "470cff6e0353b08e2a6e9b4f61729ecdc47ccb3ced335fa5520e9ce334572d59"),
            "vocab.txt": (213450, "eeaa9875b23b04b4c54ef759d03db9d1ba1554838f8fb26c5d96fa551df93d02"),
        },
    },
}


class ContractError(ValueError):
    """A frozen formal-test invariant was violated."""


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def reject_symlink_chain(path: Path) -> Path:
    candidate = absolute(path)
    for component in (candidate, *candidate.parents):
        if component.is_symlink():
            raise ContractError(f"symlinked formal-test path is forbidden: {path}")
    return candidate


def require_regular(path: Path) -> Path:
    candidate = reject_symlink_chain(path)
    try:
        mode = candidate.stat().st_mode
    except FileNotFoundError:
        raise ContractError(f"required regular file is missing: {path}") from None
    if not stat.S_ISREG(mode):
        raise ContractError(f"formal-test path is not a regular file: {path}")
    return candidate


def require_directory(path: Path) -> Path:
    candidate = reject_symlink_chain(path)
    try:
        mode = candidate.stat().st_mode
    except FileNotFoundError:
        raise ContractError(f"required directory is missing: {path}") from None
    if not stat.S_ISDIR(mode):
        raise ContractError(f"formal-test path is not a directory: {path}")
    return candidate


def read_bytes(path: Path) -> bytes:
    candidate = require_regular(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as error:
        raise ContractError(f"cannot securely open formal-test input {path}: {error}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ContractError(f"formal-test input is not regular: {path}")
        if Path(f"/proc/self/fd/{descriptor}").resolve(strict=True) != candidate.resolve(strict=True):
            raise ContractError(f"formal-test input changed during secure open: {path}")
        chunks = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        if (
            before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ContractError(f"formal-test input changed while being read: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def load_json(path: Path) -> Any:
    try:
        return json.loads(read_bytes(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"invalid JSON at {path}: {error}") from error


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        text = read_bytes(path).decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContractError(f"invalid UTF-8 at {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ContractError(f"invalid JSONL at {path}:{line_number}") from error
        if not isinstance(row, dict):
            raise ContractError(f"non-object JSONL row at {path}:{line_number}")
        rows.append(row)
    return rows


def identity(path: Path) -> dict[str, Any]:
    payload = read_bytes(path)
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _same_path(left: Any, right: Path) -> bool:
    return isinstance(left, str) and absolute(Path(left)).resolve(strict=False) == absolute(
        right
    ).resolve(strict=False)


def _identity_matches(record: Any, path: Path) -> bool:
    actual = identity(path)
    return (
        isinstance(record, dict)
        and set(record) == {"path", "bytes", "sha256"}
        and _same_path(record.get("path"), path)
        and record.get("bytes") == actual["bytes"]
        and record.get("sha256") == actual["sha256"]
    )


def atomic_json(path: Path, value: Any) -> None:
    target = absolute(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    reject_symlink_chain(target.parent)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=target.parent,
        prefix=f".{target.name}.", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)


def atomic_copy(source: Path, target: Path) -> None:
    payload = read_bytes(source)
    destination = absolute(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    reject_symlink_chain(destination.parent)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)


def stable_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _finite(value: Any, context: str = "root") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError(f"non-finite numeric metric at {context}")
        return
    if isinstance(value, list):
        for position, item in enumerate(value):
            _finite(item, f"{context}[{position}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _finite(item, f"{context}.{key}")
        return
    raise ContractError(f"unsupported value in formal artifact at {context}")


def verify_checkpoint(dataset: str) -> dict[str, Any]:
    if dataset not in DATASETS:
        raise ContractError(f"unsupported dataset: {dataset}")
    spec = CHECKPOINT_SPECS[dataset]
    pointer = Path(spec["pointer"])
    checkpoint = Path(spec["checkpoint"])
    pointer_payload = read_bytes(pointer)
    if hashlib.sha256(pointer_payload).hexdigest() != spec["pointer_sha256"]:
        raise ContractError(f"{dataset}: frozen checkpoint pointer hash changed")
    expected_pointer = f"{absolute(checkpoint)}\n".encode("utf-8")
    if pointer_payload != expected_pointer:
        raise ContractError(f"{dataset}: checkpoint pointer is not the exact frozen path")
    checkpoint_path = require_directory(checkpoint)
    children = {path.name for path in checkpoint_path.iterdir()}
    if children != set(spec["files"]):
        raise ContractError(f"{dataset}: frozen checkpoint file set changed: {sorted(children)}")
    files: dict[str, dict[str, Any]] = {}
    for name, (expected_bytes, expected_sha256) in spec["files"].items():
        path = checkpoint / name
        actual = identity(path)
        if actual["bytes"] != expected_bytes or actual["sha256"] != expected_sha256:
            raise ContractError(f"{dataset}: frozen checkpoint file changed: {name}")
        files[name] = actual

    # extra.state is a pickle.  It is loaded only after its exact frozen hash
    # has been verified above, so no unaudited serialized object is executed.
    try:
        import torch

        state = torch.load(absolute(checkpoint / "extra.state"), map_location="cpu", weights_only=False)
    except Exception as error:  # pragma: no cover - exact library text varies
        raise ContractError(f"{dataset}: cannot read frozen extra.state: {error}") from error
    if state != spec["extra_state"]:
        raise ContractError(f"{dataset}: frozen training progress state changed")

    train_args_path = Path(spec["train_args"])
    train_args_identity = identity(train_args_path)
    if train_args_identity["sha256"] != spec["train_args_sha256"]:
        raise ContractError(f"{dataset}: frozen training argument record changed")
    train_args = load_json(train_args_path)
    expected_training = {
        "model_type": "spert", "train_batch_size": 2, "eval_batch_size": 1,
        "neg_entity_count": 100, "neg_relation_count": 100, "epochs": 20,
        "lr": 5e-5, "lr_warmup": 0.1, "weight_decay": 0.01,
        "max_grad_norm": 1.0, "rel_filter_threshold": 0.4,
        "size_embedding": 25, "prop_drop": 0.1, "max_span_size": 10,
        "sampling_processes": 4, "max_pairs": 1000, "seed": 42,
        "final_eval": True, "store_predictions": True,
    }
    if any(train_args.get(key) != value for key, value in expected_training.items()):
        raise ContractError(f"{dataset}: frozen training parameters changed")
    for key, suffix in (
        ("train_path", f"/{dataset}/train.json"),
        ("valid_path", f"/{dataset}/validation.json"),
        ("types_path", f"/{dataset}/types.json"),
    ):
        value = train_args.get(key)
        if (
            not isinstance(value, str)
            or not value.endswith(suffix)
            or Path(value).name == "test.json"
        ):
            raise ContractError(f"{dataset}: unsafe frozen training split path: {key}")

    status_path = VALIDATION_ROOT / "status.json"
    status = load_json(status_path)
    status_row = status.get("datasets", {}).get(dataset, {})
    if (
        status.get("status") != "complete"
        or status.get("split") != "validation"
        or status.get("seed") != 42
        or status.get("test_split_access") != "forbidden-and-not-read"
        or not _same_path(status_row.get("checkpoint"), checkpoint)
        or status_row.get("prediction_rows") != EXPECTED_VALIDATION_ROWS[dataset]
    ):
        raise ContractError(f"{dataset}: audited validation checkpoint lineage changed")
    return {
        "dataset": dataset,
        "seed": 42,
        "pointer": identity(pointer),
        "checkpoint": {"path": str(absolute(checkpoint)), "files": files},
        "training_args": train_args_identity,
        "extra_state": state,
        "validation_status": identity(status_path),
    }


def verify_runtime() -> dict[str, Any]:
    compat = identity(COMPAT_RUNNER)
    if compat["sha256"] != COMPAT_RUNNER_SHA256:
        raise ContractError("SpERT compatibility runner changed from the frozen implementation")
    repo = require_directory(SPERT_REPO)
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ContractError(f"cannot verify SpERT source checkout: {error}") from error
    if revision != SPERT_REPO_REVISION or dirty:
        raise ContractError("SpERT source checkout is not the frozen clean revision")
    return {
        "compat_runner": compat,
        "spert_repo": str(repo),
        "spert_revision": revision,
        "tracked_worktree_clean": True,
        "contract": identity(Path(__file__).resolve()),
    }


def canonical_release(
    release_status: Path = RELEASE_STATUS_PATH,
    promotion: Path = PROMOTION_PATH,
    public_prepared_root: Path = PUBLIC_PREPARED_ROOT,
) -> dict[str, Any]:
    if Path.cwd().resolve(strict=True) != ROOT.resolve(strict=True):
        raise ContractError(f"formal SpERT contract must run from project root: {ROOT}")
    try:
        return release_contract.verify_release(
            release_status, promotion, public_prepared_root
        )
    except Exception as error:
        raise ContractError(f"canonical formal release verification failed: {error}") from error


def release_binding(value: dict[str, Any]) -> dict[str, Any]:
    result = {
        "release_status": value.get("release_status"),
        "promotion": value.get("promotion"),
        "canonical_fingerprint": value.get("canonical_fingerprint"),
    }
    if not (
        isinstance(result["release_status"], dict)
        and isinstance(result["promotion"], dict)
        and isinstance(result["canonical_fingerprint"], str)
        and len(result["canonical_fingerprint"]) == 64
    ):
        raise ContractError("canonical release identity is malformed")
    return result


def verify_prepared(
    dataset: str, manifest_path: Path, release_value: dict[str, Any]
) -> dict[str, Any]:
    try:
        result = prepared.validate_manifest(dataset, manifest_path)
    except Exception as error:
        raise ContractError(f"{dataset}: prepared SpERT test verification failed: {error}") from error
    manifest = load_json(manifest_path)
    promotion = release_value.get("promotion", {})
    if (
        manifest.get("promotion_attestation_sha256")
        != promotion.get("attestation_sha256")
        or result.get("fingerprint") != manifest.get("fingerprint")
    ):
        raise ContractError(f"{dataset}: SpERT test input is bound to another promotion")
    return {"verification": result, "manifest": manifest, "identity": identity(manifest_path)}


def expected_eval_args(
    dataset: str, checkpoint: Path, prepared_manifest: Path, log_root: Path
) -> dict[str, Any]:
    label = f"spert_{dataset}_seed42_formal_test"
    test_root = prepared_manifest.parent
    return {
        "cache_path": None,
        "config": None,
        "cpu": False,
        "dataset_path": str(absolute(test_root / "test.json")),
        "debug": False,
        "eval_batch_size": 1,
        "example_count": None,
        "freeze_transformer": False,
        "label": label,
        "log_path": str(absolute(log_root)),
        "lowercase": False,
        "max_pairs": 1000,
        "max_span_size": 10,
        "model_path": str(absolute(checkpoint)),
        "model_type": "spert",
        "no_overlapping": False,
        "prop_drop": 0.1,
        "rel_filter_threshold": 0.4,
        "sampling_processes": 4,
        "seed": 42,
        "size_embedding": 25,
        "store_examples": False,
        "store_predictions": True,
        "tokenizer_path": str(absolute(checkpoint)),
        "types_path": str(absolute(test_root / "types.json")),
    }


def _direct_run_directories(label_root: Path) -> list[str]:
    root = absolute(label_root)
    if not root.exists():
        root.mkdir(parents=True, exist_ok=False)
    require_directory(root)
    result: list[str] = []
    for child in root.iterdir():
        if child.is_symlink() or not child.is_dir():
            raise ContractError(f"unexpected non-directory in SpERT upstream log root: {child}")
        result.append(child.name)
    return sorted(result)


def begin_eval(
    dataset: str,
    prepared_manifest: Path,
    log_root: Path,
    snapshot_path: Path,
    release_value: dict[str, Any],
) -> dict[str, Any]:
    prepared_value = verify_prepared(dataset, prepared_manifest, release_value)
    checkpoint_value = verify_checkpoint(dataset)
    runtime = verify_runtime()
    checkpoint_path = Path(CHECKPOINT_SPECS[dataset]["checkpoint"])
    label_root = log_root / f"spert_{dataset}_seed42_formal_test"
    before = _direct_run_directories(label_root)
    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "status": "eval_started",
        "dataset": dataset,
        "split": "test",
        "formal_test_read": True,
        "seed": 42,
        "invocation_id": str(uuid.uuid4()),
        "release": release_binding(release_value),
        "prepared_manifest": prepared_value["identity"],
        "prepared_fingerprint": prepared_value["manifest"]["fingerprint"],
        "checkpoint": checkpoint_value,
        "runtime": runtime,
        "label_root": str(absolute(label_root)),
        "before_directories": before,
        "eval_args": expected_eval_args(
            dataset, checkpoint_path, prepared_manifest, log_root
        ),
    }
    atomic_json(snapshot_path, snapshot)
    return snapshot


def _validate_snapshot(
    dataset: str,
    prepared_manifest: Path,
    log_root: Path,
    snapshot_path: Path,
    release_value: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    prepared_value = verify_prepared(dataset, prepared_manifest, release_value)
    checkpoint_value = verify_checkpoint(dataset)
    runtime = verify_runtime()
    snapshot = load_json(snapshot_path)
    checkpoint_path = Path(CHECKPOINT_SPECS[dataset]["checkpoint"])
    expected_fields = {
        "schema_version": SCHEMA_VERSION,
        "status": "eval_started",
        "dataset": dataset,
        "split": "test",
        "formal_test_read": True,
        "seed": 42,
        "release": release_binding(release_value),
        "prepared_manifest": prepared_value["identity"],
        "prepared_fingerprint": prepared_value["manifest"]["fingerprint"],
        "checkpoint": checkpoint_value,
        "runtime": runtime,
        "label_root": str(absolute(log_root / f"spert_{dataset}_seed42_formal_test")),
        "eval_args": expected_eval_args(dataset, checkpoint_path, prepared_manifest, log_root),
    }
    for key, value in expected_fields.items():
        if snapshot.get(key) != value:
            raise ContractError(f"{dataset}: eval snapshot changed at {key}")
    try:
        uuid.UUID(snapshot["invocation_id"])
    except (KeyError, TypeError, ValueError) as error:
        raise ContractError(f"{dataset}: eval snapshot invocation ID is invalid") from error
    before = snapshot.get("before_directories")
    if not isinstance(before, list) or before != sorted(set(before)) or not all(
        isinstance(name, str) and name and "/" not in name for name in before
    ):
        raise ContractError(f"{dataset}: eval snapshot directory inventory is invalid")
    return snapshot, prepared_value


def capture_eval(
    dataset: str,
    prepared_manifest: Path,
    jobs: Path,
    log_root: Path,
    snapshot_path: Path,
    raw_output: Path,
    args_output: Path,
    capture_manifest: Path,
    release_value: dict[str, Any],
) -> dict[str, Any]:
    snapshot, prepared_value = _validate_snapshot(
        dataset, prepared_manifest, log_root, snapshot_path, release_value
    )
    label_root = Path(snapshot["label_root"])
    current = _direct_run_directories(label_root)
    before = snapshot["before_directories"]
    if not set(before) <= set(current):
        raise ContractError(f"{dataset}: pre-existing upstream log directory disappeared")
    created = sorted(set(current) - set(before))
    if len(created) != 1:
        raise ContractError(
            f"{dataset}: expected exactly one newly created upstream eval directory, got {created}"
        )
    upstream_dir = label_root / created[0]
    require_directory(upstream_dir)
    upstream_raw = upstream_dir / "predictions_test_epoch_0.json"
    upstream_args = upstream_dir / "args.json"
    actual_args = load_json(upstream_args)
    if actual_args != snapshot["eval_args"]:
        raise ContractError(f"{dataset}: upstream SpERT eval arguments differ from frozen values")
    source_rows = load_json(prepared_manifest.parent / "test.json")
    raw_rows = load_json(upstream_raw)
    job_rows = load_jsonl(jobs)
    types = load_json(prepared_manifest.parent / "types.json")
    try:
        converter.build_conversion(dataset, source_rows, raw_rows, job_rows, types)
    except Exception as error:
        raise ContractError(f"{dataset}: invalid upstream SpERT prediction payload: {error}") from error

    atomic_copy(upstream_raw, raw_output)
    atomic_copy(upstream_args, args_output)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "captured_exact_new_eval_output",
        "dataset": dataset,
        "split": "test",
        "formal_test_read": True,
        "seed": 42,
        "invocation_id": snapshot["invocation_id"],
        "release": release_binding(release_value),
        "prepared_manifest": prepared_value["identity"],
        "snapshot": identity(snapshot_path),
        "before_directories": before,
        "new_directories": created,
        "upstream_directory": str(upstream_dir),
        "upstream": {
            "predictions": identity(upstream_raw),
            "args": identity(upstream_args),
        },
        "captured": {
            "predictions": identity(raw_output),
            "args": identity(args_output),
        },
        "rows": EXPECTED_ROWS[dataset],
    }
    if result["upstream"]["predictions"]["sha256"] != result["captured"]["predictions"]["sha256"]:
        raise ContractError(f"{dataset}: raw prediction capture was not byte exact")
    if result["upstream"]["args"]["sha256"] != result["captured"]["args"]["sha256"]:
        raise ContractError(f"{dataset}: eval argument capture was not byte exact")
    atomic_json(capture_manifest, result)
    return result


def validate_capture(
    dataset: str,
    prepared_manifest: Path,
    jobs: Path,
    log_root: Path,
    snapshot_path: Path,
    raw_output: Path,
    args_output: Path,
    capture_manifest: Path,
    release_value: dict[str, Any],
) -> dict[str, Any]:
    snapshot, prepared_value = _validate_snapshot(
        dataset, prepared_manifest, log_root, snapshot_path, release_value
    )
    capture = load_json(capture_manifest)
    expected_scalars = {
        "schema_version": SCHEMA_VERSION,
        "status": "captured_exact_new_eval_output",
        "dataset": dataset,
        "split": "test",
        "formal_test_read": True,
        "seed": 42,
        "invocation_id": snapshot["invocation_id"],
        "release": release_binding(release_value),
        "prepared_manifest": prepared_value["identity"],
        "snapshot": identity(snapshot_path),
        "before_directories": snapshot["before_directories"],
        "rows": EXPECTED_ROWS[dataset],
    }
    for key, value in expected_scalars.items():
        if capture.get(key) != value:
            raise ContractError(f"{dataset}: raw capture manifest changed at {key}")
    created = capture.get("new_directories")
    if not isinstance(created, list) or len(created) != 1 or created[0] in snapshot["before_directories"]:
        raise ContractError(f"{dataset}: raw capture does not identify exactly one new run")
    upstream_dir = Path(snapshot["label_root"]) / created[0]
    if not _same_path(capture.get("upstream_directory"), upstream_dir):
        raise ContractError(f"{dataset}: captured upstream directory path changed")
    upstream_raw = upstream_dir / "predictions_test_epoch_0.json"
    upstream_args = upstream_dir / "args.json"
    records = (
        (capture.get("upstream", {}).get("predictions"), upstream_raw),
        (capture.get("upstream", {}).get("args"), upstream_args),
        (capture.get("captured", {}).get("predictions"), raw_output),
        (capture.get("captured", {}).get("args"), args_output),
    )
    if any(not _identity_matches(record, path) for record, path in records):
        raise ContractError(f"{dataset}: captured SpERT output identity changed")
    if read_bytes(upstream_raw) != read_bytes(raw_output) or read_bytes(upstream_args) != read_bytes(args_output):
        raise ContractError(f"{dataset}: captured SpERT output bytes differ from exact upstream output")
    if load_json(args_output) != snapshot["eval_args"]:
        raise ContractError(f"{dataset}: captured eval arguments differ from frozen values")
    source_rows = load_json(prepared_manifest.parent / "test.json")
    raw_rows = load_json(raw_output)
    job_rows = load_jsonl(jobs)
    types = load_json(prepared_manifest.parent / "types.json")
    try:
        expected_rows = converter.build_conversion(
            dataset, source_rows, raw_rows, job_rows, types
        )
    except Exception as error:
        raise ContractError(f"{dataset}: captured predictions are invalid: {error}") from error
    return {
        "capture": capture,
        "prepared": prepared_value,
        "expected_predictions": expected_rows,
        "job_rows": job_rows,
    }


def artifact_paths(data_root: Path, prepared_root: Path, run_root: Path, dataset: str) -> dict[str, Path]:
    base = run_root / dataset
    public = data_root / dataset
    return {
        "prepared_manifest": prepared_root / dataset / "manifest.json",
        "jobs": public / "test_baseline_jobs.jsonl",
        "gold": public / "test_gold.jsonl",
        "gold_index": public / "test_index.jsonl",
        "snapshot": base / "eval_snapshot.json",
        "raw_predictions": base / "raw_predictions.json",
        "eval_args": base / "eval_args.json",
        "capture_manifest": base / "raw_capture_manifest.json",
        "conversion_manifest": base / "conversion_manifest.json",
        "predictions": base / "test_predictions.jsonl",
        "normalized_metrics": base / "test_normalized_text_metrics.json",
        "span_metrics": base / "test_character_span_metrics.json",
        "completion": base / "completion_manifest.json",
        "log_root": base / "upstream_logs",
    }


def _validate_conversion(
    dataset: str,
    paths: dict[str, Path],
    capture: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    actual_rows = load_jsonl(paths["predictions"])
    expected_rows = capture["expected_predictions"]
    if actual_rows != expected_rows:
        raise ContractError(f"{dataset}: canonical predictions differ from exact raw conversion")
    manifest = load_json(paths["conversion_manifest"])
    prepared_manifest = capture["prepared"]["manifest"]
    expected_fields = {
        "schema_version": converter.SCHEMA_VERSION,
        "status": "complete",
        "dataset": dataset,
        "split": "test",
        "formal_test_read": True,
        "seed": 42,
        "rows": EXPECTED_ROWS[dataset],
        "entities": sum(len(row["annotation"]["entities"]) for row in actual_rows),
        "relations": sum(len(row["annotation"]["relations"]) for row in actual_rows),
        "promotion_attestation_sha256": prepared_manifest["promotion_attestation_sha256"],
        "prepared_fingerprint": prepared_manifest["fingerprint"],
        "orig_id_sha256": prepared_manifest["orig_id_sha256"],
        "job_id_sha256": converter.stable_digest([row["job_id"] for row in actual_rows]),
        "inputs": {
            "prepared_manifest": identity(paths["prepared_manifest"]),
            "test_data": identity(paths["prepared_manifest"].parent / "test.json"),
            "types": identity(paths["prepared_manifest"].parent / "types.json"),
            "raw_predictions": identity(paths["raw_predictions"]),
            "test_jobs": identity(paths["jobs"]),
            "inference_manifest": identity(paths["capture_manifest"]),
            "converter": identity(Path(converter.__file__).resolve()),
        },
        "output": identity(paths["predictions"]),
    }
    if manifest != expected_fields:
        raise ContractError(f"{dataset}: conversion manifest/content binding is invalid")
    return actual_rows, manifest


def _validate_gold_index(dataset: str, paths: dict[str, Path], job_ids: list[str]) -> None:
    gold = load_jsonl(paths["gold"])
    index = load_jsonl(paths["gold_index"])
    if len(gold) != EXPECTED_ROWS[dataset] or len(index) != EXPECTED_ROWS[dataset]:
        raise ContractError(f"{dataset}: public test gold/index count changed")
    expected_index = [
        {"job_id": job_id, "record_index": position}
        for position, job_id in enumerate(job_ids)
    ]
    if index != expected_index:
        raise ContractError(f"{dataset}: public test gold index/order changed")
    for job_id, annotation in zip(job_ids, gold, strict=True):
        expected_document = job_id.removesuffix("_C1")
        if annotation.get("document_id") != expected_document:
            raise ContractError(f"{dataset}: public test gold document order changed")


def _recompute_metrics(paths: dict[str, Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="spert-formal-metric-check-") as temporary:
        temp_root = Path(temporary)
        normalized_path = temp_root / "normalized.json"
        span_path = temp_root / "span.json"
        normalized_args = SimpleNamespace(
            gold=absolute(paths["gold"]),
            gold_index=absolute(paths["gold_index"]),
            predictions=absolute(paths["predictions"]),
            pred_index=None,
            jobs=absolute(paths["jobs"]),
            limit=None,
            offset=0,
            include_missing_as_empty=True,
            output=normalized_path,
        )
        span_args = SimpleNamespace(
            gold=absolute(paths["gold"]),
            gold_index=absolute(paths["gold_index"]),
            predictions=absolute(paths["predictions"]),
            jobs=absolute(paths["jobs"]),
            output=span_path,
            allow_non_validation=True,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            normalized_evaluator.run(normalized_args)
            span_evaluator.run(span_args)
        return load_json(normalized_path), load_json(span_path)


def _validated_payload(
    data_root: Path,
    prepared_root: Path,
    run_root: Path,
    dataset: str,
    release_value: dict[str, Any],
) -> dict[str, Any]:
    paths = artifact_paths(data_root, prepared_root, run_root, dataset)
    capture = validate_capture(
        dataset=dataset,
        prepared_manifest=paths["prepared_manifest"],
        jobs=paths["jobs"],
        log_root=paths["log_root"],
        snapshot_path=paths["snapshot"],
        raw_output=paths["raw_predictions"],
        args_output=paths["eval_args"],
        capture_manifest=paths["capture_manifest"],
        release_value=release_value,
    )
    predictions, conversion = _validate_conversion(dataset, paths, capture)
    job_ids = [row["job_id"] for row in capture["job_rows"]]
    if [row.get("job_id") for row in predictions] != job_ids:
        raise ContractError(f"{dataset}: canonical prediction coverage/order changed")
    _validate_gold_index(dataset, paths, job_ids)

    normalized = load_json(paths["normalized_metrics"])
    span = load_json(paths["span_metrics"])
    recomputed_normalized, recomputed_span = _recompute_metrics(paths)
    if normalized != recomputed_normalized:
        raise ContractError(f"{dataset}: normalized-text metrics fail independent recomputation")
    if span != recomputed_span:
        raise ContractError(f"{dataset}: strict character-span metrics fail independent recomputation")
    expected_counts = {
        "jobs_gold": EXPECTED_ROWS[dataset],
        "jobs_predicted": EXPECTED_ROWS[dataset],
        "jobs_evaluated": EXPECTED_ROWS[dataset],
        "jobs_missing_predictions": 0,
        "generation_success_rate": 1.0,
    }
    if any(normalized.get(key) != value for key, value in expected_counts.items()):
        raise ContractError(f"{dataset}: normalized metric coverage is incomplete")
    if any(span.get(key) != value for key, value in expected_counts.items()):
        raise ContractError(f"{dataset}: strict metric coverage is incomplete")
    if (
        span.get("metric") != "strict-source-character-span"
        or span.get("selection_split") != "test"
        or span.get("formal_test_read") is not True
        or list(span.get("per_job", {})) != job_ids
        or span.get("resolution", {}).get("unresolved_predicted_entities") != 0
    ):
        raise ContractError(f"{dataset}: strict formal-test metric contract is invalid")
    _finite(normalized, "normalized_metrics")
    _finite(span, "span_metrics")

    checkpoint = verify_checkpoint(dataset)
    runner_identity = identity(FORMAL_RUNNER)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "dataset": dataset,
        "split": "test",
        "formal_test_read": True,
        "test_used_for_selection": False,
        "seed": 42,
        "rows": EXPECTED_ROWS[dataset],
        "release": release_binding(release_value),
        "prepared": {
            "manifest": capture["prepared"]["identity"],
            "fingerprint": capture["prepared"]["manifest"]["fingerprint"],
        },
        "checkpoint": checkpoint,
        "eval_args": load_json(paths["eval_args"]),
        "counts": {
            "entities": conversion["entities"],
            "relations": conversion["relations"],
        },
        "implementation": {
            "runner": runner_identity,
            "contract": identity(Path(__file__).resolve()),
            "converter": identity(Path(converter.__file__).resolve()),
        },
        "artifacts": {
            name: identity(paths[name])
            for name in (
                "snapshot", "raw_predictions", "eval_args", "capture_manifest",
                "conversion_manifest", "predictions", "normalized_metrics", "span_metrics",
            )
        },
    }


def finalize_dataset(
    data_root: Path,
    prepared_root: Path,
    run_root: Path,
    dataset: str,
    release_value: dict[str, Any],
) -> dict[str, Any]:
    payload = _validated_payload(data_root, prepared_root, run_root, dataset, release_value)
    completion = artifact_paths(data_root, prepared_root, run_root, dataset)["completion"]
    atomic_json(completion, payload)
    return payload


def validate_dataset(
    data_root: Path,
    prepared_root: Path,
    run_root: Path,
    dataset: str,
    release_value: dict[str, Any],
) -> dict[str, Any]:
    expected = _validated_payload(data_root, prepared_root, run_root, dataset, release_value)
    completion = artifact_paths(data_root, prepared_root, run_root, dataset)["completion"]
    actual = load_json(completion)
    if actual != expected:
        raise ContractError(f"{dataset}: completion manifest no longer matches verified artifacts")
    return expected


def build_status(
    state: str,
    stage: str,
    formal_test_read: bool,
    active_dataset: str | None,
    error: str | None,
    release_sha256: str | None,
    release_fingerprint: str | None,
    run_root: Path,
    gpu_lock: Path,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": state,
        "stage": stage,
        "split": "test",
        "formal_test_read": formal_test_read,
        "test_used_for_selection": False,
        "seed": 42,
        "active_dataset": active_dataset,
        "gpu_lock": str(gpu_lock),
        "release": {
            "captured_sha256": release_sha256,
            "captured_canonical_fingerprint": release_fingerprint,
        },
        "datasets": {},
    }
    if error:
        result["error"] = error
    for dataset in DATASETS:
        paths = artifact_paths(DATA_ROOT, SPERT_PREPARED_ROOT, run_root, dataset)
        result["datasets"][dataset] = {
            "expected_rows": EXPECTED_ROWS[dataset],
            "checkpoint_pointer": CHECKPOINT_SPECS[dataset]["pointer"],
            "prepared_manifest": str(paths["prepared_manifest"]),
            "completion_manifest": str(paths["completion"]),
            "completion_present": absolute(paths["completion"]).is_file(),
            "test_access": "opened_after_promotion" if formal_test_read else "sealed_until_promotion",
        }
    return result


def add_release_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--release-status", type=Path, default=RELEASE_STATUS_PATH)
    parser.add_argument("--promotion", type=Path, default=PROMOTION_PATH)
    parser.add_argument("--public-prepared-root", type=Path, default=PUBLIC_PREPARED_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    checkpoint = subparsers.add_parser("verify-checkpoint")
    checkpoint.add_argument("--dataset", choices=DATASETS, required=True)
    subparsers.add_parser("verify-checkpoints")

    begin = subparsers.add_parser("begin-eval")
    begin.add_argument("--dataset", choices=DATASETS, required=True)
    begin.add_argument("--prepared-manifest", type=Path, required=True)
    begin.add_argument("--log-root", type=Path, required=True)
    begin.add_argument("--snapshot", type=Path, required=True)
    add_release_args(begin)

    capture = subparsers.add_parser("capture-eval")
    capture.add_argument("--dataset", choices=DATASETS, required=True)
    capture.add_argument("--prepared-manifest", type=Path, required=True)
    capture.add_argument("--jobs", type=Path, required=True)
    capture.add_argument("--log-root", type=Path, required=True)
    capture.add_argument("--snapshot", type=Path, required=True)
    capture.add_argument("--raw-output", type=Path, required=True)
    capture.add_argument("--args-output", type=Path, required=True)
    capture.add_argument("--capture-manifest", type=Path, required=True)
    add_release_args(capture)

    for name in ("finalize", "validate"):
        command = subparsers.add_parser(name)
        command.add_argument("--dataset", choices=DATASETS, required=True)
        command.add_argument("--data-root", type=Path, default=DATA_ROOT)
        command.add_argument("--prepared-root", type=Path, default=SPERT_PREPARED_ROOT)
        command.add_argument("--run-root", type=Path, default=RUN_ROOT)
        command.add_argument("--quiet", action="store_true")
        add_release_args(command)

    status = subparsers.add_parser("status")
    status.add_argument("--output", type=Path, required=True)
    status.add_argument("--state", required=True)
    status.add_argument("--stage", required=True)
    status.add_argument("--formal-test-read", choices=("true", "false"), required=True)
    status.add_argument("--active-dataset", choices=DATASETS)
    status.add_argument("--error")
    status.add_argument("--release-sha256")
    status.add_argument("--release-fingerprint")
    status.add_argument("--run-root", type=Path, default=RUN_ROOT)
    status.add_argument("--gpu-lock", type=Path, required=True)
    return parser.parse_args()


def _release_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return canonical_release(
        args.release_status, args.promotion, args.public_prepared_root
    )


def main() -> int:
    args = parse_args()
    if args.command == "verify-checkpoint":
        result: Any = verify_checkpoint(args.dataset)
    elif args.command == "verify-checkpoints":
        result = {
            "status": "verified_frozen_seed42_checkpoints",
            "datasets": {dataset: verify_checkpoint(dataset) for dataset in DATASETS},
        }
    elif args.command == "begin-eval":
        result = begin_eval(
            args.dataset, args.prepared_manifest, args.log_root, args.snapshot,
            _release_from_args(args),
        )
    elif args.command == "capture-eval":
        result = capture_eval(
            args.dataset, args.prepared_manifest, args.jobs, args.log_root,
            args.snapshot, args.raw_output, args.args_output,
            args.capture_manifest, _release_from_args(args),
        )
    elif args.command in {"finalize", "validate"}:
        function = finalize_dataset if args.command == "finalize" else validate_dataset
        result = function(
            args.data_root, args.prepared_root, args.run_root, args.dataset,
            _release_from_args(args),
        )
    elif args.command == "status":
        result = build_status(
            state=args.state,
            stage=args.stage,
            formal_test_read=args.formal_test_read == "true",
            active_dataset=args.active_dataset,
            error=args.error,
            release_sha256=args.release_sha256,
            release_fingerprint=args.release_fingerprint,
            run_root=args.run_root,
            gpu_lock=args.gpu_lock,
        )
        atomic_json(args.output, result)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    if not getattr(args, "quiet", False):
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
