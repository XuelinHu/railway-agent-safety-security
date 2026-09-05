#!/usr/bin/env python3
"""Verify the immutable release and safe resume state for the internal matrix.

The verifier never edits validation artifacts.  With ``--quarantine-invalid``
it may atomically move only artifacts below the formal run root into a
recoverable quarantine directory, causing the unchanged matrix runner to
recompute them instead of trusting presence-only resume markers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prepare_public_test_inputs import (
    DATASETS,
    EXPECTED_TEST_JOBS,
    GRAPH_ROOT,
    OUTPUT_ROOT,
    PROMOTION_PATH,
    SOURCE_ROOT,
    stable_digest,
)
from qwen_zeroshot_formal_contract import verify_release as verify_completed_release


SEEDS = (42, 2026, 3407)
FORMAL_TRAINING_SEEDS = (2026, 3407)
SPLITS_BY_SEED = {42: ("test",), 2026: ("validation", "test"), 3407: ("validation", "test")}
EXPECTED_VALIDATION_JOBS = {"conll04": 231, "scierc": 275, "ade": 384}
SYSTEMS = ("soe", "eae", "hrge", "evge", "cfe", "pge")
BRANCHES = ("soe", "eae", "hrge")
MODEL = Path(
    "/ds2/xuelin/cache/huggingface/hub/"
    "models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
)
RELEASE_STATUS_PATH = Path("outputs/public_formal_matrix/release_status.json")
ADAPTER_FILES = {
    "README.md", "adapter_config.json", "adapter_model.safetensors",
    "chat_template.jinja", "tokenizer.json", "tokenizer_config.json",
    "training_metrics.json",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"required regular artifact is missing or symlinked: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read JSONL artifact {path}: {error}") from error
    for number, line in enumerate(lines, 1):
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


def atomic_json(path: Path, value: dict[str, Any]) -> None:
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


def validate_finite(value: Any, label: str = "metric") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            validate_finite(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            validate_finite(item, f"{label}[{index}]")
    elif isinstance(value, bool):
        return
    elif isinstance(value, (int, float)) and not math.isfinite(value):
        raise ValueError(f"non-finite number in {label}")


def job_path(dataset: str, split: str, branch: str) -> Path:
    if branch == "soe":
        return SOURCE_ROOT / dataset / f"{split}_baseline_jobs.jsonl"
    if split == "validation":
        return GRAPH_ROOT / dataset / "jobs" / f"validation_{branch}_jobs.jsonl"
    return OUTPUT_ROOT / dataset / "jobs" / f"test_{branch}_jobs.jsonl"


def expected_jobs(dataset: str, split: str, branch: str) -> tuple[list[str], dict[str, str]]:
    rows = load_jsonl(job_path(dataset, split, branch))
    count = EXPECTED_TEST_JOBS[dataset] if split == "test" else EXPECTED_VALIDATION_JOBS[dataset]
    if len(rows) != count:
        raise ValueError(f"canonical job denominator mismatch: {dataset}/{split}/{branch}")
    ids = [str(row.get("job_id", "")) for row in rows]
    documents = {str(row.get("job_id", "")): str(row.get("document_id", "")) for row in rows}
    if "" in ids or len(set(ids)) != count or "" in documents.values():
        raise ValueError(f"canonical job IDs are invalid: {dataset}/{split}/{branch}")
    return ids, documents


def validate_prediction(
    path: Path, expected_ids: list[str], documents: dict[str, str]
) -> dict[str, Any]:
    rows = load_jsonl(path)
    actual = [str(row.get("job_id", "")) for row in rows]
    if actual != expected_ids:
        raise ValueError(f"prediction rows do not preserve full job order: {path}")
    for row in rows:
        annotation = row.get("annotation")
        job_id = str(row["job_id"])
        if not isinstance(annotation, dict) or annotation.get("document_id") != documents[job_id]:
            raise ValueError(f"prediction annotation/document mismatch: {path}/{job_id}")
        if not isinstance(annotation.get("entities"), list) or not isinstance(
            annotation.get("relations"), list
        ):
            raise ValueError(f"prediction annotation schema mismatch: {path}/{job_id}")
    return identity(path)


def adapter_path(run_root: Path, seed: int, dataset: str, branch: str) -> Path:
    if seed == 42:
        if branch == "soe":
            return Path("outputs/public_full_stage1") / f"{dataset}_baseline"
        return Path("outputs/public_pge_validation_seed42") / dataset / f"{branch}_adapter"
    return run_root / f"seed{seed}" / dataset / f"{branch}_adapter"


def validate_adapter(path: Path, seed: int, dataset: str, branch: str) -> dict[str, Any]:
    if not path.is_dir() or path.is_symlink():
        raise ValueError(f"adapter directory is missing or symlinked: {path}")
    files = {item.name for item in path.iterdir() if item.is_file() and not item.is_symlink()}
    if files != ADAPTER_FILES:
        raise ValueError(f"adapter file registry is incomplete or unexpected: {path}")
    metrics = load_json(path / "training_metrics.json")
    train_count = sum(
        1 for line in (SOURCE_ROOT / dataset / "train_baseline_jobs.jsonl").read_text(
            encoding="utf-8"
        ).splitlines() if line.strip()
    )
    expected = {
        "seed": seed,
        "base_model": str(MODEL),
        "epochs": 1,
        "max_length": 4096,
        "lora_rank": 8,
        "compact_target": True,
        "use_job_instruction": branch != "soe",
        "truncated_prompts": 0,
        "truncated_answers_with_eos": 0,
        "source_examples": train_count,
        "train_examples": train_count,
    }
    if any(metrics.get(key) != value for key, value in expected.items()):
        raise ValueError(f"adapter training contract mismatch: {path}")
    validate_finite(metrics, f"adapter:{seed}:{dataset}:{branch}")
    return {
        "seed": seed,
        "dataset": dataset,
        "branch": branch,
        "files": {name: identity(path / name) for name in sorted(ADAPTER_FILES)},
    }


def validate_materialization(
    path: Path,
    expected_ids: list[str],
    complete_path: Path,
    jobs_path: Path,
) -> dict[str, Any]:
    value = load_json(path)
    missing = value.get("missing_job_ids")
    if (
        value.get("status") != "complete"
        or value.get("gold_read") is not False
        or value.get("jobs") != len(expected_ids)
        or not isinstance(missing, list)
        or value.get("failures_materialized_as_empty") != len(missing)
        or value.get("successful_prediction_rows") + len(missing) != len(expected_ids)
        or Path(str(value.get("jobs_path", ""))).resolve(strict=False)
        != jobs_path.resolve(strict=False)
        or Path(str(value.get("output", ""))).resolve(strict=False)
        != complete_path.resolve(strict=False)
        or not set(map(str, missing)) <= set(expected_ids)
    ):
        raise ValueError(f"materialization contract mismatch: {path}")
    return identity(path)


def validate_metric(path: Path, expected_ids: list[str], split: str) -> dict[str, Any]:
    value = load_json(path)
    validate_finite(value, str(path))
    if path.name.endswith("_span.json"):
        if (
            value.get("metric") not in {
                "strict-global-character-span-one-to-one",
                "strict-global-character-span-document-deduplicated",
            }
            or value.get("jobs") != len(expected_ids)
            or not isinstance(value.get("per_job"), dict)
            or set(value["per_job"]) != set(expected_ids)
            or value.get("selection_split") != split
        ):
            raise ValueError(f"strict metric denominator/protocol mismatch: {path}")
        expected_formal = split == "test"
        if "formal_test_read" in value and value.get("formal_test_read") is not expected_formal:
            raise ValueError(f"strict metric formal-test marker mismatch: {path}")
    return identity(path)


def validate_complete_split(
    run_root: Path, seed: int, dataset: str, split: str
) -> dict[str, Any]:
    target = run_root / f"seed{seed}" / dataset / split
    marker_path = target / "complete.json"
    marker = load_json(marker_path)
    if marker.get("status") != "complete" or marker.get("seed") != seed:
        raise ValueError(f"split completion marker is invalid: {marker_path}")
    if (
        marker.get("dataset") != dataset
        or marker.get("split") != split
        or marker.get("formal_test_read") is not (split == "test")
        or marker.get("systems") != list(SYSTEMS)
    ):
        raise ValueError(f"split completion contract mismatch: {marker_path}")
    artifacts: dict[str, Any] = {"complete_marker": identity(marker_path)}
    branch_predictions: dict[str, tuple[list[str], dict[str, str]]] = {}
    for branch in BRANCHES:
        ids, documents = expected_jobs(dataset, split, branch)
        branch_predictions[branch] = (ids, documents)
        complete = target / f"{branch}_complete.jsonl"
        expanded = target / f"{branch}_expanded.jsonl"
        artifacts[f"{branch}_complete"] = validate_prediction(complete, ids, documents)
        artifacts[f"{branch}_expanded"] = validate_prediction(expanded, ids, documents)
        artifacts[f"{branch}_materialization"] = validate_materialization(
            target / f"{branch}_materialization.json", ids, complete,
            job_path(dataset, split, branch),
        )
    derived_ids, derived_documents = branch_predictions["hrge"]
    for system in ("evge", "cfe", "pge"):
        artifacts[system] = validate_prediction(
            target / f"{system}.jsonl", derived_ids, derived_documents
        )
        audit_path = target / f"{system}_audit.jsonl"
        artifacts[f"{system}_audit"] = identity(audit_path)
    span_index = target / f"{split}_span_index.jsonl"
    index_rows = load_jsonl(span_index)
    if len(index_rows) != len(derived_ids) or {
        str(row.get("parent_job_id", "")) for row in index_rows
    } != set(derived_ids) or any(row.get("split") != split for row in index_rows):
        raise ValueError(f"span index coverage mismatch: {span_index}")
    artifacts["span_index"] = identity(span_index)
    for system in SYSTEMS:
        ids = branch_predictions["soe"][0] if system == "soe" else derived_ids
        for kind in ("normalized_text", "span", "evidence"):
            metric_path = target / "metrics" / f"{system}_{kind}.json"
            artifacts[f"metric:{system}:{kind}"] = validate_metric(metric_path, ids, split)
    comparison_path = target / "soe_vs_pge.json"
    comparison = load_json(comparison_path)
    validate_finite(comparison, str(comparison_path))
    if (
        comparison.get("seed") != 20_260_830
        or comparison.get("iterations") != 20_000
        or comparison.get("dataset") != dataset
        or comparison.get("selection_split") != split
        or comparison.get("documents") != len(derived_ids)
    ):
        raise ValueError(f"paired comparison contract mismatch: {comparison_path}")
    artifacts["comparison"] = identity(comparison_path)
    return {
        "seed": seed,
        "dataset": dataset,
        "split": split,
        "jobs": len(derived_ids),
        "artifacts": artifacts,
        "state_sha256": stable_digest(artifacts),
    }


def quarantine(path: Path, run_root: Path, quarantine_root: Path) -> Path:
    try:
        relative = path.resolve(strict=False).relative_to(run_root.resolve(strict=False))
    except ValueError as error:
        raise ValueError(f"refusing to quarantine artifact outside formal run root: {path}") from error
    target = quarantine_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target = target.with_name(f"{target.name}.{len(list(target.parent.iterdir()))}")
    os.replace(path, target)
    return target


def validate_release_binding(
    release_status: Path,
    promotion: Path,
    prepared_root: Path,
    expected_release_status_sha256: str | None,
    expected_canonical_fingerprint: str | None,
    expected_prepared_release_sha256: str | None,
) -> dict[str, Any]:
    """Verify the completed publisher chain and any frozen run identities."""

    release = verify_completed_release(release_status, promotion, prepared_root)
    if (
        expected_release_status_sha256 is not None
        and release["release_status"]["sha256"]
        != expected_release_status_sha256
    ):
        raise ValueError("formal release-status identity changed during matrix execution")
    if (
        expected_canonical_fingerprint is not None
        and release["canonical_fingerprint"] != expected_canonical_fingerprint
    ):
        raise ValueError("formal canonical release fingerprint changed during matrix execution")
    if (
        expected_prepared_release_sha256 is not None
        and release["prepared_release_sha256"]
        != expected_prepared_release_sha256
    ):
        raise ValueError("formal prepared-release fingerprint changed during matrix execution")
    return release


def validate_state(
    mode: str,
    run_root: Path,
    expected_prepared_release_sha256: str | None,
    quarantine_invalid: bool,
    *,
    release_status: Path = RELEASE_STATUS_PATH,
    promotion: Path = PROMOTION_PATH,
    prepared_root: Path = OUTPUT_ROOT,
    expected_release_status_sha256: str | None = None,
    expected_canonical_fingerprint: str | None = None,
) -> dict[str, Any]:
    expected_identities = (
        expected_release_status_sha256,
        expected_canonical_fingerprint,
        expected_prepared_release_sha256,
    )
    if mode != "release-only" and any(value is None for value in expected_identities):
        raise ValueError(
            "preflight/postflight requires the complete formal release identity binding"
        )
    if any(value is not None for value in expected_identities) and any(
        value is None for value in expected_identities
    ):
        raise ValueError("formal release identity binding must include all three digests")
    # Reuse the consumer-facing release verifier rather than accepting the
    # prepared-output fingerprint alone.  This additionally proves that the
    # publisher reached its final ``complete`` state and bound that state to
    # the persisted, canonically replayed preparation attestation.
    release = validate_release_binding(
        release_status,
        promotion,
        prepared_root,
        expected_release_status_sha256,
        expected_canonical_fingerprint,
        expected_prepared_release_sha256,
    )
    if mode == "release-only":
        return {"status": "release_unchanged", "release": release}
    quarantine_root = run_root / ".invalid_resume" / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S.%fZ"
    )
    moved: list[dict[str, str]] = []
    adapters: dict[str, Any] = {}
    for seed in SEEDS:
        for dataset in DATASETS:
            for branch in BRANCHES:
                path = adapter_path(run_root, seed, dataset, branch)
                key = f"seed{seed}/{dataset}/{branch}"
                if seed == 42:
                    adapters[key] = validate_adapter(path, seed, dataset, branch)
                    continue
                if not path.exists():
                    if mode == "postflight":
                        raise ValueError(f"postflight adapter is missing: {path}")
                    continue
                try:
                    adapters[key] = validate_adapter(path, seed, dataset, branch)
                except ValueError as error:
                    if mode != "preflight" or not quarantine_invalid:
                        raise
                    destination = quarantine(path, run_root, quarantine_root)
                    moved.append({"path": str(path), "quarantined_to": str(destination), "reason": str(error)})
    splits: dict[str, Any] = {}
    for seed, expected_splits in SPLITS_BY_SEED.items():
        for dataset in DATASETS:
            for split in expected_splits:
                target = run_root / f"seed{seed}" / dataset / split
                key = f"seed{seed}/{dataset}/{split}"
                if not target.exists():
                    if mode == "postflight":
                        raise ValueError(f"postflight split is missing: {target}")
                    continue
                try:
                    splits[key] = validate_complete_split(run_root, seed, dataset, split)
                except ValueError as error:
                    if mode != "preflight" or not quarantine_invalid:
                        raise
                    destination = quarantine(target, run_root, quarantine_root)
                    moved.append({"path": str(target), "quarantined_to": str(destination), "reason": str(error)})
    # State validation can be lengthy when a resumable matrix is present.
    # Replay the complete release chain at the far side of that scan so a
    # release mutation cannot hide inside the preflight/postflight window.
    closing_release = validate_release_binding(
        release_status,
        promotion,
        prepared_root,
        expected_release_status_sha256,
        expected_canonical_fingerprint,
        expected_prepared_release_sha256,
    )
    if closing_release != release:
        raise ValueError("formal release changed while internal state was being validated")
    state = {
        "schema_version": "public-formal-internal-resume-v1",
        "status": "postflight_complete" if mode == "postflight" else "preflight_ready",
        "mode": mode,
        "release": closing_release,
        "adapters": adapters,
        "complete_splits": splits,
        "quarantined_invalid_formal_artifacts": moved,
        "validation_artifacts_modified": False,
        "generated_at": utc_now(),
    }
    state["resume_state_sha256"] = stable_digest({
        key: value for key, value in state.items() if key not in {"generated_at", "resume_state_sha256"}
    })
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "release-only", "postflight"), required=True)
    parser.add_argument("--run-root", type=Path, default=Path("outputs/public_formal_matrix/internal"))
    parser.add_argument(
        "--expected-prepared-release-sha256",
        "--expected-release-sha256",
        dest="expected_prepared_release_sha256",
        help="Expected canonical prepared-release release_sha256 (legacy alias retained).",
    )
    parser.add_argument("--expected-release-status-sha256")
    parser.add_argument("--expected-canonical-fingerprint")
    parser.add_argument("--release-status", type=Path, default=RELEASE_STATUS_PATH)
    parser.add_argument("--promotion", type=Path, default=PROMOTION_PATH)
    parser.add_argument("--prepared-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--quarantine-invalid", action="store_true")
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/public_formal_matrix/internal/resume_attestation.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state = validate_state(
        args.mode,
        args.run_root,
        args.expected_prepared_release_sha256,
        args.quarantine_invalid,
        release_status=args.release_status,
        promotion=args.promotion,
        prepared_root=args.prepared_root,
        expected_release_status_sha256=args.expected_release_status_sha256,
        expected_canonical_fingerprint=args.expected_canonical_fingerprint,
    )
    if args.mode != "release-only":
        atomic_json(args.output, state)
    print(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
