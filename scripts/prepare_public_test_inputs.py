#!/usr/bin/env python3
"""Prepare immutable leakage-safe EAE/HRGE inputs for public formal test.

The production CLI has no source, graph, model, or force override.  It accepts
only a fully revalidated promotion marker, snapshots every data input once,
and freezes retrieval code/settings/model identity to the validation-period
train-only graph manifest.  Test gold is never opened.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_experiment_jobs as eae_builder  # noqa: E402
import build_kg_v2_jobs as hrge_builder  # noqa: E402
import promote_public_validation_to_test as promoter  # noqa: E402


DATASETS = promoter.DATASETS
EXPECTED_TEST_JOBS = {"conll04": 288, "scierc": 551, "ade": 427}
PREPARATION_SCHEMA_VERSION = "public-formal-test-inputs-v2"
SOURCE_ROOT = Path("data/processed/public_benchmarks_full")
GRAPH_ROOT = Path("data/processed/public_benchmarks_hrge_v1")
OUTPUT_ROOT = Path("data/processed/public_benchmarks_hrge_test_v1")
PROMOTION_PATH = Path("outputs/public_formal_matrix/promotion.json")
FROZEN_BATCH_SIZE = 16

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
    "exact_train_test_text_quarantine": True,
}
VALIDATION_SETTINGS = {
    **{key: value for key, value in SETTINGS.items() if key != "exact_train_test_text_quarantine"},
    "exact_train_validation_text_quarantine": True,
}
BASELINE_JOB_KEYS = {
    "job_id", "document_id", "language", "category", "source_path",
    "system_instruction", "segments", "ontology", "experiment_mode",
    "prompt_version", "teacher_model", "status", "chunk_number", "chunk_count",
}
SEGMENT_KEYS = {"segment_id", "segment_type", "page", "start", "end", "text"}
FORBIDDEN_LABEL_KEYS = {
    "annotation", "annotations", "entities", "relations", "gold",
    "gold_entities", "gold_relations", "labels",
}
GRAPH_OUTPUT_NAMES = {
    "train_only_concepts", "train_only_mentions", "train_only_relations",
    "graph_conversion_audit", "train_eae_jobs", "validation_eae_jobs",
    "train_hrge_jobs", "validation_hrge_jobs", "train_validation_source_jobs",
    "train_validation_hrge_jobs", "hrge_builder_audit_before_overlap_quarantine",
    "exact_source_overlap_quarantine_audit",
}
GRAPH_INPUT_NAMES = {
    "train_jobs", "validation_jobs", "mentions", "training_edges", "split_manifest",
    "ontology", "eae_builder", "hrge_builder", "graph_converter", "preparation_runner",
}
MODEL_FILES = (
    "config.json", "modules.json", "pytorch_model.bin",
    "sentencepiece.bpe.model", "tokenizer.json",
)


def stable_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _tokens(path: Path) -> list[str]:
    return [
        token
        for part in path.parts
        for token in re.split(r"[._-]+", part.casefold())
        if token
    ]


def _reject_gold_namespace(path: Path) -> None:
    tokens = _tokens(path)
    if "gold" in tokens and "test" in tokens:
        raise ValueError(f"formal-test preparation must never open test gold: {path}")


def _assert_no_symlink(path: Path, *, regular: bool = True) -> Path:
    _reject_gold_namespace(path)
    candidate = path if path.is_absolute() else Path.cwd() / path
    for component in (candidate, *candidate.parents):
        try:
            if component.is_symlink():
                raise ValueError(f"symlinked preparation path is forbidden: {path}")
        except OSError as error:
            raise ValueError(f"cannot inspect preparation path {path}: {error}") from error
    resolved = candidate.resolve(strict=False)
    _reject_gold_namespace(resolved)
    if regular:
        try:
            mode = candidate.stat().st_mode
        except FileNotFoundError:
            raise FileNotFoundError(path) from None
        if not stat.S_ISREG(mode):
            raise ValueError(f"preparation input is not a regular file: {path}")
    return candidate


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return (value if value.is_absolute() else Path.cwd() / value).resolve(strict=False)


def _require_contained(path: Path, root: Path, label: str) -> None:
    try:
        _resolve(path).relative_to(_resolve(root))
    except ValueError as error:
        raise ValueError(f"{label} escapes its frozen root: {path}") from error


class SnapshotTracker:
    """Read each non-model input once, then serve parsed objects from its bytes."""

    def __init__(self) -> None:
        self._payloads: dict[Path, bytes] = {}

    def bytes(self, path: Path) -> bytes:
        candidate = _assert_no_symlink(path)
        key = candidate.resolve(strict=True)
        if key not in self._payloads:
            self._payloads[key] = candidate.read_bytes()
        return self._payloads[key]

    def text(self, path: Path) -> str:
        try:
            return self.bytes(path).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"preparation input is not UTF-8: {path}") from error

    def json(self, path: Path) -> dict[str, Any]:
        try:
            value = json.loads(self.text(path))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at {path}: {error}") from error
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}")
        return value

    def jsonl(self, path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(self.text(path).splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(row)
        return rows

    def identity(self, path: Path) -> dict[str, Any]:
        payload = self.bytes(path)
        return {
            "path": str(path), "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_matches(identity: Any, actual: dict[str, Any]) -> bool:
    return (
        isinstance(identity, dict)
        and isinstance(identity.get("path"), str)
        and _resolve(identity["path"]) == _resolve(actual["path"])
        and identity.get("bytes") == actual["bytes"]
        and identity.get("sha256") == actual["sha256"]
    )


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


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_snapshot(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def validate_promotion(path: Path) -> dict[str, Any]:
    """Recompute the full gate; never trust marker status fields in isolation."""

    tracker = SnapshotTracker()
    promotion = tracker.json(path)
    if (
        promotion.get("schema_version") != promoter.SCHEMA_VERSION
        or promotion.get("status") != "promoted"
        or promotion.get("promoted_systems") != ["soe", "pge"]
    ):
        raise ValueError("public formal-test promotion marker is not canonical")
    inputs = promotion.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {
        "summary_json", "summary_tsv", "input_manifest_json", "promotion_policy_document"
    }:
        raise ValueError("promotion input identities are incomplete")
    for name, identity in inputs.items():
        if not isinstance(identity, dict) or not isinstance(identity.get("path"), str):
            raise ValueError(f"promotion input identity is malformed: {name}")
        actual = tracker.identity(Path(identity["path"]))
        if not _identity_matches(identity, actual):
            raise ValueError(f"promotion input hash no longer matches: {name}")
    if _resolve(inputs["promotion_policy_document"]["path"]) != _resolve(
        promoter.POLICY_DOCUMENT
    ):
        raise ValueError("promotion is not bound to the canonical policy document")
    summary_json = Path(inputs["summary_json"]["path"])
    summary_tsv = Path(inputs["summary_tsv"]["path"])
    expected_manifest = summary_json.parent / "input_manifest.json"
    if _resolve(inputs["input_manifest_json"]["path"]) != _resolve(expected_manifest):
        raise ValueError("promotion audit input-manifest path is inconsistent")
    rebuilt = promoter.build_promotion(summary_json, summary_tsv)
    if promoter.comparable_promotion(promotion) != promoter.comparable_promotion(rebuilt):
        raise ValueError("promotion marker differs from a fresh closed-world recomputation")
    deterministic = {
        key: value
        for key, value in promotion.items()
        if key not in {"schema_version", "status", "promoted_at", "attestation_sha256"}
    }
    if promotion.get("attestation_sha256") != stable_digest(deterministic):
        raise ValueError("promotion attestation fingerprint is invalid")
    return promotion


def normalized_job_text(job: dict[str, Any]) -> str:
    segments = job.get("segments")
    if not isinstance(segments, list):
        raise ValueError(f"job {job.get('job_id')!r} has no segment list")
    text = "\n".join(str(segment.get("text", "")) for segment in segments)
    normalized = re.sub(r"\s+", " ", text).strip().casefold()
    if not normalized:
        raise ValueError(f"job {job.get('job_id')!r} has no source text")
    return normalized


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(FORBIDDEN_LABEL_KEYS & set(value)) or any(
            _contains_forbidden_key(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_key(item) for item in value)
    return False


def _validate_segments(row: dict[str, Any], path: Path) -> None:
    segments = row.get("segments")
    if not isinstance(segments, list) or len(segments) != 1:
        raise ValueError(f"{path} job {row.get('job_id')} must contain one segment")
    segment = segments[0]
    if not isinstance(segment, dict) or set(segment) != SEGMENT_KEYS:
        raise ValueError(f"{path} job {row.get('job_id')} has invalid segment schema")
    text = segment.get("text")
    if (
        segment.get("segment_id") != "S1"
        or segment.get("segment_type") != "sentence"
        or segment.get("page") is not None
        or segment.get("start") != 0
        or not isinstance(text, str)
        or not text
        or segment.get("end") != len(text)
    ):
        raise ValueError(f"{path} job {row.get('job_id')} has invalid character offsets")


def validate_source_jobs(
    path: Path, rows: list[dict[str, Any]], dataset: str, split: str
) -> None:
    expected = EXPECTED_TEST_JOBS[dataset] if split == "test" else None
    if not rows or (expected is not None and len(rows) != expected):
        raise ValueError(
            f"full {split} job count mismatch for {dataset}: {len(rows)} != {expected}"
        )
    seen_jobs: set[str] = set()
    seen_documents: set[str] = set()
    for row in rows:
        job_id, document_id = row.get("job_id"), row.get("document_id")
        if not isinstance(job_id, str) or not job_id or job_id in seen_jobs:
            raise ValueError(f"{path} contains a missing or duplicate job_id")
        if not isinstance(document_id, str) or not document_id or document_id in seen_documents:
            raise ValueError(f"{path} contains a missing or duplicate document_id")
        seen_jobs.add(job_id)
        seen_documents.add(document_id)
        prefix = f"{dataset}_{split}_"
        if not document_id.startswith(prefix) or job_id != f"{document_id}_C1":
            raise ValueError(f"{path} has invalid {split} job/document identity: {job_id}")
        suffix = document_id[len(prefix):]
        if row.get("source_path") != f"public:{dataset}:{split}:{suffix}":
            raise ValueError(f"{path} has invalid source provenance: {job_id}")
        if row.get("category") != dataset or row.get("language") != "en":
            raise ValueError(f"{path} has invalid category/language: {job_id}")
        _validate_segments(row, path)
        normalized_job_text(row)
        if _contains_forbidden_key(row):
            raise ValueError(f"{path} contains label-bearing data: {job_id}")
        if split == "test":
            if set(row) != BASELINE_JOB_KEYS:
                raise ValueError(f"{path} test job has noncanonical schema: {job_id}")
            expected_fields = {
                "experiment_mode": "baseline", "prompt_version": "public-benchmark-v1",
                "teacher_model": "benchmark_model", "status": "benchmark",
                "chunk_number": 1, "chunk_count": 1,
            }
            if any(row.get(name) != value for name, value in expected_fields.items()):
                raise ValueError(f"{path} test job violates baseline contract: {job_id}")
        elif split == "train":
            if row.get("train_graph_only") is not True:
                raise ValueError(f"{path} train job lacks train_graph_only provenance: {job_id}")
        else:
            raise ValueError(f"unsupported source split: {split}")


def validate_model_identity(record: Any) -> tuple[Path, dict[str, Any]]:
    """Verify the complete validation-period semantic-model identity."""

    if not isinstance(record, dict) or set(record) != {"path", "revision", "files"}:
        raise ValueError("validation graph manifest has invalid semantic model identity")
    model_path = Path(str(record.get("path", "")))
    if not model_path.is_dir() or model_path.is_symlink():
        raise ValueError(f"semantic model snapshot is missing or symlinked: {model_path}")
    if record.get("revision") != model_path.name:
        raise ValueError("semantic model revision/path mismatch")
    files = record.get("files")
    if not isinstance(files, dict) or set(files) != set(MODEL_FILES):
        raise ValueError("semantic model file identity is incomplete")
    verified: dict[str, Any] = {}
    repository_root = model_path.parent.parent
    for name in MODEL_FILES:
        expected = files[name]
        if not isinstance(expected, dict) or set(expected) != {"blob", "size"}:
            raise ValueError(f"semantic model identity is malformed for {name}")
        path = model_path / name
        if not path.exists():
            raise FileNotFoundError(path)
        resolved = path.resolve(strict=True)
        if not resolved.is_file() or resolved.name != expected.get("blob"):
            raise ValueError(f"semantic model blob identity mismatch for {name}")
        size = resolved.stat().st_size
        if isinstance(expected.get("size"), bool) or expected.get("size") != size:
            raise ValueError(f"semantic model size mismatch for {name}")
        if path.is_symlink():
            try:
                resolved.relative_to(repository_root / "blobs")
            except ValueError as error:
                raise ValueError(f"semantic model symlink escapes its blob store: {name}") from error
        else:
            try:
                resolved.relative_to(model_path.resolve())
            except ValueError as error:
                raise ValueError(f"semantic model file escapes its snapshot: {name}") from error
        verified[name] = {"blob": resolved.name, "size": size}
    actual = {"path": str(model_path), "revision": model_path.name, "files": verified}
    if actual != record:
        raise ValueError("semantic model identity differs from validation")
    return model_path, actual


def _validate_identity_record(
    record: Any,
    expected_path: Path,
    tracker: SnapshotTracker,
    label: str,
) -> dict[str, Any]:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError(f"{label} identity schema is invalid")
    if _resolve(str(record.get("path", ""))) != _resolve(expected_path):
        raise ValueError(f"{label} path differs from frozen provenance")
    actual = tracker.identity(expected_path)
    if record.get("sha256") != actual["sha256"]:
        raise ValueError(f"{label} hash differs from frozen provenance")
    return actual


def _manifest_output_identity(
    manifest: dict[str, Any],
    name: str,
    expected_path: Path,
    tracker: SnapshotTracker,
) -> dict[str, Any]:
    item = manifest.get("outputs", {}).get(name)
    if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
        raise ValueError(f"train-only graph manifest lacks canonical output {name}")
    if _resolve(str(item.get("path", ""))) != _resolve(expected_path):
        raise ValueError(f"train-only graph path mismatch for {name}")
    actual = tracker.identity(expected_path)
    if item.get("sha256") != actual["sha256"]:
        raise ValueError(f"train-only graph hash mismatch for {name}")
    return actual


def _validate_graph_manifest(
    dataset: str,
    manifest_path: Path,
    source_root: Path,
    graph_root: Path,
    promotion: dict[str, Any],
    tracker: SnapshotTracker,
) -> tuple[dict[str, Any], dict[str, Path], dict[str, Any], Path, dict[str, Any]]:
    manifest = tracker.json(manifest_path)
    expected_header = {
        "version": "public-eae-hrge-cpu-v1",
        "dataset": dataset,
        "status": "prepared_train_and_validation",
        "prepared_splits": ["train", "validation"],
        "test_job_file_read": False,
        "test_gold_read": False,
        "validation_gold_read": False,
        "semantic_device": "cpu",
    }
    for key, expected in expected_header.items():
        if manifest.get(key) != expected:
            raise ValueError(f"{dataset} graph manifest has invalid {key}")
    if manifest.get("settings") != VALIDATION_SETTINGS:
        raise ValueError(f"{dataset} validation retrieval settings drifted")
    if set(manifest.get("outputs", {})) != GRAPH_OUTPUT_NAMES:
        raise ValueError(f"{dataset} graph output registry is not closed")
    payload = manifest.get("fingerprint_inputs")
    if not isinstance(payload, dict) or stable_digest(payload) != manifest.get("fingerprint"):
        raise ValueError(f"{dataset} graph manifest fingerprint is invalid")
    if (
        payload.get("version") != "public-eae-hrge-cpu-v1"
        or payload.get("dataset") != dataset
        or payload.get("prepared_splits") != ["train", "validation"]
        or payload.get("settings") != VALIDATION_SETTINGS
        or payload.get("batch_size") != FROZEN_BATCH_SIZE
    ):
        raise ValueError(f"{dataset} graph fingerprint contract drifted")
    source = source_root / dataset
    prepared = graph_root / dataset
    expected_inputs = {
        "train_jobs": source / "train_baseline_jobs.jsonl",
        "validation_jobs": source / "validation_baseline_jobs.jsonl",
        "mentions": source / "knowledge_graph" / "mentions.jsonl",
        "training_edges": source / "knowledge_graph" / "training_edges.jsonl",
        "split_manifest": source / "split_manifest.jsonl",
        "ontology": source / "ontology.yaml",
        "eae_builder": SCRIPT_DIR / "build_experiment_jobs.py",
        "hrge_builder": SCRIPT_DIR / "build_kg_v2_jobs.py",
        "graph_converter": SCRIPT_DIR / "convert_public_train_graph.py",
        "preparation_runner": SCRIPT_DIR / "prepare_public_hrge_cpu.py",
    }
    input_records = payload.get("inputs")
    if not isinstance(input_records, dict) or set(input_records) != GRAPH_INPUT_NAMES:
        raise ValueError(f"{dataset} graph input registry is not closed")
    # Full mentions contain held-out annotations.  Their bytes are deliberately
    # not reopened here; the immutable graph manifest is instead hash-anchored
    # by the audited PGE run manifest below.  All inputs used by this builder
    # are independently checked from one-time snapshots.
    safe_current_inputs = {
        "train_jobs", "validation_jobs", "training_edges", "split_manifest", "ontology",
        "eae_builder", "hrge_builder", "graph_converter", "preparation_runner",
    }
    verified_inputs: dict[str, Any] = {}
    for name, expected_path in expected_inputs.items():
        record = input_records[name]
        if name in safe_current_inputs:
            verified_inputs[name] = _validate_identity_record(
                record, expected_path, tracker, f"{dataset} graph input {name}"
            )
        else:
            if (
                not isinstance(record, dict)
                or set(record) != {"path", "sha256"}
                or _resolve(str(record.get("path", ""))) != _resolve(expected_path)
                or not re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", "")))
            ):
                raise ValueError(f"{dataset} sealed graph input {name} provenance is invalid")

    expected_outputs = {
        "train_only_concepts": prepared / "knowledge_graph" / "concepts.jsonl",
        "train_only_mentions": prepared / "knowledge_graph" / "mentions.jsonl",
        "train_only_relations": prepared / "knowledge_graph" / "relations.jsonl",
        "graph_conversion_audit": prepared / "knowledge_graph" / "conversion_audit.json",
        "train_eae_jobs": prepared / "jobs" / "train_eae_jobs.jsonl",
        "validation_eae_jobs": prepared / "jobs" / "validation_eae_jobs.jsonl",
        "train_hrge_jobs": prepared / "jobs" / "train_hrge_jobs.jsonl",
        "validation_hrge_jobs": prepared / "jobs" / "validation_hrge_jobs.jsonl",
        "train_validation_source_jobs": prepared / "jobs" / "train_validation_source_jobs.jsonl",
        "train_validation_hrge_jobs": prepared / "jobs" / "train_validation_hrge_jobs.jsonl",
        "hrge_builder_audit_before_overlap_quarantine": prepared / "audits" / "train_validation_hrge_context.json",
        "exact_source_overlap_quarantine_audit": prepared / "audits" / "exact_source_overlap_quarantine.json",
    }
    verified_outputs = {
        name: _manifest_output_identity(manifest, name, path, tracker)
        for name, path in expected_outputs.items()
    }
    graph_audit = manifest.get("graph_audit")
    conversion_audit = tracker.json(expected_outputs["graph_conversion_audit"])
    if not isinstance(graph_audit, dict) or graph_audit != conversion_audit:
        raise ValueError(f"{dataset} graph audit is not identical to its frozen artifact")
    if (
        graph_audit.get("version") != "public-train-graph-v1"
        or graph_audit.get("policy") != "physical-train-only-after-manifest-cross-check"
        or graph_audit.get("dataset") != dataset
        or graph_audit.get("all_output_rows_train_only") is not True
        or graph_audit.get("non_train_graph_documents") != []
    ):
        raise ValueError(f"{dataset} graph audit safety policy failed")
    if graph_audit.get("inputs") != {
        name: input_records[name] for name in ("mentions", "split_manifest", "training_edges")
    }:
        raise ValueError(f"{dataset} graph audit input lineage is inconsistent")
    graph_output_mapping = {
        "concepts": manifest["outputs"]["train_only_concepts"],
        "mentions": manifest["outputs"]["train_only_mentions"],
        "relations": manifest["outputs"]["train_only_relations"],
    }
    if graph_audit.get("outputs") != graph_output_mapping:
        raise ValueError(f"{dataset} graph audit output lineage is inconsistent")
    job_counts = manifest.get("job_counts")
    if (
        not isinstance(job_counts, dict)
        or set(job_counts) != {"train", "validation"}
        or job_counts.get("validation") != promoter.EXPECTED_VALIDATION_JOBS[dataset]
        or isinstance(job_counts.get("train"), bool)
        or not isinstance(job_counts.get("train"), int)
        or job_counts["train"] < 1
    ):
        raise ValueError(f"{dataset} graph job-count contract is invalid")
    pge_identity = promotion.get("pge_run_manifest")
    if not isinstance(pge_identity, dict) or not isinstance(pge_identity.get("path"), str):
        raise ValueError("promotion lacks PGE run-manifest identity")
    pge_actual = tracker.identity(Path(pge_identity["path"]))
    if not _identity_matches(pge_identity, pge_actual):
        raise ValueError("promotion PGE run-manifest hash changed")
    pge_manifest = tracker.json(Path(pge_identity["path"]))
    matching = [
        digest for raw_path, digest in pge_manifest.get("files", {}).items()
        if _resolve(raw_path) == _resolve(manifest_path)
    ]
    graph_manifest_identity = tracker.identity(manifest_path)
    if len(matching) != 1 or matching[0] != graph_manifest_identity["sha256"]:
        raise ValueError(f"{dataset} graph manifest is not anchored by audited PGE")
    model_path, semantic_identity = validate_model_identity(payload.get("semantic_model"))
    return manifest, expected_outputs, {
        "inputs": verified_inputs,
        "outputs": verified_outputs,
        "graph_manifest": graph_manifest_identity,
        "graph_audit": graph_audit,
    }, model_path, semantic_identity


def validate_physical_train_graph(
    concepts: list[dict[str, Any]],
    mentions: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    train_document_ids: set[str],
    graph_audit: dict[str, Any],
) -> dict[str, int]:
    concept_ids = [str(row.get("concept_id", "")) for row in concepts]
    mention_ids = [str(row.get("mention_id", "")) for row in mentions]
    relation_ids = [str(row.get("relation_id", "")) for row in relations]
    for label, identifiers in (
        ("concept", concept_ids), ("mention", mention_ids), ("relation", relation_ids)
    ):
        if not identifiers or "" in identifiers or len(set(identifiers)) != len(identifiers):
            raise ValueError(f"train-only graph has missing or duplicate {label} IDs")
    concept_set = set(concept_ids)
    mention_documents: dict[str, set[str]] = {}
    for row in mentions:
        document_id = str(row.get("document_id", ""))
        concept_id = str(row.get("concept_id", ""))
        if row.get("split") != "train" or document_id not in train_document_ids:
            raise ValueError("mention graph is not physically train-only")
        if concept_id not in concept_set:
            raise ValueError("mention graph references an unknown concept")
        mention_documents.setdefault(concept_id, set()).add(document_id)
    if set(mention_documents) != concept_set:
        raise ValueError("train-only graph contains an orphan concept")
    for row in concepts:
        concept_id = str(row["concept_id"])
        documents = row.get("source_documents")
        if (
            not isinstance(documents, list)
            or set(map(str, documents)) != mention_documents[concept_id]
            or not set(map(str, documents)) <= train_document_ids
        ):
            raise ValueError("concept provenance disagrees with physical train mentions")
    for row in relations:
        document_id = str(row.get("document_id", ""))
        if row.get("split") != "train" or document_id not in train_document_ids:
            raise ValueError("relation graph is not physically train-only")
        provenance = row.get("provenance")
        if not isinstance(provenance, dict) or provenance.get("source_split") != "train":
            raise ValueError("relation graph lacks train-only provenance")
        if (
            str(row.get("source_concept_id", "")) not in concept_set
            or str(row.get("target_concept_id", "")) not in concept_set
        ):
            raise ValueError("relation graph references an unknown concept")
    counts = {
        "concepts": len(concepts),
        "mentions": len(mentions),
        "relations": len(relations),
        "training_documents": len(train_document_ids),
    }
    expected = {
        "concepts": graph_audit.get("concepts"),
        "mentions": graph_audit.get("mentions"),
        "relations": graph_audit.get("relations"),
        "training_documents": graph_audit.get("train_documents_in_manifest"),
    }
    if counts != expected:
        raise ValueError(f"physical train graph counts disagree with graph audit: {counts} != {expected}")
    return counts


def exact_text_quarantine(
    train_jobs: list[dict[str, Any]], test_jobs: list[dict[str, Any]]
) -> dict[str, list[str]]:
    training_documents_by_text: dict[str, set[str]] = {}
    for job in train_jobs:
        training_documents_by_text.setdefault(normalized_job_text(job), set()).add(
            str(job["document_id"])
        )
    return {
        str(job["job_id"]): sorted(training_documents_by_text[normalized_job_text(job)])
        for job in test_jobs
        if normalized_job_text(job) in training_documents_by_text
    }


def _shared_ontology(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    ontology = rows[0].get("ontology") if rows else None
    if not isinstance(ontology, dict) or not ontology:
        raise ValueError(f"{label} jobs do not embed an ontology")
    digest = stable_digest(ontology)
    if any(stable_digest(row.get("ontology")) != digest for row in rows):
        raise ValueError(f"{label} jobs embed inconsistent ontologies")
    return ontology


def build_eae_rows(
    test_jobs: list[dict[str, Any]],
    mentions: list[dict[str, Any]],
    quarantine: dict[str, list[str]],
) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str, str], int] = {}
    documents: dict[tuple[str, str, str], set[str]] = {}
    for mention in mentions:
        key = (
            str(mention.get("language", "unknown")),
            str(mention.get("type", "UNKNOWN")),
            str(
                mention.get("canonical_name")
                or mention.get("normalized_name")
                or mention.get("text", "")
            ),
        )
        counts[key] = counts.get(key, 0) + 1
        documents.setdefault(key, set()).add(str(mention.get("document_id", "")))
    concepts = [
        {
            "language": key[0], "type": key[1], "name": key[2],
            "count": count, "source_documents": documents[key],
        }
        for key, count in counts.items()
        if key[1] and key[2]
    ]
    ontology = _shared_ontology(test_jobs, "test baseline")
    output: list[dict[str, Any]] = []
    for source in test_jobs:
        row = copy.deepcopy(source)
        job_id = str(row["job_id"])
        if job_id in quarantine:
            row["graph_context_quarantine"] = {
                "reason": "exact_source_text_occurs_in_training_split",
                "excluded_training_documents": quarantine[job_id],
            }
        else:
            context = eae_builder.retrieve_concept_context(
                row, concepts, SETTINGS["eae_concept_limit"],
                SETTINGS["eae_balanced_concepts"],
            )
            row["system_instruction"] = str(row["system_instruction"]) + (
                eae_builder.constraint_context(ontology, context)
            )
        row["experiment_mode"] = "eae_exact_anchor"
        row["method_name"] = "EAE"
        row["train_graph_only"] = True
        output.append(row)
    return output


def quarantine_hrge_rows(
    rows: list[dict[str, Any]], quarantine: dict[str, list[str]]
) -> list[dict[str, Any]]:
    for row in rows:
        job_id = str(row.get("job_id", ""))
        context = row.get("kg_v2_context")
        if not isinstance(context, dict) or context.get("train_graph_only") is not True:
            raise ValueError(f"HRGE builder omitted train_graph_only for {job_id}")
        if context.get("leave_current_document_out") is not True:
            raise ValueError(f"HRGE builder omitted leave-current-document-out for {job_id}")
        row["method_name"] = "HRGE"
        # Keep the method-level provenance contract symmetric with EAE.  The
        # nested kg_v2_context guard proves graph construction details; this
        # top-level bit is the common release-runner discriminator.
        row["train_graph_only"] = True
        if job_id not in quarantine:
            continue
        base_instruction = str(row.get("system_instruction", "")).split("\n\nKG_RULES:", 1)[0].rstrip()
        row["system_instruction"] = f"{base_instruction}\n\n{hrge_builder.render_context([], [], [])}"
        context["anchors"] = []
        context["edge_priors"] = []
        context["semantic_relation_patterns"] = []
        context["exact_source_overlap_quarantine"] = {
            "reason": "exact_source_text_occurs_in_training_split",
            "excluded_training_documents": quarantine[job_id],
        }
    return rows


def validate_generated_method_provenance(
    dataset: str,
    expected_ids: list[str],
    eae_rows: list[dict[str, Any]],
    hrge_rows: list[dict[str, Any]],
) -> None:
    """Apply one closed provenance contract to in-memory and stored outputs."""

    for method, rows in (("EAE", eae_rows), ("HRGE", hrge_rows)):
        actual_ids = [str(row.get("job_id", "")) for row in rows]
        if actual_ids != expected_ids or len(set(actual_ids)) != len(actual_ids):
            raise ValueError(
                f"formal-test {method} IDs/order are incomplete: {dataset}"
            )
        if any(
            row.get("method_name") != method
            or row.get("train_graph_only") is not True
            for row in rows
        ):
            raise ValueError(
                f"formal-test {method} provenance is invalid: {dataset}"
            )
    for row in hrge_rows:
        context = row.get("kg_v2_context")
        if (
            not isinstance(context, dict)
            or context.get("train_graph_only") is not True
            or context.get("leave_current_document_out") is not True
        ):
            raise ValueError(
                f"formal-test HRGE leakage guard is invalid: {row.get('job_id')}"
            )


def dry_run_generated_provenance(dataset: str, context: dict[str, Any]) -> None:
    """Exercise the same writer/validator contract without semantic inference."""

    eae_rows = build_eae_rows(
        context["test_jobs"], context["mentions"], context["quarantine"]
    )
    # The semantic builder already has its own frozen preflight and physical
    # graph checks.  For a no-write dry run, construct its documented envelope
    # and pass it through the exact same post-builder writer that production
    # uses.  This specifically catches writer/verifier schema drift.
    hrge_envelopes: list[dict[str, Any]] = []
    for source in context["test_jobs"]:
        row = copy.deepcopy(source)
        row["kg_v2_context"] = {
            "train_graph_only": True,
            "leave_current_document_out": True,
            "anchors": [],
            "edge_priors": [],
            "semantic_relation_patterns": [],
        }
        hrge_envelopes.append(row)
    hrge_rows = quarantine_hrge_rows(hrge_envelopes, context["quarantine"])
    validate_generated_method_provenance(
        dataset,
        [str(row["job_id"]) for row in context["test_jobs"]],
        eae_rows,
        hrge_rows,
    )


def _verified_context(
    dataset: str,
    promotion_path: Path,
    source_root: Path,
    graph_root: Path,
) -> dict[str, Any]:
    if dataset not in DATASETS:
        raise ValueError(f"unsupported public dataset {dataset!r}")
    promotion = validate_promotion(promotion_path)
    tracker = SnapshotTracker()
    if tracker.json(promotion_path) != promotion:
        raise ValueError("promotion marker changed during revalidation")
    source = source_root / dataset
    prepared = graph_root / dataset
    test_path = source / "test_baseline_jobs.jsonl"
    manifest_path = prepared / "preparation_manifest.json"
    _require_contained(test_path, source, "test baseline input")
    if test_path.name != "test_baseline_jobs.jsonl" or test_path.parent != source:
        raise ValueError("only the canonical test_baseline_jobs.jsonl may be opened")
    graph_manifest, graph_paths, graph_proof, model_path, semantic_identity = (
        _validate_graph_manifest(
            dataset, manifest_path, source_root, graph_root, promotion, tracker
        )
    )
    test_jobs = tracker.jsonl(test_path)
    train_jobs = tracker.jsonl(graph_paths["train_eae_jobs"])
    validate_source_jobs(test_path, test_jobs, dataset, "test")
    validate_source_jobs(graph_paths["train_eae_jobs"], train_jobs, dataset, "train")
    if len(train_jobs) != graph_manifest["job_counts"]["train"]:
        raise ValueError(f"{dataset} train job count differs from graph manifest")
    train_ontology = _shared_ontology(train_jobs, "train EAE")
    test_ontology = _shared_ontology(test_jobs, "test baseline")
    if stable_digest(train_ontology) != stable_digest(test_ontology):
        raise ValueError(f"{dataset} test ontology differs from training ontology")
    train_document_ids = {str(row["document_id"]) for row in train_jobs}
    test_document_ids = {str(row["document_id"]) for row in test_jobs}
    if train_document_ids & test_document_ids:
        raise ValueError(f"{dataset} train and test documents overlap by ID")
    concepts = tracker.jsonl(graph_paths["train_only_concepts"])
    mentions = tracker.jsonl(graph_paths["train_only_mentions"])
    relations = tracker.jsonl(graph_paths["train_only_relations"])
    graph_counts = validate_physical_train_graph(
        concepts, mentions, relations, train_document_ids, graph_proof["graph_audit"]
    )
    quarantine = exact_text_quarantine(train_jobs, test_jobs)
    preparation_runner = tracker.identity(Path(__file__).resolve())
    promotion_identity = tracker.identity(promotion_path)
    test_identity = tracker.identity(test_path)
    fingerprint_payload = {
        "schema_version": PREPARATION_SCHEMA_VERSION,
        "dataset": dataset,
        "promotion": promotion_identity,
        "promotion_attestation_sha256": promotion["attestation_sha256"],
        "test_baseline_jobs": test_identity,
        "train_only_graph_manifest": graph_proof["graph_manifest"],
        "train_only_graph_outputs": {
            name: graph_proof["outputs"][name]
            for name in ("train_only_concepts", "train_only_mentions", "train_only_relations")
        },
        "validation_frozen_builders": {
            name: graph_proof["inputs"][name]
            for name in ("eae_builder", "hrge_builder")
        },
        "preparation_runner": preparation_runner,
        "settings": SETTINGS,
        "batch_size": FROZEN_BATCH_SIZE,
        "semantic_model": semantic_identity,
    }
    return {
        "promotion": promotion,
        "promotion_identity": promotion_identity,
        "tracker": tracker,
        "test_path": test_path,
        "graph_paths": graph_paths,
        "test_jobs": test_jobs,
        "train_jobs": train_jobs,
        "concepts": concepts,
        "mentions": mentions,
        "relations": relations,
        "graph_counts": graph_counts,
        "graph_proof": graph_proof,
        "model_path": model_path,
        "semantic_identity": semantic_identity,
        "quarantine": quarantine,
        "fingerprint_payload": fingerprint_payload,
        "fingerprint": stable_digest(fingerprint_payload),
    }


def _existing_outputs_are_valid(
    target: Path, fingerprint: str, tracker: SnapshotTracker
) -> bool:
    manifest_path = target / "preparation_manifest.json"
    try:
        manifest = tracker.json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if (
        manifest.get("schema_version") != PREPARATION_SCHEMA_VERSION
        or manifest.get("status") != "prepared_test"
        or manifest.get("fingerprint") != fingerprint
        or manifest.get("test_gold_read") is not False
        or manifest.get("validation_gold_read") is not False
    ):
        return False
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != {
        "test_eae_jobs", "test_hrge_jobs", "hrge_context_audit",
        "train_test_exact_text_quarantine_audit",
    }:
        return False
    for item in outputs.values():
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            return False
        path = Path(item["path"])
        try:
            path.resolve(strict=False).relative_to(target.resolve(strict=False))
            actual = tracker.identity(path)
        except (OSError, ValueError):
            return False
        if not _identity_matches(item, actual):
            return False
    return True


def validate_preparation_manifest(
    dataset: str,
    manifest_path: Path,
    promotion_path: Path = PROMOTION_PATH,
    source_root: Path = SOURCE_ROOT,
    graph_root: Path = GRAPH_ROOT,
) -> dict[str, Any]:
    """Revalidate one prepared dataset and return its immutable identity."""

    context = _verified_context(dataset, promotion_path, source_root, graph_root)
    target = manifest_path.parent
    # Tests may place an isolated release elsewhere, but a production default
    # must still be exactly dataset-scoped; no recorded output may escape it.
    if manifest_path.name != "preparation_manifest.json":
        raise ValueError("formal-test preparation manifest has a noncanonical name")
    tracker: SnapshotTracker = context["tracker"]
    manifest = tracker.json(manifest_path)
    expected_header = {
        "schema_version": PREPARATION_SCHEMA_VERSION,
        "status": "prepared_test",
        "dataset": dataset,
        "prepared_split": "test",
        "fingerprint": context["fingerprint"],
        "promotion_attestation_sha256": context["promotion"]["attestation_sha256"],
        "test_namespace_inputs_opened": [str(context["test_path"])],
        "test_baseline_jobs_read": True,
        "test_gold_read": False,
        "validation_gold_read": False,
        "job_count": EXPECTED_TEST_JOBS[dataset],
        "settings": SETTINGS,
        "batch_size": FROZEN_BATCH_SIZE,
        "semantic_device": "cpu",
        "semantic_model": context["semantic_identity"],
        "source_snapshot_policy": "read-once-hash-and-build-from-identical-bytes",
        "train_only_graph_manifest_pge_anchored": True,
    }
    for key, expected in expected_header.items():
        if manifest.get(key) != expected:
            raise ValueError(f"formal-test manifest has invalid {key}: {dataset}")
    if not _identity_matches(manifest.get("promotion"), context["promotion_identity"]):
        raise ValueError(f"formal-test manifest promotion identity mismatch: {dataset}")
    if manifest.get("promotion_dataset_decision") != context["promotion"]["datasets"][dataset]:
        raise ValueError(f"formal-test manifest promotion decision mismatch: {dataset}")
    if manifest.get("train_only_graph_manifest") != context["graph_proof"]["graph_manifest"]:
        raise ValueError(f"formal-test graph-manifest identity mismatch: {dataset}")
    expected_graph_outputs = {
        name: context["graph_proof"]["outputs"][name]
        for name in ("train_only_concepts", "train_only_mentions", "train_only_relations")
    }
    if manifest.get("train_only_graph_manifest_verified_outputs") != expected_graph_outputs:
        raise ValueError(f"formal-test graph-output lineage mismatch: {dataset}")
    if manifest.get("validation_frozen_builders") != context["fingerprint_payload"][
        "validation_frozen_builders"
    ]:
        raise ValueError(f"formal-test builder identity mismatch: {dataset}")
    expected_test_schema = {
        "expected_jobs": EXPECTED_TEST_JOBS[dataset],
        "exact_top_level_keys": sorted(BASELINE_JOB_KEYS),
        "exact_segment_keys": sorted(SEGMENT_KEYS),
        "experiment_mode": "baseline",
        "id_source_and_character_offsets_verified": True,
        "ontology_matches_train": True,
    }
    if manifest.get("test_schema") != expected_test_schema:
        raise ValueError(f"formal-test source schema proof mismatch: {dataset}")
    isolation = manifest.get("exact_train_test_text_isolation")
    quarantine = context["quarantine"]
    if (
        not isinstance(isolation, dict)
        or isolation.get("policy")
        != "clear_all_graph_context_for_exact_train_test_source_overlap"
        or isolation.get("quarantined_test_jobs") != len(quarantine)
        or isolation.get("test_job_ids") != sorted(quarantine)
        or isolation.get("excluded_training_documents_by_test_job")
        != dict(sorted(quarantine.items()))
        or isolation.get("all_quarantined_eae_context_cleared") is not True
        or isolation.get("all_quarantined_hrge_context_cleared") is not True
    ):
        raise ValueError(f"formal-test overlap quarantine proof mismatch: {dataset}")
    outputs = manifest.get("outputs")
    expected_paths = {
        "test_eae_jobs": target / "jobs" / "test_eae_jobs.jsonl",
        "test_hrge_jobs": target / "jobs" / "test_hrge_jobs.jsonl",
        "hrge_context_audit": target / "audits" / "test_hrge_context.json",
        "train_test_exact_text_quarantine_audit": target / "audits" / "train_test_exact_text_quarantine.json",
    }
    if not isinstance(outputs, dict) or set(outputs) != set(expected_paths):
        raise ValueError(f"formal-test output registry is not closed: {dataset}")
    verified_outputs: dict[str, Any] = {}
    for name, path in expected_paths.items():
        item = outputs[name]
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError(f"formal-test output identity is malformed: {dataset}/{name}")
        if _resolve(item["path"]) != _resolve(path):
            raise ValueError(f"formal-test output path mismatch: {dataset}/{name}")
        _require_contained(path, target, f"formal-test output {name}")
        actual = tracker.identity(path)
        if not _identity_matches(item, actual):
            raise ValueError(f"formal-test output hash mismatch: {dataset}/{name}")
        verified_outputs[name] = actual
    expected_ids = [str(row["job_id"]) for row in context["test_jobs"]]
    eae_rows = tracker.jsonl(expected_paths["test_eae_jobs"])
    hrge_rows = tracker.jsonl(expected_paths["test_hrge_jobs"])
    validate_generated_method_provenance(
        dataset, expected_ids, eae_rows, hrge_rows
    )
    for row in hrge_rows:
        job_id = str(row.get("job_id", ""))
        context_value = row.get("kg_v2_context")
        if job_id in quarantine and (
            context_value.get("anchors")
            or context_value.get("edge_priors")
            or context_value.get("semantic_relation_patterns")
            or not isinstance(context_value.get("exact_source_overlap_quarantine"), dict)
        ):
            raise ValueError(f"formal-test HRGE quarantine is not empty: {job_id}")
    for row in eae_rows:
        if str(row["job_id"]) in quarantine and "KG_RULES:" in str(
            row.get("system_instruction", "")
        ):
            raise ValueError(f"formal-test EAE quarantine is not empty: {row['job_id']}")
    overlap_artifact = tracker.json(expected_paths["train_test_exact_text_quarantine_audit"])
    if overlap_artifact != isolation:
        raise ValueError(f"formal-test overlap audit hash/content mismatch: {dataset}")
    hrge_audit = tracker.json(expected_paths["hrge_context_audit"])
    if (
        hrge_audit.get("split") != "test"
        or hrge_audit.get("test_gold_read") is not False
        or hrge_audit.get("promotion_attestation_sha256")
        != context["promotion"]["attestation_sha256"]
        or hrge_audit.get("source_snapshot_policy") != "single-read-verified-bytes"
    ):
        raise ValueError(f"formal-test HRGE audit contract mismatch: {dataset}")
    return {
        "dataset": dataset,
        "manifest": tracker.identity(manifest_path),
        "outputs": verified_outputs,
        "fingerprint": context["fingerprint"],
        "status": "verified_prepared_test",
    }


def validate_prepared_release(
    promotion_path: Path = PROMOTION_PATH,
    output_root: Path = OUTPUT_ROOT,
    source_root: Path = SOURCE_ROOT,
    graph_root: Path = GRAPH_ROOT,
) -> dict[str, Any]:
    """Return a canonical all-dataset release fingerprint for runners."""

    promotion = validate_promotion(promotion_path)
    promotion_identity = SnapshotTracker().identity(promotion_path)
    datasets = {
        dataset: validate_preparation_manifest(
            dataset,
            output_root / dataset / "preparation_manifest.json",
            promotion_path,
            source_root,
            graph_root,
        )
        for dataset in DATASETS
    }
    payload = {
        "schema_version": "public-formal-test-release-v2",
        "status": "verified_release",
        "promotion": promotion_identity,
        "promotion_attestation_sha256": promotion["attestation_sha256"],
        "datasets": datasets,
    }
    return {**payload, "release_sha256": stable_digest(payload)}


def _staged_identity(staged_path: Path, final_path: Path) -> dict[str, Any]:
    payload = staged_path.read_bytes()
    return {
        "path": str(final_path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def prepare_dataset(
    dataset: str,
    promotion_path: Path = PROMOTION_PATH,
    source_root: Path = SOURCE_ROOT,
    graph_root: Path = GRAPH_ROOT,
    output_root: Path = OUTPUT_ROOT,
    semantic_model: Path | None = None,
    batch_size: int = FROZEN_BATCH_SIZE,
    force: bool = False,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Validate and atomically materialize one frozen test-input directory.

    ``semantic_model`` exists only to fail legacy callers closed: model identity
    is always derived from the PGE-anchored validation manifest.
    """

    if semantic_model is not None:
        raise ValueError("semantic-model override is forbidden after validation")
    if batch_size != FROZEN_BATCH_SIZE:
        raise ValueError(f"batch size is frozen at {FROZEN_BATCH_SIZE}")
    if force:
        raise ValueError("force/overwrite is forbidden for formal-test inputs")
    context = _verified_context(dataset, promotion_path, source_root, graph_root)
    target = output_root / dataset
    if dry_run:
        dry_run_generated_provenance(dataset, context)
        return {
            "dataset": dataset,
            "status": "ready_no_writes",
            "fingerprint": context["fingerprint"],
            "job_count": len(context["test_jobs"]),
            "train_only_graph_counts": context["graph_counts"],
            "quarantined_test_jobs": len(context["quarantine"]),
            "semantic_model_revision": context["semantic_identity"]["revision"],
            "promotion_attestation_sha256": context["promotion"]["attestation_sha256"],
        }

    if target.exists():
        if target.is_dir() and not target.is_symlink():
            try:
                verified = validate_preparation_manifest(
                    dataset,
                    target / "preparation_manifest.json",
                    promotion_path,
                    source_root,
                    graph_root,
                )
            except (OSError, ValueError, json.JSONDecodeError):
                verified = None
            if verified is not None and verified["fingerprint"] == context["fingerprint"]:
                return {
                    "dataset": dataset,
                    "status": "skipped_unchanged",
                    "fingerprint": context["fingerprint"],
                    "manifest": str(target / "preparation_manifest.json"),
                }
        raise FileExistsError(
            f"refusing to overwrite noncanonical formal-test directory: {target}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink(output_root, regular=False)
    staging = Path(tempfile.mkdtemp(prefix=f".{dataset}.formal-test.", dir=output_root))
    try:
        jobs_dir = staging / "jobs"
        audits_dir = staging / "audits"
        eae_rows = build_eae_rows(
            context["test_jobs"], context["mentions"], context["quarantine"]
        )
        with tempfile.TemporaryDirectory(
            prefix=f".{dataset}.verified-snapshots.", dir=output_root
        ) as temporary_name:
            temporary = Path(temporary_name)
            snapshot_jobs = temporary / "test_baseline_jobs.jsonl"
            snapshot_concepts = temporary / "concepts.jsonl"
            snapshot_relations = temporary / "relations.jsonl"
            temporary_hrge = temporary / "test_hrge_jobs.jsonl"
            temporary_audit = temporary / "test_hrge_context.json"
            tracker: SnapshotTracker = context["tracker"]
            _write_snapshot(snapshot_jobs, tracker.bytes(context["test_path"]))
            _write_snapshot(
                snapshot_concepts, tracker.bytes(context["graph_paths"]["train_only_concepts"])
            )
            _write_snapshot(
                snapshot_relations, tracker.bytes(context["graph_paths"]["train_only_relations"])
            )
            hrge_builder.run(
                SimpleNamespace(
                    jobs=snapshot_jobs,
                    concepts=snapshot_concepts,
                    relations=snapshot_relations,
                    output=temporary_hrge,
                    audit=temporary_audit,
                    semantic_model=context["model_path"],
                    device="cpu",
                    batch_size=FROZEN_BATCH_SIZE,
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
            generated_tracker = SnapshotTracker()
            hrge_rows = quarantine_hrge_rows(
                generated_tracker.jsonl(temporary_hrge), context["quarantine"]
            )
            hrge_audit = generated_tracker.json(temporary_audit)

        expected_ids = [str(row["job_id"]) for row in context["test_jobs"]]
        validate_generated_method_provenance(
            dataset, expected_ids, eae_rows, hrge_rows
        )

        eae_staged = jobs_dir / "test_eae_jobs.jsonl"
        hrge_staged = jobs_dir / "test_hrge_jobs.jsonl"
        hrge_audit_staged = audits_dir / "test_hrge_context.json"
        overlap_staged = audits_dir / "train_test_exact_text_quarantine.json"
        write_jsonl_atomic(eae_staged, eae_rows)
        write_jsonl_atomic(hrge_staged, hrge_rows)
        hrge_audit.update({
            "output": str(target / "jobs" / "test_hrge_jobs.jsonl"),
            "split": "test",
            "test_gold_read": False,
            "promotion_attestation_sha256": context["promotion"]["attestation_sha256"],
            "source_snapshot_policy": "single-read-verified-bytes",
        })
        write_json_atomic(hrge_audit_staged, hrge_audit)
        quarantine = context["quarantine"]
        overlap_audit = {
            "policy": "clear_all_graph_context_for_exact_train_test_source_overlap",
            "normalized_text": "Unicode casefold plus collapsed whitespace",
            "quarantined_test_jobs": len(quarantine),
            "test_job_ids": sorted(quarantine),
            "excluded_training_documents_by_test_job": dict(sorted(quarantine.items())),
            "all_quarantined_eae_context_cleared": all(
                "KG_RULES:" not in str(row.get("system_instruction", ""))
                for row in eae_rows if str(row["job_id"]) in quarantine
            ),
            "all_quarantined_hrge_context_cleared": all(
                not row["kg_v2_context"].get("anchors")
                and not row["kg_v2_context"].get("edge_priors")
                and not row["kg_v2_context"].get("semantic_relation_patterns")
                for row in hrge_rows if str(row["job_id"]) in quarantine
            ),
        }
        if not overlap_audit["all_quarantined_eae_context_cleared"] or not overlap_audit[
            "all_quarantined_hrge_context_cleared"
        ]:
            raise ValueError("exact train-test quarantine did not clear every graph channel")
        write_json_atomic(overlap_staged, overlap_audit)

        staged_outputs = {
            "test_eae_jobs": eae_staged,
            "test_hrge_jobs": hrge_staged,
            "hrge_context_audit": hrge_audit_staged,
            "train_test_exact_text_quarantine_audit": overlap_staged,
        }
        final_outputs = {
            "test_eae_jobs": target / "jobs" / "test_eae_jobs.jsonl",
            "test_hrge_jobs": target / "jobs" / "test_hrge_jobs.jsonl",
            "hrge_context_audit": target / "audits" / "test_hrge_context.json",
            "train_test_exact_text_quarantine_audit": target / "audits" / "train_test_exact_text_quarantine.json",
        }
        manifest = {
            "schema_version": PREPARATION_SCHEMA_VERSION,
            "status": "prepared_test",
            "dataset": dataset,
            "prepared_split": "test",
            "fingerprint": context["fingerprint"],
            "promotion": context["promotion_identity"],
            "promotion_attestation_sha256": context["promotion"]["attestation_sha256"],
            "promotion_dataset_decision": context["promotion"]["datasets"][dataset],
            "test_namespace_inputs_opened": [str(context["test_path"])],
            "test_baseline_jobs_read": True,
            "test_gold_read": False,
            "validation_gold_read": False,
            "job_count": len(context["test_jobs"]),
            "test_schema": {
                "expected_jobs": EXPECTED_TEST_JOBS[dataset],
                "exact_top_level_keys": sorted(BASELINE_JOB_KEYS),
                "exact_segment_keys": sorted(SEGMENT_KEYS),
                "experiment_mode": "baseline",
                "id_source_and_character_offsets_verified": True,
                "ontology_matches_train": True,
            },
            "train_only_graph_counts": context["graph_counts"],
            "train_only_graph_manifest": context["graph_proof"]["graph_manifest"],
            "train_only_graph_manifest_pge_anchored": True,
            "train_only_graph_manifest_verified_outputs": {
                name: context["graph_proof"]["outputs"][name]
                for name in ("train_only_concepts", "train_only_mentions", "train_only_relations")
            },
            "exact_train_test_text_isolation": overlap_audit,
            "settings": SETTINGS,
            "batch_size": FROZEN_BATCH_SIZE,
            "semantic_device": "cpu",
            "semantic_model": context["semantic_identity"],
            "validation_frozen_builders": context["fingerprint_payload"]["validation_frozen_builders"],
            "source_snapshot_policy": "read-once-hash-and-build-from-identical-bytes",
            "fingerprint_inputs": context["fingerprint_payload"],
            "outputs": {
                name: _staged_identity(path, final_outputs[name])
                for name, path in staged_outputs.items()
            },
        }
        write_json_atomic(staging / "preparation_manifest.json", manifest)
        os.replace(staging, target)
        staging = Path()  # ownership moved atomically
        return {
            "dataset": dataset,
            "status": "prepared_test",
            "fingerprint": context["fingerprint"],
            "manifest": str(target / "preparation_manifest.json"),
            "job_count": len(context["test_jobs"]),
            "quarantined_test_jobs": len(quarantine),
        }
    finally:
        if staging != Path() and staging.exists():
            shutil.rmtree(staging)


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
    except ImportError as error:
        raise RuntimeError("PyTorch is required for semantic CPU preparation") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--batch-size", type=int, default=FROZEN_BATCH_SIZE)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Revalidate all release/data/graph/model contracts without writing outputs.",
    )
    parser.add_argument(
        "--verify-release", action="store_true",
        help="Verify all prepared outputs and print their canonical release fingerprint.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size != FROZEN_BATCH_SIZE:
        raise ValueError(f"batch size is frozen at {FROZEN_BATCH_SIZE}")
    if args.dry_run and args.verify_release:
        raise ValueError("--dry-run and --verify-release are mutually exclusive")
    if args.verify_release:
        print(json.dumps(validate_prepared_release(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if not args.dry_run:
        configure_cpu(args.threads)
    results = [
        prepare_dataset(dataset, batch_size=args.batch_size, dry_run=args.dry_run)
        for dataset in args.datasets
    ]
    print(json.dumps({
        "status": "ready_no_writes" if args.dry_run else "prepared_test",
        "datasets": results,
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
