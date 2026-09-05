#!/usr/bin/env python3
"""Convert a promoted SpERT test prediction array to project annotation JSONL.

The converter deliberately revalidates the canonical promotion and the
materialized SpERT test manifest before it opens any test input.  Rows are
converted in native SpERT order; no ID recovery or best-effort alignment is
allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import prepare_spert_fresh_splits as training  # noqa: E402
import prepare_spert_fresh_test as prepared  # noqa: E402


DATASETS = training.DATASETS
EXPECTED_ROWS = prepared.EXPECTED_TEST_ROWS
SCHEMA_VERSION = "spert-fresh-formal-test-conversion-v1"
FORMAL_ROOT = Path("outputs/public_formal_matrix/horizontal/spert_fresh_seed42")


def require_project_cwd() -> None:
    if Path.cwd().resolve(strict=True) != PROJECT_ROOT.resolve(strict=True):
        raise ValueError(
            f"SpERT formal-test conversion must run from the canonical project root: {PROJECT_ROOT}"
        )


def _candidate(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def assert_regular(path: Path) -> Path:
    candidate = _candidate(path)
    for component in (candidate, *candidate.parents):
        if component.is_symlink():
            raise ValueError(f"symlinked SpERT conversion input is forbidden: {path}")
    try:
        mode = candidate.stat().st_mode
    except FileNotFoundError:
        raise FileNotFoundError(path) from None
    if not stat.S_ISREG(mode):
        raise ValueError(f"SpERT conversion input is not a regular file: {path}")
    return candidate


def read_bytes(path: Path) -> bytes:
    candidate = assert_regular(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as error:
        raise ValueError(f"cannot securely open SpERT conversion input {path}: {error}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"SpERT conversion input is not regular: {path}")
        if Path(f"/proc/self/fd/{descriptor}").resolve(strict=True) != candidate.resolve(strict=True):
            raise ValueError(f"SpERT conversion input changed during secure open: {path}")
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
            raise ValueError(f"SpERT conversion input changed while being read: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def read_json(path: Path) -> Any:
    try:
        return json.loads(read_bytes(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON at {path}: {error}") from error


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        text = read_bytes(path).decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"invalid UTF-8 at {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
        if not isinstance(value, dict):
            raise ValueError(f"non-object JSONL row at {path}:{line_number}")
        rows.append(value)
    return rows


def identity(path: Path) -> dict[str, Any]:
    payload = read_bytes(path)
    return payload_identity(path, payload)


def payload_identity(path: Path, payload: bytes) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def json_bytes(path: Path, payload: bytes) -> Any:
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON at {path}: {error}") from error


def jsonl_bytes(path: Path, payload: bytes) -> list[dict[str, Any]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"invalid UTF-8 at {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
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


def identity_matches(record: Any, actual: dict[str, Any]) -> bool:
    if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
        return False
    return (
        _candidate(Path(str(record["path"]))).resolve(strict=False)
        == _candidate(Path(actual["path"])).resolve(strict=False)
        and record["bytes"] == actual["bytes"]
        and record["sha256"] == actual["sha256"]
    )


def validate_destinations(args: argparse.Namespace, inputs: list[Path]) -> None:
    expected_base = FORMAL_ROOT / args.dataset
    expected_output = expected_base / "test_predictions.jsonl"
    expected_manifest = expected_base / "conversion_manifest.json"
    lexical_output = _candidate(args.output).absolute()
    lexical_manifest = _candidate(args.manifest).absolute()
    if (
        lexical_output != _candidate(expected_output).absolute()
        or lexical_manifest != _candidate(expected_manifest).absolute()
    ):
        raise ValueError("SpERT formal conversion outputs must use the canonical dataset run root")
    destinations = [lexical_output, lexical_manifest]
    if destinations[0] == destinations[1]:
        raise ValueError("SpERT prediction output and conversion manifest must be distinct")
    input_paths = {_candidate(path).resolve(strict=False) for path in inputs}
    if any(path.resolve(strict=False) in input_paths for path in destinations):
        raise ValueError("SpERT conversion destination aliases a protected input")
    for destination in destinations:
        for component in (destination, destination.parent, *destination.parent.parents):
            if component.is_symlink():
                raise ValueError(f"symlinked SpERT conversion output root is forbidden: {destination}")


def stable_digest(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
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


def token_offsets(tokens: list[str]) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    position = 0
    for token in tokens:
        offsets.append((position, position + len(token)))
        position += len(token) + 1
    return offsets


def _prediction_lists(
    prediction: dict[str, Any], *, context: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if set(prediction) != {"tokens", "entities", "relations"}:
        raise ValueError(f"{context}: unexpected SpERT prediction schema")
    entities, relations = prediction["entities"], prediction["relations"]
    if not isinstance(entities, list) or not all(isinstance(row, dict) for row in entities):
        raise ValueError(f"{context}: predicted entities must be object rows")
    if not isinstance(relations, list) or not all(isinstance(row, dict) for row in relations):
        raise ValueError(f"{context}: predicted relations must be object rows")
    return entities, relations


def convert_document(
    dataset: str,
    source: dict[str, Any],
    prediction: dict[str, Any],
    job: dict[str, Any],
    types: dict[str, Any],
) -> dict[str, Any]:
    identifier = training.row_id(source)
    document_id = f"{dataset}_test_{identifier}"
    job_id = f"{document_id}_C1"
    context = f"{dataset}/test/{identifier}"
    if job.get("document_id") != document_id or job.get("job_id") != job_id:
        raise ValueError(f"{context}: public test job ID/order mismatch")

    source_tokens = source.get("tokens")
    if not isinstance(source_tokens, list) or not all(
        isinstance(token, str) for token in source_tokens
    ):
        raise ValueError(f"{context}: invalid source tokens")
    if prediction.get("tokens") != source_tokens:
        raise ValueError(f"{context}: prediction tokens do not match promoted test input")
    predicted_entities, predicted_relations = _prediction_lists(
        prediction, context=context
    )

    segments = job.get("segments")
    if not isinstance(segments, list) or len(segments) != 1:
        raise ValueError(f"{context}: expected exactly one public source segment")
    segment = segments[0]
    if not isinstance(segment, dict):
        raise ValueError(f"{context}: invalid public source segment")
    text = " ".join(source_tokens)
    if (
        segment.get("text") != text
        or segment.get("start") != 0
        or segment.get("end") != len(text)
        or not isinstance(segment.get("segment_id"), str)
    ):
        raise ValueError(f"{context}: promoted test text/offset mismatch")

    type_entities = set(types.get("entities", {}))
    type_relations = set(types.get("relations", {}))
    ontology = job.get("ontology")
    if not isinstance(ontology, dict):
        raise ValueError(f"{context}: public test ontology is missing")
    if set(ontology.get("entity_types", {})) != type_entities:
        raise ValueError(f"{context}: SpERT/public entity ontology mismatch")
    if set(ontology.get("relation_types", {})) != type_relations:
        raise ValueError(f"{context}: SpERT/public relation ontology mismatch")

    offsets = token_offsets(source_tokens)
    entities: list[dict[str, Any]] = []
    character_spans: list[tuple[int, int]] = []
    for position, entity in enumerate(predicted_entities, 1):
        if set(entity) != {"type", "start", "end"}:
            raise ValueError(f"{context}: malformed predicted entity {position}")
        start_token, end_token = entity["start"], entity["end"]
        if (
            not isinstance(start_token, int)
            or isinstance(start_token, bool)
            or not isinstance(end_token, int)
            or isinstance(end_token, bool)
            or not 0 <= start_token < end_token <= len(source_tokens)
        ):
            raise ValueError(f"{context}: invalid predicted entity span {position}")
        entity_type = entity["type"]
        if entity_type not in type_entities:
            raise ValueError(f"{context}: unknown predicted entity type {entity_type!r}")
        start = offsets[start_token][0]
        end = offsets[end_token - 1][1]
        entity_text = text[start:end]
        character_spans.append((start, end))
        entities.append(
            {
                "id": f"E{position}",
                "text": entity_text,
                "type": entity_type,
                "normalized_name": None,
                "evidence": {
                    "text": entity_text,
                    "segment_id": segment["segment_id"],
                    "page": segment.get("page"),
                    "start": start,
                    "end": end,
                },
                "confidence": 1.0,
                "review_status": "pending",
                "created_by": "spert-fresh-seed42",
            }
        )

    relations: list[dict[str, Any]] = []
    for position, relation in enumerate(predicted_relations, 1):
        if set(relation) != {"type", "head", "tail"}:
            raise ValueError(f"{context}: malformed predicted relation {position}")
        head, tail = relation["head"], relation["tail"]
        if (
            not isinstance(head, int)
            or isinstance(head, bool)
            or not isinstance(tail, int)
            or isinstance(tail, bool)
            or not 0 <= head < len(entities)
            or not 0 <= tail < len(entities)
        ):
            raise ValueError(f"{context}: invalid predicted relation endpoints {position}")
        relation_type = relation["type"]
        if relation_type not in type_relations:
            raise ValueError(f"{context}: unknown predicted relation type {relation_type!r}")
        evidence_start = min(character_spans[head][0], character_spans[tail][0])
        evidence_end = max(character_spans[head][1], character_spans[tail][1])
        relations.append(
            {
                "id": f"R{position}",
                "source_id": f"E{head + 1}",
                "type": relation_type,
                "target_id": f"E{tail + 1}",
                "claim_status": "explicit",
                "evidence": [
                    {
                        "text": text[evidence_start:evidence_end],
                        "segment_id": segment["segment_id"],
                        "page": segment.get("page"),
                        "start": evidence_start,
                        "end": evidence_end,
                    }
                ],
                "confidence": 1.0,
                "review_status": "pending",
                "created_by": "spert-fresh-seed42",
            }
        )

    return {
        "job_id": job_id,
        "annotation": {
            "schema_version": "0.1.0",
            "document_id": document_id,
            "language": job.get("language", "en"),
            "entities": entities,
            "relations": relations,
            "review": {
                "status": "unreviewed",
                "reviewers": [],
                "notes": (
                    "Fresh SpERT seed-42 formal-test prediction; discrete labels "
                    "do not export confidence scores."
                ),
            },
        },
    }


def build_conversion(
    dataset: str,
    source_rows: Any,
    prediction_rows: Any,
    jobs: list[dict[str, Any]],
    types: dict[str, Any],
) -> list[dict[str, Any]]:
    expected = EXPECTED_ROWS[dataset]
    if not isinstance(source_rows, list) or not all(isinstance(row, dict) for row in source_rows):
        raise ValueError("SpERT promoted test data must be an object array")
    if not isinstance(prediction_rows, list) or not all(
        isinstance(row, dict) for row in prediction_rows
    ):
        raise ValueError("SpERT raw predictions must be an object array")
    if not len(source_rows) == len(prediction_rows) == len(jobs) == expected:
        raise ValueError(
            f"{dataset}: exact row-count mismatch: test={len(source_rows)} "
            f"predictions={len(prediction_rows)} jobs={len(jobs)} expected={expected}"
        )
    expected_job_ids = [
        f"{dataset}_test_{training.row_id(source)}_C1" for source in source_rows
    ]
    actual_job_ids = [job.get("job_id") for job in jobs]
    if actual_job_ids != expected_job_ids or len(set(actual_job_ids)) != expected:
        raise ValueError(f"{dataset}: public test jobs do not exactly match native row order")
    return [
        convert_document(dataset, source, prediction, job, types)
        for source, prediction, job in zip(
            source_rows, prediction_rows, jobs, strict=True
        )
    ]


def convert(args: argparse.Namespace) -> dict[str, Any]:
    require_project_cwd()
    # Reject an unsafe/non-canonical publication target before the gate opens
    # or any test input is read.
    validate_destinations(
        args,
        [args.prepared_manifest, args.predictions, args.jobs, args.inference_manifest],
    )
    # This is intentionally first: validating the manifest recomputes the
    # canonical promotion before any test-data path is opened here.
    verified = prepared.validate_manifest(args.dataset, args.prepared_manifest)
    test_data = args.prepared_manifest.parent / "test.json"
    types_path = args.prepared_manifest.parent / "types.json"
    input_paths = [
        args.prepared_manifest,
        test_data,
        types_path,
        args.predictions,
        args.jobs,
        args.inference_manifest,
        Path(__file__).resolve(),
    ]
    validate_destinations(args, input_paths)
    snapshots = {path: read_bytes(path) for path in input_paths}
    identities = {
        path: payload_identity(path, payload) for path, payload in snapshots.items()
    }
    if not identity_matches(verified.get("manifest"), identities[args.prepared_manifest]):
        raise ValueError("prepared SpERT manifest changed after promotion verification")
    prepared_manifest = json_bytes(
        args.prepared_manifest, snapshots[args.prepared_manifest]
    )
    verified_outputs = verified["outputs"]
    if not identity_matches(verified_outputs["test.json"], identities[test_data]):
        raise ValueError("prepared SpERT test data changed after promotion verification")
    if not identity_matches(verified_outputs["types.json"], identities[types_path]):
        raise ValueError("prepared SpERT test types changed after promotion verification")

    prepared_jobs = prepared_manifest.get("inputs", {}).get("test_jobs")
    if not identity_matches(prepared_jobs, identities[args.jobs]):
        raise ValueError("test jobs do not match the hash-bound SpERT preparation input")
    inference = json_bytes(args.inference_manifest, snapshots[args.inference_manifest])
    expected_inference = {
        "status": "captured_exact_new_eval_output",
        "dataset": args.dataset,
        "split": "test",
        "formal_test_read": True,
        "seed": 42,
        "prepared_manifest": identities[args.prepared_manifest],
    }
    if any(inference.get(key) != value for key, value in expected_inference.items()):
        raise ValueError("raw predictions lack a valid frozen SpERT inference attestation")
    if not identity_matches(
        inference.get("captured", {}).get("predictions"), identities[args.predictions]
    ):
        raise ValueError("raw predictions differ from the attested SpERT inference output")

    source_rows = json_bytes(test_data, snapshots[test_data])
    prediction_rows = json_bytes(args.predictions, snapshots[args.predictions])
    jobs = jsonl_bytes(args.jobs, snapshots[args.jobs])
    types = json_bytes(types_path, snapshots[types_path])
    prepared.release.validate_source_jobs(args.jobs, jobs, args.dataset, "test")
    converted = build_conversion(args.dataset, source_rows, prediction_rows, jobs, types)

    # Re-run the entire release/preparation verification before publication.
    # Conversion uses the immutable snapshots above, while this catches source
    # or gate drift that occurred during conversion.
    final_verified = prepared.validate_manifest(args.dataset, args.prepared_manifest)
    if final_verified != verified:
        raise ValueError("canonical SpERT preparation changed during conversion")
    atomic_write_jsonl(args.output, converted)

    inputs = {
        "prepared_manifest": identities[args.prepared_manifest],
        "test_data": identities[test_data],
        "types": identities[types_path],
        "raw_predictions": identities[args.predictions],
        "test_jobs": identities[args.jobs],
        "inference_manifest": identities[args.inference_manifest],
        "converter": identities[Path(__file__).resolve()],
    }
    output_identity = identity(args.output)
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "dataset": args.dataset,
        "split": "test",
        "formal_test_read": True,
        "seed": 42,
        "rows": len(converted),
        "entities": sum(len(row["annotation"]["entities"]) for row in converted),
        "relations": sum(len(row["annotation"]["relations"]) for row in converted),
        "promotion_attestation_sha256": prepared_manifest[
            "promotion_attestation_sha256"
        ],
        "prepared_fingerprint": prepared_manifest["fingerprint"],
        "orig_id_sha256": prepared_manifest["orig_id_sha256"],
        "job_id_sha256": stable_digest([row["job_id"] for row in converted]),
        "inputs": inputs,
        "output": output_identity,
    }
    for value in (summary["entities"], summary["relations"]):
        if not isinstance(value, int) or value < 0 or not math.isfinite(float(value)):
            raise ValueError("invalid aggregate prediction count")
    atomic_write_json(args.manifest, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--inference-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    result = convert(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
