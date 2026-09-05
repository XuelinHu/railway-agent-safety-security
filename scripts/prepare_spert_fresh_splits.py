#!/usr/bin/env python3
"""Materialize train/validation-only SpERT inputs for the public benchmarks.

The script deliberately has no test-split input. CoNLL04 and SciERC retain
their published train/dev split. ADE is split with the same deterministic rule
and seed as ``import_spert_benchmarks.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DATASETS = ("conll04", "scierc", "ade")
PROTOCOL = "spert-fresh-train-validation-v1"


@dataclass(frozen=True)
class SourceSpec:
    train_name: str
    validation_name: str | None
    types_name: str


SOURCES = {
    "conll04": SourceSpec("conll04_train.json", "conll04_dev.json", "conll04_types.json"),
    "scierc": SourceSpec("scierc_train.json", "scierc_dev.json", "scierc_types.json"),
    "ade": SourceSpec("ade_train.json", None, "ade_types.json"),
}


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def row_id(row: dict[str, Any]) -> str:
    if "orig_id" not in row:
        raise ValueError("Every SpERT row must have orig_id for split auditing")
    return str(row["orig_id"])


def validate_rows(dataset: str, split: str, rows: Any, types: dict[str, Any]) -> None:
    if not isinstance(rows, list):
        raise TypeError(f"{dataset}/{split}: expected a JSON array")
    entity_types = set(types.get("entities", {}))
    relation_types = set(types.get("relations", {}))
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        prefix = f"{dataset}/{split}[{index}]"
        identifier = row_id(row)
        if identifier in seen_ids:
            raise ValueError(f"{prefix}: duplicate orig_id {identifier}")
        seen_ids.add(identifier)
        tokens = row.get("tokens")
        entities = row.get("entities")
        relations = row.get("relations")
        if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
            raise ValueError(f"{prefix}: tokens must be a list of strings")
        if not isinstance(entities, list) or not isinstance(relations, list):
            raise ValueError(f"{prefix}: entities and relations must be lists")
        for entity_index, entity in enumerate(entities):
            start, end = entity.get("start"), entity.get("end")
            if entity.get("type") not in entity_types:
                raise ValueError(f"{prefix}: unknown entity type {entity.get('type')!r}")
            if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(tokens):
                raise ValueError(f"{prefix}: invalid entity span at index {entity_index}")
        for relation_index, relation in enumerate(relations):
            head, tail = relation.get("head"), relation.get("tail")
            if relation.get("type") not in relation_types:
                raise ValueError(f"{prefix}: unknown relation type {relation.get('type')!r}")
            if not isinstance(head, int) or not isinstance(tail, int):
                raise ValueError(f"{prefix}: relation endpoints must be integers at index {relation_index}")
            if not 0 <= head < len(entities) or not 0 <= tail < len(entities):
                raise ValueError(f"{prefix}: invalid relation endpoint at index {relation_index}")


def split_ade(rows: list[dict[str, Any]], seed: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    keyed = sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{seed}:{row_id(row)}".encode("utf-8")).hexdigest(),
    )
    validation_size = max(1, len(keyed) // 10)
    return keyed[validation_size:], keyed[:validation_size]


def native_annotation_fingerprint(row: dict[str, Any]) -> dict[str, Any]:
    tokens = row["tokens"]
    offsets: list[tuple[int, int]] = []
    position = 0
    for token in tokens:
        offsets.append((position, position + len(token)))
        position += len(token) + 1
    entities = []
    for entity in row["entities"]:
        start_token, end_token = entity["start"], entity["end"]
        entities.append(
            {
                "type": entity["type"],
                "text": " ".join(tokens[start_token:end_token]),
                "start": offsets[start_token][0],
                "end": offsets[end_token - 1][1],
            }
        )
    relations = [
        {"type": relation["type"], "head": relation["head"], "tail": relation["tail"]}
        for relation in row["relations"]
    ]
    return {"text": " ".join(tokens), "entities": entities, "relations": relations}


def reference_annotation_fingerprint(annotation: dict[str, Any]) -> dict[str, Any]:
    entities = annotation.get("entities", [])
    entity_index = {entity["id"]: index for index, entity in enumerate(entities)}
    normalized_entities = [
        {
            "type": entity["type"],
            "text": entity["text"],
            "start": entity["evidence"]["start"],
            "end": entity["evidence"]["end"],
        }
        for entity in entities
    ]
    relations = [
        {
            "type": relation["type"],
            "head": entity_index[relation["source_id"]],
            "tail": entity_index[relation["target_id"]],
        }
        for relation in annotation.get("relations", [])
    ]
    text = None
    # Gold annotations store entity-local evidence. Reconstruct sentence text
    # from relation evidence when available; entity/relation fingerprints still
    # fully audit rows without relying on that convenience field.
    relation_evidence = annotation.get("relations", [])
    if relation_evidence:
        text = relation_evidence[0].get("evidence", [{}])[0].get("text")
    return {"text": text, "entities": normalized_entities, "relations": relations}


def verify_reference(
    dataset: str,
    split: str,
    rows: list[dict[str, Any]],
    reference_root: Path,
) -> dict[str, Any]:
    path = reference_root / dataset / f"{split}_gold.jsonl"
    annotations = read_jsonl(path)
    prefix = f"{dataset}_{split}_"
    by_id: dict[str, dict[str, Any]] = {}
    for annotation in annotations:
        document_id = annotation.get("document_id", "")
        if not document_id.startswith(prefix):
            raise ValueError(f"{path}: unexpected document_id {document_id!r}")
        by_id[document_id[len(prefix) :]] = annotation
    source_ids = {row_id(row) for row in rows}
    if source_ids != set(by_id):
        missing = sorted(source_ids - set(by_id))[:5]
        extra = sorted(set(by_id) - source_ids)[:5]
        raise ValueError(f"{dataset}/{split}: reference ID mismatch; missing={missing}, extra={extra}")
    for row in rows:
        identifier = row_id(row)
        native = native_annotation_fingerprint(row)
        reference = reference_annotation_fingerprint(by_id[identifier])
        if native["entities"] != reference["entities"] or native["relations"] != reference["relations"]:
            raise ValueError(f"{dataset}/{split}/{identifier}: reference annotation mismatch")
        if reference["text"] is not None and native["text"] != reference["text"]:
            raise ValueError(f"{dataset}/{split}/{identifier}: reference text mismatch")
    return {"path": str(path), "sha256": sha256_file(path), "rows": len(annotations), "matched": True}


def prepare_dataset(args: argparse.Namespace, dataset: str) -> dict[str, Any]:
    spec = SOURCES[dataset]
    source_dir = args.input_root / dataset
    train_source = source_dir / spec.train_name
    types_source = source_dir / spec.types_name
    source_train = read_json(train_source)
    types = read_json(types_source)
    sources = {
        "train": {"path": str(train_source), "sha256": sha256_file(train_source)},
        "types": {"path": str(types_source), "sha256": sha256_file(types_source)},
    }
    if spec.validation_name is None:
        train_rows, validation_rows = split_ade(source_train, args.ade_seed)
    else:
        validation_source = source_dir / spec.validation_name
        validation_rows = read_json(validation_source)
        train_rows = source_train
        sources["validation"] = {"path": str(validation_source), "sha256": sha256_file(validation_source)}

    validate_rows(dataset, "train", train_rows, types)
    validate_rows(dataset, "validation", validation_rows, types)
    overlap = {row_id(row) for row in train_rows} & {row_id(row) for row in validation_rows}
    if overlap:
        raise ValueError(f"{dataset}: train/validation overlap: {sorted(overlap)[:5]}")

    references = {}
    if args.reference_root is not None:
        references = {
            split: verify_reference(dataset, split, rows, args.reference_root)
            for split, rows in (("train", train_rows), ("validation", validation_rows))
        }

    target = args.output_root / dataset
    payloads = {
        "train.json": canonical_bytes(train_rows),
        "validation.json": canonical_bytes(validation_rows),
        "types.json": canonical_bytes(types),
    }
    for name, content in payloads.items():
        atomic_write(target / name, content)

    summary = {
        "dataset": dataset,
        "protocol": PROTOCOL,
        "ade_seed": args.ade_seed if dataset == "ade" else None,
        "test_split_access": "forbidden-and-not-materialized",
        "sources": sources,
        "references": references,
        "outputs": {
            name: {"path": str(target / name), "sha256": sha256_bytes(content)}
            for name, content in payloads.items()
        },
        "rows": {"train": len(train_rows), "validation": len(validation_rows)},
        "orig_id_sha256": {
            "train": sha256_bytes(canonical_bytes([row_id(row) for row in train_rows])),
            "validation": sha256_bytes(canonical_bytes([row_id(row) for row in validation_rows])),
        },
    }
    atomic_write(target / "manifest.json", json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("data/external/spert"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/spert_fresh_train_v1"))
    parser.add_argument("--reference-root", type=Path, default=Path("data/processed/public_benchmarks_full"))
    parser.add_argument("--skip-reference-check", action="store_true")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--ade-seed", default="public-full-42")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.skip_reference_check:
        args.reference_root = None
    summaries = [prepare_dataset(args, dataset) for dataset in args.datasets]
    status = {
        "status": "ready",
        "protocol": PROTOCOL,
        "test_split_access": "forbidden-and-not-materialized",
        "datasets": {summary["dataset"]: summary for summary in summaries},
    }
    atomic_write(args.output_root / "status.json", json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    print(json.dumps({"status": status["status"], "rows": {item["dataset"]: item["rows"] for item in summaries}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
