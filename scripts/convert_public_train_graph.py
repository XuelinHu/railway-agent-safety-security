#!/usr/bin/env python3
"""Convert public benchmark graph exports into a physical train-only KG.

The public importer stores mentions for every benchmark split in one file, while
``training_edges.jsonl`` contains only training relations.  KG-v2/HRGE expects
the provenance-bearing ``concepts.jsonl`` and ``relations.jsonl`` schema used by
the paper pipeline.  This converter filters mentions before aggregation and
fails closed if an edge, endpoint, or split assignment is not demonstrably from
the training split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ALLOWED_SPLITS = {"train", "validation", "test"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary_name = stream.name
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temporary_name, path)
    finally:
        if temporary_name and Path(temporary_name).exists():
            Path(temporary_name).unlink()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary_name = stream.name
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_name, path)
    finally:
        if temporary_name and Path(temporary_name).exists():
            Path(temporary_name).unlink()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def required_text(row: dict[str, Any], field: str, label: str) -> str:
    value = str(row.get(field, "")).strip()
    if not value:
        raise ValueError(f"{label} is missing {field}")
    return value


def split_by_document(manifest: list[dict[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for index, row in enumerate(manifest):
        document_id = required_text(row, "document_id", f"manifest row {index}")
        split = required_text(row, "split", f"manifest row {index}")
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"manifest row {index} has unsupported split {split!r}")
        previous = result.setdefault(document_id, split)
        if previous != split:
            raise ValueError(f"document {document_id} occurs in multiple splits")
    if not any(split == "train" for split in result.values()):
        raise ValueError("manifest contains no training documents")
    return result


def concept_id_for(mention: dict[str, Any], language: str) -> str:
    explicit = str(mention.get("concept_id", "")).strip()
    if explicit:
        return explicit
    key = "|".join(
        (
            language,
            required_text(mention, "type", "mention"),
            normalize_text(
                mention.get("canonical_name")
                or mention.get("normalized_name")
                or mention.get("text")
            ),
        )
    )
    return "public_concept_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def build_train_graph(
    dataset: str,
    mentions: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
    default_language: str = "en",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    assignments = split_by_document(manifest)
    train_documents = {document_id for document_id, split in assignments.items() if split == "train"}
    excluded_splits: Counter[str] = Counter()
    train_mentions: list[dict[str, Any]] = []
    concepts: dict[str, dict[str, Any]] = {}
    concept_keys: dict[str, tuple[str, str, str]] = {}
    endpoint_mentions: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)

    for index, source in enumerate(mentions):
        label = f"mention row {index}"
        document_id = required_text(source, "document_id", label)
        split = required_text(source, "split", label)
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"{label} has unsupported split {split!r}")
        assigned_split = assignments.get(document_id)
        if assigned_split is None:
            raise ValueError(f"{label} references a document absent from the manifest: {document_id}")
        if assigned_split != split:
            raise ValueError(
                f"{label} split mismatch for {document_id}: row={split}, manifest={assigned_split}"
            )
        if split != "train":
            excluded_splits[split] += 1
            continue

        entity_type = required_text(source, "type", label)
        surface = required_text(source, "text", label)
        language = str(source.get("language") or default_language).strip() or default_language
        canonical_name = normalize_text(
            source.get("canonical_name") or source.get("normalized_name") or surface
        )
        if not canonical_name:
            raise ValueError(f"{label} has no usable canonical name")
        concept_id = concept_id_for(source, language)
        concept_key = (language, entity_type, canonical_name)
        previous_key = concept_keys.setdefault(concept_id, concept_key)
        if previous_key != concept_key:
            raise ValueError(
                f"concept_id {concept_id!r} maps to inconsistent concepts: "
                f"{previous_key!r} and {concept_key!r}"
            )

        mention = {
            **source,
            "concept_id": concept_id,
            "split": "train",
            "language": language,
            "canonical_name": canonical_name,
        }
        train_mentions.append(mention)
        endpoint_mentions[(document_id, entity_type, normalize_text(surface))].append(mention)
        concept = concepts.setdefault(
            concept_id,
            {
                "concept_id": concept_id,
                "canonical_name": canonical_name,
                "language": language,
                "type": entity_type,
                "mention_count": 0,
                "source_documents": set(),
            },
        )
        concept["mention_count"] += 1
        concept["source_documents"].add(document_id)

    if not train_mentions:
        raise ValueError("no training mentions remain after split filtering")

    ambiguous_endpoint_rows = 0
    relation_rows: list[dict[str, Any]] = []
    for index, edge in enumerate(edges):
        label = f"training edge row {index}"
        provenance_split = required_text(edge, "provenance_split", label)
        if provenance_split != "train":
            raise ValueError(f"{label} is not train-only: provenance_split={provenance_split!r}")
        document_id = required_text(edge, "document_id", label)
        if assignments.get(document_id) != "train" or document_id not in train_documents:
            raise ValueError(f"{label} references non-training document {document_id!r}")
        source_type = required_text(edge, "source_type", label)
        target_type = required_text(edge, "target_type", label)
        source_text = required_text(edge, "source_text", label)
        target_text = required_text(edge, "target_text", label)
        relation_type = required_text(edge, "relation_type", label)
        source_matches = endpoint_mentions.get(
            (document_id, source_type, normalize_text(source_text)), []
        )
        target_matches = endpoint_mentions.get(
            (document_id, target_type, normalize_text(target_text)), []
        )
        if not source_matches or not target_matches:
            raise ValueError(
                f"{label} cannot resolve endpoints in train mentions: "
                f"source_matches={len(source_matches)}, target_matches={len(target_matches)}"
            )
        source_concepts = {row["concept_id"] for row in source_matches}
        target_concepts = {row["concept_id"] for row in target_matches}
        if len(source_concepts) != 1 or len(target_concepts) != 1:
            raise ValueError(f"{label} has ambiguous endpoint concept identities")
        if len(source_matches) > 1 or len(target_matches) > 1:
            ambiguous_endpoint_rows += 1

        job_ids = {
            str(row.get("job_id", "")).strip()
            for row in (*source_matches, *target_matches)
            if str(row.get("job_id", "")).strip()
        }
        if len(job_ids) != 1:
            raise ValueError(f"{label} does not resolve to exactly one source job: {sorted(job_ids)}")
        evidence = edge.get("evidence")
        if not isinstance(evidence, dict) or not normalize_text(evidence.get("text")):
            raise ValueError(f"{label} has no usable local evidence quote")
        identity = json.dumps(
            {
                "dataset": dataset,
                "index": index,
                "document_id": document_id,
                "source": source_text,
                "source_type": source_type,
                "relation": relation_type,
                "target": target_text,
                "target_type": target_type,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        relation_rows.append(
            {
                "relation_id": "public_relation_"
                + hashlib.sha1(identity.encode("utf-8")).hexdigest()[:20],
                "source_concept_id": next(iter(source_concepts)),
                "target_concept_id": next(iter(target_concepts)),
                "source_mention_id": sorted(
                    str(row.get("mention_id", "")) for row in source_matches
                )[0],
                "target_mention_id": sorted(
                    str(row.get("mention_id", "")) for row in target_matches
                )[0],
                "document_id": document_id,
                "job_id": next(iter(job_ids)),
                "split": "train",
                "type": relation_type,
                "claim_status": "explicit",
                "confidence": 1.0,
                "evidence": [evidence],
                "provenance": {
                    "source_path": f"public:{dataset}:train:{document_id}",
                    "source_split": "train",
                    "converter": "convert_public_train_graph.py",
                },
            }
        )

    concept_rows = []
    for concept_id in sorted(concepts):
        concept = concepts[concept_id]
        concept_rows.append(
            {
                **concept,
                "source_documents": sorted(concept["source_documents"]),
            }
        )
    train_mentions.sort(key=lambda row: str(row.get("mention_id", "")))
    relation_rows.sort(key=lambda row: row["relation_id"])
    relation_ids = [row["relation_id"] for row in relation_rows]
    if len(relation_ids) != len(set(relation_ids)):
        raise ValueError("deterministic relation IDs are not unique")

    graph_documents = {
        row["document_id"] for row in train_mentions
    } | {row["document_id"] for row in relation_rows}
    non_train_graph_documents = sorted(graph_documents - train_documents)
    if non_train_graph_documents:
        raise ValueError(f"converted graph contains non-training documents: {non_train_graph_documents}")
    audit = {
        "version": "public-train-graph-v1",
        "dataset": dataset,
        "policy": "physical-train-only-after-manifest-cross-check",
        "input_mentions": len(mentions),
        "input_edges": len(edges),
        "excluded_mentions_by_split": dict(sorted(excluded_splits.items())),
        "train_documents_in_manifest": len(train_documents),
        "train_documents_with_graph_items": len(graph_documents),
        "concepts": len(concept_rows),
        "mentions": len(train_mentions),
        "relations": len(relation_rows),
        "ambiguous_endpoint_rows_same_concept": ambiguous_endpoint_rows,
        "non_train_graph_documents": non_train_graph_documents,
        "all_output_rows_train_only": True,
    }
    return concept_rows, train_mentions, relation_rows, audit


def convert_paths(
    dataset: str,
    mentions_path: Path,
    edges_path: Path,
    manifest_path: Path,
    output_dir: Path,
    default_language: str = "en",
) -> dict[str, Any]:
    concepts, mentions, relations, audit = build_train_graph(
        dataset,
        load_jsonl(mentions_path),
        load_jsonl(edges_path),
        load_jsonl(manifest_path),
        default_language,
    )
    output_paths = {
        "concepts": output_dir / "concepts.jsonl",
        "mentions": output_dir / "mentions.jsonl",
        "relations": output_dir / "relations.jsonl",
    }
    write_jsonl_atomic(output_paths["concepts"], concepts)
    write_jsonl_atomic(output_paths["mentions"], mentions)
    write_jsonl_atomic(output_paths["relations"], relations)
    audit.update(
        {
            "inputs": {
                "mentions": {"path": str(mentions_path), "sha256": sha256(mentions_path)},
                "training_edges": {"path": str(edges_path), "sha256": sha256(edges_path)},
                "split_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
            },
            "outputs": {
                name: {"path": str(path), "sha256": sha256(path)}
                for name, path in output_paths.items()
            },
        }
    )
    write_json_atomic(output_dir / "conversion_audit.json", audit)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mentions", type=Path, required=True)
    parser.add_argument("--training-edges", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--default-language", default="en")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = convert_paths(
        args.dataset,
        args.mentions,
        args.training_edges,
        args.manifest,
        args.output_dir,
        args.default_language,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
