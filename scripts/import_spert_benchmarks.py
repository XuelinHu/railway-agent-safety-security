#!/usr/bin/env python3
"""Convert public SpERT-format IE datasets into evidence-grounded artifacts.

The converter intentionally derives the benchmark ontology and graph only from
the selected training rows.  Validation and test labels are never used to make
KG hints or endpoint-type constraints.  This makes the public benchmarks a
clean transfer test for the evidence-gated graph-prior method.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


SPLIT_NAMES = {"train": "train", "validation": "dev", "test": "test"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_subset(rows: list[dict[str, Any]], limit: int, seed: str) -> list[dict[str, Any]]:
    if not limit or limit >= len(rows):
        return rows
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{seed}:{row['document_id']}".encode()).hexdigest(),
    )[:limit]


def text_and_offsets(tokens: list[str]) -> tuple[str, list[tuple[int, int]]]:
    text = " ".join(tokens)
    offsets: list[tuple[int, int]] = []
    position = 0
    for token in tokens:
        offsets.append((position, position + len(token)))
        position += len(token) + 1
    return text, offsets


def span_text(tokens: list[str], offsets: list[tuple[int, int]], start: int, end: int) -> tuple[str, int, int]:
    if not 0 <= start < end <= len(tokens):
        raise ValueError(f"Invalid token span [{start}, {end}) for {len(tokens)} tokens")
    char_start, char_end = offsets[start][0], offsets[end - 1][1]
    return " ".join(tokens[start:end]), char_start, char_end


def import_row(dataset: str, split: str, raw: dict[str, Any], index: int) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    tokens = raw["tokens"]
    text, offsets = text_and_offsets(tokens)
    raw_id = str(raw.get("orig_id", index))
    document_id = f"{dataset}_{split}_{raw_id}"
    segment = {"segment_id": "S1", "segment_type": "sentence", "page": None, "start": 0, "end": len(text), "text": text}
    entities: list[dict[str, Any]] = []
    for entity_index, raw_entity in enumerate(raw.get("entities", []), start=1):
        mention, start, end = span_text(tokens, offsets, raw_entity["start"], raw_entity["end"])
        entities.append(
            {
                "id": f"E{entity_index}", "text": mention, "type": raw_entity["type"],
                "normalized_name": None,
                "evidence": {"text": mention, "segment_id": "S1", "page": None, "start": start, "end": end},
                "confidence": 1.0, "review_status": "accepted", "created_by": "public_benchmark_import",
            }
        )
    relations: list[dict[str, Any]] = []
    for relation_index, raw_relation in enumerate(raw.get("relations", []), start=1):
        head, tail = raw_relation["head"], raw_relation["tail"]
        if not 0 <= head < len(entities) or not 0 <= tail < len(entities):
            raise ValueError(f"{document_id}: invalid relation endpoint")
        relations.append(
            {
                "id": f"R{relation_index}", "source_id": entities[head]["id"], "type": raw_relation["type"], "target_id": entities[tail]["id"],
                "claim_status": "explicit", "evidence": [{"text": text, "segment_id": "S1", "page": None, "start": 0, "end": len(text)}],
                "confidence": 1.0, "review_status": "accepted", "created_by": "public_benchmark_import",
            }
        )
    annotation = {"schema_version": "0.1.0", "document_id": document_id, "language": "en", "entities": entities, "relations": relations,
                  "review": {"status": "unreviewed", "reviewers": [], "notes": "Imported public benchmark gold label; not locally re-annotated."}}
    job = {"job_id": f"{document_id}_C1", "document_id": document_id, "language": "en", "category": dataset, "source_path": f"public:{dataset}:{split}:{raw_id}",
           "chunk_number": 1, "chunk_count": 1, "teacher_model": "benchmark_model", "prompt_version": "public-benchmark-v1",
           "system_instruction": "Extract only exact source spans and relations supported by the supplied text. For each retained relation, provide one contiguous evidence quote containing both endpoints. Omit unsupported predictions.",
           "segments": [segment], "status": "benchmark"}
    mentions = [
        {"mention_id": f"{document_id}_C1:{entity['id']}", "concept_id": f"{dataset}:{entity['type']}:{entity['text'].casefold()}", "document_id": document_id,
         "job_id": job["job_id"], "split": split, "source_id": entity["id"], "text": entity["text"], "type": entity["type"],
         "normalized_name": None, "confidence": 1.0, "evidence": entity["evidence"], "provenance": {"source_path": job["source_path"], "teacher_model": "public_benchmark_import"}}
        for entity in entities
    ]
    return job, annotation, mentions, [{"document_id": document_id, "source_group": dataset, "category": dataset, "relative_path": job["source_path"], "split": split}]


def train_ontology(dataset: str, annotations: list[dict[str, Any]]) -> dict[str, Any]:
    entity_types = sorted({entity["type"] for annotation in annotations for entity in annotation["entities"]})
    relation_types = sorted({relation["type"] for annotation in annotations for relation in annotation["relations"]})
    signatures: dict[str, dict[str, set[str]]] = defaultdict(lambda: {"source": set(), "target": set()})
    for annotation in annotations:
        entities = {entity["id"]: entity for entity in annotation["entities"]}
        for relation in annotation["relations"]:
            signatures[relation["type"]]["source"].add(entities[relation["source_id"]]["type"])
            signatures[relation["type"]]["target"].add(entities[relation["target_id"]]["type"])
    return {
        "version": "public-benchmark-v1", "annotation_schema_version": "0.1.0", "name": f"{dataset}_training_only_ontology",
        "scope": "Automatically derived from the selected training split only.",
        "entity_types": {name: {"description": f"Published {dataset} entity type: {name}."} for name in entity_types},
        "relation_types": {name: {"description": f"Published {dataset} relation type: {name}."} for name in relation_types},
        "claim_statuses": {"explicit": "Directly annotated in the public benchmark."},
        "allowed_relation_signatures": {name: {"source": sorted(signatures[name]["source"]), "target": sorted(signatures[name]["target"])} for name in relation_types},
    }


def build_train_graph(annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges = []
    for annotation in annotations:
        entities = {entity["id"]: entity for entity in annotation["entities"]}
        for relation in annotation["relations"]:
            source, target = entities[relation["source_id"]], entities[relation["target_id"]]
            edges.append({"document_id": annotation["document_id"], "source_text": source["text"], "source_type": source["type"],
                          "relation_type": relation["type"], "target_text": target["text"], "target_type": target["type"],
                          "evidence": relation["evidence"][0], "provenance_split": "train"})
    return edges


def run(args: argparse.Namespace) -> int:
    dataset_dir = args.input_root / args.dataset
    all_rows: dict[str, list[dict[str, Any]]] = {}
    for split, upstream in SPLIT_NAMES.items():
        path = dataset_dir / f"{args.dataset}_{upstream}.json"
        if path.exists():
            all_rows[split] = read_json(path)
        elif split == "validation" and args.dataset == "ade":
            # ADE publishes train/test rather than a dev split.  Derive a
            # deterministic 10% development slice from train, then remove it
            # from the training pool so no document is reused across splits.
            published_train = read_json(dataset_dir / "ade_train.json")
            keyed = sorted(
                published_train,
                key=lambda row: hashlib.sha256(f"{args.seed}:{row.get('orig_id', '')}".encode()).hexdigest(),
            )
            dev_size = max(1, len(keyed) // 10)
            all_rows[split] = keyed[:dev_size]
            all_rows["train"] = keyed[dev_size:]
        else:
            raise FileNotFoundError(f"Expected published file {path}")
    limits = {"train": args.train_limit, "validation": args.validation_limit, "test": args.test_limit}
    selected: dict[str, list[dict[str, Any]]] = {}
    for split, rows in all_rows.items():
        converted = [import_row(args.dataset, split, row, index) for index, row in enumerate(rows)]
        selected[split] = stable_subset(
            [{"document_id": item[0]["document_id"], "value": item} for item in converted], limits[split], args.seed
        )
    jobs: list[dict[str, Any]] = []
    annotations_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    mentions: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for split in SPLIT_NAMES:
        for selected_row in selected[split]:
            job, annotation, row_mentions, row_manifest = selected_row["value"]
            jobs.append(job)
            annotations_by_split[split].append(annotation)
            mentions.extend(row_mentions)
            manifest.extend(row_manifest)
    ontology = train_ontology(args.dataset, annotations_by_split["train"])
    # The existing QLoRA prompt consumes the ontology from each job.  Embed the
    # training-derived copy so the same trainer/inference path is reusable.
    for job in jobs:
        job["ontology"] = ontology
    root = args.output_root / args.dataset
    write_jsonl(root / "jobs.jsonl", jobs)
    gold_rows = [{"job_id": f"{item['document_id']}_C1", "annotation": item} for split in SPLIT_NAMES for item in annotations_by_split[split]]
    # Training consumes raw annotation objects; job IDs live in the parallel
    # index file. Keep a wrapped copy only for convenient prediction auditing.
    write_jsonl(root / "gold.jsonl", gold_rows)
    for split in SPLIT_NAMES:
        split_ids = {item["document_id"] for item in annotations_by_split[split]}
        write_jsonl(root / f"{split}_gold.jsonl", [row["annotation"] for row in gold_rows if row["annotation"]["document_id"] in split_ids])
        split_rows = [row for row in gold_rows if row["annotation"]["document_id"] in split_ids]
        write_jsonl(root / f"{split}_index.jsonl", [{"job_id": row["job_id"], "record_index": index} for index, row in enumerate(split_rows)])
    write_jsonl(root / "split_manifest.jsonl", manifest)
    write_jsonl(root / "knowledge_graph" / "mentions.jsonl", mentions)
    write_jsonl(root / "knowledge_graph" / "training_edges.jsonl", build_train_graph(annotations_by_split["train"]))
    (root / "ontology.yaml").write_text(yaml.safe_dump(ontology, sort_keys=False, allow_unicode=True), encoding="utf-8")
    summary = {"dataset": args.dataset, "rows": {split: len(annotations_by_split[split]) for split in SPLIT_NAMES},
               "train_entity_types": sorted(ontology["entity_types"]), "train_relation_types": sorted(ontology["relation_types"]), "output": str(root)}
    (root / "README.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["conll04", "scierc", "ade"], required=True)
    parser.add_argument("--input-root", type=Path, default=Path("data/external/spert"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/public_benchmarks"))
    parser.add_argument("--train-limit", type=int, default=0, help="0 keeps the full published split.")
    parser.add_argument("--validation-limit", type=int, default=0, help="0 keeps the full published split.")
    parser.add_argument("--test-limit", type=int, default=0, help="0 keeps the full published split.")
    parser.add_argument("--seed", default="public-benchmark-v1")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
