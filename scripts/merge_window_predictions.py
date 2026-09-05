#!/usr/bin/env python3
"""Merge compact predictions from overlapping document windows."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def merge_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for row in rows:
        parent = row.get("parent_job_id", row["job_id"].split("_W", 1)[0])
        grouped.setdefault(parent, []).append(row)

    merged: list[dict[str, Any]] = []
    for parent_job_id, group in grouped.items():
        group.sort(key=lambda row: row.get("window_index", 1))
        first = group[0]["annotation"]
        entities: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
        entity_ids: dict[tuple[str, str], str] = {}
        for row in group:
            for entity in row["annotation"].get("entities", []):
                key = (entity.get("text", ""), entity.get("type", ""))
                if key not in entities:
                    entity_id = f"E{len(entities) + 1}"
                    entity_ids[key] = entity_id
                    entities[key] = {**entity, "id": entity_id, "text": key[0], "type": key[1]}

        relations: OrderedDict[tuple[str, str, str, str], dict[str, Any]] = OrderedDict()
        for row in group:
            local_entities = {
                entity.get("id"): (entity.get("text", ""), entity.get("type", ""))
                for entity in row["annotation"].get("entities", [])
            }
            for relation in row["annotation"].get("relations", []):
                source_key = local_entities.get(relation.get("source_id"))
                target_key = local_entities.get(relation.get("target_id"))
                if not source_key or not target_key or source_key not in entity_ids or target_key not in entity_ids:
                    continue
                claim_status = relation.get("claim_status", "uncertain")
                key = (source_key[0], relation.get("type", ""), target_key[0], claim_status)
                if key not in relations:
                    relations[key] = {
                        **relation,
                        "id": f"R{len(relations) + 1}",
                        "source_id": entity_ids[source_key],
                        "type": relation.get("type", ""),
                        "target_id": entity_ids[target_key],
                        "claim_status": claim_status,
                    }
        merged.append(
            {
                "job_id": parent_job_id,
                "annotation": {
                    "schema_version": first.get("schema_version", "0.1.0"),
                    "document_id": first["document_id"],
                    "language": first["language"],
                    "entities": list(entities.values()),
                    "relations": list(relations.values()),
                },
            }
        )
    return merged


def main(args: argparse.Namespace) -> int:
    rows = merge_rows(load_jsonl(args.predictions))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"window_rows": len(load_jsonl(args.predictions)), "merged_jobs": len(rows), "output": str(args.output)}, ensure_ascii=False))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
