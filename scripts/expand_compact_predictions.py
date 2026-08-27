#!/usr/bin/env python3
"""Restore evidence spans around compact model predictions using source segments."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def locate(value: str, segments: list[dict[str, Any]], used: dict[str, int]) -> dict[str, Any]:
    tokens = re.split(r"\s+", value.strip())
    token_patterns = [re.escape(token).replace(r"\-", r"-\s*") for token in tokens]
    pattern = re.compile(r"\s+".join(token_patterns))
    matches = []
    for segment in segments:
        match = pattern.search(segment["text"])
        while match:
            matches.append((segment, match))
            match = pattern.search(segment["text"], match.end())
    if not matches:
        raise ValueError(f"cannot locate compact span: {value[:80]!r}")
    occurrence = used.get(value, 0)
    segment, match = matches[min(occurrence, len(matches) - 1)]
    used[value] = occurrence + 1
    return {
        "text": segment["text"][match.start() : match.end()],
        "segment_id": segment["segment_id"],
        "page": segment.get("page"),
        "start": segment["start"] + match.start(),
        "end": segment["start"] + match.end(),
    }


def run(args: argparse.Namespace) -> int:
    jobs = {job["job_id"]: job for job in load_jsonl(args.jobs)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in load_jsonl(args.predictions):
            job_id = row["job_id"]
            compact = row["annotation"]
            job = jobs[job_id]
            used_spans: dict[str, int] = {}
            entities = []
            for entity in compact.get("entities", []):
                evidence = locate(entity["text"], job["segments"], used_spans)
                contracted_entity = {
                    key: entity[key]
                    for key in ("id", "text", "normalized_name", "type")
                    if key in entity
                }
                entities.append(
                    {
                        **contracted_entity,
                        "evidence": evidence,
                        "confidence": 1.0,
                        "review_status": "pending",
                        "created_by": "qwen3-4b-qlora",
                    }
                )
            by_id = {entity["id"]: entity for entity in entities}
            relations = []
            for relation in compact.get("relations", []):
                source = by_id.get(relation["source_id"])
                target = by_id.get(relation["target_id"])
                if not source or not target:
                    continue
                evidence = next((s for s in job["segments"] if source["evidence"]["segment_id"] == s["segment_id"] and target["evidence"]["segment_id"] == s["segment_id"]), source["evidence"])
                if "segment_type" in evidence:
                    evidence = {"text": evidence["text"], "segment_id": evidence["segment_id"], "page": evidence.get("page"), "start": evidence["start"], "end": evidence["end"]}
                contracted_relation = {
                    key: relation[key]
                    for key in ("id", "source_id", "type", "target_id", "claim_status")
                    if key in relation
                }
                relations.append(
                    {
                        **contracted_relation,
                        "evidence": [evidence],
                        "confidence": 1.0,
                        "review_status": "pending",
                        "created_by": "qwen3-4b-qlora",
                    }
                )
            annotation = {
                "schema_version": "0.1.0",
                "document_id": job["document_id"],
                "language": job["language"],
                "entities": entities,
                "relations": relations,
                "review": {"status": "unreviewed", "reviewers": [], "notes": "expanded compact QLoRA output"},
            }
            stream.write(json.dumps({"job_id": job_id, "annotation": annotation}, ensure_ascii=False) + "\n")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
