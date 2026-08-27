#!/usr/bin/env python3
"""Prepare provider-neutral JSONL jobs for teacher-model pre-annotation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml


SYSTEM_INSTRUCTION = """You are pre-annotating safety-critical documents. Extract only claims supported by the supplied text. Do not invent missing causes, controls, regulations, or consequences. Return one complete JSON object conforming to preannotation_candidate.schema.json. Entity IDs must be E1, E2, ...; relation IDs must be R1, R2, .... Each entity text must be an exact, contiguous source substring; never paraphrase, expand, normalize, or add punctuation. For every entity, set evidence.text byte-for-byte equal to entity.text and select a segment_id that contains that exact occurrence. For relation evidence, copy an exact, contiguous quote from one supplied segment. Return only evidence segment_id and text; do not calculate offsets or page numbers. Every relation source and target entity type must follow ontology.allowed_relation_signatures. Set entity and relation review_status to pending, document review.status to unreviewed, and created_by to the supplied teacher_model. Distinguish explicit, inferred, normative, and uncertain relations."""
PROMPT_VERSION = "teacher-preannotation-v1.1.0"


def chunk_segments(segments: list[dict[str, Any]], max_characters: int) -> list[list[dict[str, Any]]]:
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_size = 0
    for segment in segments:
        segment_size = len(segment["text"])
        if current and current_size + segment_size > max_characters:
            chunks.append(current)
            current = []
            current_size = 0
        if segment_size > max_characters:
            text = segment["text"]
            for local_start in range(0, len(text), max_characters):
                piece = dict(segment)
                piece["segment_id"] = f"{segment['segment_id']}.{local_start // max_characters + 1}"
                piece["text"] = text[local_start : local_start + max_characters]
                piece["start"] = segment["start"] + local_start
                piece["end"] = piece["start"] + len(piece["text"])
                if current:
                    chunks.append(current)
                    current = []
                    current_size = 0
                chunks.append([piece])
                current_size = 0
            continue
        current.append(segment)
        current_size += segment_size
    if current:
        chunks.append(current)
    return chunks


def representative_chunk_indices(chunk_count: int, limit: int) -> list[int]:
    """Select evenly distributed chunks while always retaining both ends."""
    if limit <= 0 or chunk_count <= limit:
        return list(range(chunk_count))
    if limit == 1:
        return [chunk_count // 2]
    denominator = limit - 1
    return [
        (position * (chunk_count - 1) + denominator // 2) // denominator
        for position in range(limit)
    ]


def prepare(args: argparse.Namespace) -> None:
    ontology = yaml.safe_load(args.ontology.read_text(encoding="utf-8"))
    with args.pilot_set.open(encoding="utf-8-sig", newline="") as stream:
        pilot_rows = list(csv.DictReader(stream))
    if args.pending_only:
        pilot_rows = [row for row in pilot_rows if row.get("review_status") == "pending"]

    compact_ontology = {
        "version": ontology["version"],
        "entity_types": {key: value["description"] for key, value in ontology["entity_types"].items()},
        "relation_types": {key: value["description"] for key, value in ontology["relation_types"].items()},
        "claim_statuses": ontology["claim_statuses"],
        "allowed_relation_signatures": ontology["allowed_relation_signatures"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    job_count = 0
    with args.output.open("w", encoding="utf-8") as stream:
        for row in pilot_rows:
            document_path = args.document_root / f"{row['document_id']}.json"
            document = json.loads(document_path.read_text(encoding="utf-8"))
            chunks = chunk_segments(document["segments"], args.max_characters)
            chunk_indices = representative_chunk_indices(len(chunks), args.max_chunks_per_document)
            for chunk_index in chunk_indices:
                chunk = chunks[chunk_index]
                chunk_number = chunk_index + 1
                job_count += 1
                job = {
                    "job_id": f"{row['document_id']}_C{chunk_number}",
                    "document_id": row["document_id"],
                    "language": document["language"],
                    "category": row["category"],
                    "source_path": document["relative_path"],
                    "chunk_number": chunk_number,
                    "chunk_count": len(chunks),
                    "teacher_model": args.teacher_model,
                    "prompt_version": PROMPT_VERSION,
                    "system_instruction": SYSTEM_INSTRUCTION,
                    "ontology": compact_ontology,
                    "segments": chunk,
                    "expected_output_schema": "schemas/preannotation_candidate.schema.json",
                    "status": "pending",
                }
                stream.write(json.dumps(job, ensure_ascii=False) + "\n")
    print(json.dumps({"documents": len(pilot_rows), "jobs": job_count, "output": str(args.output)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-set", type=Path, default=Path("data/catalog/pilot_set.csv"))
    parser.add_argument("--document-root", type=Path, default=Path("data/processed/corpus/documents"))
    parser.add_argument("--ontology", type=Path, default=Path("configs/risk_ontology.yaml"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/preannotation/jobs.jsonl"))
    parser.add_argument("--max-characters", type=int, default=12000)
    parser.add_argument(
        "--max-chunks-per-document",
        type=int,
        default=0,
        help="Select at most this many evenly distributed chunks per document; 0 keeps all chunks",
    )
    parser.add_argument("--teacher-model", default="unspecified-teacher")
    parser.add_argument("--pending-only", action="store_true", help="Skip rows already marked as pilot_accepted")
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
