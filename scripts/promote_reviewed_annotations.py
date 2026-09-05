#!/usr/bin/env python3
"""Promote screened annotations into gold splits and a provenance-bearing graph."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def canonical_text(value: str) -> str:
    text = re.sub(r"\s+", " ", value).strip().casefold()
    outer_markup = {
        "《": "》",
        "（": "）",
        "(": ")",
        "【": "】",
        "[": "]",
        "<": ">",
        "“": "”",
        "「": "」",
        "『": "』",
    }
    changed = True
    while changed and len(text) >= 2:
        changed = False
        closing = outer_markup.get(text[0])
        if closing and text.endswith(closing):
            text = text[1:-1].strip()
            changed = True
    return text


def concept_id(language: str, entity_type: str, name: str) -> str:
    key = f"{language}|{entity_type}|{canonical_text(name)}".encode("utf-8")
    return "concept_" + hashlib.sha1(key).hexdigest()[:16]


def stable_rank(document_id: str) -> str:
    return hashlib.sha256(document_id.encode("utf-8")).hexdigest()


def choose_splits(documents: list[dict[str, str]]) -> dict[str, str]:
    """Choose a reproducible 20/4/4 document split, balanced by source group."""
    by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in documents:
        by_group[row["source_group"]].append(row)
    assignment: dict[str, str] = {}
    for group, rows in sorted(by_group.items()):
        ordered = sorted(rows, key=lambda row: stable_rank(row["document_id"]))
        test_count = max(1, round(len(ordered) * 0.15))
        validation_count = max(1, round(len(ordered) * 0.15))
        for row in ordered[:test_count]:
            assignment[row["document_id"]] = "test"
        for row in ordered[test_count : test_count + validation_count]:
            assignment[row["document_id"]] = "validation"
        for row in ordered[test_count + validation_count :]:
            assignment[row["document_id"]] = "train"
    return assignment


def promote_annotation(annotation: dict[str, Any]) -> dict[str, Any]:
    promoted = json.loads(json.dumps(annotation, ensure_ascii=False))
    promoted["review"] = {
        "status": "single_reviewed",
        "reviewers": ["human_screened"],
        "notes": "Bulk accepted after human screening of the pre-annotation review set.",
    }
    for entity in promoted["entities"]:
        entity["review_status"] = "accepted"
    for relation in promoted["relations"]:
        relation["review_status"] = "accepted"
    return promoted


def read_pilot(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {row["document_id"]: row for row in csv.DictReader(stream)}


def validate_annotations(annotations: list[dict[str, Any]], schema_path: Path) -> None:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    errors = [error.message for annotation in annotations for error in validator.iter_errors(annotation)]
    if errors:
        raise ValueError("promoted annotations failed schema validation: " + "; ".join(errors[:10]))


def build_graph(
    annotations: list[dict[str, Any]],
    jobs_by_id: dict[str, dict[str, Any]],
    split_by_document: dict[str, str],
    output_dir: Path,
) -> dict[str, int]:
    concepts: dict[str, dict[str, Any]] = {}
    mention_rows: list[dict[str, Any]] = []
    relation_rows: list[dict[str, Any]] = []
    for annotation in annotations:
        job_id = annotation["_job_id"]
        job = jobs_by_id[job_id]
        document_id = annotation["document_id"]
        split = split_by_document[document_id]
        mention_by_source_id: dict[str, str] = {}
        for entity in annotation["entities"]:
            name = canonical_text(entity.get("normalized_name") or entity["text"])
            cid = concept_id(annotation["language"], entity["type"], name)
            mention_id = f"{job_id}:{entity['id']}"
            mention_by_source_id[entity["id"]] = mention_id
            concepts.setdefault(
                cid,
                {
                    "concept_id": cid,
                    "canonical_name": name,
                    "language": annotation["language"],
                    "type": entity["type"],
                    "mention_count": 0,
                    "source_documents": [],
                },
            )
            concept = concepts[cid]
            concept["mention_count"] += 1
            if document_id not in concept["source_documents"]:
                concept["source_documents"].append(document_id)
            mention_rows.append(
                {
                    "mention_id": mention_id,
                    "concept_id": cid,
                    "document_id": document_id,
                    "job_id": job_id,
                    "split": split,
                    "language": annotation["language"],
                    "source_id": entity["id"],
                    "text": entity["text"],
                    "type": entity["type"],
                    "canonical_name": name,
                    "normalized_name": entity.get("normalized_name"),
                    "confidence": entity["confidence"],
                    "evidence": entity["evidence"],
                    "provenance": {"source_path": job["source_path"], "teacher_model": entity.get("created_by")},
                }
            )
        for relation in annotation["relations"]:
            source_mention = mention_by_source_id[relation["source_id"]]
            target_mention = mention_by_source_id[relation["target_id"]]
            source_concept = next(row["concept_id"] for row in mention_rows if row["mention_id"] == source_mention)
            target_concept = next(row["concept_id"] for row in mention_rows if row["mention_id"] == target_mention)
            relation_rows.append(
                {
                    "relation_id": f"{job_id}:{relation['id']}",
                    "source_concept_id": source_concept,
                    "target_concept_id": target_concept,
                    "source_mention_id": source_mention,
                    "target_mention_id": target_mention,
                    "document_id": document_id,
                    "job_id": job_id,
                    "split": split,
                    "type": relation["type"],
                    "claim_status": relation["claim_status"],
                    "confidence": relation["confidence"],
                    "evidence": relation["evidence"],
                    "provenance": {"source_path": job["source_path"], "teacher_model": relation.get("created_by")},
                }
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "concepts.jsonl", list(concepts.values()))
    write_jsonl(output_dir / "mentions.jsonl", mention_rows)
    write_jsonl(output_dir / "relations.jsonl", relation_rows)
    return {"concepts": len(concepts), "mentions": len(mention_rows), "relations": len(relation_rows)}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> int:
    normalized = load_jsonl(args.annotations)
    jobs = load_jsonl(args.jobs)
    candidates = load_jsonl(args.candidates) if args.candidates.exists() else []
    jobs_by_id = {job["job_id"]: job for job in jobs}
    candidate_job_ids = [row["job_id"] for row in candidates]
    if len(normalized) != len(jobs):
        raise ValueError(f"expected one normalized record per job: {len(normalized)} vs {len(jobs)}")
    if candidate_job_ids and set(candidate_job_ids) != set(jobs_by_id):
        raise ValueError("candidate job IDs do not exactly match jobs; refuse ambiguous block-to-job mapping")
    enriched = []
    for index, annotation in enumerate(normalized):
        job_id = candidate_job_ids[index] if candidate_job_ids else jobs[index]["job_id"]
        if annotation["document_id"] != jobs_by_id[job_id]["document_id"]:
            raise ValueError(f"document mismatch for {job_id}")
        item = promote_annotation(annotation)
        item["_job_id"] = job_id
        enriched.append(item)
    validate_annotations([{key: value for key, value in annotation.items() if not key.startswith("_")} for annotation in enriched], args.schema)

    pilot = read_pilot(args.pilot_set)
    documents = [{"document_id": doc_id, "source_group": row["source_group"], "category": row["category"], "relative_path": row["relative_path"]} for doc_id, row in pilot.items()]
    split_by_document = choose_splits(documents)
    output_root = args.output
    gold_dir = output_root / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(gold_dir / "all.jsonl", [{k: v for k, v in annotation.items() if not k.startswith("_")} for annotation in enriched])
    split_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for annotation in enriched:
        split_rows[split_by_document[annotation["document_id"]]].append({k: v for k, v in annotation.items() if not k.startswith("_")})
    for split, rows in split_rows.items():
        write_jsonl(gold_dir / f"{split}.jsonl", rows)
        write_jsonl(
            gold_dir / f"{split}_index.jsonl",
            [
                {
                    "record_index": index,
                    "job_id": annotation["_job_id"],
                    "document_id": annotation["document_id"],
                    "split": split,
                }
                for index, annotation in enumerate(
                    [item for item in enriched if split_by_document[item["document_id"]] == split]
                )
            ],
        )
    write_jsonl(
        gold_dir / "record_index.jsonl",
        [
            {
                "record_index": index,
                "job_id": annotation["_job_id"],
                "document_id": annotation["document_id"],
                "split": split_by_document[annotation["document_id"]],
            }
            for index, annotation in enumerate(enriched)
        ],
    )
    manifest = []
    for row in sorted(documents, key=lambda item: item["document_id"]):
        manifest.append({**row, "split": split_by_document[row["document_id"]]})
    write_jsonl(output_root / "split_manifest.jsonl", manifest)
    graph_stats = build_graph(enriched, jobs_by_id, split_by_document, output_root / "knowledge_graph")
    summary = {
        "jobs": len(enriched),
        "documents": len(split_by_document),
        "split_documents": dict(Counter(split_by_document.values())),
        "split_chunks": dict(Counter(split_by_document[a["document_id"]] for a in enriched)),
        "entities": sum(len(a["entities"]) for a in enriched),
        "relations": sum(len(a["relations"]) for a in enriched),
        "graph": graph_stats,
        "review_policy": "bulk_accepted_after_human_screening",
        "schema_version": "0.1.0",
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path("data/processed/preannotation/sub2api_terra_normalized.jsonl"))
    parser.add_argument("--jobs", type=Path, default=Path("data/processed/preannotation/jobs.jsonl"))
    parser.add_argument("--candidates", type=Path, default=Path("data/processed/preannotation/sub2api_terra_candidates.jsonl"))
    parser.add_argument("--pilot-set", type=Path, default=Path("data/catalog/pilot_set.csv"))
    parser.add_argument("--schema", type=Path, default=Path("schemas/risk_annotation.schema.json"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/reviewed"))
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
