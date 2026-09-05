#!/usr/bin/env python3
"""Audit and deterministically sample the public GenIE Geo-NRE dataset.

This utility does not convert Geo-NRE into the railway ontology.  It measures
the compatibility barriers first so that an incompatible external benchmark is
not silently reported as a railway extraction result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def triples(row: dict[str, Any]) -> list[tuple[str, str, str]]:
    values: list[tuple[str, str, str]] = []
    for output in row.get("output", []):
        for triple in output.get("non_formatted_surface_output", []):
            if isinstance(triple, list) and len(triple) == 3:
                values.append((str(triple[0]), str(triple[1]), str(triple[2])))
    return values


def normalize_relation(value: str) -> str:
    return "_".join(value.casefold().replace("-", " ").split())


def main(args: argparse.Namespace) -> int:
    rows = load_jsonl(args.input)
    ontology = yaml.safe_load(args.ontology.read_text(encoding="utf-8"))
    railway_relations = set(ontology["relation_types"])

    relation_counts: Counter[str] = Counter()
    unique_entities: set[str] = set()
    triple_count = 0
    subject_exact = 0
    object_exact = 0
    both_exact = 0
    subject_casefold = 0
    object_casefold = 0
    both_casefold = 0
    rows_without_triples = 0
    rows_with_multiple_triples = 0
    by_relation: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        row_triples = triples(row)
        if not row_triples:
            rows_without_triples += 1
            continue
        if len(row_triples) > 1:
            rows_with_multiple_triples += 1
        text = str(row.get("input", ""))
        folded = text.casefold()
        for subject, relation, obj in row_triples:
            triple_count += 1
            relation_counts[relation] += 1
            unique_entities.update((subject, obj))
            subject_in = subject in text
            object_in = obj in text
            subject_folded_in = subject.casefold() in folded
            object_folded_in = obj.casefold() in folded
            subject_exact += subject_in
            object_exact += object_in
            both_exact += subject_in and object_in
            subject_casefold += subject_folded_in
            object_casefold += object_folded_in
            both_casefold += subject_folded_in and object_folded_in
            by_relation[relation].append(row)

    rng = random.Random(args.seed)
    sampled: list[dict[str, Any]] = []
    sampled_ids: set[str] = set()
    for relation in sorted(by_relation):
        candidates = list(by_relation[relation])
        rng.shuffle(candidates)
        selected = 0
        for row in candidates:
            row_id = str(row.get("id"))
            if row_id in sampled_ids:
                continue
            sampled.append(row)
            sampled_ids.add(row_id)
            selected += 1
            if selected >= args.per_relation:
                break
    sampled.sort(key=lambda row: int(row["id"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.output_dir / "geo_nre_stratified_sample.jsonl"
    sample_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in sampled),
        encoding="utf-8",
    )

    normalized_geo_relations = {
        normalize_relation(relation): relation for relation in relation_counts
    }
    exact_relation_overlap = sorted(
        railway_relations.intersection(normalized_geo_relations)
    )
    report = {
        "dataset": "GenIE Geo-NRE",
        "source": {
            "paper_doi": "10.18653/v1/2022.naacl-main.342",
            "dataset_doi": "10.5281/zenodo.6139236",
            "license": "CC-BY-4.0",
            "input_path": str(args.input),
            "input_sha256": sha256(args.input),
        },
        "dataset_statistics": {
            "rows": len(rows),
            "triples": triple_count,
            "rows_without_triples": rows_without_triples,
            "rows_with_multiple_triples": rows_with_multiple_triples,
            "unique_surface_entities": len(unique_entities),
            "relations": len(relation_counts),
            "relation_counts": dict(sorted(relation_counts.items())),
        },
        "source_grounding_compatibility": {
            "subject_exact_rate": subject_exact / triple_count if triple_count else 0.0,
            "object_exact_rate": object_exact / triple_count if triple_count else 0.0,
            "both_endpoints_exact_rate": both_exact / triple_count if triple_count else 0.0,
            "subject_casefold_rate": subject_casefold / triple_count if triple_count else 0.0,
            "object_casefold_rate": object_casefold / triple_count if triple_count else 0.0,
            "both_endpoints_casefold_rate": both_casefold / triple_count if triple_count else 0.0,
        },
        "ontology_compatibility": {
            "geo_nre_relation_count": len(relation_counts),
            "railway_relation_count": len(railway_relations),
            "exact_normalized_relation_overlap": exact_relation_overlap,
            "geo_nre_has_entity_type_labels": False,
            "geo_nre_has_global_character_offsets": False,
            "geo_nre_has_claim_status": False,
            "geo_nre_has_relation_evidence_spans": False,
        },
        "sample": {
            "seed": args.seed,
            "per_relation_target": args.per_relation,
            "rows": len(sampled),
            "path": str(sample_path),
            "sha256": sha256(sample_path),
        },
        "formal_test_read": False,
    }
    report_path = args.output_dir / "compatibility_audit.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, default=Path("configs/risk_ontology.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--per-relation", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
