#!/usr/bin/env python3
"""Evaluate auditable V3-style selection proxies on Geo-NRE quickcheck output.

Geo-NRE does not provide railway entity types, legal relation signatures,
claim status, or evidence spans.  The proxies below therefore use only the
checks that the public dataset can support and never claim to reproduce the
full railway KG V3 pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable


Triple = tuple[str, str, str]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def dataset_triples(row: dict[str, Any]) -> list[Triple]:
    values: list[Triple] = []
    for output in row.get("output", []):
        for triple in output.get("non_formatted_surface_output", []):
            if isinstance(triple, list) and len(triple) == 3:
                values.append(tuple(str(part) for part in triple))
    return values


def prediction_map(path: Path) -> dict[str, list[Triple]]:
    values: dict[str, list[Triple]] = {}
    for row in load_jsonl(path):
        values[str(row["id"])] = [tuple(map(str, triple)) for triple in row.get("predicted", [])]
    return values


def normalized(triple: Triple) -> Triple:
    return tuple(part.casefold() for part in triple)  # type: ignore[return-value]


def metrics(tp: int, fp: int, fn: int, total_rows: int, covered_rows: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "coverage": covered_rows / total_rows if total_rows else 0.0,
        "covered_rows": covered_rows,
        "rows": total_rows,
    }


def main(args: argparse.Namespace) -> int:
    sample = load_jsonl(args.sample)
    full = load_jsonl(args.catalogue_source)
    kg1 = prediction_map(args.kg1)
    kg2 = prediction_map(args.kg2)
    all_gold = [triple for row in full for triple in dataset_triples(row)]
    entity_catalogue = {part for triple in all_gold for part in (triple[0], triple[2])}
    relation_catalogue = {triple[1] for triple in all_gold}

    def verified_proxy(row: dict[str, Any], _: list[Triple], v2: list[Triple]) -> list[Triple]:
        text = str(row["input"]).casefold()
        return [
            triple
            for triple in v2
            if triple[0] in entity_catalogue
            and triple[2] in entity_catalogue
            and triple[1] in relation_catalogue
            and triple[0].casefold() in text
            and triple[2].casefold() in text
        ]

    def entity_agreement(row: dict[str, Any], v1: list[Triple], v2: list[Triple]) -> list[Triple]:
        allowed = verified_proxy(row, v1, v2)
        v1_entities = {part.casefold() for triple in v1 for part in (triple[0], triple[2])}
        return [
            triple
            for triple in allowed
            if triple[0].casefold() in v1_entities and triple[2].casefold() in v1_entities
        ]

    def triple_agreement(row: dict[str, Any], v1: list[Triple], v2: list[Triple]) -> list[Triple]:
        allowed = verified_proxy(row, v1, v2)
        v1_triples = {normalized(triple) for triple in v1}
        return [triple for triple in allowed if normalized(triple) in v1_triples]

    variants: dict[str, Callable[[dict[str, Any], list[Triple], list[Triple]], list[Triple]]] = {
        "v2_catalogue_and_source_verified_proxy": verified_proxy,
        "v3_entity_agreement_proxy": entity_agreement,
        "v3_exact_triple_agreement_proxy": triple_agreement,
    }
    totals = {name: [0, 0, 0, 0] for name in variants}
    outputs: list[dict[str, Any]] = []
    for row in sample:
        row_id = str(row["id"])
        gold = dataset_triples(row)
        gold_set = {normalized(triple) for triple in gold}
        row_output: dict[str, Any] = {"id": row["id"], "gold": gold, "variants": {}}
        for name, selector in variants.items():
            selected = selector(row, kg1.get(row_id, []), kg2.get(row_id, []))
            selected_set = {normalized(triple) for triple in selected}
            tp = len(gold_set.intersection(selected_set))
            fp = len(selected_set - gold_set)
            fn = len(gold_set - selected_set)
            totals[name][0] += tp
            totals[name][1] += fp
            totals[name][2] += fn
            totals[name][3] += bool(selected_set)
            row_output["variants"][name] = {
                "predicted": selected,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        outputs.append(row_output)

    summary = {
        "experiment": "Geo-NRE V3-style selective proxy",
        "interpretation_boundary": (
            "Not the full railway KG V3 pipeline. Geo-NRE lacks entity types, railway "
            "signatures, claim status, source offsets, and relation evidence spans."
        ),
        "kg_v1_predictions": str(args.kg1),
        "kg_v2_predictions": str(args.kg2),
        "variants": {
            name: metrics(tp, fp, fn, len(sample), covered)
            for name, (tp, fp, fn, covered) in totals.items()
        },
        "formal_test_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in outputs),
        encoding="utf-8",
    )
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--catalogue-source", type=Path, required=True)
    parser.add_argument("--kg1", type=Path, required=True)
    parser.add_argument("--kg2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
