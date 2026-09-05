#!/usr/bin/env python3
"""Document-level bootstrap and paired permutation comparison for two systems."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def norm(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def relation_key(relation: dict[str, Any], entities: dict[str, dict[str, Any]]) -> tuple[str, str, str]:
    source = entities.get(relation.get("source_id"), {})
    target = entities.get(relation.get("target_id"), {})
    return (
        norm(source.get("text", "")),
        relation.get("type", ""),
        norm(target.get("text", "")),
    )


def counts(gold: dict[str, Any], predicted: dict[str, Any]) -> dict[str, dict[str, int | float]]:
    gold_entities = {(norm(entity["text"]), entity["type"]) for entity in gold.get("entities", [])}
    predicted_entities = {
        (norm(entity["text"]), entity["type"]) for entity in predicted.get("entities", [])
    }
    gold_by_id = {entity["id"]: entity for entity in gold.get("entities", [])}
    predicted_by_id = {entity["id"]: entity for entity in predicted.get("entities", [])}
    gold_relations = {
        relation_key(relation, gold_by_id) for relation in gold.get("relations", [])
    }
    predicted_relations = {
        relation_key(relation, predicted_by_id) for relation in predicted.get("relations", [])
    }
    return {
        "entity": {
            "gold": len(gold_entities),
            "predicted": len(predicted_entities),
            "correct": len(gold_entities & predicted_entities),
        },
        "relation": {
            "gold": len(gold_relations),
            "predicted": len(predicted_relations),
            "correct": len(gold_relations & predicted_relations),
        },
    }


def f1(item: dict[str, int | float]) -> float:
    gold = int(item["gold"])
    predicted = int(item["predicted"])
    correct = int(item["correct"])
    if not gold or not predicted:
        return 0.0
    precision = correct / predicted
    recall = correct / gold
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def add_counts(left: dict[str, int], right: dict[str, int | float]) -> None:
    for key in ("gold", "predicted", "correct"):
        left[key] += int(right[key])


def pooled(items: list[dict[str, Any]], indices: Iterable[int], field: str) -> float:
    total = {"gold": 0, "predicted": 0, "correct": 0}
    for index in indices:
        add_counts(total, items[index][field])
    return f1(total)


def macro(items: list[dict[str, Any]], indices: Iterable[int], field: str) -> float:
    selected = [items[index][field] for index in indices]
    return sum(f1(item) for item in selected) / len(selected) if selected else 0.0


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def interval(values: list[float]) -> dict[str, float]:
    return {
        "lower": round(quantile(values, 0.025), 4),
        "upper": round(quantile(values, 0.975), 4),
    }


def observed(items: list[dict[str, Any]], field: str) -> dict[str, float]:
    total = {"gold": 0, "predicted": 0, "correct": 0}
    for item in items:
        add_counts(total, item[field])
    return {
        "pooled_f1": round(f1(total), 4),
        "macro_f1": round(macro(items, range(len(items)), field), 4),
        "gold": total["gold"],
        "predicted": total["predicted"],
        "correct": total["correct"],
    }


def compare_field(
    baseline: list[dict[str, Any]],
    kg: list[dict[str, Any]],
    field: str,
    iterations: int,
    rng: random.Random,
) -> dict[str, Any]:
    baseline_observed = observed(baseline, field)
    kg_observed = observed(kg, field)
    n = len(baseline)
    bootstrap_baseline_pooled: list[float] = []
    bootstrap_kg_pooled: list[float] = []
    bootstrap_baseline_macro: list[float] = []
    bootstrap_kg_macro: list[float] = []
    bootstrap_difference_pooled: list[float] = []
    bootstrap_difference_macro: list[float] = []
    permutation_pooled: list[float] = []
    permutation_macro: list[float] = []
    for _ in range(iterations):
        indices = [rng.randrange(n) for _ in range(n)]
        baseline_pooled = pooled(baseline, indices, field)
        kg_pooled = pooled(kg, indices, field)
        baseline_macro = macro(baseline, indices, field)
        kg_macro = macro(kg, indices, field)
        bootstrap_baseline_pooled.append(baseline_pooled)
        bootstrap_kg_pooled.append(kg_pooled)
        bootstrap_baseline_macro.append(baseline_macro)
        bootstrap_kg_macro.append(kg_macro)
        bootstrap_difference_pooled.append(kg_pooled - baseline_pooled)
        bootstrap_difference_macro.append(kg_macro - baseline_macro)

        swapped_baseline: list[dict[str, Any]] = []
        swapped_kg: list[dict[str, Any]] = []
        for index in range(n):
            if rng.randrange(2):
                swapped_baseline.append(kg[index])
                swapped_kg.append(baseline[index])
            else:
                swapped_baseline.append(baseline[index])
                swapped_kg.append(kg[index])
        permutation_pooled.append(
            pooled(swapped_kg, range(n), field) - pooled(swapped_baseline, range(n), field)
        )
        permutation_macro.append(
            macro(swapped_kg, range(n), field) - macro(swapped_baseline, range(n), field)
        )

    observed_pooled_difference = kg_observed["pooled_f1"] - baseline_observed["pooled_f1"]
    observed_macro_difference = kg_observed["macro_f1"] - baseline_observed["macro_f1"]

    def permutation_p(values: list[float], value: float) -> float:
        extreme = sum(abs(candidate) >= abs(value) for candidate in values)
        # Five decimals preserves the finite-sample minimum for the default
        # 20,000 iterations instead of serializing it misleadingly as 0.0000.
        return round((extreme + 1) / (len(values) + 1), 5)

    return {
        "baseline": {
            **baseline_observed,
            "pooled_f1_ci95": interval(bootstrap_baseline_pooled),
            "macro_f1_ci95": interval(bootstrap_baseline_macro),
        },
        "kg": {
            **kg_observed,
            "pooled_f1_ci95": interval(bootstrap_kg_pooled),
            "macro_f1_ci95": interval(bootstrap_kg_macro),
        },
        "kg_minus_baseline": {
            "pooled_f1": round(observed_pooled_difference, 4),
            "macro_f1": round(observed_macro_difference, 4),
            "pooled_f1_ci95": interval(bootstrap_difference_pooled),
            "macro_f1_ci95": interval(bootstrap_difference_macro),
            "paired_permutation_p_pooled_f1": permutation_p(
                permutation_pooled, observed_pooled_difference
            ),
            "paired_permutation_p_macro_f1": permutation_p(
                permutation_macro, observed_macro_difference
            ),
        },
    }


def build_units(
    gold_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    kg_rows: list[dict[str, Any]],
    job_rows: list[dict[str, Any]],
    gold_index: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    def keyed_annotations(rows: list[dict[str, Any]], index: list[dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
        result = {}
        for position, row in enumerate(rows):
            if "annotation" in row:
                result[row["job_id"]] = row["annotation"]
            elif index:
                result[index[position]["job_id"]] = row
            else:
                raise ValueError("annotation rows without job_id require an index file")
        return result

    requested = [row["job_id"] for row in job_rows]
    gold = keyed_annotations(gold_rows, gold_index)
    baseline = keyed_annotations(baseline_rows)
    kg = keyed_annotations(kg_rows)
    missing = {
        "gold": sorted(set(requested) - set(gold)),
        "baseline": sorted(set(requested) - set(baseline)),
        "kg": sorted(set(requested) - set(kg)),
    }
    if any(missing.values()):
        raise ValueError(f"missing requested job IDs: {missing}")
    units: dict[str, dict[str, Any]] = {}
    for job_id in requested:
        document_id = gold[job_id]["document_id"]
        unit = units.setdefault(
            document_id,
            {"document_id": document_id, "language": gold[job_id].get("language", "unknown")},
        )
        for system, annotations in (("baseline", baseline), ("kg", kg)):
            system_counts = counts(gold[job_id], annotations[job_id])
            target = unit.setdefault(system, {"entity": {"gold": 0, "predicted": 0, "correct": 0}, "relation": {"gold": 0, "predicted": 0, "correct": 0}})
            for field in ("entity", "relation"):
                add_counts(target[field], system_counts[field])
    return [units[key] for key in sorted(units)]


def run(args: argparse.Namespace) -> int:
    gold_rows = load_jsonl(args.gold)
    gold_index = load_jsonl(args.gold_index) if args.gold_index else None
    baseline_rows = load_jsonl(args.baseline)
    kg_rows = load_jsonl(args.kg)
    job_rows = load_jsonl(args.jobs)
    units = build_units(gold_rows, baseline_rows, kg_rows, job_rows, gold_index)
    rng = random.Random(args.seed)
    result = {
        "unit": "document",
        "documents": len(units),
        "iterations": args.iterations,
        "seed": args.seed,
        "systems": {"baseline": args.baseline.name, "kg": args.kg.name},
        "fields": {
            field: compare_field(
                [unit["baseline"] for unit in units],
                [unit["kg"] for unit in units],
                field,
                args.iterations,
                rng,
            )
            for field in ("entity", "relation")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--gold-index", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--kg", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260829)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
