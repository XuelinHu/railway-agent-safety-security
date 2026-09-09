#!/usr/bin/env python3
"""Aggregate repeated official CrossNER runs and compute paired uncertainty."""

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "paper/cross-residual/results/official_crossner"
rows = list(csv.DictReader((RESULT / "runs.csv").open()))


def write_csv(path, records, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


grouped = defaultdict(list)
for row in rows:
    grouped[(row["method"], row["domain"])].append(float(row["f1"]) * 100)

aggregate = []
for method in ["zero-shot", "domain-adapted", "score-residual"]:
    for domain in ["ai", "literature", "music", "politics", "science"]:
        values = grouped[(method, domain)]
        aggregate.append({
            "method": method,
            "domain": domain,
            "mean_f1": statistics.mean(values),
            "sample_sd": statistics.stdev(values) if len(set(values)) > 1 else 0.0,
        })
write_csv(RESULT / "aggregate.csv", aggregate,
          ["method", "domain", "mean_f1", "sample_sd"])

by_seed = defaultdict(lambda: defaultdict(list))
for row in rows:
    by_seed[int(row["seed"])][row["method"]].append(float(row["f1"]) * 100)
repeats = []
for seed in sorted(by_seed):
    adapted = statistics.mean(by_seed[seed]["domain-adapted"])
    residual = statistics.mean(by_seed[seed]["score-residual"])
    repeats.append({"run": len(repeats) + 1, "adapted_f1": adapted,
                    "residual_f1": residual, "difference": residual - adapted})
write_csv(RESULT / "repeated_macro.csv", repeats,
          ["run", "adapted_f1", "residual_f1", "difference"])

differences = [row["difference"] for row in repeats]
test = stats.ttest_1samp(differences, 0.0)
interval = stats.t.interval(0.95, len(differences) - 1,
                            loc=statistics.mean(differences), scale=stats.sem(differences))
(RESULT / "statistics.json").write_text(json.dumps({
    "paired_unit": "run-level five-domain macro F1",
    "n": len(differences),
    "mean_difference_points": statistics.mean(differences),
    "t_statistic": float(test.statistic),
    "two_sided_p_value": float(test.pvalue),
    "confidence_level": 0.95,
    "confidence_interval_points": [float(interval[0]), float(interval[1])],
}, indent=2))
