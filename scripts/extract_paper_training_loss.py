#!/usr/bin/env python3
"""Recover logged seed-42 EAE/HRGE losses with adapter-level provenance."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "outputs/public_pge_validation_seed42"
LOG = RUN_ROOT / "runner.log"
OUT = ROOT / "paper/results/ade_conll04/training_loss_seed42.csv"
META = ROOT / "paper/results/ade_conll04/training_loss_seed42_provenance.json"


def adapter_index() -> dict[str, tuple[str, str, dict]]:
    result = {}
    for dataset in ("ade", "conll04"):
        for system in ("eae", "hrge"):
            path = RUN_ROOT / dataset / f"{system}_adapter/training_metrics.json"
            metrics = json.loads(path.read_text(encoding="utf-8"))
            result[metrics["started_at_utc"]] = (dataset, system, metrics)
    return result


def completed_blocks(text: str) -> list[tuple[list[dict], dict]]:
    blocks, losses, collecting = [], [], False
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if line.startswith('{"prepared_examples"'):
            losses, collecting = [], True
        elif collecting and re.fullmatch(
                r'\{"epoch":\s*\d+,\s*"step":\s*\d+,\s*"loss":\s*[^}]+\}', line):
            losses.append(json.loads(line))
        elif collecting and line == "{":
            candidate, depth = [line], 1
            while depth and index + 1 < len(lines):
                index += 1
                candidate.append(lines[index])
                depth += lines[index].count("{") - lines[index].count("}")
            try:
                metrics = json.loads("\n".join(candidate))
            except json.JSONDecodeError:
                metrics = {}
            if "started_at_utc" in metrics and "steps" in metrics:
                blocks.append((losses, metrics))
                collecting = False
        index += 1
    return blocks


def main() -> None:
    source = LOG.read_bytes()
    adapters = adapter_index()
    rows, matched = [], {}
    for losses, logged_metrics in completed_blocks(source.decode("utf-8", errors="replace")):
        started = logged_metrics["started_at_utc"]
        if started not in adapters:
            continue
        dataset, system, stored_metrics = adapters[started]
        assert logged_metrics["steps"] == stored_metrics["steps"]
        assert logged_metrics["final_loss"] == stored_metrics["final_loss"]
        assert losses and losses[-1]["step"] <= stored_metrics["steps"]
        assert all(right["step"] - left["step"] == 5
                   for left, right in zip(losses, losses[1:]))
        matched[f"{dataset}:{system}"] = {
            "started_at_utc": started,
            "optimizer_steps": stored_metrics["steps"],
            "logged_observations": len(losses),
            "logging_interval_steps": 5,
            "final_unlogged_step": stored_metrics["steps"] % 5 != 0,
        }
        rows.extend({"dataset": dataset, "system": system,
                     "seed": stored_metrics["seed"], "epoch": observation["epoch"],
                     "step": observation["step"], "loss": observation["loss"],
                     "started_at_utc": started} for observation in losses)
    assert set(matched) == {"ade:eae", "ade:hrge", "conll04:eae", "conll04:hrge"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    META.write_text(json.dumps({
        "source_log": str(LOG.relative_to(ROOT)),
        "source_log_sha256": hashlib.sha256(source).hexdigest(),
        "scope": "seed-42 EAE and HRGE adapters; logged every five optimizer steps",
        "adapters": matched,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} genuine loss observations to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
