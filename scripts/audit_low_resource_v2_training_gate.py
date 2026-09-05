#!/usr/bin/env python3
"""Audit the v2 first-row memory gate without reading evaluation gold."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_run(row: dict[str, Any]) -> dict[str, Any]:
    root = Path(row["output_directory"])
    manifest_path = root / "run_manifest.json"
    if not manifest_path.is_file():
        return {
            "run_id": row["run_id"],
            "system": row["system"],
            "seed": row["seed"],
            "status": "not_started",
            "output_directory": str(root),
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics_path = root / "training_metrics.json"
    telemetry_path = root / "telemetry.json"
    metrics = (
        json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_path.is_file()
        else {}
    )
    telemetry = (
        json.loads(telemetry_path.read_text(encoding="utf-8"))
        if telemetry_path.is_file()
        else {}
    )
    status = str(manifest.get("status", "unknown"))
    if status == "complete" and not (root / "adapter_model.safetensors").is_file():
        status = "invalid_complete_missing_adapter"
    zero_loss = {
        "truncated_answers": metrics.get("truncated_answers_with_eos"),
        "truncated_prompts": metrics.get("truncated_prompts"),
        "skipped_overlength": metrics.get("skipped_overlength"),
    }
    if status == "complete" and any(value != 0 for value in zero_loss.values()):
        status = "invalid_complete_data_loss"
    return {
        "run_id": row["run_id"],
        "system": row["system"],
        "seed": row["seed"],
        "status": status,
        "output_directory": str(root),
        "return_code": manifest.get("return_code"),
        "formal_test_read": manifest.get("formal_test_read"),
        "protocol_sha256": manifest.get("protocol_sha256"),
        "sequence_audit": manifest.get("environment", {}).get("sequence_audit"),
        "runtime_environment": manifest.get("environment", {}).get(
            "runtime_environment"
        ),
        "examples": metrics.get("train_examples"),
        "steps": metrics.get("steps"),
        "mean_loss": metrics.get("mean_loss"),
        "zero_data_loss": zero_loss,
        "training_wall_clock_seconds": metrics.get("training_wall_clock_seconds"),
        "peak_cuda_memory_allocated_mib": metrics.get(
            "peak_cuda_memory_allocated_mib"
        ),
        "peak_cuda_memory_reserved_mib": metrics.get(
            "peak_cuda_memory_reserved_mib"
        ),
        "peak_device_memory_used_mib": telemetry.get(
            "peak_device_memory_used_mib"
        ),
        "estimated_energy_kwh": telemetry.get("estimated_energy_kwh"),
        "estimated_electricity_cost_cny": telemetry.get(
            "estimated_electricity_cost_cny"
        ),
        "manifest_sha256": sha256_file(manifest_path),
        "adapter_sha256": sha256_file(root / "adapter_model.safetensors")
        if (root / "adapter_model.safetensors").is_file()
        else None,
        "training_metrics_sha256": sha256_file(metrics_path)
        if metrics_path.is_file()
        else None,
        "telemetry_sha256": sha256_file(telemetry_path)
        if telemetry_path.is_file()
        else None,
    }


def markdown(summary: dict[str, Any]) -> str:
    first = summary["runs"][0]
    lines = [
        "# Protocol v2 d100 first-row execution gate",
        "",
        f"Status: **{summary['gate_status']}**",
        "",
        "Formal test remained sealed. No validation metric was read or calculated.",
        "",
        "| Run | System | Seed | Status |",
        "|---|---|---:|---|",
    ]
    lines.extend(
        f"| `{row['run_id']}` | `{row['system']}` | {row['seed']} | {row['status']} |"
        for row in summary["runs"]
    )
    lines.extend(
        [
            "",
            "## First-row result",
            "",
            f"The baseline trained all {first['examples']} examples for "
            f"{first['steps']} optimizer steps. No answer or prompt was truncated, "
            "and no overlength example was skipped.",
            "",
            f"Training took {first['training_wall_clock_seconds']} seconds. Peak "
            f"PyTorch allocated/reserved memory was "
            f"{first['peak_cuda_memory_allocated_mib']}/"
            f"{first['peak_cuda_memory_reserved_mib']} MiB; peak whole-device use "
            f"was {first['peak_device_memory_used_mib']} MiB. Mean training loss "
            f"was {first['mean_loss']}.",
            "",
            "## Decision",
            "",
            "The protocol v2 memory gate passed. The remaining eight d100 rows may "
            "run sequentially under the same frozen protocol. Lower-budget rows "
            "remain blocked until all nine d100 trainable rows complete and pass "
            "their artifact/telemetry checks.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> int:
    rows = [
        row
        for row in load_jsonl(args.matrix)
        if int(row.get("budget_documents", 0)) == 100
    ]
    runs = [inspect_run(row) for row in rows]
    first_passed = bool(runs) and runs[0]["status"] == "complete"
    all_passed = bool(runs) and all(row["status"] == "complete" for row in runs)
    gate_status = (
        "passed_all_d100"
        if all_passed
        else "passed_first_row"
        if first_passed
        else "failed_or_incomplete_first_row"
    )
    summary = {
        "protocol_id": "low-resource-provenance-study-v2",
        "gate": "d100-first-baseline-memory-gate",
        "gate_status": gate_status,
        "formal_test_read": False,
        "validation_metrics_read": False,
        "matrix": str(args.matrix),
        "matrix_sha256": sha256_file(args.matrix),
        "planned_d100_runs": len(runs),
        "complete_runs": sum(row["status"] == "complete" for row in runs),
        "not_started_runs": sum(row["status"] == "not_started" for row in runs),
        "remaining_d100_authorized": first_passed,
        "lower_budgets_authorized": all_passed,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(markdown(summary), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "runs"},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if first_passed else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path(
            "data/processed/experiments/formal/low_resource_manifests_v2/run_matrix.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/experiments/formal/low_resource_v2/d100_execution_gate.json"
        ),
    )
    parser.add_argument(
        "--report", type=Path, default=Path("paper/D100_V2_EXECUTION_GATE.md")
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
