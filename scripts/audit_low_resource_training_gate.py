#!/usr/bin/env python3
"""Audit the 100-document execution gate without reading evaluation gold."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_attempt(root: Path) -> dict[str, Any]:
    manifest_path = root / "run_manifest.json"
    if not manifest_path.is_file():
        return {"directory": str(root), "status": "not_started"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    log_path = root / "training.log"
    log = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
    if manifest.get("status") == "complete":
        failure_class = None
        terminal = False
    elif "torch.OutOfMemoryError" in log:
        failure_class = "cuda_out_of_memory_after_training_started"
        terminal = True
    elif "Invalid device argument" in log and "prepared_examples" not in log:
        failure_class = "telemetry_incompatibility_before_model_output"
        terminal = False
    else:
        failure_class = "unclassified_execution_failure"
        terminal = "prepared_examples" in log
    telemetry_path = root / "telemetry.json"
    telemetry = (
        json.loads(telemetry_path.read_text(encoding="utf-8"))
        if telemetry_path.is_file()
        else {}
    )
    return {
        "directory": str(root),
        "status": manifest.get("status", "unknown"),
        "attempt_id": manifest.get("attempt_id", manifest.get("run_id")),
        "failure_class": failure_class,
        "terminal_under_frozen_retry_policy": terminal,
        "model_or_training_output_present": bool(
            (root / "adapter_config.json").exists()
            or (root / "training_metrics.json").exists()
        ),
        "prepared_examples_logged": "prepared_examples" in log,
        "return_code": manifest.get("return_code"),
        "wall_clock_seconds": manifest.get("wall_clock_seconds"),
        "peak_device_memory_used_mib": telemetry.get(
            "peak_device_memory_used_mib"
        ),
        "peak_gpu_utilization_percent": telemetry.get(
            "peak_gpu_utilization_percent"
        ),
        "peak_power_draw_watts": telemetry.get("peak_power_draw_watts"),
        "estimated_energy_kwh": telemetry.get("estimated_energy_kwh"),
        "manifest_sha256": sha256_file(manifest_path),
        "training_log_sha256": sha256_file(log_path) if log_path.is_file() else None,
        "telemetry_sha256": sha256_file(telemetry_path)
        if telemetry_path.is_file()
        else None,
    }


def run_status(row: dict[str, Any]) -> dict[str, Any]:
    root = Path(row["output_directory"])
    attempts = [classify_attempt(root)]
    attempts.extend(
        classify_attempt(path)
        for path in sorted(root.parent.glob(f"{root.name}_retry*"))
        if path.is_dir()
    )
    attempted = [item for item in attempts if item["status"] != "not_started"]
    if any(item["status"] == "complete" for item in attempted):
        status = "complete"
    elif any(item.get("terminal_under_frozen_retry_policy") for item in attempted):
        status = "failed_terminal"
    elif attempted:
        status = "retry_pending"
    else:
        status = "not_started"
    return {
        "run_id": row["run_id"],
        "seed": row["seed"],
        "system": row["system"],
        "status": status,
        "attempts": attempts,
    }


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# 100-document execution gate audit",
        "",
        f"Status: **{summary['gate_status']}**",
        "",
        "The formal test remained sealed. No validation metric was read or",
        "calculated during this gate.",
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
            "## Finding",
            "",
            "The first baseline attempt failed before model loading because the",
            "new telemetry call was incompatible with the frozen PyTorch build.",
            "The one allowed mechanical retry preserved all model and training",
            "parameters, loaded 986 examples without truncation or dropping, and",
            "then failed during the first long-sequence backward pass with CUDA",
            "out-of-memory. No adapter checkpoint or training metrics were emitted.",
            "",
            "The retry reached peak whole-device use of "
            f"{summary['terminal_failure']['peak_device_memory_used_mib']} MiB on the",
            "24 GiB RTX 3090. The retained maximum sequence lengths are 5,027",
            "tokens for baseline, 5,086 for KG V1, and 5,007 for KG V2, so the",
            "remaining eight rows were not launched after the execution gate failed.",
            "",
            "## Decision",
            "",
            "Protocol v1 is immutable and failed its hardware execution gate. Do not",
            "lower the frozen sequence length, alter batch settings, discard the",
            "failed row, or start lower-budget runs under v1. Continuing requires a",
            "new versioned runtime-only protocol amendment, frozen before another",
            "attempt and applied identically to every system and seed.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> int:
    matrix_rows = [
        row
        for row in load_jsonl(args.matrix)
        if int(row.get("budget_documents", 0)) == args.budget
    ]
    runs = [run_status(row) for row in matrix_rows]
    counts = Counter(row["status"] for row in runs)
    terminal_attempts = [
        attempt
        for row in runs
        for attempt in row["attempts"]
        if attempt.get("terminal_under_frozen_retry_policy")
    ]
    gate_status = (
        "passed"
        if runs and counts["complete"] == len(runs)
        else "failed_terminal"
        if terminal_attempts
        else "incomplete"
    )
    summary = {
        "protocol_id": "low-resource-provenance-study-v1",
        "gate": "100-document-three-seed-three-trainable-system",
        "gate_status": gate_status,
        "formal_test_read": False,
        "validation_metrics_read": False,
        "matrix": str(args.matrix),
        "matrix_sha256": sha256_file(args.matrix),
        "planned_runs": len(runs),
        "status_counts": dict(sorted(counts.items())),
        "attempts_recorded": sum(
            attempt["status"] != "not_started"
            for row in runs
            for attempt in row["attempts"]
        ),
        "terminal_failure": terminal_attempts[0] if terminal_attempts else None,
        "remaining_rows_launched": sum(row["status"] != "not_started" for row in runs) - 1
        if terminal_attempts
        else None,
        "decision": (
            "halt protocol v1; preserve failures; require a new versioned "
            "runtime-only protocol amendment before another attempt"
            if terminal_attempts
            else "continue only when all gate rows are complete"
        ),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(markdown(summary), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "runs"}, ensure_ascii=False, indent=2))
    return 0 if gate_status == "passed" else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path(
            "data/processed/experiments/formal/low_resource_manifests_v1/run_matrix.jsonl"
        ),
    )
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/experiments/formal/low_resource_v1/d100_execution_gate.json"
        ),
    )
    parser.add_argument(
        "--report", type=Path, default=Path("paper/D100_EXECUTION_GATE.md")
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
