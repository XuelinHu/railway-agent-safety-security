#!/usr/bin/env python3
"""Preflight and execute one immutable row from the low-resource run matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def query_gpu() -> dict[str, Any] | None:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
        "--id=0",
    ]
    try:
        row = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=10
        ).stdout.strip().splitlines()[0]
        fields = [value.strip() for value in row.split(",")]
        return {
            "timestamp_utc": utc_now(),
            "index": int(fields[0]),
            "name": fields[1],
            "uuid": fields[2],
            "memory_total_mib": float(fields[3]),
            "memory_used_mib": float(fields[4]),
            "utilization_percent": float(fields[5]),
            "power_draw_watts": float(fields[6]),
            "temperature_celsius": float(fields[7]),
        }
    except (FileNotFoundError, IndexError, ValueError, subprocess.SubprocessError):
        return None


class HardwareMonitor:
    def __init__(self, path: Path, interval_seconds: float = 2.0):
        self.path = path
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as stream:
            while not self.stop_event.is_set():
                sample = query_gpu()
                if sample:
                    self.samples.append(sample)
                    stream.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    stream.flush()
                self.stop_event.wait(self.interval_seconds)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=max(10.0, self.interval_seconds * 2))

    def summary(self, wall_clock_seconds: float) -> dict[str, Any]:
        powers = [float(row["power_draw_watts"]) for row in self.samples]
        energy_kwh = (
            sum(powers) / len(powers) * wall_clock_seconds / 3_600_000
            if powers
            else None
        )
        electricity_cny_per_kwh = 0.60
        usd_per_cny = 0.14
        return {
            "sample_interval_seconds": self.interval_seconds,
            "samples": len(self.samples),
            "gpu": self.samples[0] if self.samples else None,
            "peak_device_memory_used_mib": max(
                (float(row["memory_used_mib"]) for row in self.samples), default=None
            ),
            "peak_gpu_utilization_percent": max(
                (float(row["utilization_percent"]) for row in self.samples),
                default=None,
            ),
            "average_power_draw_watts": (
                round(sum(powers) / len(powers), 3) if powers else None
            ),
            "peak_power_draw_watts": max(powers, default=None),
            "estimated_energy_kwh": round(energy_kwh, 6)
            if energy_kwh is not None
            else None,
            "estimated_electricity_cost_cny": round(
                energy_kwh * electricity_cny_per_kwh, 6
            )
            if energy_kwh is not None
            else None,
            "estimated_electricity_cost_usd": round(
                energy_kwh * electricity_cny_per_kwh * usd_per_cny, 6
            )
            if energy_kwh is not None
            else None,
            "cost_assumption": {
                "scope": "electricity only; excludes hardware amortization and labor",
                "electricity_cny_per_kwh": electricity_cny_per_kwh,
                "usd_per_cny": usd_per_cny,
                "frozen_at": "2026-08-30",
            },
        }


def find_run(matrix: Path, run_id: str) -> dict[str, Any]:
    matches = [row for row in load_jsonl(matrix) if row.get("run_id") == run_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one matrix row for {run_id}, got {len(matches)}")
    return matches[0]


def verify_protocol(protocol_path: Path, protocol: dict[str, Any]) -> None:
    if protocol.get("status") != "frozen-pre-training":
        raise ValueError("protocol is not frozen-pre-training")
    if protocol.get("formal_test_status") != "sealed":
        raise ValueError("formal test must remain sealed")
    for relative_path, expected in protocol.get("implementation_sha256", {}).items():
        path = Path(relative_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"frozen implementation hash mismatch for {path}: {actual} != {expected}"
            )
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)


def validate_inputs(row: dict[str, Any]) -> dict[str, str]:
    paths = {
        "document_manifest": Path(row["document_manifest"]),
        **{key: Path(value) for key, value in row["training_inputs"].items()},
        "validation_jobs": Path(row["validation_jobs"]),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label}: {path}")
        lower_parts = [part.casefold() for part in path.parts]
        if any(
            part == "test" or "windowed_test" in part or "test_" in part
            for part in lower_parts
        ):
            raise ValueError(f"formal-test path refused: {path}")
    actual_manifest_hash = sha256_file(paths["document_manifest"])
    if actual_manifest_hash != row["document_manifest_sha256"]:
        raise ValueError("document manifest hash mismatch")
    index = load_jsonl(paths["index"])
    jobs = load_jsonl(paths["jobs"])
    gold = load_jsonl(paths["gold"])
    if len(index) != len(gold):
        raise ValueError("training index/gold counts differ")
    if {item["job_id"] for item in index} != {item["job_id"] for item in jobs}:
        raise ValueError("training index/job IDs differ")
    if any(item.get("split") != "train" for item in index):
        raise ValueError("training index contains non-train rows")
    validation_jobs = load_jsonl(paths["validation_jobs"])
    train_documents = {item["document_id"] for item in index}
    validation_documents = {item["document_id"] for item in validation_jobs}
    if train_documents & validation_documents:
        raise ValueError("training and validation documents overlap")
    return {label: sha256_file(path) for label, path in paths.items()}


def training_command(
    python: Path,
    row: dict[str, Any],
    protocol: dict[str, Any],
    output: Path,
) -> list[str]:
    training = protocol["training"]
    command = [
        str(python),
        str(training["trainer"]),
        "--model-path",
        str(protocol["model"]["local_snapshot"]),
        "--gold",
        str(row["training_inputs"]["gold"]),
        "--index",
        str(row["training_inputs"]["index"]),
        "--jobs",
        str(row["training_inputs"]["jobs"]),
        "--output",
        str(output),
        "--epochs",
        str(training["epochs"]),
        "--batch-size",
        str(training["batch_size"]),
        "--gradient-accumulation",
        str(training["gradient_accumulation"]),
        "--learning-rate",
        str(training["learning_rate"]),
        "--max-length",
        str(training["max_length"]),
        "--lora-rank",
        str(training["lora_rank"]),
        "--lora-alpha",
        str(training["lora_alpha"]),
        "--seed",
        str(row["seed"]),
        "--length-bucket-size",
        str(training["length_bucket_size"]),
    ]
    if training.get("bucket_by_length"):
        command.append("--bucket-by-length")
    if training.get("compact_target"):
        command.append("--compact-target")
    if training.get("skip_overlength"):
        command.append("--skip-overlength")
    system_config = next(
        item
        for item in protocol["systems"]["trainable"]
        if item["id"] == row["system"]
    )
    if system_config.get("use_job_instruction"):
        command.append("--use-job-instruction")
    return command


def environment_versions(python: Path) -> dict[str, Any]:
    source = (
        "import json,sys,torch,transformers,peft,accelerate,bitsandbytes;"
        "print(json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,"
        "'transformers':transformers.__version__,'peft':peft.__version__,"
        "'accelerate':accelerate.__version__,'bitsandbytes':bitsandbytes.__version__,"
        "'cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available()}))"
    )
    result = subprocess.run(
        [str(python), "-c", source], check=True, capture_output=True, text=True
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def output_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name not in {"run_manifest.json", "hardware_samples.jsonl"}
    }


def run(args: argparse.Namespace) -> int:
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    verify_protocol(args.protocol, protocol)
    row = find_run(args.matrix, args.run_id)
    input_hashes = validate_inputs(row)
    python = args.python.resolve()
    if not python.is_file():
        raise FileNotFoundError(python)
    versions = environment_versions(python)
    if not versions.get("cuda_available"):
        raise RuntimeError(f"training environment has no CUDA: {python}")
    base_output = Path(row["output_directory"])
    output = (
        Path(f"{base_output}_retry{args.retry_attempt:02d}")
        if args.retry_attempt
        else base_output
    )
    retry_record = None
    if args.retry_attempt:
        retry_manifest_path = base_output / "run_manifest.json"
        if not retry_manifest_path.is_file():
            raise ValueError("mechanical retry requires a preserved failed base attempt")
        retry_manifest = json.loads(retry_manifest_path.read_text(encoding="utf-8"))
        if retry_manifest.get("status") != "failed":
            raise ValueError("mechanical retry is allowed only after a failed base attempt")
        if (base_output / "training_metrics.json").exists() or (
            base_output / "adapter_config.json"
        ).exists():
            raise ValueError("failed base attempt produced model/training output; retry refused")
        retry_record = {
            "attempt": args.retry_attempt,
            "retry_of": str(base_output),
            "retry_of_manifest_sha256": sha256_file(retry_manifest_path),
            "eligibility": "infrastructure failure before model or training output",
            "generation_or_training_parameters_changed": False,
        }
    command = training_command(python, row, protocol, output)
    preflight = {
        "run_id": args.run_id,
        "attempt_id": (
            f"{args.run_id}_retry{args.retry_attempt:02d}"
            if args.retry_attempt
            else args.run_id
        ),
        "retry": retry_record,
        "status": "preflight_passed",
        "formal_test_read": False,
        "protocol": str(args.protocol),
        "protocol_sha256": sha256_file(args.protocol),
        "matrix": str(args.matrix),
        "matrix_sha256": sha256_file(args.matrix),
        "row": row,
        "input_sha256": input_hashes,
        "environment": {
            **versions,
            "python_executable": str(python),
            "platform": platform.platform(),
        },
        "gpu_preflight": query_gpu(),
        "command": command,
        "command_shell": shlex.join(command),
        "checked_at_utc": utc_now(),
    }
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"refusing non-empty immutable output directory: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    log_path = output / "training.log"
    samples_path = output / "hardware_samples.jsonl"
    manifest = {
        **preflight,
        "status": "running",
        "started_at_utc": utc_now(),
        "output_directory": str(output),
    }
    json_write(manifest_path, manifest)
    monitor = HardwareMonitor(samples_path, args.monitor_interval)
    started = time.perf_counter()
    monitor.start()
    env = os.environ.copy()
    env.update(
        {
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return_code = -1
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=Path.cwd(),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
            return_code = process.wait()
    finally:
        monitor.stop()
    wall_clock_seconds = time.perf_counter() - started
    telemetry = {
        "run_id": args.run_id,
        "wall_clock_seconds": round(wall_clock_seconds, 3),
        **monitor.summary(wall_clock_seconds),
    }
    metrics_path = output / "training_metrics.json"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        telemetry.update(
            {
                "train_examples": metrics.get("train_examples"),
                "prompt_tokens": metrics.get("prompt_tokens"),
                "target_tokens": metrics.get("target_tokens"),
                "examples_per_training_second": metrics.get(
                    "examples_per_training_second"
                ),
                "tokens_per_training_second": metrics.get(
                    "tokens_per_training_second"
                ),
                "peak_cuda_memory_allocated_mib": metrics.get(
                    "peak_cuda_memory_allocated_mib"
                ),
                "peak_cuda_memory_reserved_mib": metrics.get(
                    "peak_cuda_memory_reserved_mib"
                ),
            }
        )
    json_write(output / "telemetry.json", telemetry)
    manifest.update(
        {
            "status": "complete" if return_code == 0 else "failed",
            "return_code": return_code,
            "finished_at_utc": utc_now(),
            "wall_clock_seconds": round(wall_clock_seconds, 3),
            "output_sha256": output_hashes(output),
        }
    )
    json_write(manifest_path, manifest)
    if return_code:
        raise RuntimeError(f"training failed with return code {return_code}")
    print(json.dumps({"run_id": args.run_id, "status": "complete", **telemetry}, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/low_resource_protocol_v1.yaml")
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path(
            "data/processed/experiments/formal/low_resource_manifests_v1/run_matrix.jsonl"
        ),
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("/home/xuelin/miniconda3/envs/rc-llm-comet/bin/python"),
    )
    parser.add_argument("--monitor-interval", type=float, default=2.0)
    parser.add_argument("--retry-attempt", type=int, choices=[0, 1], default=0)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
