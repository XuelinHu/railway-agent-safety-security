#!/usr/bin/env python3
"""Execute a v2 row after enforcing its allocator and full-sequence token gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

import run_low_resource_training as v1
from train_qlora import build_examples


def sequence_audit(
    row: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    system = next(
        item
        for item in protocol["systems"]["trainable"]
        if item["id"] == row["system"]
    )
    examples = build_examples(
        Path(row["training_inputs"]["gold"]),
        Path(row["training_inputs"]["index"]),
        Path(row["training_inputs"]["jobs"]),
        compact_target=bool(protocol["training"]["compact_target"]),
        use_job_instruction=bool(system.get("use_job_instruction")),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        protocol["model"]["local_snapshot"],
        local_files_only=True,
        trust_remote_code=True,
    )
    lengths = []
    for example in examples:
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": example["system"]},
                {"role": "user", "content": example["user"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_tokens = len(
            tokenizer(prompt, add_special_tokens=False)["input_ids"]
        )
        target_tokens = len(
            tokenizer(
                example["target"] + "<|im_end|>", add_special_tokens=False
            )["input_ids"]
        )
        lengths.append((prompt_tokens + target_tokens, prompt_tokens, target_tokens))
    max_length = int(protocol["training"]["max_length"])
    overlength = sum(total > max_length for total, _, _ in lengths)
    audit = {
        "examples": len(lengths),
        "max_sequence_tokens": max((row[0] for row in lengths), default=0),
        "max_prompt_tokens": max((row[1] for row in lengths), default=0),
        "max_target_tokens": max((row[2] for row in lengths), default=0),
        "hard_max_length": max_length,
        "overlength_examples": overlength,
        "truncation_or_skip_allowed": False,
    }
    if not lengths or overlength:
        raise ValueError(f"v2 full-sequence token gate failed: {audit}")
    return audit


def run(args: argparse.Namespace) -> int:
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    row = v1.find_run(args.matrix, args.run_id)
    if row.get("protocol_config") != str(args.protocol):
        raise ValueError("matrix row does not reference the selected v2 protocol")
    audit = sequence_audit(row, protocol)
    runtime_environment = {
        str(key): str(value)
        for key, value in protocol.get("runtime_environment", {}).items()
    }
    expected_allocator = "expandable_segments:True"
    if runtime_environment.get("PYTORCH_CUDA_ALLOC_CONF") != expected_allocator:
        raise ValueError("v2 allocator configuration is not frozen as expected")

    previous = {key: os.environ.get(key) for key in runtime_environment}
    os.environ.update(runtime_environment)
    original_versions = v1.environment_versions

    def audited_versions(python: Path) -> dict[str, Any]:
        values = original_versions(python)
        values["runtime_environment"] = runtime_environment
        values["sequence_audit"] = audit
        values["sequence_audit_sha256"] = hashlib.sha256(
            json.dumps(audit, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return values

    v1.environment_versions = audited_versions
    try:
        print(json.dumps({"v2_sequence_audit": audit}, indent=2), flush=True)
        return v1.run(args)
    finally:
        v1.environment_versions = original_versions
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/low_resource_protocol_v2.yaml"),
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path(
            "data/processed/experiments/formal/low_resource_manifests_v2/run_matrix.jsonl"
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
