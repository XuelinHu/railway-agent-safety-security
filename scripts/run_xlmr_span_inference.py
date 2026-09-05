#!/usr/bin/env python3
"""Run a saved XLM-R span-boundary model on validation jobs only."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

try:
    from train_xlmr_span_ner import (
        SpanBoundaryModel,
        collect_segments,
        encode_span_segments,
        load_jsonl,
        predict_segments,
        span_prediction_rows,
        write_jsonl,
    )
except ModuleNotFoundError:
    from scripts.train_xlmr_span_ner import (
        SpanBoundaryModel,
        collect_segments,
        encode_span_segments,
        load_jsonl,
        predict_segments,
        span_prediction_rows,
        write_jsonl,
    )


def run(args: argparse.Namespace) -> int:
    config = json.loads((args.model_output / "span_config.json").read_text(encoding="utf-8"))
    type_names = list(config["entity_types"])
    type_to_id = {name: index + 1 for index, name in enumerate(type_names)}
    jobs = load_jsonl(args.validation_jobs)
    segments, _ = collect_segments(jobs)
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_output), local_files_only=True, use_fast=True)
    encoded, encode_stats = encode_span_segments(
        segments,
        tokenizer,
        type_to_id,
        args.max_length,
        args.stride,
        int(config["max_span_tokens"]),
        0.0,
        0,
        args.seed,
        False,
    )
    encoder = AutoModel.from_pretrained(str(args.model_output), local_files_only=True)
    model = SpanBoundaryModel(
        encoder,
        len(type_names),
        int(config["max_span_tokens"]),
        int(config["span_dim"]),
        float(config.get("dropout", 0.1)),
    )
    model.load_head(args.model_output / "span_head.pt", map_location="cpu")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    predicted = predict_segments(
        model,
        encoded,
        type_names,
        device,
        args.batch_size,
        float(config["score_threshold"]),
    )
    outputs = span_prediction_rows(jobs, segments, predicted)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.time() - started
    write_jsonl(args.output, outputs)
    summary = {
        "version": "xlm-roberta-large-span-boundary-inference-1.0",
        "stage": "validation-only-inference",
        "formal_test_read": False,
        "model_output": str(args.model_output),
        "validation_jobs": len(jobs),
        "validation_segments": len(segments),
        "validation_chunks": len(encoded),
        "encode_stats": encode_stats,
        "score_threshold": float(config["score_threshold"]),
        "predicted_entities": sum(len(row["annotation"]["entities"]) for row in outputs),
        "predictions": str(args.output),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "milliseconds_per_job": round(elapsed_seconds * 1000 / len(jobs), 3) if jobs else 0.0,
        "chunks_per_second": round(len(encoded) / elapsed_seconds, 3) if elapsed_seconds else 0.0,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "peak_cuda_memory_allocated_mib": round(
            torch.cuda.max_memory_allocated(device) / (1024**2), 2
        ) if device.type == "cuda" else None,
        "peak_cuda_memory_reserved_mib": round(
            torch.cuda.max_memory_reserved(device) / (1024**2), 2
        ) if device.type == "cuda" else None,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--validation-jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--stride", type=int, default=48)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
