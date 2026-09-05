#!/usr/bin/env python3
"""Calibrate GLiREL labels and thresholds on a deterministic train-only subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_gliner_glirel_validation as validation


DEFAULT_THRESHOLDS = (0.0, 0.01, 0.1, 0.2, 0.3, 0.5)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank_key(seed: int, job_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}:{job_id}".encode()).hexdigest()
    return digest, job_id


def index_annotations(
    jobs: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    index: list[dict[str, Any]],
    dataset: str,
) -> dict[str, dict[str, Any]]:
    job_ids = {str(job.get("job_id")) for job in jobs}
    if len(job_ids) != len(jobs):
        raise ValueError("train jobs contain missing or duplicate job IDs")
    result: dict[str, dict[str, Any]] = {}
    for row in index:
        job_id = row.get("job_id")
        record_index = row.get("record_index")
        if job_id not in job_ids or not isinstance(record_index, int):
            raise ValueError("train index references an unknown job or malformed record index")
        if not 0 <= record_index < len(gold) or job_id in result:
            raise ValueError("train index contains an out-of-range or duplicate record")
        annotation = gold[record_index]
        if not str(job_id).startswith(f"{dataset}_train_"):
            raise ValueError("calibration accepts train job IDs only")
        if annotation.get("document_id") != str(job_id).rsplit("_C", 1)[0]:
            raise ValueError("train index does not align with the gold document ID")
        result[str(job_id)] = annotation
    if set(result) != job_ids:
        raise ValueError("train index does not cover every train job exactly once")
    return result


def select_job_ids(
    jobs: list[dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
    relation_labels: list[str],
    max_jobs: int,
    seed: int,
) -> list[str]:
    if max_jobs < len(relation_labels):
        raise ValueError("max_jobs must be at least the number of relation labels")
    ranked = sorted((str(job["job_id"]) for job in jobs), key=lambda value: rank_key(seed, value))
    selected: set[str] = set()
    for label in relation_labels:
        candidates = [
            job_id
            for job_id in ranked
            if any(rel.get("type") == label for rel in annotations[job_id].get("relations", []))
        ]
        if not candidates:
            raise ValueError(f"train split has no example for relation label {label!r}")
        selected.add(candidates[0])
    for job_id in ranked:
        if len(selected) >= min(max_jobs, len(ranked)):
            break
        selected.add(job_id)
    return sorted(selected, key=lambda value: rank_key(seed, value))


def prepare_gold_entities(
    job: dict[str, Any], annotation: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    segments = {segment["segment_id"]: segment for segment in job["segments"]}
    prepared: dict[str, dict[str, Any]] = {}
    for segment_id, segment in segments.items():
        tokens, offsets = validation.tokenize_with_offsets(segment["text"])
        prepared[segment_id] = {
            "segment": segment,
            "tokens": tokens,
            "offsets": offsets,
            "entities": [],
            "ner": [],
            "by_span": {},
        }

    for entity in annotation.get("entities", []):
        evidence = entity.get("evidence")
        if not isinstance(evidence, dict) or evidence.get("segment_id") not in prepared:
            raise ValueError(f"entity {entity.get('id')} has no usable train-segment evidence")
        bucket = prepared[evidence["segment_id"]]
        segment = bucket["segment"]
        local_start = evidence.get("start", -1) - segment["start"]
        local_end = evidence.get("end", -1) - segment["start"]
        token_span = validation.token_span_for_chars(bucket["offsets"], local_start, local_end)
        if token_span is None or segment["text"][local_start:local_end] != entity.get("text"):
            raise ValueError(f"entity {entity.get('id')} is not aligned to exact token boundaries")
        start, end = token_span
        if token_span in bucket["by_span"]:
            raise ValueError("multiple gold entities share one token span")
        compact = {"id": entity["id"], "type": entity["type"], "text": entity["text"]}
        bucket["entities"].append(compact)
        bucket["ner"].append([start, end - 1, entity["type"], entity["text"]])
        bucket["by_span"][token_span] = compact
    return prepared


def predict_train_job(
    dataset: str,
    job: dict[str, Any],
    annotation: dict[str, Any],
    model: Any,
    label_mode: str,
) -> dict[str, Any]:
    ontology = job["ontology"]
    prepared = prepare_gold_entities(job, annotation)
    candidates: dict[tuple[str, str, str], float] = {}
    for bucket in prepared.values():
        entities = bucket["entities"]
        canonical_labels = validation.viable_relation_labels(entities, ontology)
        if len(entities) < 2 or not canonical_labels:
            continue
        prompt_labels, canonical_by_prompt = validation.relation_prompt_labels(
            dataset, canonical_labels, label_mode
        )
        raw = model.predict_relations(
            bucket["tokens"],
            prompt_labels,
            flat_ner=True,
            threshold=0.0,
            ner=bucket["ner"],
            top_k=-1,
        )
        for relation in raw:
            canonical = canonical_by_prompt.get(relation.get("label"))
            head_pos = validation._relation_position(relation.get("head_pos"))
            tail_pos = validation._relation_position(relation.get("tail_pos"))
            head = bucket["by_span"].get(head_pos) if head_pos else None
            tail = bucket["by_span"].get(tail_pos) if tail_pos else None
            if canonical is None or head is None or tail is None or head["id"] == tail["id"]:
                continue
            signature = ontology["allowed_relation_signatures"][canonical]
            if head["type"] not in signature["source"] or tail["type"] not in signature["target"]:
                continue
            key = (head["id"], canonical, tail["id"])
            candidates[key] = max(candidates.get(key, 0.0), float(relation["score"]))

    gold = sorted(
        (str(row["source_id"]), str(row["type"]), str(row["target_id"]))
        for row in annotation.get("relations", [])
    )
    serialized_candidates = [
        {"source_id": source, "type": label, "target_id": target, "score": score}
        for (source, label, target), score in sorted(candidates.items())
    ]
    return {
        "job_id": job["job_id"],
        "label_mode": label_mode,
        "gold_relations": [
            {"source_id": source, "type": label, "target_id": target}
            for source, label, target in gold
        ],
        "candidates": serialized_candidates,
    }


def predictions_at_threshold(
    row: dict[str, Any], threshold: float
) -> set[tuple[str, str, str]]:
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in row["candidates"]:
        if float(candidate["score"]) > threshold:
            by_pair[(candidate["source_id"], candidate["target_id"])].append(candidate)
    predictions: set[tuple[str, str, str]] = set()
    for (source, target), candidates in by_pair.items():
        best = min(candidates, key=lambda item: (-float(item["score"]), item["type"]))
        predictions.add((source, best["type"], target))
    return predictions


def score_rows(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    tp = fp = fn = predictions = gold_count = 0
    for row in rows:
        gold = {
            (item["source_id"], item["type"], item["target_id"])
            for item in row["gold_relations"]
        }
        predicted = predictions_at_threshold(row, threshold)
        tp += len(gold & predicted)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
        predictions += len(predicted)
        gold_count += len(gold)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "predictions": predictions,
        "gold_relations": gold_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def choose_configuration(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not metrics:
        raise ValueError("no calibration metrics are available")
    return max(
        metrics,
        key=lambda row: (
            row["f1"],
            row["recall"],
            row["precision"],
            -row["predictions"],
            row["threshold"],
            row["label_mode"] == "canonical",
        ),
    )


def load_glirel_model(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    os.environ["HF_HOME"] = str(args.hf_home.expanduser().resolve())
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    source = args.glirel_source.expanduser().resolve()
    sys.path.insert(0, str(source))

    checkpoint = validation.resolve_local_model(args.glirel_model, args.hf_home)
    config = json.loads((checkpoint / "glirel_config.json").read_text(encoding="utf-8"))
    backbone = validation.resolve_local_model(
        str(args.glirel_backbone or config["model_name"]), args.hf_home
    )
    validation.require_files(checkpoint, ("glirel_config.json",), validation.MODEL_WEIGHT_NAMES)
    validation.require_files(backbone, ("config.json",), validation.TOKENIZER_NAMES)

    import torch
    import transformers
    from glirel import GLiREL, __version__ as glirel_version

    with validation.patched_checkpoint(checkpoint, "glirel_config.json", backbone) as local:
        with validation.glirel_config_only_backbone(True):
            model = GLiREL.from_pretrained(
                str(local), local_files_only=True, map_location="cpu", strict=True
            )
    if args.dtype == "float16":
        model.half()
    elif args.dtype == "bfloat16":
        model.bfloat16()
    model.to(args.device)
    model.eval()
    return model, {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "glirel": glirel_version,
        "glirel_source": str(source),
        "checkpoint": str(checkpoint),
        "backbone": str(backbone),
        "device": args.device,
        "dtype": args.dtype,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=validation.SUPPORTED_DATASETS, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--max-jobs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS)
    )
    parser.add_argument(
        "--label-modes", nargs="+", choices=validation.RELATION_LABEL_MODES,
        default=list(validation.RELATION_LABEL_MODES),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="float32"
    )
    parser.add_argument(
        "--hf-home", type=Path,
        default=Path(os.environ.get("HF_HOME", "/ds2/xuelin/cache/huggingface")),
    )
    parser.add_argument("--glirel-model", default="jackboyla/glirel-large-v0")
    parser.add_argument("--glirel-backbone")
    parser.add_argument(
        "--glirel-source", type=Path, default=Path("tools/external-baselines/glirel")
    )
    parser.add_argument(
        "--compatibility-canary", type=Path,
        default=Path(
            "outputs/public_horizontal_validation/gliner_glirel_t0/compatibility_canary.json"
        ),
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.max_jobs < 1 or len(set(args.thresholds)) != len(args.thresholds):
        parser.error("--max-jobs must be positive and thresholds must be unique")
    if any(not 0.0 <= value <= 1.0 for value in args.thresholds):
        parser.error("thresholds must be between 0 and 1")
    if args.device == "cpu" and args.dtype == "float16":
        parser.error("float16 CPU inference is unsupported")
    return args


def run(args: argparse.Namespace) -> int:
    root = Path("data/processed/public_benchmarks_full") / args.dataset
    jobs_path = root / "train_baseline_jobs.jsonl"
    gold_path = root / "train_gold.jsonl"
    index_path = root / "train_index.jsonl"
    output_root = args.output_root or (
        Path("outputs/public_horizontal_validation/glirel_train_calibration") / args.dataset
    )
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    raw_path = output_root / "raw_scores.jsonl"
    result_path = output_root / "calibration.json"
    status_path = output_root / "status.json"

    canary = json.loads(args.compatibility_canary.read_text(encoding="utf-8"))
    if canary.get("status") != "passed" or canary.get("runtime_compatible") is not True:
        raise RuntimeError("GLiREL runtime compatibility canary has not passed")

    jobs = load_jsonl(jobs_path)
    gold = load_jsonl(gold_path)
    index = load_jsonl(index_path)
    by_id = {str(job["job_id"]): job for job in jobs}
    annotations = index_annotations(jobs, gold, index, args.dataset)
    ontology = jobs[0]["ontology"]
    if any(validation.canonical_digest(job["ontology"]) != validation.canonical_digest(ontology) for job in jobs):
        raise ValueError("train ontology changes within the calibration source")
    relation_labels = list(ontology["relation_types"])
    selected_ids = select_job_ids(
        jobs, annotations, relation_labels, args.max_jobs, args.seed
    )
    source_hashes = {
        str(path): file_sha256(path) for path in (jobs_path, gold_path, index_path)
    }
    protocol = {
        "schema_version": "glirel-train-calibration-v1",
        "dataset": args.dataset,
        "split": "train_inner_calibration",
        "validation_gold_read": False,
        "test_gold_read": False,
        "seed": args.seed,
        "max_jobs": args.max_jobs,
        "selected_job_ids": selected_ids,
        "label_modes": args.label_modes,
        "label_aliases": validation.NATURALIZED_RELATION_LABELS[args.dataset],
        "thresholds": args.thresholds,
        "gold_ner_oracle": True,
        "top_k_after_signature_filter": 1,
        "threshold_comparison": "strict_score_greater_than_threshold",
        "source_sha256": source_hashes,
        "compatibility_canary": str(args.compatibility_canary),
        "created_at": utc_now(),
    }
    fingerprint_payload = {key: value for key, value in protocol.items() if key != "created_at"}
    fingerprint = validation.canonical_digest(fingerprint_payload)
    protocol["fingerprint"] = fingerprint
    if raw_path.exists() and manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("fingerprint") != fingerprint:
            raise RuntimeError("existing raw calibration scores use a different protocol")
        protocol["created_at"] = previous.get("created_at", protocol["created_at"])
    write_json_atomic(manifest_path, protocol)
    write_json_atomic(
        status_path,
        {
            "status": "prepared" if args.prepare_only else "loading_model",
            "dataset": args.dataset,
            "fingerprint": fingerprint,
            "selected_jobs": len(selected_ids),
            "completed_mode_jobs": 0,
            "expected_mode_jobs": len(selected_ids) * len(args.label_modes),
            "updated_at": utc_now(),
        },
    )
    if args.prepare_only:
        return 0

    existing_rows = load_jsonl(raw_path) if raw_path.exists() else []
    for row in existing_rows:
        if row.get("fingerprint") != fingerprint:
            raise RuntimeError("raw calibration row has a mismatched protocol fingerprint")
    completed = {(row["label_mode"], row["job_id"]) for row in existing_rows}
    model, software = load_glirel_model(args)
    import torch

    started = time.monotonic()
    mode = "a" if raw_path.exists() else "w"
    with raw_path.open(mode, encoding="utf-8") as stream, torch.inference_mode():
        for label_mode in args.label_modes:
            for job_id in selected_ids:
                if (label_mode, job_id) in completed:
                    continue
                row = predict_train_job(
                    args.dataset, by_id[job_id], annotations[job_id], model, label_mode
                )
                row["fingerprint"] = fingerprint
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                existing_rows.append(row)
                completed.add((label_mode, job_id))
                write_json_atomic(
                    status_path,
                    {
                        "status": "running",
                        "dataset": args.dataset,
                        "active_label_mode": label_mode,
                        "active_job_id": job_id,
                        "fingerprint": fingerprint,
                        "selected_jobs": len(selected_ids),
                        "completed_mode_jobs": len(completed),
                        "expected_mode_jobs": len(selected_ids) * len(args.label_modes),
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "updated_at": utc_now(),
                    },
                )

    metrics: list[dict[str, Any]] = []
    for label_mode in args.label_modes:
        rows = [row for row in existing_rows if row["label_mode"] == label_mode]
        for threshold in args.thresholds:
            metrics.append(
                {"label_mode": label_mode, **score_rows(rows, float(threshold))}
            )
    selected = choose_configuration(metrics)
    result = {
        "schema_version": "glirel-train-calibration-result-v1",
        "status": "complete",
        "dataset": args.dataset,
        "split": "train_inner_calibration",
        "validation_gold_read": False,
        "test_gold_read": False,
        "fingerprint": fingerprint,
        "selected_jobs": len(selected_ids),
        "selected_configuration": selected,
        "has_positive_signal": selected["true_positives"] > 0,
        "metrics": metrics,
        "software": software,
        "manifest": str(manifest_path),
        "raw_scores": str(raw_path),
        "finished_at": utc_now(),
    }
    write_json_atomic(result_path, result)
    write_json_atomic(
        status_path,
        {
            "status": "complete",
            "dataset": args.dataset,
            "fingerprint": fingerprint,
            "selected_jobs": len(selected_ids),
            "completed_mode_jobs": len(completed),
            "expected_mode_jobs": len(selected_ids) * len(args.label_modes),
            "selected_configuration": selected,
            "result": str(result_path),
            "updated_at": utc_now(),
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
