#!/usr/bin/env python3
"""Run validation-only XLM-R NER decoder ablations from a trained model.

The script never reads gold annotations.  It compares greedy token decoding
with a hard BIO-transition constraint and, when supplied, a learned CRF path;
it then consolidates overlapping tokenizer windows either by confidence or by
span length.  Gold-based scoring remains a separate explicit validation step.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml

from train_xlmr_ner_baseline import (
    collect_segments,
    consolidate_predictions,
    encode_segments,
    label_names,
    load_jsonl,
    prediction_rows,
    write_jsonl,
)

try:
    from xlmr_crf import LinearChainCRF
except ModuleNotFoundError:  # importlib-based tests load scripts from the repo root
    from scripts.xlmr_crf import LinearChainCRF


def bio_viterbi_paths(
    logits: torch.Tensor,
    offsets: list[list[tuple[int, int]]],
    id_to_label: dict[int, str],
) -> torch.Tensor:
    """Return highest-scoring paths that prohibit sentence-initial O-to-I."""
    batch_size, sequence_length, num_labels = logits.shape
    device = logits.device
    negative_infinity = torch.finfo(logits.dtype).min
    transition = torch.zeros((num_labels, num_labels), device=device)
    start_scores = torch.zeros(num_labels, device=device)
    for current_id, current_label in id_to_label.items():
        if not current_label.startswith("I-"):
            continue
        entity_type = current_label[2:]
        transition[:, current_id] = negative_infinity
        for previous_id, previous_label in id_to_label.items():
            if previous_label in {f"B-{entity_type}", f"I-{entity_type}"}:
                transition[previous_id, current_id] = 0.0
        start_scores[current_id] = negative_infinity

    emissions = logits.log_softmax(dim=-1)
    valid = torch.tensor(
        [[end > start for start, end in row] for row in offsets],
        dtype=torch.bool,
        device=device,
    )
    invalid = ~valid
    emissions = emissions.masked_fill(invalid.unsqueeze(-1), negative_infinity)
    emissions[:, :, 0] = torch.where(
        invalid, torch.zeros_like(emissions[:, :, 0]), emissions[:, :, 0]
    )

    scores = emissions[:, 0, :] + start_scores
    backpointers = []
    for token_index in range(1, sequence_length):
        candidate_scores = scores.unsqueeze(2) + transition.unsqueeze(0)
        best_scores, previous = candidate_scores.max(dim=1)
        scores = best_scores + emissions[:, token_index, :]
        backpointers.append(previous)

    paths = torch.zeros(
        (batch_size, sequence_length), dtype=torch.long, device=device
    )
    paths[:, -1] = scores.argmax(dim=-1)
    for token_index in range(sequence_length - 1, 0, -1):
        paths[:, token_index - 1] = backpointers[token_index - 1].gather(
            1, paths[:, token_index].unsqueeze(1)
        ).squeeze(1)
    return paths


def decode_path(
    label_ids: torch.Tensor,
    token_scores: torch.Tensor,
    offsets: list[tuple[int, int]],
    id_to_label: dict[int, str],
) -> list[tuple[int, int, str, float]]:
    entities = []
    current: list[Any] | None = None
    for token_index, ((start, end), label_id) in enumerate(
        zip(offsets, label_ids.tolist())
    ):
        if end <= start:
            continue
        label = id_to_label[label_id]
        if label == "O":
            if current:
                entities.append(tuple(current))
                current = None
            continue
        prefix, entity_type = label.split("-", 1)
        score = float(token_scores[token_index])
        if (
            current is None
            or prefix == "B"
            or current[2] != entity_type
            or start > current[1] + 1
        ):
            if current:
                entities.append(tuple(current))
            current = [start, end, entity_type, score]
        else:
            current[1] = end
            current[3] = min(current[3], score)
    if current:
        entities.append(tuple(current))
    return entities


def predict_variants(
    model: Any,
    encoded_rows: list[dict[str, Any]],
    id_to_label: dict[int, str],
    device: torch.device,
    batch_size: int,
    crf: LinearChainCRF | None = None,
) -> dict[str, dict[int, list[tuple[int, int, str, float]]]]:
    candidates = {
        "greedy": defaultdict(list),
        "bio_viterbi": defaultdict(list),
    }
    if crf is not None:
        candidates["learned_crf"] = defaultdict(list)
    model.eval()
    if crf is not None:
        crf.eval()
    for start in range(0, len(encoded_rows), batch_size):
        batch = encoded_rows[start : start + batch_size]
        input_ids = torch.tensor(
            [row["input_ids"] for row in batch], dtype=torch.long, device=device
        )
        attention_mask = torch.tensor(
            [row["attention_mask"] for row in batch],
            dtype=torch.long,
            device=device,
        )
        offsets = [row["offset_mapping"] for row in batch]
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits.float()
        probabilities = logits.softmax(dim=-1)
        greedy_paths = probabilities.argmax(dim=-1)
        viterbi_paths = bio_viterbi_paths(logits, offsets, id_to_label)
        paths_by_decoder = {
            "greedy": greedy_paths,
            "bio_viterbi": viterbi_paths,
        }
        if crf is not None:
            valid_mask = torch.tensor(
                [[end > start for start, end in row] for row in offsets],
                dtype=torch.bool,
                device=device,
            )
            paths_by_decoder["learned_crf"] = crf.decode(logits, valid_mask)
        for decoder, paths in paths_by_decoder.items():
            selected_scores = probabilities.gather(2, paths.unsqueeze(-1)).squeeze(-1)
            for row, path, scores in zip(batch, paths.cpu(), selected_scores.cpu()):
                candidates[decoder][row["segment_index"]].extend(
                    decode_path(path, scores, row["offset_mapping"], id_to_label)
                )

    variants = {}
    for decoder, by_segment in candidates.items():
        for span_selection in ("confidence", "longest"):
            name = f"{decoder}_{span_selection}"
            variants[name] = {
                segment_index: consolidate_predictions(rows, span_selection)
                for segment_index, rows in by_segment.items()
            }
    return variants


def run(args: argparse.Namespace) -> int:
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    ontology = yaml.safe_load(args.ontology.read_text(encoding="utf-8"))
    labels = label_names(list(ontology["entity_types"]))
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    jobs = load_jsonl(args.validation_jobs)
    segments, _ = collect_segments(jobs)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    encoded = encode_segments(
        segments,
        tokenizer,
        label_to_id,
        args.max_length,
        args.stride,
        include_labels=False,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        str(args.model), local_files_only=True
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    crf = None
    if args.crf:
        crf = LinearChainCRF.load(args.crf, map_location=device).to(device)
        if crf.num_labels != len(labels):
            raise ValueError(
                f"CRF has {crf.num_labels} labels but ontology defines {len(labels)}"
            )
    started = time.time()
    variants = predict_variants(
        model, encoded, id_to_label, device, args.batch_size, crf=crf
    )
    summary = {
        "stage": "validation-only-decoder-ablation",
        "formal_test_read": False,
        "model": str(args.model),
        "crf": str(args.crf) if args.crf else None,
        "validation_jobs": len(jobs),
        "validation_unique_segments": len(segments),
        "validation_chunks": len(encoded),
        "variants": {},
    }
    for name, predicted in variants.items():
        rows = prediction_rows(
            jobs,
            segments,
            predicted,
            decoder="crf" if name.startswith("learned_crf") else "greedy",
        )
        output = args.output_dir / f"{name}.jsonl"
        write_jsonl(output, rows)
        summary["variants"][name] = {
            "predictions": str(output),
            "predicted_entities": sum(
                len(row["annotation"]["entities"]) for row in rows
            ),
        }
    summary["elapsed_seconds"] = round(time.time() - started, 3)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--crf",
        type=Path,
        default=None,
        help="Optional separately saved learned CRF parameters.",
    )
    parser.add_argument("--validation-jobs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, default=Path("configs/risk_ontology.yaml"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--stride", type=int, default=48)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
