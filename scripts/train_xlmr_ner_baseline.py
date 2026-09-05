#!/usr/bin/env python3
"""Train a local XLM-R token-classification NER baseline.

Gold is read only for the training split.  Validation inference consumes jobs
without annotations and emits the same JSONL annotation envelope as the
generative systems.  Source segments are deduplicated before training so
overlapping LLM windows do not silently multiply examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

try:
    from xlmr_crf import LinearChainCRF
except ModuleNotFoundError:  # importlib-based tests load scripts from the repo root
    from scripts.xlmr_crf import LinearChainCRF


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def indexed_annotations(
    annotations: list[dict[str, Any]], index: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    return {
        row["job_id"]: annotations[row["record_index"]]
        for row in index
    }


def find_all_spans(text: str, mention: str) -> list[tuple[int, int]]:
    """Find exact mentions, falling back to case-insensitive matching."""
    if not mention:
        return []
    spans = []
    offset = 0
    while True:
        start = text.find(mention, offset)
        if start < 0:
            break
        spans.append((start, start + len(mention)))
        offset = start + max(len(mention), 1)
    if spans:
        return spans
    folded_text = text.casefold()
    folded_mention = mention.casefold()
    offset = 0
    while True:
        start = folded_text.find(folded_mention, offset)
        if start < 0:
            break
        spans.append((start, start + len(folded_mention)))
        offset = start + max(len(folded_mention), 1)
    return spans


def select_non_overlapping(
    candidates: list[tuple[int, int, str]]
) -> list[tuple[int, int, str]]:
    """Prefer longer mentions, then restore source order."""
    selected = []
    occupied: set[int] = set()
    for start, end, entity_type in sorted(
        candidates, key=lambda item: (-(item[1] - item[0]), item[0], item[2])
    ):
        positions = set(range(start, end))
        if not positions or positions & occupied:
            continue
        selected.append((start, end, entity_type))
        occupied.update(positions)
    return sorted(selected)


def collect_segments(
    jobs: list[dict[str, Any]],
    annotations: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]]]:
    """Return unique source segments and their unioned training mentions."""
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for job in jobs:
        job_id = job["job_id"]
        annotation = annotations.get(job_id, {}) if annotations else {}
        entities = annotation.get("entities", [])
        for segment in job.get("segments", []):
            key = (
                str(job.get("document_id", "")),
                str(segment.get("segment_id", "")),
                str(segment.get("text", "")),
            )
            row = by_key.setdefault(
                key,
                {
                    "key": key,
                    "document_id": job.get("document_id"),
                    "language": job.get("language", "unknown"),
                    "segment": segment,
                    "mentions": set(),
                },
            )
            text = str(segment.get("text", ""))
            for entity in entities:
                mention = str(entity.get("text", ""))
                entity_type = str(entity.get("type", ""))
                for start, end in find_all_spans(text, mention):
                    row["mentions"].add((start, end, entity_type))
    rows = []
    for row in by_key.values():
        rows.append(
            {
                **row,
                "mentions": select_non_overlapping(list(row["mentions"])),
            }
        )
    return rows, by_key


def label_names(entity_types: list[str]) -> list[str]:
    return ["O"] + [
        label
        for entity_type in entity_types
        for label in (f"B-{entity_type}", f"I-{entity_type}")
    ]


def label_for_offset(
    start: int,
    end: int,
    mentions: list[tuple[int, int, str]],
    label_to_id: dict[str, int],
) -> int:
    for mention_start, mention_end, entity_type in mentions:
        if start >= mention_start and end <= mention_end and end > start:
            prefix = "B" if start == mention_start else "I"
            return label_to_id[f"{prefix}-{entity_type}"]
    return label_to_id["O"]


def repair_bio_labels(
    labels: list[int], label_to_id: dict[str, int]
) -> tuple[list[int], int, dict[str, int]]:
    """Repair chunk-initial or interrupted ``I-X`` labels to ``B-X``.

    Special/padding positions are represented by ``-100`` and do not count as
    a preceding valid token.  The repair is intentionally local to each
    tokenizer chunk, since a chunk boundary cannot safely inherit a tag from a
    different overlapping window.
    """
    id_to_label = {value: key for key, value in label_to_id.items()}
    repaired = list(labels)
    previous_label: str | None = None
    counts: Counter[str] = Counter()
    for index, label_id in enumerate(labels):
        if label_id < 0:
            continue
        label = id_to_label.get(label_id)
        if label is None:
            previous_label = None
            continue
        if label.startswith("I-"):
            entity_type = label[2:]
            if previous_label not in {f"B-{entity_type}", f"I-{entity_type}"}:
                repaired[index] = label_to_id[f"B-{entity_type}"]
                counts[entity_type] += 1
                label = f"B-{entity_type}"
        previous_label = label
    return repaired, sum(counts.values()), dict(counts)


def encode_segments(
    segments: list[dict[str, Any]],
    tokenizer: Any,
    label_to_id: dict[str, int],
    max_length: int,
    stride: int,
    include_labels: bool,
    repair_stats: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    encoded_rows = []
    for segment_index, row in enumerate(segments):
        text = str(row["segment"].get("text", ""))
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        for chunk_index in range(len(encoded["input_ids"])):
            offsets = [tuple(item) for item in encoded["offset_mapping"][chunk_index]]
            item = {
                "segment_index": segment_index,
                "chunk_index": chunk_index,
                "input_ids": encoded["input_ids"][chunk_index],
                "attention_mask": encoded["attention_mask"][chunk_index],
                "offset_mapping": offsets,
            }
            if include_labels:
                labels = [
                    -100
                    if end <= start
                    else label_for_offset(
                        start, end, row["mentions"], label_to_id
                    )
                    for start, end in offsets
                ]
                labels, repair_count, by_type = repair_bio_labels(labels, label_to_id)
                item["labels"] = labels
                if repair_stats is not None:
                    repair_stats["repaired_i_to_b_count"] = (
                        int(repair_stats.get("repaired_i_to_b_count", 0))
                        + repair_count
                    )
                    repair_stats["repaired_i_to_b_chunks"] = int(
                        repair_stats.get("repaired_i_to_b_chunks", 0)
                    ) + int(repair_count > 0)
                    aggregate = Counter(repair_stats.get("repaired_i_to_b_by_type", {}))
                    aggregate.update(by_type)
                    repair_stats["repaired_i_to_b_by_type"] = dict(aggregate)
            encoded_rows.append(item)
    return encoded_rows


class TokenDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        return {
            key: torch.tensor(row[key], dtype=torch.long)
            for key in ("input_ids", "attention_mask", "labels")
        }


def sample_training_chunks(
    rows: list[dict[str, Any]], negative_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    positive = [row for row in rows if any(label > 0 for label in row["labels"])]
    negative = [row for row in rows if not any(label > 0 for label in row["labels"])]
    limit = min(len(negative), math.ceil(len(positive) * negative_ratio))
    rng = random.Random(seed)
    rng.shuffle(negative)
    selected = positive + negative[:limit]
    rng.shuffle(selected)
    return selected, {
        "positive_chunks": len(positive),
        "available_negative_chunks": len(negative),
        "selected_negative_chunks": limit,
    }


def class_weights(
    rows: list[dict[str, Any]], num_labels: int, mode: str
) -> torch.Tensor:
    counts = Counter(
        label
        for row in rows
        for label in row["labels"]
        if label >= 0
    )
    if mode == "none":
        return torch.ones(num_labels)
    if mode == "binary_sqrt":
        outside = max(counts[0], 1)
        positive = max(sum(counts[label] for label in range(1, num_labels)), 1)
        outside_weight = math.sqrt(positive / outside)
        return torch.tensor([outside_weight] + [1.0] * (num_labels - 1))
    if mode != "per_label_sqrt":
        raise ValueError(f"unknown class-weighting mode: {mode}")
    total = sum(counts.values())
    weights = []
    for label in range(num_labels):
        count = max(counts[label], 1)
        weights.append(math.sqrt(total / (num_labels * count)))
    mean = sum(weights) / len(weights)
    return torch.tensor([min(value / mean, 8.0) for value in weights])


def configure_trainable_layers(model: Any, trainable_top_layers: int) -> dict[str, int]:
    """Freeze the encoder except for its last N layers; -1 trains all layers."""
    if trainable_top_layers < 0:
        for parameter in model.parameters():
            parameter.requires_grad = True
    else:
        for parameter in model.base_model.parameters():
            parameter.requires_grad = False
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
        layers = model.base_model.encoder.layer
        if trainable_top_layers > len(layers):
            raise ValueError(
                f"trainable_top_layers={trainable_top_layers} exceeds {len(layers)}"
            )
        if trainable_top_layers:
            for layer in layers[-trainable_top_layers:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True
    return {
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def decode_entities(
    logits: torch.Tensor,
    offsets: list[tuple[int, int]],
    id_to_label: dict[int, str],
) -> list[tuple[int, int, str, float]]:
    probabilities = logits.softmax(dim=-1)
    scores, predicted = probabilities.max(dim=-1)
    entities = []
    current: list[Any] | None = None
    for token_index, ((start, end), label_id) in enumerate(
        zip(offsets, predicted.tolist())
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
        score = float(scores[token_index])
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


def decode_label_path(
    label_ids: torch.Tensor,
    token_scores: torch.Tensor,
    offsets: list[tuple[int, int]],
    id_to_label: dict[int, str],
) -> list[tuple[int, int, str, float]]:
    """Decode a supplied BIO path using the same span rules as greedy logits."""
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


def consolidate_predictions(
    candidates: list[tuple[int, int, str, float]],
    span_selection: str = "confidence",
) -> list[tuple[int, int, str, float]]:
    if span_selection not in {"confidence", "longest"}:
        raise ValueError(f"unknown span-selection mode: {span_selection}")
    best = {}
    for start, end, entity_type, score in candidates:
        key = (start, end, entity_type)
        if key not in best or score > best[key]:
            best[key] = score
    selected = []
    occupied: set[int] = set()
    if span_selection == "confidence":
        rank_key = lambda item: (-item[3], -(item[1] - item[0]), item[0])
    else:
        # Overlapping tokenizer windows frequently emit a high-confidence
        # suffix of a longer mention at a chunk boundary.  The longest policy
        # is kept as an explicit validation ablation rather than silently
        # changing the baseline decoder.
        rank_key = lambda item: (-(item[1] - item[0]), -item[3], item[0])
    ranked = sorted(((*key, score) for key, score in best.items()), key=rank_key)
    for start, end, entity_type, score in ranked:
        positions = set(range(start, end))
        if not positions or positions & occupied:
            continue
        selected.append((start, end, entity_type, score))
        occupied.update(positions)
    return sorted(selected)


def predict_segments(
    model: Any,
    encoded_rows: list[dict[str, Any]],
    id_to_label: dict[int, str],
    device: torch.device,
    batch_size: int,
    decoder: str = "greedy",
    crf: LinearChainCRF | None = None,
) -> dict[int, list[tuple[int, int, str, float]]]:
    candidates: dict[int, list[tuple[int, int, str, float]]] = defaultdict(list)
    model.eval()
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
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits.float()
        for row, row_logits in zip(batch, logits.cpu()):
            if decoder == "crf":
                if crf is None:
                    raise ValueError("decoder='crf' requires a trained CRF")
                valid_mask = torch.tensor(
                    [end > start for start, end in row["offset_mapping"]],
                    dtype=torch.bool,
                    device=device,
                ).unsqueeze(0)
                paths = crf.decode(row_logits.unsqueeze(0).to(device), valid_mask)[0].cpu()
                probabilities = row_logits.softmax(dim=-1)
                scores = probabilities.gather(1, paths.unsqueeze(1)).squeeze(1)
                decoded = decode_label_path(
                    paths.cpu(), scores.cpu(), row["offset_mapping"], id_to_label
                )
            else:
                decoded = decode_entities(row_logits, row["offset_mapping"], id_to_label)
            candidates[row["segment_index"]].extend(decoded)
    return {
        index: consolidate_predictions(rows) for index, rows in candidates.items()
    }


def prediction_rows(
    jobs: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    predicted: dict[int, list[tuple[int, int, str, float]]],
    decoder: str = "greedy",
) -> list[dict[str, Any]]:
    index_by_key = {row["key"]: index for index, row in enumerate(segments)}
    outputs = []
    for job in jobs:
        entities = []
        seen = set()
        for segment in job.get("segments", []):
            key = (
                str(job.get("document_id", "")),
                str(segment.get("segment_id", "")),
                str(segment.get("text", "")),
            )
            segment_index = index_by_key[key]
            text = str(segment.get("text", ""))
            for local_start, local_end, entity_type, score in predicted.get(
                segment_index, []
            ):
                raw_mention = text[local_start:local_end]
                leading_space = len(raw_mention) - len(raw_mention.lstrip())
                trailing_space = len(raw_mention) - len(raw_mention.rstrip())
                local_start += leading_space
                local_end -= trailing_space
                mention = text[local_start:local_end]
                entity_key = (normalize(mention), entity_type)
                if not entity_key[0] or entity_key in seen:
                    continue
                seen.add(entity_key)
                global_start = int(segment.get("start", 0)) + local_start
                entities.append(
                    {
                        "id": f"E{len(entities) + 1}",
                        "text": mention,
                        "type": entity_type,
                        "evidence": [
                            {
                                "text": text,
                                "segment_id": segment.get("segment_id"),
                                "page": segment.get("page"),
                                "start": global_start,
                                "end": global_start + len(mention),
                            }
                        ],
                        "confidence": round(score, 6),
                        "review_status": "pending",
                        "created_by": (
                            "xlm-roberta-large-token-classifier-crf"
                            if decoder == "crf"
                            else "xlm-roberta-large-token-classifier"
                        ),
                    }
                )
        outputs.append(
            {
                "job_id": job["job_id"],
                "annotation": {
                    "schema_version": "0.1.0",
                    "document_id": job.get("document_id"),
                    "language": job.get("language", "unknown"),
                    "entities": entities,
                    "relations": [],
                    "review": {
                        "status": "unreviewed",
                        "reviewers": [],
                        "notes": "XLM-R large token-classification baseline; entity stage only",
                    },
                },
            }
        )
    return outputs


def run(args: argparse.Namespace) -> int:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ontology = yaml.safe_load(args.ontology.read_text(encoding="utf-8"))
    entity_types = list(ontology["entity_types"])
    labels = label_names(entity_types)
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}

    train_gold = indexed_annotations(
        load_jsonl(args.train_gold), load_jsonl(args.train_index)
    )
    train_jobs = load_jsonl(args.train_jobs)
    validation_jobs = load_jsonl(args.validation_jobs)
    train_segments, _ = collect_segments(train_jobs, train_gold)
    validation_segments, _ = collect_segments(validation_jobs)

    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model), local_files_only=True, use_fast=True
    )
    bio_repair_stats: dict[str, Any] = {}
    train_encoded = encode_segments(
        train_segments,
        tokenizer,
        label_to_id,
        args.max_length,
        args.stride,
        include_labels=True,
        repair_stats=bio_repair_stats,
    )
    selected, sampling = sample_training_chunks(
        train_encoded, args.negative_ratio, args.seed
    )
    weights = class_weights(selected, len(labels), args.class_weighting)

    model = AutoModelForTokenClassification.from_pretrained(
        str(args.model),
        local_files_only=True,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
        ignore_mismatched_sizes=True,
    )
    parameter_counts = configure_trainable_layers(model, args.trainable_top_layers)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    crf = LinearChainCRF(len(labels)).to(device) if args.decoder == "crf" else None
    loader = DataLoader(
        TokenDataset(selected),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ] + ([] if crf is None else list(crf.parameters()))
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    update_steps = math.ceil(len(loader) / args.gradient_accumulation)
    total_steps = update_steps * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, round(total_steps * args.warmup_ratio)),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    weights = weights.to(device)
    losses = []
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for batch_index, batch in enumerate(loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                ).logits
                if crf is None:
                    loss = F.cross_entropy(
                        logits.view(-1, len(labels)),
                        batch["labels"].view(-1),
                        weight=weights,
                        ignore_index=-100,
                    )
                else:
                    token_mask = batch["attention_mask"].bool() & batch["labels"].ne(-100)
                    safe_labels = batch["labels"].clamp_min(0)
                    loss = crf.neg_log_likelihood(logits, safe_labels, token_mask)
                scaled_loss = loss / args.gradient_accumulation
            scaler.scale(scaled_loss).backward()
            epoch_loss += float(loss.detach())
            should_step = (
                (batch_index + 1) % args.gradient_accumulation == 0
                or batch_index + 1 == len(loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        mean_loss = epoch_loss / max(len(loader), 1)
        losses.append(mean_loss)
        print(
            json.dumps(
                {"epoch": epoch + 1, "mean_loss": mean_loss}, ensure_ascii=False
            ),
            flush=True,
        )

    validation_encoded = encode_segments(
        validation_segments,
        tokenizer,
        label_to_id,
        args.max_length,
        args.stride,
        include_labels=False,
    )
    predicted = predict_segments(
        model,
        validation_encoded,
        id_to_label,
        device,
        args.inference_batch_size,
        decoder=args.decoder,
        crf=crf,
    )
    outputs = prediction_rows(
        validation_jobs, validation_segments, predicted, decoder=args.decoder
    )
    write_jsonl(args.output, outputs)

    args.model_output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.model_output)
    tokenizer.save_pretrained(args.model_output)
    crf_path = None
    if crf is not None:
        crf_path = args.crf_output or (args.model_output / "crf.pt")
        crf.save(crf_path)
    label_counts = Counter(
        labels[label]
        for row in selected
        for label in row["labels"]
        if label >= 0
    )
    summary = {
        "version": "xlm-roberta-large-token-classifier-1.0",
        "stage": "entity-only",
        "decoder": args.decoder,
        "formal_test_read": False,
        "seed": args.seed,
        "model": str(args.model),
        "model_sha256": sha256(args.model / "model.safetensors"),
        "train_jobs": len(train_jobs),
        "train_unique_segments": len(train_segments),
        "train_chunks": len(train_encoded),
        "selected_train_chunks": len(selected),
        "sampling": sampling,
        "label_counts": dict(label_counts),
        "bio_repairs": bio_repair_stats,
        "validation_jobs": len(validation_jobs),
        "validation_unique_segments": len(validation_segments),
        "validation_chunks": len(validation_encoded),
        "predicted_entities": sum(
            len(row["annotation"]["entities"]) for row in outputs
        ),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "stride": args.stride,
        "negative_ratio": args.negative_ratio,
        "class_weighting": args.class_weighting,
        "class_weights": [round(float(value), 6) for value in weights.cpu()],
        "trainable_top_layers": args.trainable_top_layers,
        **parameter_counts,
        "epoch_losses": losses,
        "elapsed_seconds": round(time.time() - started, 3),
        "predictions": str(args.output),
        "model_output": str(args.model_output),
        "crf_output": str(crf_path) if crf_path else None,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-gold", type=Path, required=True)
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--train-jobs", type=Path, required=True)
    parser.add_argument("--validation-jobs", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, default=Path("configs/risk_ontology.yaml"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--inference-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument(
        "--decoder",
        choices=("greedy", "crf"),
        default="greedy",
        help="Train with token cross-entropy or a learned linear-chain CRF.",
    )
    parser.add_argument(
        "--crf-output",
        type=Path,
        default=None,
        help="Separate CRF parameter file; defaults to <model-output>/crf.pt.",
    )
    parser.add_argument("--negative-ratio", type=float, default=2.0)
    parser.add_argument(
        "--class-weighting",
        choices=("binary_sqrt", "per_label_sqrt", "none"),
        default="binary_sqrt",
        help="Use train-only token counts; binary_sqrt avoids amplifying rare types independently.",
    )
    parser.add_argument(
        "--trainable-top-layers",
        type=int,
        default=4,
        help="Train the last N encoder layers plus the token head; -1 trains all layers.",
    )
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
