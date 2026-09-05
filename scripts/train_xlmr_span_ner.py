#!/usr/bin/env python3
"""Train an XLM-R span-boundary NER candidate.

The candidate has independent token start/end heads and a typed span head.  A
document-level inner-dev split is used only to freeze the span score
threshold.  The final model is then refit on all training documents and run
against the supplied validation jobs.  This module intentionally never opens
the formal test artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

try:
    from train_xlmr_ner_baseline import (
        collect_segments,
        indexed_annotations,
        load_jsonl,
        write_jsonl,
    )
except ModuleNotFoundError:  # importlib-based tests load scripts from repo root
    from scripts.train_xlmr_ner_baseline import (
        collect_segments,
        indexed_annotations,
        load_jsonl,
        write_jsonl,
    )


def stable_document_split(
    jobs: list[dict[str, Any]], seed: int, dev_fraction: float
) -> tuple[set[str], set[str]]:
    """Return a deterministic document split, never splitting windows."""
    documents = sorted({str(job["document_id"]) for job in jobs})
    if len(documents) < 2:
        raise ValueError("at least two training documents are required")
    ranked = sorted(
        documents,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(),
    )
    dev_count = max(1, min(len(documents) - 1, round(len(documents) * dev_fraction)))
    dev = set(ranked[:dev_count])
    return set(documents) - dev, dev


def filter_documents(
    jobs: list[dict[str, Any]], documents: set[str]
) -> list[dict[str, Any]]:
    return [job for job in jobs if str(job.get("document_id")) in documents]


def align_span_to_tokens(
    offsets: list[tuple[int, int]],
    start: int,
    end: int,
    entity_type: str,
    type_to_id: dict[str, int],
    max_span_tokens: int,
) -> tuple[int, int, int] | None:
    """Align a gold character interval to exact fast-tokenizer boundaries."""
    starts = [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_end > token_start and token_start == start
    ]
    ends = [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_end > token_start and token_end == end
    ]
    if not starts or not ends:
        return None
    token_start = starts[0]
    token_end = ends[-1]
    if token_end < token_start or token_end - token_start + 1 > max_span_tokens:
        return None
    if any(
        offsets[index][0] < start or offsets[index][1] > end
        for index in range(token_start, token_end + 1)
        if offsets[index][1] > offsets[index][0]
    ):
        return None
    return token_start, token_end, type_to_id[entity_type]


def _sample_negative_pairs(
    valid_tokens: list[int],
    positive_keys: set[tuple[int, int]],
    max_span_tokens: int,
    count: int,
    rng: random.Random,
) -> list[tuple[int, int, int]]:
    """Sample valid (start, width) pairs without materialising all pairs."""
    if not valid_tokens or count <= 0:
        return []
    valid = set(valid_tokens)
    result: list[tuple[int, int, int]] = []
    seen = set(positive_keys)
    attempts = 0
    max_attempts = max(100, count * 40)
    while len(result) < count and attempts < max_attempts:
        attempts += 1
        start = rng.choice(valid_tokens)
        end = min(start + rng.randrange(max_span_tokens), valid_tokens[-1])
        if end not in valid or end < start:
            continue
        if any(index not in valid for index in range(start, end + 1)):
            continue
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)
        result.append((start, end - start + 1, 0))
    if len(result) >= count:
        return result
    # The fallback is deterministic and only runs for unusually short chunks.
    for start in valid_tokens:
        for end in valid_tokens:
            if end < start or end - start + 1 > max_span_tokens:
                continue
            if any(index not in valid for index in range(start, end + 1)):
                continue
            key = (start, end)
            if key in seen:
                continue
            seen.add(key)
            result.append((start, end - start + 1, 0))
            if len(result) >= count:
                return result
    return result


def encode_span_segments(
    segments: list[dict[str, Any]],
    tokenizer: Any,
    type_to_id: dict[str, int],
    max_length: int,
    stride: int,
    max_span_tokens: int,
    negative_ratio: float,
    negative_minimum: int,
    seed: int,
    include_labels: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Tokenize source segments and create sparse span-training targets."""
    rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "segments": len(segments),
        "chunks": 0,
        "positive_chunks": 0,
        "gold_mentions_seen": 0,
        "gold_mentions_aligned": 0,
        "gold_mentions_unaligned": 0,
        "gold_mentions_over_width": 0,
        "span_widths": Counter(),
        "sampled_negative_pairs": 0,
    }
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
            valid_tokens = [
                index
                for index, (start, end) in enumerate(offsets)
                if end > start and encoded["attention_mask"][chunk_index][index]
            ]
            positives: list[tuple[int, int, int]] = []
            seen_positive: set[tuple[int, int]] = set()
            if include_labels:
                for start, end, entity_type in row.get("mentions", []):
                    stats["gold_mentions_seen"] += 1
                    aligned = align_span_to_tokens(
                        offsets,
                        start,
                        end,
                        entity_type,
                        type_to_id,
                        max_span_tokens,
                    )
                    if aligned is None:
                        if end - start > max_span_tokens * 8:
                            stats["gold_mentions_over_width"] += 1
                        else:
                            stats["gold_mentions_unaligned"] += 1
                        continue
                    token_start, token_end, type_id = aligned
                    if token_start not in valid_tokens or token_end not in valid_tokens:
                        stats["gold_mentions_unaligned"] += 1
                        continue
                    key = (token_start, token_end)
                    if key in seen_positive:
                        continue
                    seen_positive.add(key)
                    positives.append((token_start, token_end - token_start + 1, type_id))
                    stats["gold_mentions_aligned"] += 1
                    stats["span_widths"][str(token_end - token_start + 1)] += 1
            negatives = _sample_negative_pairs(
                valid_tokens,
                seen_positive,
                max_span_tokens,
                max(negative_minimum, math.ceil(len(positives) * negative_ratio)),
                random.Random(seed + segment_index * 1009 + chunk_index),
            )
            item = {
                "segment_index": segment_index,
                "chunk_index": chunk_index,
                "input_ids": encoded["input_ids"][chunk_index],
                "attention_mask": encoded["attention_mask"][chunk_index],
                "offset_mapping": offsets,
                "span_pairs": positives + negatives if include_labels else [],
            }
            if include_labels:
                start_targets = [0] * len(offsets)
                end_targets = [0] * len(offsets)
                for token_start, width, type_id in positives:
                    start_targets[token_start] = type_id
                    end_targets[token_start + width - 1] = type_id
                item["start_targets"] = start_targets
                item["end_targets"] = end_targets
            rows.append(item)
            stats["chunks"] += 1
            stats["positive_chunks"] += int(bool(positives))
            stats["sampled_negative_pairs"] += len(negatives)
    stats["span_widths"] = dict(stats["span_widths"])
    return rows, stats


class SpanDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),
            "start_targets": torch.tensor(row.get("start_targets", []), dtype=torch.long),
            "end_targets": torch.tensor(row.get("end_targets", []), dtype=torch.long),
            "span_pairs": row.get("span_pairs", []),
            "row_index": index,
        }


def span_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "start_targets": torch.stack([item["start_targets"] for item in batch]),
        "end_targets": torch.stack([item["end_targets"] for item in batch]),
        "span_pairs": [item["span_pairs"] for item in batch],
        "row_index": [item["row_index"] for item in batch],
    }


class SpanBoundaryModel(nn.Module):
    """XLM-R encoder with start, end, and typed span scoring heads."""

    def __init__(
        self,
        encoder: nn.Module,
        num_types: int,
        max_span_tokens: int,
        span_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        hidden_size = int(encoder.config.hidden_size)
        self.num_types = num_types
        self.max_span_tokens = max_span_tokens
        self.span_dim = span_dim
        self.start_head = nn.Linear(hidden_size, num_types + 1)
        self.end_head = nn.Linear(hidden_size, num_types + 1)
        self.dropout = nn.Dropout(dropout)
        self.start_projection = nn.Linear(hidden_size, span_dim)
        self.end_projection = nn.Linear(hidden_size, span_dim)
        self.width_embedding = nn.Embedding(max_span_tokens, span_dim)
        self.span_classifier = nn.Sequential(
            nn.Linear(span_dim, span_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(span_dim, num_types + 1),
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        hidden = self.dropout(hidden)
        start_logits = self.start_head(hidden)
        end_logits = self.end_head(hidden)
        start_repr = torch.tanh(self.start_projection(hidden))
        end_repr = torch.tanh(self.end_projection(hidden))
        sequence_length = hidden.shape[1]
        starts = torch.arange(sequence_length, device=hidden.device)
        widths = torch.arange(self.max_span_tokens, device=hidden.device)
        end_indices = starts[:, None] + widths[None, :]
        end_indices = end_indices.clamp_max(sequence_length - 1)
        gathered_end = end_repr[:, end_indices, :]
        pair_repr = torch.tanh(
            start_repr[:, :, None, :]
            + gathered_end
            + self.width_embedding.weight[None, None, :, :]
        )
        span_logits = self.span_classifier(pair_repr)
        return start_logits, end_logits, span_logits

    def save_head(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
            if not key.startswith("encoder.")
        }
        torch.save(
            {
                "state_dict": state,
                "num_types": self.num_types,
                "max_span_tokens": self.max_span_tokens,
                "span_dim": self.span_dim,
            },
            path,
        )

    def load_head(self, path: Path, map_location: str | torch.device = "cpu") -> None:
        payload = torch.load(path, map_location=map_location, weights_only=True)
        missing, unexpected = self.load_state_dict(payload["state_dict"], strict=False)
        missing = [key for key in missing if not key.startswith("encoder.")]
        if missing or unexpected:
            raise ValueError(f"span head state mismatch: missing={missing}, unexpected={unexpected}")


def configure_trainable_layers(model: SpanBoundaryModel, trainable_top_layers: int) -> dict[str, int]:
    if trainable_top_layers < 0:
        for parameter in model.parameters():
            parameter.requires_grad = True
    else:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
        for name, parameter in model.named_parameters():
            if not name.startswith("encoder."):
                parameter.requires_grad = True
        layers = getattr(getattr(model.encoder, "encoder", None), "layer", [])
        if trainable_top_layers > len(layers):
            raise ValueError(f"trainable_top_layers={trainable_top_layers} exceeds {len(layers)}")
        for layer in layers[-trainable_top_layers:] if trainable_top_layers else []:
            for parameter in layer.parameters():
                parameter.requires_grad = True
    return {
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def make_model(
    model_path: Path,
    num_types: int,
    max_span_tokens: int,
    span_dim: int,
    dropout: float,
) -> SpanBoundaryModel:
    from transformers import AutoModel

    encoder = AutoModel.from_pretrained(str(model_path), local_files_only=True)
    return SpanBoundaryModel(encoder, num_types, max_span_tokens, span_dim, dropout)


def train_epoch(
    model: SpanBoundaryModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    gradient_accumulation: int,
    max_grad_norm: float,
    endpoint_none_weight: float,
    span_none_weight: float,
    num_labels: int,
) -> dict[str, float]:
    model.train()
    endpoint_weights = torch.tensor(
        [endpoint_none_weight] + [1.0] * (num_labels - 1), device=device
    )
    span_weights = torch.tensor(
        [span_none_weight] + [1.0] * (num_labels - 1), device=device
    )
    optimizer.zero_grad(set_to_none=True)
    sums = Counter()
    for batch_index, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_targets = batch["start_targets"].to(device)
        end_targets = batch["end_targets"].to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            start_logits, end_logits, span_logits = model(input_ids, attention_mask)
            start_loss = F.cross_entropy(
                start_logits.reshape(-1, num_labels),
                start_targets.reshape(-1),
                weight=endpoint_weights,
            )
            end_loss = F.cross_entropy(
                end_logits.reshape(-1, num_labels),
                end_targets.reshape(-1),
                weight=endpoint_weights,
            )
            pair_logits: list[torch.Tensor] = []
            pair_targets: list[int] = []
            for row_index, pairs in enumerate(batch["span_pairs"]):
                for token_start, width, type_id in pairs:
                    pair_logits.append(span_logits[row_index, token_start, width - 1])
                    pair_targets.append(type_id)
            if pair_logits:
                span_loss = F.cross_entropy(
                    torch.stack(pair_logits),
                    torch.tensor(pair_targets, dtype=torch.long, device=device),
                    weight=span_weights,
                )
            else:
                span_loss = start_loss.new_zeros(())
            loss = start_loss + end_loss + span_loss
            scaled_loss = loss / gradient_accumulation
        scaled_loss.backward()
        sums["loss"] += float(loss.detach())
        sums["start_loss"] += float(start_loss.detach())
        sums["end_loss"] += float(end_loss.detach())
        sums["span_loss"] += float(span_loss.detach())
        should_step = (
            (batch_index + 1) % gradient_accumulation == 0
            or batch_index + 1 == len(loader)
        )
        if should_step:
            trainable = [p for p in model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
    count = max(len(loader), 1)
    return {key: value / count for key, value in sums.items()}


def _valid_pair_mask(offsets: list[tuple[int, int]], max_span_tokens: int) -> list[tuple[int, int, int]]:
    valid = [index for index, (start, end) in enumerate(offsets) if end > start]
    valid_set = set(valid)
    pairs = []
    for start in valid:
        for end in valid:
            if end < start or end - start + 1 > max_span_tokens:
                continue
            if any(index not in valid_set for index in range(start, end + 1)):
                continue
            pairs.append((start, end, end - start + 1))
    return pairs


def predict_segments(
    model: SpanBoundaryModel,
    encoded_rows: list[dict[str, Any]],
    type_names: list[str],
    device: torch.device,
    batch_size: int,
    score_threshold: float | None,
) -> dict[int, list[tuple[int, int, str, float]]]:
    """Decode all candidate widths and retain deterministic non-overlap spans."""
    candidates: dict[int, list[tuple[int, int, str, float]]] = defaultdict(list)
    model.eval()
    for offset in range(0, len(encoded_rows), batch_size):
        batch = encoded_rows[offset : offset + batch_size]
        input_ids = torch.tensor([row["input_ids"] for row in batch], dtype=torch.long, device=device)
        attention_mask = torch.tensor([row["attention_mask"] for row in batch], dtype=torch.long, device=device)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            start_logits, end_logits, span_logits = model(input_ids, attention_mask)
            start_probs = start_logits.float().softmax(-1)
            end_probs = end_logits.float().softmax(-1)
            span_probs = span_logits.float().softmax(-1)
            sequence_length = input_ids.shape[1]
            starts = torch.arange(sequence_length, device=device)
            widths = torch.arange(model.max_span_tokens, device=device)
            end_indices = (starts[:, None] + widths[None, :]).clamp_max(sequence_length - 1)
            endpoint_end = end_probs[:, end_indices, 1:]
            joint = span_probs[..., 1:] * torch.sqrt(
                start_probs[:, :, None, 1:] * endpoint_end.clamp_min(1e-9)
            )
            best_score, best_type = joint.max(-1)
        for row, scores, type_ids in zip(batch, best_score.cpu(), best_type.cpu()):
            row_candidates = []
            offsets_map = row["offset_mapping"]
            for start, end, width in _valid_pair_mask(offsets_map, model.max_span_tokens):
                score = float(scores[start, width - 1])
                if score_threshold is not None and score < score_threshold:
                    continue
                entity_type = type_names[int(type_ids[start, width - 1])]
                row_candidates.append(
                    (offsets_map[start][0], offsets_map[end][1], entity_type, score)
                )
            candidates[row["segment_index"]].extend(row_candidates)
    if score_threshold is None:
        return dict(candidates)
    return {
        segment_index: consolidate_span_predictions(
            [item for item in items if item[3] >= score_threshold]
        )
        for segment_index, items in candidates.items()
    }


def consolidate_span_predictions(
    candidates: list[tuple[int, int, str, float]]
) -> list[tuple[int, int, str, float]]:
    best: dict[tuple[int, int, str], float] = {}
    for start, end, entity_type, score in candidates:
        key = (start, end, entity_type)
        best[key] = max(score, best.get(key, float("-inf")))
    ranked = sorted(
        ((*key, score) for key, score in best.items()),
        key=lambda item: (-item[3], -(item[1] - item[0]), item[0], item[2]),
    )
    selected = []
    occupied: set[int] = set()
    for start, end, entity_type, score in ranked:
        positions = set(range(start, end))
        if not positions or positions & occupied:
            continue
        selected.append((start, end, entity_type, score))
        occupied.update(positions)
    return sorted(selected)


def span_prediction_rows(
    jobs: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    predicted: dict[int, list[tuple[int, int, str, float]]],
) -> list[dict[str, Any]]:
    index_by_key = {row["key"]: index for index, row in enumerate(segments)}
    outputs = []
    for job in jobs:
        best: dict[tuple[int, int, str], tuple[dict[str, Any], float]] = {}
        for segment in job.get("segments", []):
            key = (
                str(job.get("document_id", "")),
                str(segment.get("segment_id", "")),
                str(segment.get("text", "")),
            )
            segment_index = index_by_key[key]
            source_text = str(segment.get("text", ""))
            source_start = int(segment.get("start", 0))
            for local_start, local_end, entity_type, score in predicted.get(segment_index, []):
                mention = source_text[local_start:local_end].strip()
                if not mention:
                    continue
                leading = len(source_text[local_start:local_end]) - len(
                    source_text[local_start:local_end].lstrip()
                )
                trailing = len(source_text[local_start:local_end]) - len(
                    source_text[local_start:local_end].rstrip()
                )
                local_start += leading
                local_end -= trailing
                global_start = source_start + local_start
                global_end = source_start + local_end
                key = (global_start, global_end, entity_type)
                entity = {
                    "text": source_text[local_start:local_end],
                    "type": entity_type,
                    "evidence": [
                        {
                            "text": source_text,
                            "segment_id": segment.get("segment_id"),
                            "page": segment.get("page"),
                            "start": global_start,
                            "end": global_end,
                        }
                    ],
                    "confidence": round(score, 6),
                }
                if key not in best or score > best[key][1]:
                    best[key] = (entity, score)
        selected_global = consolidate_span_predictions(
            [(*key, score) for key, (_, score) in best.items()]
        )
        selected_keys = {
            (start, end, entity_type)
            for start, end, entity_type, _ in selected_global
        }
        entities = []
        selected_entities = [
            value
            for key, value in best.items()
            if key in selected_keys
        ]
        for entity, _ in sorted(
            selected_entities,
            key=lambda item: (
                item[0]["evidence"][0]["start"],
                item[0]["type"],
                item[0]["text"],
            ),
        ):
            entities.append({"id": f"E{len(entities) + 1}", **entity})
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
                        "notes": "XLM-R span-boundary entity candidate; entity stage only",
                    },
                },
            }
        )
    return outputs


def document_span_sets(
    segments: list[dict[str, Any]],
    predicted: dict[int, list[tuple[int, int, str, float]]] | None = None,
) -> dict[str, set[tuple[int, int, str]]]:
    result: dict[str, set[tuple[int, int, str]]] = defaultdict(set)
    document_candidates: dict[
        str, list[tuple[int, int, str, float]]
    ] = defaultdict(list)
    for index, row in enumerate(segments):
        base = int(row["segment"].get("start", 0))
        if predicted is None:
            for start, end, entity_type in row.get("mentions", []):
                result[str(row["document_id"])].add((base + start, base + end, entity_type))
        else:
            for start, end, entity_type, score in predicted.get(index, []):
                document_candidates[str(row["document_id"])].append(
                    (base + start, base + end, entity_type, score)
                )
    for document_id, candidates in document_candidates.items():
        result[document_id] = {
            (start, end, entity_type)
            for start, end, entity_type, _ in consolidate_span_predictions(candidates)
        }
    return result


def micro_span_metrics(
    gold: dict[str, set[tuple[int, int, str]]],
    predicted: dict[str, set[tuple[int, int, str]]],
) -> dict[str, int | float]:
    gold_set = {
        (document_id, *span)
        for document_id, spans in gold.items()
        for span in spans
    }
    predicted_set = {
        (document_id, *span)
        for document_id, spans in predicted.items()
        for span in spans
    }
    correct = len(gold_set & predicted_set)
    precision = correct / len(predicted_set) if predicted_set else 0.0
    recall = correct / len(gold_set) if gold_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "gold": len(gold_set),
        "predicted": len(predicted_set),
        "correct": correct,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def select_threshold(
    model: SpanBoundaryModel,
    encoded_rows: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    type_names: list[str],
    thresholds: list[float],
    device: torch.device,
    batch_size: int,
) -> tuple[float, list[dict[str, Any]]]:
    gold = document_span_sets(segments)
    results = []
    raw_predicted = predict_segments(
        model, encoded_rows, type_names, device, batch_size, None
    )
    for threshold in thresholds:
        predicted = {
            index: consolidate_span_predictions(
                [item for item in items if item[3] >= threshold]
            )
            for index, items in raw_predicted.items()
        }
        metrics = micro_span_metrics(gold, document_span_sets(segments, predicted))
        results.append({"threshold": threshold, **metrics})
    selected = max(results, key=lambda item: (item["f1"], item["precision"], item["threshold"]))
    return float(selected["threshold"]), results


def run(args: argparse.Namespace) -> int:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    ontology = yaml.safe_load(args.ontology.read_text(encoding="utf-8"))
    type_names = list(ontology["entity_types"])
    type_to_id = {name: index + 1 for index, name in enumerate(type_names)}
    train_gold = indexed_annotations(
        load_jsonl(args.train_gold), load_jsonl(args.train_index)
    )
    all_jobs = load_jsonl(args.train_jobs)
    validation_jobs = load_jsonl(args.validation_jobs)
    train_documents, inner_documents = stable_document_split(
        all_jobs, args.seed, args.inner_dev_fraction
    )
    inner_jobs = filter_documents(all_jobs, inner_documents)
    fit_jobs = filter_documents(all_jobs, train_documents)
    all_segments, _ = collect_segments(all_jobs, train_gold)
    fit_segments, _ = collect_segments(fit_jobs, {job_id: train_gold[job_id] for job_id in train_gold if any(j["job_id"] == job_id for j in fit_jobs)})
    inner_gold = {job_id: train_gold[job_id] for job_id in train_gold if any(j["job_id"] == job_id for j in inner_jobs)}
    inner_segments, _ = collect_segments(inner_jobs, inner_gold)
    validation_segments, _ = collect_segments(validation_jobs)

    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, use_fast=True)
    fit_rows, fit_stats = encode_span_segments(
        fit_segments, tokenizer, type_to_id, args.max_length, args.stride,
        args.max_span_tokens, args.negative_ratio, args.negative_minimum,
        args.seed, True,
    )
    inner_rows, inner_stats = encode_span_segments(
        inner_segments, tokenizer, type_to_id, args.max_length, args.stride,
        args.max_span_tokens, args.negative_ratio, args.negative_minimum,
        args.seed + 17, True,
    )
    validation_rows, validation_stats = encode_span_segments(
        validation_segments, tokenizer, type_to_id, args.max_length, args.stride,
        args.max_span_tokens, args.negative_ratio, args.negative_minimum,
        args.seed + 31, False,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.time()

    def fit_model(rows: list[dict[str, Any]], phase_seed: int) -> tuple[SpanBoundaryModel, list[dict[str, float]], dict[str, int]]:
        random.seed(phase_seed)
        torch.manual_seed(phase_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(phase_seed)
        model = make_model(args.model, len(type_names), args.max_span_tokens, args.span_dim, args.dropout)
        parameter_counts = configure_trainable_layers(model, args.trainable_top_layers)
        if args.gradient_checkpointing:
            model.encoder.gradient_checkpointing_enable()
        model.to(device)
        loader = DataLoader(
            SpanDataset(rows), batch_size=args.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(phase_seed), collate_fn=span_collate,
        )
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
        updates = math.ceil(len(loader) / args.gradient_accumulation)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, max(1, round(updates * args.epochs * args.warmup_ratio)), updates * args.epochs
        )
        losses = []
        for epoch in range(args.epochs):
            loss = train_epoch(
                model, loader, optimizer, scheduler, device, args.gradient_accumulation,
                args.max_grad_norm, args.endpoint_none_weight, args.span_none_weight,
                len(type_names) + 1,
            )
            loss["epoch"] = epoch + 1
            losses.append(loss)
            print(json.dumps({"phase": "inner" if rows is fit_rows else "final", **loss}, ensure_ascii=False), flush=True)
        return model, losses, parameter_counts

    inner_model, inner_losses, inner_parameter_counts = fit_model(fit_rows, args.seed)
    thresholds = [float(value) for value in args.thresholds.split(",") if value.strip()]
    selected_threshold, threshold_results = select_threshold(
        inner_model, inner_rows, inner_segments, type_names, thresholds, device, args.inference_batch_size
    )
    del inner_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    final_rows, final_stats = encode_span_segments(
        all_segments, tokenizer, type_to_id, args.max_length, args.stride,
        args.max_span_tokens, args.negative_ratio, args.negative_minimum,
        args.seed + 101, True,
    )
    final_model, final_losses, parameter_counts = fit_model(final_rows, args.seed + 101)
    predicted = predict_segments(
        final_model, validation_rows, type_names, device, args.inference_batch_size, selected_threshold
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    outputs = span_prediction_rows(validation_jobs, validation_segments, predicted)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, outputs)
    args.model_output.mkdir(parents=True, exist_ok=True)
    final_model.encoder.save_pretrained(args.model_output)
    tokenizer.save_pretrained(args.model_output)
    final_model.save_head(args.model_output / "span_head.pt")
    (args.model_output / "span_config.json").write_text(
        json.dumps({
            "entity_types": type_names,
            "max_span_tokens": args.max_span_tokens,
            "span_dim": args.span_dim,
            "dropout": args.dropout,
            "score_threshold": selected_threshold,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    split_manifest = {
        "seed": args.seed,
        "dev_fraction": args.inner_dev_fraction,
        "fit_documents": sorted(train_documents),
        "inner_dev_documents": sorted(inner_documents),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    (args.summary.parent / "inner_dev_split.json").write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "version": "xlm-roberta-large-span-boundary-1.0",
        "stage": "entity-only",
        "formal_test_read": False,
        "seed": args.seed,
        "model": str(args.model),
        "train_jobs": len(all_jobs),
        "train_documents": len({job["document_id"] for job in all_jobs}),
        "fit_documents": len(train_documents),
        "inner_dev_documents": len(inner_documents),
        "fit_segments": len(fit_segments),
        "fit_chunks": len(fit_rows),
        "fit_stats": fit_stats,
        "final_segments": len(all_segments),
        "final_chunks": len(final_rows),
        "final_stats": final_stats,
        "inner_segments": len(inner_segments),
        "inner_chunks": len(inner_rows),
        "inner_stats": inner_stats,
        "validation_jobs": len(validation_jobs),
        "validation_segments": len(validation_segments),
        "validation_chunks": len(validation_rows),
        "validation_stats": validation_stats,
        "max_span_tokens": args.max_span_tokens,
        "span_dim": args.span_dim,
        "negative_ratio": args.negative_ratio,
        "negative_minimum": args.negative_minimum,
        "endpoint_none_weight": args.endpoint_none_weight,
        "span_none_weight": args.span_none_weight,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "inference_batch_size": args.inference_batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "stride": args.stride,
        "trainable_top_layers": args.trainable_top_layers,
        "inner_threshold_candidates": thresholds,
        "selected_score_threshold": selected_threshold,
        "inner_threshold_results": threshold_results,
        "inner_parameter_counts": inner_parameter_counts,
        "parameter_counts": parameter_counts,
        "inner_epoch_losses": inner_losses,
        "epoch_losses": final_losses,
        "predictions": str(args.output),
        "predicted_entities": sum(
            len(row["annotation"]["entities"]) for row in outputs
        ),
        "model_output": str(args.model_output),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "peak_cuda_memory_allocated_mib": round(
            torch.cuda.max_memory_allocated(device) / (1024**2), 2
        ) if device.type == "cuda" else None,
        "peak_cuda_memory_reserved_mib": round(
            torch.cuda.max_memory_reserved(device) / (1024**2), 2
        ) if device.type == "cuda" else None,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--stride", type=int, default=48)
    parser.add_argument("--max-span-tokens", type=int, default=64)
    parser.add_argument("--span-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--negative-ratio", type=float, default=4.0)
    parser.add_argument("--negative-minimum", type=int, default=8)
    parser.add_argument("--endpoint-none-weight", type=float, default=0.25)
    parser.add_argument("--span-none-weight", type=float, default=1.0)
    parser.add_argument("--inner-dev-fraction", type=float, default=0.2)
    parser.add_argument(
        "--thresholds",
        default="0.005,0.0075,0.01,0.015,0.02,0.03,0.05,0.075,0.1,0.15,0.2,0.3,0.4,0.5",
    )
    parser.add_argument("--trainable-top-layers", type=int, default=-1)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
