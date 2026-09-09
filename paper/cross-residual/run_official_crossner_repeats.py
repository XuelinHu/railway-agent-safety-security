#!/usr/bin/env python3
"""Train and evaluate score-residual NER on the official CrossNER BIO splits."""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from gliner import GLiNER
from transformers import set_seed

ROOT = Path(__file__).resolve().parents[2]
DOMAINS = ["ai", "literature", "music", "politics", "science"]
THRESHOLDS = (0.25, 0.35, 0.45, 0.55, 0.65)
ALPHAS = (0.75, 1.00, 1.25, 1.50)


def read_bio(path):
    sentences, tokens, tags = [], [], []
    for line in path.read_text(encoding="utf-8").splitlines() + [""]:
        if line.strip():
            token, tag = line.rsplit(maxsplit=1)
            tokens.append(token)
            tags.append(tag)
        elif tokens:
            offsets, cursor = [], 0
            for token in tokens:
                offsets.append((cursor, cursor + len(token)))
                cursor += len(token) + 1
            entities, start, entity_type = [], None, None
            for index, tag in enumerate(tags + ["O"]):
                prefix, current_type = (tag.split("-", 1) if "-" in tag else ("O", None))
                if start is not None and (prefix != "I" or current_type != entity_type):
                    entities.append((offsets[start][0], offsets[index - 1][1], entity_type))
                    start, entity_type = None, None
                if prefix == "B" or (prefix == "I" and start is None):
                    start, entity_type = index, current_type
            sentences.append({
                "sentence": " ".join(tokens),
                "tokens": tokens,
                "ner": [(next(i for i, offset in enumerate(offsets) if offset[0] == b),
                         next(i for i, offset in enumerate(offsets) if offset[1] == e), y)
                        for b, e, y in entities],
                "gold": set(entities),
            })
            tokens, tags = [], []
    return sentences


def label_inventory(*splits):
    return sorted({entity[2] for split in splits for row in split for entity in row["gold"]})


def training_records(rows):
    return [{"tokenized_text": row["tokens"], "ner": row["ner"]} for row in rows]


def prediction_cache(model, rows, labels):
    return [
        {(item["start"], item["end"], item["label"]): item["score"]
         for item in model.predict_entities(row["sentence"], labels, threshold=0.03)}
        for row in rows
    ]


def measure(rows, first, second=None, alpha=0.0, threshold=0.5):
    tp = fp = fn = 0
    for row, score0, score1 in zip(rows, first, second or first):
        keys = set(score0) | set(score1)
        predicted = {
            candidate for candidate in keys
            if score0.get(candidate, 0.0)
            + alpha * (score1.get(candidate, 0.0) - score0.get(candidate, 0.0)) >= threshold
        }
        gold = row["gold"]
        tp += len(predicted & gold)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def best_threshold(rows, cache):
    trials = [(threshold, measure(rows, cache, alpha=0.0, threshold=threshold))
              for threshold in THRESHOLDS]
    return max(trials, key=lambda item: item[1][2])


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


parser = argparse.ArgumentParser()
parser.add_argument("--data-root", type=Path, required=True,
                    help="Path to the official CrossNER ner_data directory")
parser.add_argument("--output", type=Path,
                    default=ROOT / "paper/cross-residual/results/official_crossner")
parser.add_argument("--seeds", nargs="+", type=int, default=[13, 29, 47])
parser.add_argument("--steps", type=int, default=300)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)

all_results, all_losses, start_time = [], [], time.time()
base = GLiNER.from_pretrained("urchade/gliner_small-v2.1", local_files_only=True).to("cuda")

for domain in DOMAINS:
    domain_root = args.data_root / domain
    train = read_bio(domain_root / "train.txt")
    validation = read_bio(domain_root / "dev.txt")
    test = read_bio(domain_root / "test.txt")
    labels = label_inventory(train, validation, test)
    base_validation = prediction_cache(base, validation, labels)
    base_test = prediction_cache(base, test, labels)
    base_threshold, _ = best_threshold(validation, base_validation)
    base_metrics = measure(test, base_test, alpha=0.0, threshold=base_threshold)

    for seed in args.seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        set_seed(seed)
        output_dir = args.output / "checkpoints" / f"seed_{seed}" / domain
        adapted = GLiNER.from_pretrained("urchade/gliner_small-v2.1", local_files_only=True)
        training_args = adapted.create_training_args(
            output_dir=str(output_dir), learning_rate=1e-5, others_lr=5e-5,
            max_steps=args.steps, per_device_train_batch_size=8,
            per_device_eval_batch_size=8, logging_steps=50, save_steps=args.steps,
            save_total_limit=1, bf16=True, report_to="none", seed=seed, data_seed=seed,
        )
        adapted.train_model(training_records(train), training_records(validation),
                            training_args=training_args)
        adapted.to("cuda")
        state_path = output_dir / f"checkpoint-{args.steps}" / "trainer_state.json"
        history = json.loads(state_path.read_text())
        all_losses.extend({"seed": seed, "domain": domain, "step": item["step"],
                           "loss": item["loss"]}
                          for item in history["log_history"] if "loss" in item)

        adapted_validation = prediction_cache(adapted, validation, labels)
        adapted_test = prediction_cache(adapted, test, labels)
        adapted_threshold, _ = best_threshold(validation, adapted_validation)
        adapted_metrics = measure(test, adapted_test, alpha=0.0,
                                  threshold=adapted_threshold)
        residual_trials = [
            (alpha, threshold, measure(validation, base_validation,
                                       adapted_validation, alpha, threshold))
            for alpha in ALPHAS for threshold in THRESHOLDS
        ]
        alpha, residual_threshold, _ = max(residual_trials, key=lambda item: item[2][2])
        residual_metrics = measure(test, base_test, adapted_test, alpha, residual_threshold)

        for method, threshold, gate, metrics in [
            ("zero-shot", base_threshold, 0.0, base_metrics),
            ("domain-adapted", adapted_threshold, 1.0, adapted_metrics),
            ("score-residual", residual_threshold, alpha, residual_metrics),
        ]:
            row = {"seed": seed, "domain": domain, "method": method,
                   "alpha": gate, "threshold": threshold,
                   "precision": metrics[0], "recall": metrics[1], "f1": metrics[2]}
            all_results.append(row)
            print(json.dumps(row), flush=True)
        del adapted
        torch.cuda.empty_cache()

write_csv(args.output / "runs.csv", all_results,
          ["seed", "domain", "method", "alpha", "threshold", "precision", "recall", "f1"])
write_csv(args.output / "loss_history.csv", all_losses,
          ["seed", "domain", "step", "loss"])
(args.output / "status.json").write_text(json.dumps({
    "status": "complete", "seeds": args.seeds, "steps": args.steps,
    "seconds": time.time() - start_time,
}, indent=2))
