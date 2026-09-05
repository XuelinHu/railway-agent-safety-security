#!/usr/bin/env python3
"""Run a small closed-schema Geo-NRE transfer check with Qwen3-4B.

The result is an exploratory external-domain diagnostic, not a reproduction of
GenIE and not a replacement for the railway validation or sealed test set.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """
You are a closed-schema information extraction model. Read the input text and
return only one JSON object with exactly one top-level field named "triples".
Each triple must contain exactly "subject", "relation", and "object". Copy the
subject and object exactly from the supplied entity catalogue and use only a
relation from the supplied relation catalogue. Do not add explanations or
markdown. If no supported triple exists, return {"triples": []}.
""".strip()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def dataset_triples(row: dict[str, Any]) -> list[tuple[str, str, str]]:
    values: list[tuple[str, str, str]] = []
    for output in row.get("output", []):
        for triple in output.get("non_formatted_surface_output", []):
            if isinstance(triple, list) and len(triple) == 3:
                values.append(tuple(str(value) for value in triple))
    return values


def parse_json_object(text: str) -> dict[str, Any]:
    for start in (match.start() for match in re.finditer(r"\{", text)):
        try:
            value, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("triples"), list):
            return value
    raise ValueError("no complete triples JSON object")


def predicted_triples(value: dict[str, Any]) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    for item in value.get("triples", []):
        if not isinstance(item, dict):
            continue
        subject = item.get("subject")
        relation = item.get("relation")
        obj = item.get("object")
        if all(isinstance(part, str) for part in (subject, relation, obj)):
            triples.append((subject, relation, obj))
    return triples


def score_sets(
    gold: list[tuple[str, str, str]], predicted: list[tuple[str, str, str]]
) -> tuple[int, int, int]:
    gold_set = set(gold)
    predicted_set = set(predicted)
    true_positive = len(gold_set.intersection(predicted_set))
    return true_positive, len(predicted_set) - true_positive, len(gold_set) - true_positive


def main(args: argparse.Namespace) -> int:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    full_rows = load_jsonl(args.catalogue_source)
    sample_rows = load_jsonl(args.sample)
    if args.limit:
        sample_rows = sample_rows[: args.limit]

    all_triples = [triple for row in full_rows for triple in dataset_triples(row)]
    entity_catalogue = sorted({part for triple in all_triples for part in (triple[0], triple[2])})
    relation_catalogue = sorted({triple[1] for triple in all_triples})

    tokenizer_source = args.adapter if args.adapter else args.model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, local_files_only=True, trust_remote_code=True
    )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True)
    model.eval()
    device = next(model.parameters()).device

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict[str, Any]] = []
    total_tp = total_fp = total_fn = 0
    normalized_tp = normalized_fp = normalized_fn = 0
    json_successes = 0
    valid_relation_predictions = 0
    valid_entity_predictions = 0
    predicted_count = 0
    grounded_endpoint_count = 0
    endpoint_count = 0
    started_all = time.monotonic()

    for index, row in enumerate(sample_rows, 1):
        user_payload = json.dumps(
            {
                "text": row["input"],
                "entity_catalogue": entity_catalogue,
                "relation_catalogue": relation_catalogue,
            },
            ensure_ascii=False,
        )
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_payload},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        started = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.monotonic() - started
        raw = tokenizer.decode(
            generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        error = None
        try:
            parsed = parse_json_object(raw)
            predicted = predicted_triples(parsed)
            json_successes += 1
        except Exception as exc:
            predicted = []
            error = str(exc)

        gold = dataset_triples(row)
        tp, fp, fn = score_sets(gold, predicted)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        gold_normalized = [tuple(part.casefold() for part in triple) for triple in gold]
        predicted_normalized = [tuple(part.casefold() for part in triple) for triple in predicted]
        tp_n, fp_n, fn_n = score_sets(gold_normalized, predicted_normalized)
        normalized_tp += tp_n
        normalized_fp += fp_n
        normalized_fn += fn_n

        text_folded = str(row["input"]).casefold()
        for subject, relation, obj in predicted:
            predicted_count += 1
            valid_relation_predictions += relation in relation_catalogue
            valid_entity_predictions += subject in entity_catalogue and obj in entity_catalogue
            grounded_endpoint_count += subject.casefold() in text_folded
            grounded_endpoint_count += obj.casefold() in text_folded
            endpoint_count += 2
        rows_out.append(
            {
                "id": row["id"],
                "gold": gold,
                "predicted": predicted,
                "exact_tp": tp,
                "exact_fp": fp,
                "exact_fn": fn,
                "json_success": error is None,
                "error": error,
                "elapsed_seconds": round(elapsed, 3),
                "raw_output": raw[:4000],
            }
        )
        print(f"{index}/{len(sample_rows)} id={row['id']} tp={tp} fp={fp} fn={fn}", flush=True)

    def metrics(tp: int, fp: int, fn: int) -> dict[str, float | int]:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}

    wall_clock = time.monotonic() - started_all
    summary = {
        "experiment": "exploratory GenIE Geo-NRE closed-schema transfer quickcheck",
        "interpretation_boundary": (
            "Not a GenIE reproduction and not directly comparable to the railway benchmark; "
            "uses the public Geo-NRE entity/relation catalogue in the prompt."
        ),
        "model_path": str(args.model_path),
        "adapter": str(args.adapter) if args.adapter else None,
        "rows": len(sample_rows),
        "entity_catalogue_size": len(entity_catalogue),
        "relation_catalogue_size": len(relation_catalogue),
        "exact_surface_triplet_micro": metrics(total_tp, total_fp, total_fn),
        "casefold_surface_triplet_micro": metrics(normalized_tp, normalized_fp, normalized_fn),
        "json_success_rate": json_successes / len(sample_rows) if sample_rows else 0.0,
        "predicted_triples": predicted_count,
        "valid_relation_rate": valid_relation_predictions / predicted_count if predicted_count else 0.0,
        "valid_entity_pair_rate": valid_entity_predictions / predicted_count if predicted_count else 0.0,
        "predicted_endpoint_source_grounding_rate": grounded_endpoint_count / endpoint_count if endpoint_count else 0.0,
        "wall_clock_seconds": wall_clock,
        "seconds_per_row": wall_clock / len(sample_rows) if sample_rows else 0.0,
        "formal_test_read": False,
    }
    with args.output.open("w", encoding="utf-8") as handle:
        for output_row in rows_out:
            handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--catalogue-source", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
