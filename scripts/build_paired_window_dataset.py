#!/usr/bin/env python3
"""Build paired baseline/KG tokenizer windows and aligned compact gold targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_qlora_inference import COMPACT_INSTRUCTION, COMPACT_SYSTEM_INSTRUCTION, payload, system_instruction


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_long_segments(segments: list[dict[str, Any]], max_chars: int, overlap: int) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for segment in segments:
        text = segment.get("text", "")
        if len(text) <= max_chars:
            expanded.append({**segment, "original_segment_id": segment["segment_id"]})
            continue
        start = 0
        part = 1
        while start < len(text):
            end = min(start + max_chars, len(text))
            global_start = segment.get("start")
            row = {
                **segment,
                "segment_id": f"{segment['segment_id']}_P{part:03d}",
                "original_segment_id": segment["segment_id"],
                "text": text[start:end],
            }
            if isinstance(global_start, int):
                row["start"] = global_start + start
                row["end"] = global_start + end
            expanded.append(row)
            if end == len(text):
                break
            start = max(start + 1, end - overlap)
            part += 1
    return expanded


def compact_for_segments(annotation: dict[str, Any], segments: list[dict[str, Any]]) -> dict[str, Any]:
    selected_entities = []
    for entity in annotation.get("entities", []):
        evidence = entity.get("evidence", {})
        evidence_segment = evidence.get("segment_id")
        evidence_start = evidence.get("start")
        evidence_end = evidence.get("end")
        for segment in segments:
            if segment.get("original_segment_id", segment.get("segment_id")) != evidence_segment:
                continue
            # Entity evidence can quote a whole paragraph while entity.text is
            # a short span inside it. Window membership follows the entity span.
            if entity.get("text", "") in segment.get("text", ""):
                selected_entities.append(entity)
                break
            segment_start = segment.get("start")
            segment_end = segment.get("end")
            if all(isinstance(value, int) for value in (evidence_start, evidence_end, segment_start, segment_end)):
                if evidence_start < segment_start or evidence_end > segment_end:
                    continue
            elif entity.get("text", "") not in segment.get("text", ""):
                continue
            selected_entities.append(entity)
            break
    selected_ids = {entity["id"] for entity in selected_entities}
    selected_relations = [
        relation
        for relation in annotation.get("relations", [])
        if relation.get("source_id") in selected_ids and relation.get("target_id") in selected_ids
    ]
    return {
        "schema_version": annotation["schema_version"],
        "document_id": annotation["document_id"],
        "language": annotation["language"],
        "entities": [
            {"id": entity["id"], "text": entity["text"], "type": entity["type"]}
            for entity in selected_entities
        ],
        "relations": [
            {
                "id": relation["id"],
                "source_id": relation["source_id"],
                "type": relation["type"],
                "target_id": relation["target_id"],
                "claim_status": relation["claim_status"],
            }
            for relation in selected_relations
        ],
    }


def prompt_tokens(tokenizer: Any, job: dict[str, Any], use_job_instruction: bool) -> int:
    system = system_instruction(job, compact_target=True, use_job_instruction=use_job_instruction)
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": payload(job)}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return len(tokenizer(prompt, add_special_tokens=False)["input_ids"])


def build_ranges(
    tokenizer: Any,
    baseline_job: dict[str, Any],
    kg_job: dict[str, Any],
    segments: list[dict[str, Any]],
    max_prompt_tokens: int,
    overlap: int,
) -> list[tuple[int, int]]:
    def fits(start: int, end: int) -> bool:
        baseline = {**baseline_job, "segments": segments[start:end]}
        kg = {**kg_job, "segments": segments[start:end]}
        return max(
            prompt_tokens(tokenizer, baseline, False),
            prompt_tokens(tokenizer, kg, True),
        ) <= max_prompt_tokens

    ranges = []
    start = 0
    while start < len(segments):
        if not fits(start, start + 1):
            raise ValueError(
                f"single segment exceeds prompt budget: {baseline_job['job_id']} "
                f"{segments[start]['segment_id']}"
            )
        end = start + 1
        while end < len(segments) and fits(start, end + 1):
            end += 1
        ranges.append((start, end))
        if end == len(segments):
            break
        start = max(start + 1, end - overlap)
    return ranges


def run(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=True)
    baseline_rows = load_jsonl(args.baseline_jobs)
    baseline_by_id = {row["job_id"]: row for row in baseline_rows}
    kg_by_id = {row["job_id"]: row for row in load_jsonl(args.kg_jobs)}
    gold_rows = load_jsonl(args.gold)
    index_rows = load_jsonl(args.index)
    if len(baseline_rows) != len(gold_rows) or len(gold_rows) != len(index_rows):
        raise ValueError("baseline jobs, gold, and index counts differ")
    index_ids = {row["job_id"] for row in index_rows}
    if set(baseline_by_id) != index_ids or set(kg_by_id) != index_ids:
        raise ValueError("baseline jobs, KG jobs, and index job IDs differ")

    baseline_windows = []
    kg_windows = []
    gold_windows = []
    index_windows = []
    dropped_empty = 0
    source_windows: dict[str, int] = {}
    for annotation, index in zip(gold_rows, index_rows):
        job_id = index["job_id"]
        baseline_job = baseline_by_id[job_id]
        kg_job = kg_by_id[job_id]
        if baseline_job["segments"] != kg_job["segments"]:
            raise ValueError(f"baseline/KG source mismatch: {job_id}")
        segments = split_long_segments(
            baseline_job["segments"], args.max_segment_chars, args.segment_char_overlap
        )
        ranges = build_ranges(
            tokenizer,
            baseline_job,
            kg_job,
            segments,
            args.max_prompt_tokens,
            args.segment_overlap,
        )
        kept = 0
        for source_window_index, (start, end) in enumerate(ranges, 1):
            selected_segments = segments[start:end]
            compact = compact_for_segments(annotation, selected_segments)
            if not args.keep_empty and not compact["entities"]:
                dropped_empty += 1
                continue
            kept += 1
            window_job_id = f"{job_id}_W{source_window_index:03d}"
            metadata = {
                "job_id": window_job_id,
                "parent_job_id": job_id,
                "window_index": source_window_index,
                "window_start": start,
                "window_end": end,
                "segments": selected_segments,
            }
            baseline_windows.append({**baseline_job, **metadata})
            kg_windows.append({**kg_job, **metadata})
            gold_windows.append(compact)
            index_windows.append(
                {
                    "record_index": len(index_windows),
                    "job_id": window_job_id,
                    "parent_job_id": job_id,
                    "document_id": baseline_job["document_id"],
                    "split": index.get("split"),
                }
            )
        source_windows[job_id] = kept

    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "baseline_jobs.jsonl", baseline_windows)
    write_jsonl(args.output / "kg_jobs.jsonl", kg_windows)
    write_jsonl(args.output / "gold.jsonl", gold_windows)
    write_jsonl(args.output / "index.jsonl", index_windows)
    summary = {
        "source_jobs": len(baseline_rows),
        "window_jobs": len(gold_windows),
        "dropped_empty_windows": dropped_empty,
        "max_prompt_tokens": args.max_prompt_tokens,
        "keep_empty": args.keep_empty,
        "max_windows_per_source": max(source_windows.values(), default=0),
        "output": str(args.output),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-jobs", type=Path, required=True)
    parser.add_argument("--kg-jobs", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-prompt-tokens", type=int, default=3072)
    parser.add_argument("--max-segment-chars", type=int, default=1500)
    parser.add_argument("--segment-char-overlap", type=int, default=150)
    parser.add_argument("--segment-overlap", type=int, default=3)
    parser.add_argument("--keep-empty", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
