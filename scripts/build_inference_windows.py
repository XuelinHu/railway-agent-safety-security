#!/usr/bin/env python3
"""Split annotation jobs into tokenizer-budgeted segment windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def window_ranges(
    segment_count: int,
    fits: Callable[[int, int], bool],
    overlap: int = 1,
) -> list[tuple[int, int]]:
    """Return half-open ranges, extending each window until the next segment would not fit."""
    if segment_count < 1:
        return []
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < segment_count:
        end = start + 1
        while end < segment_count and fits(start, end + 1):
            end += 1
        ranges.append((start, end))
        if end >= segment_count:
            break
        next_start = max(start + 1, end - overlap)
        start = next_start
    return ranges


def build_windows(
    jobs: list[dict[str, Any]],
    tokenizer: Any,
    system_instruction: str,
    max_input_tokens: int,
    overlap: int,
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for job in jobs:
        segments = job.get("segments", [])

        def prompt_length(start: int, end: int) -> int:
            window_job = {**job, "segments": segments[start:end]}
            user = json.dumps(
                {
                    "document_id": window_job["document_id"],
                    "language": window_job["language"],
                    "teacher_model": window_job.get("teacher_model", "qwen3-4b-qlora"),
                    "ontology": window_job["ontology"],
                    "segments": window_job["segments"],
                },
                ensure_ascii=False,
            )
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            return len(tokenizer(prompt, add_special_tokens=False)["input_ids"])

        ranges = window_ranges(len(segments), lambda start, end: prompt_length(start, end) <= max_input_tokens, overlap)
        for index, (start, end) in enumerate(ranges, 1):
            if end - start == 1 and prompt_length(start, end) > max_input_tokens:
                raise ValueError(f"single segment exceeds input budget for {job['job_id']} at index {start}")
            windows.append(
                {
                    **job,
                    "job_id": f"{job['job_id']}_W{index:03d}",
                    "segments": segments[start:end],
                    "parent_job_id": job["job_id"],
                    "window_index": index,
                    "window_count": len(ranges),
                    "window_start": start,
                    "window_end": end,
                }
            )
    return windows


def main(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    from run_qlora_inference import COMPACT_SYSTEM_INSTRUCTION

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=True)
    jobs = load_jsonl(args.jobs)
    if args.job_id:
        jobs = [job for job in jobs if job["job_id"] == args.job_id]
        if not jobs:
            raise SystemExit(f"job not found: {args.job_id}")
    windows = build_windows(jobs, tokenizer, COMPACT_SYSTEM_INSTRUCTION, args.max_input_tokens, args.overlap)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for window in windows:
            stream.write(json.dumps(window, ensure_ascii=False) + "\n")
    print(json.dumps({"source_jobs": len(jobs), "window_jobs": len(windows), "output": str(args.output)}, ensure_ascii=False))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job-id")
    parser.add_argument("--max-input-tokens", type=int, default=10000)
    parser.add_argument("--overlap", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
