#!/usr/bin/env python3
"""Build token-bounded semantic windows with complete entity/relation coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from run_qlora_inference import payload, system_instruction


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_non_test_path(path: Path, label: str) -> None:
    lowered = [part.casefold() for part in path.parts]
    if any(
        part == "test" or part.startswith("test_") or part.endswith("_test")
        for part in lowered
    ):
        raise ValueError(f"{label} must not reference formal test: {path}")


def compact_for_segments(
    annotation: dict[str, Any], segments: list[dict[str, Any]]
) -> dict[str, Any]:
    selected_entities = []
    for entity in annotation.get("entities", []):
        evidence = entity.get("evidence", {})
        evidence_segment = evidence.get("segment_id")
        entity_text = str(entity.get("text", ""))
        if not entity_text:
            continue
        if any(
            segment.get("original_segment_id", segment.get("segment_id"))
            == evidence_segment
            and entity_text in str(segment.get("text", ""))
            for segment in segments
        ):
            selected_entities.append(entity)
    selected_ids = {str(entity["id"]) for entity in selected_entities}
    selected_relations = [
        relation
        for relation in annotation.get("relations", [])
        if str(relation.get("source_id")) in selected_ids
        and str(relation.get("target_id")) in selected_ids
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


def protected_intervals(
    annotation: dict[str, Any], segment: dict[str, Any]
) -> list[tuple[int, int]]:
    segment_id = segment.get("segment_id")
    text = str(segment.get("text", ""))
    intervals: set[tuple[int, int]] = set()
    for entity in annotation.get("entities", []):
        if entity.get("evidence", {}).get("segment_id") != segment_id:
            continue
        entity_text = str(entity.get("text", ""))
        if not entity_text:
            continue
        start = 0
        while True:
            match = text.find(entity_text, start)
            if match < 0:
                break
            intervals.add((match, match + len(entity_text)))
            start = match + 1
    for relation in annotation.get("relations", []):
        evidence = relation.get("evidence", [])
        if isinstance(evidence, dict):
            evidence = [evidence]
        for item in evidence:
            if item.get("segment_id") != segment_id:
                continue
            quote = str(item.get("text", ""))
            if not quote:
                continue
            start = 0
            while True:
                match = text.find(quote, start)
                if match < 0:
                    break
                intervals.add((match, match + len(quote)))
                start = match + 1
    return sorted(intervals)


def semantic_end(
    text: str,
    start: int,
    hard_end: int,
    protected: list[tuple[int, int]],
) -> int:
    minimum = start + max(1, int((hard_end - start) * 0.55))
    view = text[start:hard_end]
    candidates: list[int] = []
    patterns = (
        r"(?:[.!?。！？；;][\"'”’）)]?)(?=\s|$)",
        r"\n\s*\n",
        r"\n",
        r"\s",
    )
    for pattern in patterns:
        values = [
            start + match.end()
            for match in re.finditer(pattern, view)
            if start + match.end() >= minimum
        ]
        if values:
            candidates = values
            break
    end = candidates[-1] if candidates else hard_end
    for interval_start, interval_end in protected:
        if interval_start < end < interval_end:
            if interval_end <= hard_end:
                end = interval_end
            elif interval_start > start:
                end = interval_start
            break
    return max(start + 1, min(end, hard_end))


def split_semantic_segments(
    segments: list[dict[str, Any]],
    annotation: dict[str, Any],
    max_chars: int,
    overlap_chars: int,
) -> list[dict[str, Any]]:
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be in [0, max_chars)")
    expanded: list[dict[str, Any]] = []
    for segment in segments:
        text = str(segment.get("text", ""))
        original_id = segment.get("original_segment_id", segment["segment_id"])
        if len(text) <= max_chars:
            expanded.append({**segment, "original_segment_id": original_id})
            continue
        protected = protected_intervals(annotation, segment)
        start = 0
        part = 1
        while start < len(text):
            hard_end = min(start + max_chars, len(text))
            end = (
                len(text)
                if hard_end == len(text)
                else semantic_end(text, start, hard_end, protected)
            )
            global_start = segment.get("start")
            row = {
                **segment,
                "segment_id": f"{segment['segment_id']}_V3P{part:03d}",
                "original_segment_id": original_id,
                "text": text[start:end],
                "semantic_part": part,
                "semantic_local_start": start,
                "semantic_local_end": end,
            }
            if isinstance(global_start, int):
                row["start"] = global_start + start
                row["end"] = global_start + end
            expanded.append(row)
            if end == len(text):
                break
            next_start = max(start + 1, end - overlap_chars)
            start = next_start
            part += 1
    return expanded


def sequence_tokens(
    tokenizer: Any,
    job: dict[str, Any],
    annotation: dict[str, Any],
    use_job_instruction: bool,
) -> tuple[int, int, int]:
    system = system_instruction(
        job, compact_target=True, use_job_instruction=use_job_instruction
    )
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": payload(job)}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    target = json.dumps(annotation, ensure_ascii=False) + "<|im_end|>"
    prompt_count = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    target_count = len(tokenizer(target, add_special_tokens=False)["input_ids"])
    return prompt_count + target_count, prompt_count, target_count


def candidate_measurement(
    tokenizer: Any,
    baseline_job: dict[str, Any],
    kg_job: dict[str, Any],
    annotation: dict[str, Any],
    segments: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, tuple[int, int, int]]]:
    compact = compact_for_segments(annotation, segments)
    measurements = {
        "baseline": sequence_tokens(
            tokenizer, {**baseline_job, "segments": segments}, compact, False
        ),
        "kg_boundary": sequence_tokens(
            tokenizer, {**kg_job, "segments": segments}, compact, True
        ),
    }
    return compact, measurements


def fits_cap(measurements: dict[str, tuple[int, int, int]], cap: int) -> bool:
    return max(value[0] for value in measurements.values()) <= cap


def sequential_ranges(
    tokenizer: Any,
    baseline_job: dict[str, Any],
    kg_job: dict[str, Any],
    annotation: dict[str, Any],
    segments: list[dict[str, Any]],
    max_sequence_tokens: int,
    overlap_segments: int,
) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < len(segments):
        _, single = candidate_measurement(
            tokenizer, baseline_job, kg_job, annotation, segments[start : start + 1]
        )
        if not fits_cap(single, max_sequence_tokens):
            raise ValueError(
                f"single semantic segment exceeds sequence cap: "
                f"{baseline_job['job_id']} {segments[start]['segment_id']} {single}"
            )
        end = start + 1
        while end < len(segments):
            _, measured = candidate_measurement(
                tokenizer,
                baseline_job,
                kg_job,
                annotation,
                segments[start : end + 1],
            )
            if not fits_cap(measured, max_sequence_tokens):
                break
            end += 1
        ranges.append((start, end))
        if end == len(segments):
            break
        start = max(start + 1, end - overlap_segments)
    return ranges


def entity_segment_indices(
    entity: dict[str, Any], segments: list[dict[str, Any]]
) -> list[int]:
    evidence_segment = entity.get("evidence", {}).get("segment_id")
    text = str(entity.get("text", ""))
    return [
        index
        for index, segment in enumerate(segments)
        if segment.get("original_segment_id", segment.get("segment_id"))
        == evidence_segment
        and text in str(segment.get("text", ""))
    ]


def relation_evidence_indices(
    relation: dict[str, Any], segments: list[dict[str, Any]]
) -> list[int]:
    evidence = relation.get("evidence", [])
    if isinstance(evidence, dict):
        evidence = [evidence]
    indices: set[int] = set()
    for item in evidence:
        segment_id = item.get("segment_id")
        quote = str(item.get("text", ""))
        matching_original = [
            index
            for index, segment in enumerate(segments)
            if segment.get("original_segment_id", segment.get("segment_id"))
            == segment_id
        ]
        quoted = [
            index
            for index in matching_original
            if quote and quote in str(segments[index].get("text", ""))
        ]
        indices.update(quoted or matching_original)
    return sorted(indices)


def focused_segment(
    segment: dict[str, Any], needles: list[str], max_chars: int = 480
) -> dict[str, Any]:
    text = str(segment.get("text", ""))
    spans = []
    for needle in needles:
        if not needle:
            continue
        start = text.find(needle)
        if start >= 0:
            spans.append((start, start + len(needle)))
    if not spans or len(text) <= max_chars:
        return segment
    required_start = min(start for start, _ in spans)
    required_end = max(end for _, end in spans)
    max_chars = min(
        len(text), max(max_chars, required_end - required_start + 120)
    )
    remaining = max_chars - (required_end - required_start)
    start = max(0, required_start - remaining // 2)
    end = min(len(text), start + max_chars)
    start = max(0, end - max_chars)

    left = text.rfind("\n", start, required_start)
    if left >= start:
        start = left + 1
    sentence_ends = [
        match.end()
        for match in re.finditer(
            r"(?:[.!?。！？；;][\"'”’）)]?)(?=\s|$)|\n",
            text[required_end:end],
        )
    ]
    if sentence_ends:
        candidate_end = required_end + sentence_ends[0]
        if candidate_end - start <= max_chars:
            end = candidate_end
    global_start = segment.get("start")
    row = {
        **segment,
        "segment_id": f"{segment['segment_id']}_RF",
        "text": text[start:end],
        "relation_focus_local_start": start,
        "relation_focus_local_end": end,
    }
    if isinstance(global_start, int):
        row["start"] = global_start + start
        row["end"] = global_start + end
    return row


def relation_rescue_segments(
    tokenizer: Any,
    baseline_job: dict[str, Any],
    kg_job: dict[str, Any],
    annotation: dict[str, Any],
    relation: dict[str, Any],
    segments: list[dict[str, Any]],
    max_sequence_tokens: int,
) -> tuple[
    list[int],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, tuple[int, int, int]],
]:
    entities = {str(row["id"]): row for row in annotation.get("entities", [])}
    source = entities.get(str(relation.get("source_id")))
    target = entities.get(str(relation.get("target_id")))
    if not source or not target:
        raise ValueError(f"relation has missing endpoint: {relation.get('id')}")
    source_indices = entity_segment_indices(source, segments)
    target_indices = entity_segment_indices(target, segments)
    evidence_indices = relation_evidence_indices(relation, segments)
    if not source_indices or not target_indices:
        raise ValueError(f"relation endpoint text is absent: {relation.get('id')}")

    candidates = []
    for source_index in source_indices:
        for target_index in target_indices:
            base = {source_index, target_index}
            evidence_options = evidence_indices or [None]
            for evidence_index in evidence_options:
                selected = set(base)
                if evidence_index is not None:
                    selected.add(evidence_index)
                ordered = sorted(selected)
                focused = []
                for index in ordered:
                    needles = []
                    if index == source_index:
                        needles.append(str(source.get("text", "")))
                    if index == target_index:
                        needles.append(str(target.get("text", "")))
                    if index == evidence_index:
                        evidence = relation.get("evidence", [])
                        if isinstance(evidence, dict):
                            evidence = [evidence]
                        needles.extend(str(item.get("text", "")) for item in evidence)
                    focused.append(focused_segment(segments[index], needles))
                compact, measured = candidate_measurement(
                    tokenizer,
                    baseline_job,
                    kg_job,
                    annotation,
                    focused,
                )
                relation_ids = {
                    str(row["id"]) for row in compact.get("relations", [])
                }
                if str(relation["id"]) not in relation_ids or not fits_cap(
                    measured, max_sequence_tokens
                ):
                    continue
                includes_evidence = evidence_index is not None
                max_tokens = max(value[0] for value in measured.values())
                candidates.append(
                    (
                        0 if includes_evidence else 1,
                        len(ordered),
                        max_tokens,
                        ordered,
                        focused,
                        compact,
                        measured,
                    )
                )
    if not candidates:
        raise ValueError(
            f"cannot construct token-bounded relation rescue: {relation.get('id')}"
        )
    _, _, _, ordered, focused, compact, measured = min(
        candidates, key=lambda row: row[:3]
    )
    return ordered, focused, compact, measured


def run(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    for label, path in (
        ("baseline jobs", args.baseline_jobs),
        ("KG boundary jobs", args.kg_jobs),
        ("gold", args.gold),
        ("index", args.index),
        ("output", args.output),
    ):
        require_non_test_path(path, label)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=True
    )
    baseline_rows = load_jsonl(args.baseline_jobs)
    baseline_by_id = {str(row["job_id"]): row for row in baseline_rows}
    kg_by_id = {str(row["job_id"]): row for row in load_jsonl(args.kg_jobs)}
    gold_rows = load_jsonl(args.gold)
    index_rows = load_jsonl(args.index)
    if len(gold_rows) != len(index_rows):
        raise ValueError("gold and index counts differ")
    index_ids = {str(row["job_id"]) for row in index_rows}
    if set(baseline_by_id) != index_ids or set(kg_by_id) != index_ids:
        raise ValueError("baseline, KG, and index job IDs differ")

    baseline_windows: list[dict[str, Any]] = []
    kg_windows: list[dict[str, Any]] = []
    gold_windows: list[dict[str, Any]] = []
    index_windows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    maximums = {"baseline": [0, 0, 0], "kg_boundary": [0, 0, 0]}
    dropped_empty = 0
    rescue_windows = 0

    def append_window(
        baseline_job: dict[str, Any],
        kg_job: dict[str, Any],
        index: dict[str, Any],
        selected_segments: list[dict[str, Any]],
        compact: dict[str, Any],
        measured: dict[str, tuple[int, int, int]],
        window_id: str,
        metadata: dict[str, Any],
    ) -> None:
        for system, values in measured.items():
            maximums[system] = [
                max(current, value)
                for current, value in zip(maximums[system], values)
            ]
        shared = {
            "job_id": window_id,
            "parent_job_id": index["job_id"],
            "segments": selected_segments,
            **metadata,
        }
        baseline_windows.append({**baseline_job, **shared})
        kg_windows.append({**kg_job, **shared})
        gold_windows.append(compact)
        index_windows.append(
            {
                "record_index": len(index_windows),
                "job_id": window_id,
                "parent_job_id": index["job_id"],
                "document_id": baseline_job["document_id"],
                "split": index.get("split"),
                "source_record_index": index.get("record_index"),
                "window_kind": metadata["window_kind"],
            }
        )

    for annotation, index in zip(gold_rows, index_rows):
        job_id = str(index["job_id"])
        baseline_job = baseline_by_id[job_id]
        kg_job = kg_by_id[job_id]
        if baseline_job.get("segments") != kg_job.get("segments"):
            raise ValueError(f"baseline/KG source mismatch: {job_id}")
        segments = split_semantic_segments(
            baseline_job.get("segments", []),
            annotation,
            args.max_segment_chars,
            args.segment_char_overlap,
        )
        ranges = sequential_ranges(
            tokenizer,
            baseline_job,
            kg_job,
            annotation,
            segments,
            args.max_sequence_tokens,
            args.segment_overlap,
        )
        covered_entities: set[str] = set()
        covered_relations: set[str] = set()
        kept = 0
        for sequence_index, (start, end) in enumerate(ranges, 1):
            selected = segments[start:end]
            compact, measured = candidate_measurement(
                tokenizer, baseline_job, kg_job, annotation, selected
            )
            if not args.keep_empty and not compact["entities"]:
                dropped_empty += 1
                continue
            kept += 1
            window_id = f"{job_id}_V3W{sequence_index:03d}"
            append_window(
                baseline_job,
                kg_job,
                index,
                selected,
                compact,
                measured,
                window_id,
                {
                    "window_kind": "semantic_sequential",
                    "window_index": sequence_index,
                    "window_start": start,
                    "window_end": end,
                    "noncontiguous": False,
                },
            )
            covered_entities.update(str(row["id"]) for row in compact["entities"])
            covered_relations.update(str(row["id"]) for row in compact["relations"])

        original_relations = {
            str(row["id"]): row for row in annotation.get("relations", [])
        }
        rescue_index = 0
        for relation_id, relation in original_relations.items():
            if relation_id in covered_relations:
                continue
            try:
                selected_indices, selected, compact, measured = relation_rescue_segments(
                    tokenizer,
                    baseline_job,
                    kg_job,
                    annotation,
                    relation,
                    segments,
                    args.max_sequence_tokens,
                )
            except ValueError as error:
                raise ValueError(f"{job_id}: {error}") from error
            rescue_index += 1
            rescue_windows += 1
            rescued_ids = [
                str(row["id"])
                for row in compact["relations"]
                if str(row["id"]) not in covered_relations
            ]
            append_window(
                baseline_job,
                kg_job,
                index,
                selected,
                compact,
                measured,
                f"{job_id}_V3R{rescue_index:03d}",
                {
                    "window_kind": "relation_rescue",
                    "window_index": rescue_index,
                    "source_segment_indices": selected_indices,
                    "noncontiguous": any(
                        right != left + 1
                        for left, right in zip(
                            selected_indices, selected_indices[1:]
                        )
                    ),
                    "relation_ids_rescued": rescued_ids,
                },
            )
            covered_entities.update(str(row["id"]) for row in compact["entities"])
            covered_relations.update(str(row["id"]) for row in compact["relations"])

        original_entities = {
            str(row["id"]) for row in annotation.get("entities", [])
        }
        missing_entities = sorted(original_entities - covered_entities)
        missing_relations = sorted(set(original_relations) - covered_relations)
        coverage_rows.append(
            {
                "job_id": job_id,
                "document_id": baseline_job["document_id"],
                "source_segments": len(baseline_job.get("segments", [])),
                "semantic_segments": len(segments),
                "sequential_ranges": len(ranges),
                "kept_sequential_windows": kept,
                "relation_rescue_windows": rescue_index,
                "source_entities": len(original_entities),
                "covered_entities": len(covered_entities),
                "source_relations": len(original_relations),
                "covered_relations": len(covered_relations),
                "missing_entities": missing_entities,
                "missing_relations": missing_relations,
            }
        )
        if missing_entities or missing_relations:
            raise ValueError(
                f"coverage failure for {job_id}: entities={missing_entities}, "
                f"relations={missing_relations}"
            )

    args.output.mkdir(parents=True, exist_ok=True)
    outputs = {
        "baseline_jobs.jsonl": baseline_windows,
        "kg_jobs.jsonl": kg_windows,
        "gold.jsonl": gold_windows,
        "index.jsonl": index_windows,
        "coverage_audit.jsonl": coverage_rows,
    }
    for name, rows in outputs.items():
        write_jsonl(args.output / name, rows)
    summary = {
        "protocol_id": "semantic-token-windows-v3",
        "formal_test_read": False,
        "source_jobs": len(index_rows),
        "window_jobs": len(gold_windows),
        "sequential_windows": len(gold_windows) - rescue_windows,
        "relation_rescue_windows": rescue_windows,
        "dropped_empty_windows": dropped_empty,
        "keep_empty": args.keep_empty,
        "max_sequence_tokens": args.max_sequence_tokens,
        "max_segment_chars": args.max_segment_chars,
        "segment_char_overlap": args.segment_char_overlap,
        "segment_overlap": args.segment_overlap,
        "builder": {
            "path": str(Path(__file__)),
            "sha256": sha256_file(Path(__file__)),
        },
        "coverage": {
            "source_entities": sum(row["source_entities"] for row in coverage_rows),
            "covered_entities": sum(row["covered_entities"] for row in coverage_rows),
            "source_relations": sum(row["source_relations"] for row in coverage_rows),
            "covered_relations": sum(row["covered_relations"] for row in coverage_rows),
            "jobs_with_missing_entities": sum(
                bool(row["missing_entities"]) for row in coverage_rows
            ),
            "jobs_with_missing_relations": sum(
                bool(row["missing_relations"]) for row in coverage_rows
            ),
        },
        "token_maxima": {
            system: {
                "sequence": values[0],
                "prompt": values[1],
                "target": values[2],
            }
            for system, values in maximums.items()
        },
        "inputs": {
            "baseline_jobs": {
                "path": str(args.baseline_jobs),
                "sha256": sha256_file(args.baseline_jobs),
            },
            "kg_jobs": {
                "path": str(args.kg_jobs),
                "sha256": sha256_file(args.kg_jobs),
            },
            "gold": {"path": str(args.gold), "sha256": sha256_file(args.gold)},
            "index": {"path": str(args.index), "sha256": sha256_file(args.index)},
        },
        "outputs": {
            name: sha256_file(args.output / name) for name in outputs
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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
    parser.add_argument("--max-sequence-tokens", type=int, default=3840)
    parser.add_argument("--max-segment-chars", type=int, default=800)
    parser.add_argument("--segment-char-overlap", type=int, default=160)
    parser.add_argument("--segment-overlap", type=int, default=1)
    parser.add_argument("--keep-empty", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
