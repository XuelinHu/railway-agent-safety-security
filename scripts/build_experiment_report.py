#!/usr/bin/env python3
"""Compile available experiment metrics into a reproducible Markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.2f}%"


def run(args: argparse.Namespace) -> int:
    baseline = read_json(args.root / "test_baseline_metrics.json")
    kg = read_json(args.root / "test_kg_metrics.json")
    qlora = read_json(args.root / "test_qlora_metrics.json")
    compact = read_json(args.root / "test_compact_metrics.json")
    training = read_json(args.root / "qwen3_4b_kg_qlora" / "training_metrics.json")
    compact_training = read_json(args.root / "qwen3_4b_compact_qlora" / "training_metrics.json")
    expanded_full_training = read_json(args.root / "qwen3_4b_kg_qlora_v1_1_full" / "training_metrics.json")
    expanded_compact_training = read_json(args.root / "qwen3_4b_compact_qlora_v1_1_eos" / "training_metrics.json")
    system_compact_training = read_json(args.root / "qwen3_4b_compact_qlora_v1_2_system" / "training_metrics.json")
    system_compact = read_json(args.root / "test_compact_v1_2_system_metrics.json")
    window_compact = read_json(args.root / "test_compact_v1_2_window_metrics.json")
    qwen38 = read_json(args.root / "test_qwen38_27b_metrics.json")
    summary = read_json(Path("data/processed/reviewed/summary.json"))
    rows = [
        ("Qwen3-14B baseline", baseline),
        ("Qwen3-14B + KG prompt", kg),
        ("Qwen3-4B + QLoRA + KG prompt", qlora),
        ("Qwen3-4B + compact QLoRA + KG prompt", compact),
        ("Qwen3-4B + v1.2 compact QLoRA + KG prompt", system_compact),
        ("Qwen3-4B + v1.2 windowed compact QLoRA + KG prompt", window_compact),
        ("Qwen3.8-27B GGUF + windowed compact extraction", qwen38),
    ]
    lines = [
        "# Initial Safety Extraction Experiments",
        "",
        "> This is an initial pilot report. The frozen test split contains four documents and is too small for final manuscript claims.",
        "",
        "## Data",
        "",
        f"- Gold schema version: `{summary.get('schema_version', 'unknown')}`; ontology version: `{summary.get('ontology_version', 'unknown')}`; review policy: `{summary.get('review_policy', 'unknown')}`.",
        f"- Documents: {summary.get('documents', 0)}; reviewed chunks: {summary.get('jobs', 0)}; train/validation/test: {summary.get('split_documents', {})}.",
        f"- Test entities: 88; test relations: 26.",
        "",
        "## Strict Results",
        "",
        "| System | Entity precision | Entity recall | Entity F1 | Relation precision | Relation recall | Relation F1 | Evaluated jobs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in rows:
        entity = result.get("entity_strict", {})
        relation = result.get("relation_strict", {})
        valid = result.get("jobs_evaluated", 0) > 0
        lines.append(
            f"| {name} | {pct(entity.get('precision')) if valid else '-'} | {pct(entity.get('recall')) if valid else '-'} | {pct(entity.get('f1')) if valid else '-'} | "
            f"{pct(relation.get('precision')) if valid else '-'} | {pct(relation.get('recall')) if valid else '-'} | {pct(relation.get('f1')) if valid else '-'} | {result.get('jobs_evaluated', 0)} |"
        )
    lines.extend(
        [
            "",
            "## QLoRA Status",
            "",
            f"- Base model: local Qwen3-4B; LoRA rank {training.get('lora_rank', '-')}; training examples {training.get('train_examples', '-')}; optimization steps {training.get('steps', '-')}.",
            f"- Final training loss: `{training.get('final_loss', '-')}`; mean loss: `{training.get('mean_loss', '-')}`.",
            f"- Compact target training: `{compact_training.get('compact_target', False)}`; final loss: `{compact_training.get('final_loss', '-')}`; mean loss: `{compact_training.get('mean_loss', '-')}`.",
            "- Both adapter training runs completed successfully.",
            "- The full-schema adapter did not produce the required top-level `entities`/`relations` JSON object on the four test jobs; its F1 is not treated as a valid extraction result.",
            "- The compact adapter produced parseable envelopes for all four jobs, but Chinese jobs still returned incomplete structures and relation candidates were removed by schema constraints; the compact F1 is an engineering pilot result, not a final claim.",
            "- Auditable inverse-direction repair recovered five schema-valid compact relations after missing claim statuses were conservatively marked `uncertain`; none matched the 26 gold relations, so strict relation F1 remains 0%. Relation-signature repair is retained as a trustworthiness constraint, not claimed as a standalone accuracy innovation.",
            "",
            "## Interpretation",
            "",
            "- The KG prompt variant improved entity F1 in this pilot from 11.11% to 13.10%, but relation F1 remained 0%; this is a smoke signal, not a statistically supported claim.",
            "- The compact target improved entity F1 in this pilot, but relation extraction and Chinese structured output remain unresolved; the next engineering task is constrained decoding with explicit relation-signature repair, followed by regeneration on validation and test splits.",
            "- Final experiments should add a second reviewer, expand the gold set, and report confidence intervals or paired tests.",
            "- The teacher-prompt evidence-copy gate reduced normalization findings on three fixed Chinese chunks from 75 to 47 and increased retained relations from 14 to 19. These are pipeline-quality measurements, not test-set extraction-accuracy results; the three accepted chunks are in train and are excluded from the frozen pilot test.",
            "",
        ]
    )
    if expanded_full_training or expanded_compact_training:
        lines.extend(
            [
                "## Updated QLoRA Run",
                "",
                f"- The reviewed training set now contains {expanded_full_training.get('train_examples', expanded_compact_training.get('train_examples', '-'))} chunks from 21 documents. Full-schema and compact-target adapters used Qwen3-4B, NF4, rank 8, one epoch, and 15 optimization steps.",
            f"- Expanded full-schema mean loss: `{expanded_full_training.get('mean_loss', '-')}`; compact-target mean loss: `{expanded_compact_training.get('mean_loss', '-')}`.",
            "- Full-schema generations did not produce usable annotation envelopes on the four pilot test jobs and are recorded as format failures rather than extraction F1.",
            "- The EOS-preserving compact adapter produced valid annotations on 2/4 pilot test jobs: both English jobs were parseable, while both Chinese jobs were format failures. On the valid English jobs, pooled strict entity F1 was 25.32% and relation F1 was 0%. These are engineering diagnostics, not final cross-language claims.",
            f"- The v1.2 compact adapter replaced the conflicting full-schema prompt, preserved a 12K default input window, and trained with mean loss `{system_compact_training.get('mean_loss', '-')}`. It produced valid structured outputs on 3/4 jobs; the long English job over-generated entities until the 4K and 8K output budgets were exhausted.",
            f"- On the 3 valid v1.2 jobs, pooled strict entity F1 was {pct(system_compact.get('entity_strict', {}).get('f1'))} and relation F1 was {pct(system_compact.get('relation_strict', {}).get('f1'))}; Chinese entity F1 was {pct(system_compact.get('by_language', {}).get('zh', {}).get('entity_strict', {}).get('f1'))} and Chinese relation F1 was {pct(system_compact.get('by_language', {}).get('zh', {}).get('relation_strict', {}).get('f1'))}. Because one test job failed structurally, these remain engineering diagnostics and are not final manuscript claims.",
            "- The v1.2 long-Chinese diagnostic required an explicit 32K input window but exceeded the 24GB GPU memory budget when a concurrent local model occupied the GPU; its 29-entity/28-relation output was nevertheless recovered from the saved log and evaluated after structural repair.",
            f"- Tokenizer-budgeted windowing split the long Chinese job into two overlapping windows and produced 3 valid window outputs from 5. After document-level merge, pooled strict entity F1 was {pct(window_compact.get('entity_strict', {}).get('f1'))}, relation F1 was {pct(window_compact.get('relation_strict', {}).get('f1'))}, and Chinese entity F1 was {pct(window_compact.get('by_language', {}).get('zh', {}).get('entity_strict', {}).get('f1'))}; this is a resource/robustness diagnostic over 3 evaluated documents.",
            f"- Qwen3.8-27B-UD-Q4_K_M GGUF was evaluated through llama.cpp with 4K output budgets and tokenizer-budgeted windows. All 5 windows and 4 documents produced valid parsed outputs; pooled strict entity F1 was {pct(qwen38.get('entity_strict', {}).get('f1'))} and relation F1 was {pct(qwen38.get('relation_strict', {}).get('f1'))}. Chinese entity F1 was {pct(qwen38.get('by_language', {}).get('zh', {}).get('entity_strict', {}).get('f1'))}, while Chinese relation F1 remained {pct(qwen38.get('by_language', {}).get('zh', {}).get('relation_strict', {}).get('f1'))}.",
            "- The Qwen3.8 GGUF grammar-constrained path failed during llama.cpp sampler initialization on the current build, including a minimal schema. The reported Qwen3.8 result therefore uses ordinary generation followed by JSON parsing, evidence recovery, ontology validation, and ID normalization; constrained decoding remains a separate pending experiment after a compatible runtime is available.",
            "",
            ]
        )
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/processed/experiments"))
    parser.add_argument("--output", type=Path, default=Path("docs/experiment-report.md"))
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
