#!/usr/bin/env python3
"""Build a fixed train-only double-annotation package without copying labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_low_resource_manifests import (
    build_inventory,
    coverage_summary,
    load_cluster_metadata,
    load_jsonl,
    require_train_only_path,
    select_nested_groups,
    sha256_file,
    write_jsonl,
)


def reviewer_template(job: dict[str, Any], reviewer_id: str) -> dict[str, Any]:
    return {
        "job_id": job["job_id"],
        "document_id": job["document_id"],
        "reviewer_id": reviewer_id,
        "review_status": "pending_independent_review",
        "annotation": {
            "schema_version": "0.1.0",
            "document_id": job["document_id"],
            "language": job.get("language", "unknown"),
            "entities": [],
            "relations": [],
            "review": {
                "status": "unreviewed",
                "reviewers": [],
                "notes": "Complete independently; do not inspect the other reviewer file.",
            },
        },
    }


def source_row(job: dict[str, Any]) -> dict[str, Any]:
    """Keep source/protocol fields but exclude all existing annotation labels."""
    return {
        key: value
        for key, value in job.items()
        if key
        in {
            "job_id",
            "parent_job_id",
            "document_id",
            "language",
            "category",
            "source_path",
            "window_index",
            "window_start",
            "window_end",
            "chunk_count",
            "chunk_number",
            "prompt_version",
            "segments",
            "ontology",
        }
    }


def run(args: argparse.Namespace) -> int:
    for label, path in (
        ("train annotations", args.train_annotations),
        ("train index", args.train_index),
        ("train window jobs", args.train_jobs),
        ("train source jobs", args.train_source_jobs),
    ):
        require_train_only_path(path, label)
    annotations = load_jsonl(args.train_annotations)
    index = load_jsonl(args.train_index)
    train_jobs = load_jsonl(args.train_jobs)
    train_source_jobs = load_jsonl(args.train_source_jobs)
    train_documents = {str(row["document_id"]) for row in index}
    train_job_ids = {str(row["job_id"]) for row in index}
    if {str(row.get("job_id")) for row in train_source_jobs} != train_job_ids:
        raise ValueError("train source jobs do not exactly match the train index")
    cluster_metadata, missing_cluster_rows = load_cluster_metadata(
        args.clusters, train_documents
    )
    inventory = build_inventory(
        annotations, index, train_jobs, cluster_metadata
    )

    selection = select_nested_groups(
        inventory, [args.documents], args.selection_salt
    )[args.documents]
    selected_documents = {str(row["document_id"]) for row in selection}
    selected_jobs = sorted(
        (
            source_row(row)
            for row in train_source_jobs
            if str(row.get("document_id")) in selected_documents
        ),
        key=lambda row: str(row["job_id"]),
    )
    if {str(row["document_id"]) for row in selected_jobs} != selected_documents:
        raise ValueError("one or more selected documents has no source review job")

    args.output.mkdir(parents=True, exist_ok=True)
    selection_path = args.output / "selection_manifest.jsonl"
    source_path = args.output / "source_queue.jsonl"
    reviewer_a_path = args.output / "reviewer_a_annotations.jsonl"
    reviewer_b_path = args.output / "reviewer_b_annotations.jsonl"
    write_jsonl(
        selection_path,
        (
            {
                **row,
                "selection_rank": rank,
                "split": "train",
                "agreement_protocol": "annotation-agreement-v1",
            }
            for rank, row in enumerate(selection, 1)
        ),
    )
    write_jsonl(source_path, selected_jobs)
    write_jsonl(
        reviewer_a_path,
        (reviewer_template(row, "reviewer_a") for row in selected_jobs),
    )
    write_jsonl(
        reviewer_b_path,
        (reviewer_template(row, "reviewer_b") for row in selected_jobs),
    )

    instructions = """# Independent annotation agreement package v1

Status: **awaiting two independent human reviews**

This package contains only complete training-document source annotation units.
`source_queue.jsonl`
is shared source material; it contains no gold annotations. Reviewer A edits only
`reviewer_a_annotations.jsonl`, and reviewer B edits only
`reviewer_b_annotations.jsonl`. Neither reviewer may inspect the other file before
both are frozen.

For every row, annotate the supplied source unit under `annotation`, then set
`review_status` to `complete_independent_review`, add exactly one real reviewer ID
under `annotation.review.reviewers`, and set `annotation.review.status` to
`independently_reviewed`. Follow `docs/annotation-guidelines.md`. Entity evidence
must preserve segment ID and global character offsets; relation evidence must
preserve exact quotes and claim status.

After both files are complete, run:

```bash
python scripts/evaluate_annotation_agreement.py \
  --selection-manifest data/processed/reviewed/annotation_agreement_v1/selection_manifest.jsonl \
  --source-jobs data/processed/reviewed/annotation_agreement_v1/source_queue.jsonl \
  --reviewer-a data/processed/reviewed/annotation_agreement_v1/reviewer_a_annotations.jsonl \
  --reviewer-b data/processed/reviewed/annotation_agreement_v1/reviewer_b_annotations.jsonl \
  --output data/processed/reviewed/annotation_agreement_v1/annotation_agreement_summary.json \
  --table data/processed/reviewed/annotation_agreement_v1/annotation_agreement_table.md \
  --disagreements data/processed/reviewed/annotation_agreement_v1/disagreement_queue.jsonl
```

Do not adjudicate until the evaluator has frozen the pre-adjudication summary and
disagreement queue. Preserve both reviewer files and record every adjudication
decision in a separate JSONL log.
"""
    (args.output / "README.md").write_text(instructions, encoding="utf-8")
    outputs = [
        selection_path,
        source_path,
        reviewer_a_path,
        reviewer_b_path,
        args.output / "README.md",
    ]
    summary = {
        "protocol_id": "annotation-agreement-v1",
        "status": "awaiting_two_independent_human_reviews",
        "formal_test_read": False,
        "validation_read": False,
        "selection_uses_training_annotations_for_type_coverage_tiebreak": True,
        "selection_uses_formal_validation_or_test_gold": False,
        "documents": len(selected_documents),
        "source_annotation_units": len(selected_jobs),
        "selection": coverage_summary(selection),
        "eligible_training_documents": len(inventory),
        "train_source_jobs": {
            "path": str(args.train_source_jobs),
            "sha256": sha256_file(args.train_source_jobs),
        },
        "missing_cluster_rows": missing_cluster_rows,
        "bootstrap": {
            "unit": "document",
            "resamples": 20000,
            "seed": 20260830,
            "confidence_level": 0.95,
        },
        "required_metrics": [
            "exact_entity_span_agreement",
            "entity_type_agreement_conditional_on_matched_spans",
            "relation_agreement_conditional_on_matched_endpoints",
            "entity_evidence_span_quote_agreement",
            "relation_evidence_span_quote_agreement",
            "claim_status_agreement",
            "adjudication_rate",
        ],
        "outputs": {
            str(path.relative_to(args.output)): sha256_file(path) for path in outputs
        },
    }
    (args.output / "package_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-annotations",
        type=Path,
        default=Path("data/processed/reviewed/formal_split/train.jsonl"),
    )
    parser.add_argument(
        "--train-index",
        type=Path,
        default=Path("data/processed/reviewed/formal_split/train_index.jsonl"),
    )
    parser.add_argument(
        "--train-jobs",
        type=Path,
        default=Path(
            "data/processed/experiments/formal/windowed_train_v2/baseline_jobs.jsonl"
        ),
    )
    parser.add_argument(
        "--train-source-jobs",
        type=Path,
        default=Path("data/processed/experiments/formal/train_baseline_jobs.jsonl"),
    )
    parser.add_argument(
        "--clusters",
        type=Path,
        default=Path("data/catalog/near_duplicate_clusters.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/reviewed/annotation_agreement_v1"),
    )
    parser.add_argument("--documents", type=int, default=20)
    parser.add_argument("--selection-salt", default="annotation-agreement-v1-20260830")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
