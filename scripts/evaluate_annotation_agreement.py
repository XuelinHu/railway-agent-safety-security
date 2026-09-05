#!/usr/bin/env python3
"""Evaluate reproducible document-bootstrap agreement for two human reviews."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from build_low_resource_manifests import require_train_only_path, sha256_file


MetricCounts = dict[str, int]


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
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def evidence_items(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    return []


def matching_spans(source: str, value: str) -> list[tuple[int, int]]:
    if not value.strip():
        return []
    exact = [(match.start(), match.end()) for match in re.finditer(re.escape(value), source)]
    if exact:
        return exact
    tokens = re.split(r"\s+", value.strip())
    pattern = r"\s+".join(re.escape(token) for token in tokens if token)
    return (
        [(match.start(), match.end()) for match in re.finditer(pattern, source)]
        if pattern
        else []
    )


def resolve_entity_span(
    entity: dict[str, Any], job: dict[str, Any]
) -> tuple[tuple[int, int] | None, str]:
    text = str(entity.get("text", ""))
    segments = job.get("segments", [])
    candidates: set[tuple[int, int]] = set()
    for evidence in evidence_items(entity.get("evidence")):
        start = evidence.get("start")
        end = evidence.get("end")
        quote = str(evidence.get("text", ""))
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if end - start == len(text):
            candidates.add((start, end))
        elif end - start == len(quote):
            candidates.update(
                (start + local_start, start + local_end)
                for local_start, local_end in matching_spans(quote, text)
            )
    if not candidates:
        for segment in segments:
            segment_start = segment.get("start")
            if not isinstance(segment_start, int):
                continue
            candidates.update(
                (segment_start + local_start, segment_start + local_end)
                for local_start, local_end in matching_spans(
                    str(segment.get("text", "")), text
                )
            )
    if len(candidates) == 1:
        return next(iter(candidates)), "resolved"
    return None, "not_found" if not candidates else "ambiguous"


def evidence_key(value: Any) -> tuple[tuple[Any, Any, Any, str], ...]:
    return tuple(
        sorted(
            [
                (
                row.get("segment_id"),
                row.get("start"),
                row.get("end"),
                normalize_text(row.get("text")),
                )
                for row in evidence_items(value)
            ],
            key=str,
        )
    )


def collect_review(
    rows: list[dict[str, Any]], jobs: dict[str, dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], str]:
    if not rows:
        raise ValueError("review file is empty")
    reviewer_ids = {str(row.get("reviewer_id", "")) for row in rows}
    if len(reviewer_ids) != 1 or "" in reviewer_ids:
        raise ValueError("each review file must contain exactly one non-empty reviewer_id")
    if any(row.get("review_status") != "complete_independent_review" for row in rows):
        raise ValueError("review file still contains an incomplete row")
    collected: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "entity_types": defaultdict(set),
            "entity_evidence": defaultdict(set),
            "relations": set(),
            "relation_claims": defaultdict(set),
            "relation_evidence": defaultdict(set),
        }
    )
    issues: list[dict[str, Any]] = []
    for row in rows:
        job_id = str(row.get("job_id", ""))
        job = jobs.get(job_id)
        if not job:
            raise ValueError(f"review row has no source job: {job_id}")
        annotation = row.get("annotation")
        if not isinstance(annotation, dict):
            raise ValueError(f"review row has no annotation object: {job_id}")
        document_id = str(job["document_id"])
        if str(annotation.get("document_id")) != document_id:
            raise ValueError(f"review/source document mismatch: {job_id}")
        entity_by_id: dict[str, tuple[int, int]] = {}
        for entity in annotation.get("entities", []):
            span, status = resolve_entity_span(entity, job)
            if span is None:
                issues.append(
                    {
                        "job_id": job_id,
                        "document_id": document_id,
                        "kind": f"entity_span_{status}",
                        "item_id": entity.get("id"),
                        "text": entity.get("text"),
                    }
                )
                continue
            entity_id = str(entity.get("id", ""))
            if not entity_id:
                issues.append(
                    {
                        "job_id": job_id,
                        "document_id": document_id,
                        "kind": "entity_id_missing",
                    }
                )
                continue
            entity_by_id[entity_id] = span
            collected[document_id]["entity_types"][span].add(
                str(entity.get("type", "unknown"))
            )
            collected[document_id]["entity_evidence"][span].add(
                evidence_key(entity.get("evidence"))
            )
        for relation in annotation.get("relations", []):
            source = entity_by_id.get(str(relation.get("source_id", "")))
            target = entity_by_id.get(str(relation.get("target_id", "")))
            if source is None or target is None:
                issues.append(
                    {
                        "job_id": job_id,
                        "document_id": document_id,
                        "kind": "relation_endpoint_unresolved",
                        "item_id": relation.get("id"),
                    }
                )
                continue
            key = (source, target, str(relation.get("type", "unknown")))
            collected[document_id]["relations"].add(key)
            collected[document_id]["relation_claims"][key].add(
                str(relation.get("claim_status", "missing"))
            )
            collected[document_id]["relation_evidence"][key].add(
                evidence_key(relation.get("evidence"))
            )
    return collected, issues, next(iter(reviewer_ids))


def disagreement_id(kind: str, document_id: str, key: Any) -> str:
    value = json.dumps(
        [kind, document_id, key], ensure_ascii=False, sort_keys=True, default=list
    )
    return "D_" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def compare_document(
    document_id: str, left: dict[str, Any], right: dict[str, Any]
) -> tuple[dict[str, MetricCounts], list[dict[str, Any]]]:
    counts: dict[str, MetricCounts] = {}
    disagreements: list[dict[str, Any]] = []
    left_spans = set(left["entity_types"])
    right_spans = set(right["entity_types"])
    matched_spans = left_spans & right_spans
    counts["exact_entity_span_agreement"] = {
        "numerator": 2 * len(matched_spans),
        "denominator": len(left_spans) + len(right_spans),
        "left": len(left_spans),
        "right": len(right_spans),
        "matched": len(matched_spans),
    }
    for span in sorted(left_spans ^ right_spans):
        side = "reviewer_a_only" if span in left_spans else "reviewer_b_only"
        disagreements.append(
            {
                "disagreement_id": disagreement_id(
                    "entity_presence", document_id, [span, side]
                ),
                "document_id": document_id,
                "kind": "entity_presence",
                "span": span,
                "side": side,
            }
        )

    type_matches = 0
    entity_evidence_matches = 0
    for span in sorted(matched_spans):
        left_types = sorted(left["entity_types"][span])
        right_types = sorted(right["entity_types"][span])
        if left_types == right_types:
            type_matches += 1
        else:
            disagreements.append(
                {
                    "disagreement_id": disagreement_id(
                        "entity_type", document_id, span
                    ),
                    "document_id": document_id,
                    "kind": "entity_type",
                    "span": span,
                    "reviewer_a": left_types,
                    "reviewer_b": right_types,
                }
            )
        left_evidence = left["entity_evidence"][span]
        right_evidence = right["entity_evidence"][span]
        if left_evidence and left_evidence == right_evidence:
            entity_evidence_matches += 1
        else:
            disagreements.append(
                {
                    "disagreement_id": disagreement_id(
                        "entity_evidence", document_id, span
                    ),
                    "document_id": document_id,
                    "kind": "entity_evidence",
                    "span": span,
                    "reviewer_a": sorted(left_evidence, key=str),
                    "reviewer_b": sorted(right_evidence, key=str),
                }
            )
    counts["entity_type_agreement_conditional_on_matched_spans"] = {
        "numerator": type_matches,
        "denominator": len(matched_spans),
    }
    counts["entity_evidence_span_quote_agreement"] = {
        "numerator": entity_evidence_matches,
        "denominator": len(matched_spans),
    }

    def eligible_relation(key: tuple[Any, Any, str]) -> bool:
        return key[0] in matched_spans and key[1] in matched_spans

    left_relations = {key for key in left["relations"] if eligible_relation(key)}
    right_relations = {key for key in right["relations"] if eligible_relation(key)}
    matched_relations = left_relations & right_relations
    counts["relation_agreement_conditional_on_matched_endpoints"] = {
        "numerator": 2 * len(matched_relations),
        "denominator": len(left_relations) + len(right_relations),
        "left": len(left_relations),
        "right": len(right_relations),
        "matched": len(matched_relations),
    }
    for key in sorted(left_relations ^ right_relations, key=str):
        side = "reviewer_a_only" if key in left_relations else "reviewer_b_only"
        disagreements.append(
            {
                "disagreement_id": disagreement_id(
                    "relation_presence", document_id, [key, side]
                ),
                "document_id": document_id,
                "kind": "relation_presence",
                "relation": key,
                "side": side,
            }
        )

    claim_matches = 0
    relation_evidence_matches = 0
    for key in sorted(matched_relations, key=str):
        left_claims = sorted(left["relation_claims"][key])
        right_claims = sorted(right["relation_claims"][key])
        if left_claims == right_claims:
            claim_matches += 1
        else:
            disagreements.append(
                {
                    "disagreement_id": disagreement_id(
                        "claim_status", document_id, key
                    ),
                    "document_id": document_id,
                    "kind": "claim_status",
                    "relation": key,
                    "reviewer_a": left_claims,
                    "reviewer_b": right_claims,
                }
            )
        left_evidence = left["relation_evidence"][key]
        right_evidence = right["relation_evidence"][key]
        if left_evidence and left_evidence == right_evidence:
            relation_evidence_matches += 1
        else:
            disagreements.append(
                {
                    "disagreement_id": disagreement_id(
                        "relation_evidence", document_id, key
                    ),
                    "document_id": document_id,
                    "kind": "relation_evidence",
                    "relation": key,
                    "reviewer_a": sorted(left_evidence, key=str),
                    "reviewer_b": sorted(right_evidence, key=str),
                }
            )
    counts["claim_status_agreement"] = {
        "numerator": claim_matches,
        "denominator": len(matched_relations),
    }
    counts["relation_evidence_span_quote_agreement"] = {
        "numerator": relation_evidence_matches,
        "denominator": len(matched_relations),
    }
    return counts, disagreements


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot take percentile of empty values")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize_metric(
    per_document: dict[str, dict[str, MetricCounts]],
    metric: str,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    documents = sorted(per_document)
    numerator = sum(per_document[doc][metric]["numerator"] for doc in documents)
    denominator = sum(per_document[doc][metric]["denominator"] for doc in documents)
    point = numerator / denominator if denominator else None
    bootstrap = []
    if documents and denominator:
        rng = random.Random(f"{seed}|{metric}")
        for _ in range(resamples):
            sampled = [rng.choice(documents) for _ in documents]
            boot_num = sum(
                per_document[doc][metric]["numerator"] for doc in sampled
            )
            boot_den = sum(
                per_document[doc][metric]["denominator"] for doc in sampled
            )
            if boot_den:
                bootstrap.append(boot_num / boot_den)
    result: dict[str, Any] = {
        "numerator": numerator,
        "denominator": denominator,
        "agreement": round(point, 4) if point is not None else None,
        "ci_95": (
            [
                round(percentile(bootstrap, 0.025), 4),
                round(percentile(bootstrap, 0.975), 4),
            ]
            if bootstrap
            else None
        ),
        "bootstrap_valid_resamples": len(bootstrap),
    }
    for field in ("left", "right", "matched"):
        values = [per_document[doc][metric].get(field) for doc in documents]
        if any(value is not None for value in values):
            result[field] = sum(int(value or 0) for value in values)
    return result


def wilson_interval(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    probability = successes / total
    denominator = 1 + z * z / total
    center = (probability + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            probability * (1 - probability) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return [round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4)]


def adjudication_summary(
    disagreements: list[dict[str, Any]], adjudication_path: Path | None
) -> dict[str, Any]:
    expected = {row["disagreement_id"] for row in disagreements}
    if adjudication_path is None:
        return {
            "status": "pending",
            "disagreements": len(expected),
            "adjudicated": 0,
            "rate": 0.0 if expected else None,
            "ci_95": wilson_interval(0, len(expected)),
        }
    decisions = load_jsonl(adjudication_path)
    ids = [str(row.get("disagreement_id", "")) for row in decisions]
    if len(ids) != len(set(ids)):
        raise ValueError("adjudication log contains duplicate disagreement IDs")
    unknown = set(ids) - expected
    if unknown:
        raise ValueError(f"adjudication log contains unknown IDs: {sorted(unknown)[:5]}")
    completed = sum(bool(row.get("decision")) for row in decisions)
    total = len(expected)
    return {
        "status": "complete" if completed == total else "partial",
        "disagreements": total,
        "adjudicated": completed,
        "rate": round(completed / total, 4) if total else None,
        "ci_95": wilson_interval(completed, total),
        "log": str(adjudication_path),
        "log_sha256": sha256_file(adjudication_path),
    }


def markdown_table(metrics: dict[str, dict[str, Any]], adjudication: dict[str, Any]) -> str:
    labels = {
        "exact_entity_span_agreement": "Exact entity span agreement (Dice)",
        "entity_type_agreement_conditional_on_matched_spans": "Entity type agreement | matched span",
        "relation_agreement_conditional_on_matched_endpoints": "Relation agreement | matched endpoints (Dice)",
        "entity_evidence_span_quote_agreement": "Entity evidence span/quote agreement",
        "relation_evidence_span_quote_agreement": "Relation evidence span/quote agreement",
        "claim_status_agreement": "Claim-status agreement | matched relation",
    }
    lines = [
        "| Measure | Agreement | 95% document-bootstrap CI | Numerator / denominator |",
        "|---|---:|---:|---:|",
    ]
    for key, label in labels.items():
        row = metrics[key]
        agreement = "NA" if row["agreement"] is None else f"{100 * row['agreement']:.2f}%"
        interval = (
            "NA"
            if row["ci_95"] is None
            else f"[{100 * row['ci_95'][0]:.2f}%, {100 * row['ci_95'][1]:.2f}%]"
        )
        lines.append(
            f"| {label} | {agreement} | {interval} | "
            f"{row['numerator']} / {row['denominator']} |"
        )
    rate = adjudication["rate"]
    interval = adjudication["ci_95"]
    lines.append(
        "| Adjudication completion rate | "
        + ("NA" if rate is None else f"{100 * rate:.2f}%")
        + " | "
        + (
            "NA"
            if interval is None
            else f"[{100 * interval[0]:.2f}%, {100 * interval[1]:.2f}%]"
        )
        + f" | {adjudication['adjudicated']} / {adjudication['disagreements']} |"
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> int:
    for label, path in (
        ("selection manifest", args.selection_manifest),
        ("source jobs", args.source_jobs),
        ("reviewer A", args.reviewer_a),
        ("reviewer B", args.reviewer_b),
    ):
        require_train_only_path(path, label)
    selection = load_jsonl(args.selection_manifest)
    if any(row.get("split") != "train" for row in selection):
        raise ValueError("selection manifest contains a non-train row")
    selected_documents = {str(row["document_id"]) for row in selection}
    jobs = {str(row["job_id"]): row for row in load_jsonl(args.source_jobs)}
    if {str(row["document_id"]) for row in jobs.values()} != selected_documents:
        raise ValueError("source jobs do not exactly cover selected documents")
    reviewer_a_rows = load_jsonl(args.reviewer_a)
    reviewer_b_rows = load_jsonl(args.reviewer_b)
    job_ids = set(jobs)
    if {str(row.get("job_id")) for row in reviewer_a_rows} != job_ids:
        raise ValueError("reviewer A job set differs from source queue")
    if {str(row.get("job_id")) for row in reviewer_b_rows} != job_ids:
        raise ValueError("reviewer B job set differs from source queue")
    reviewer_a, issues_a, reviewer_a_id = collect_review(reviewer_a_rows, jobs)
    reviewer_b, issues_b, reviewer_b_id = collect_review(reviewer_b_rows, jobs)
    if reviewer_a_id == reviewer_b_id:
        raise ValueError("the two review files must identify different humans")

    per_document: dict[str, dict[str, MetricCounts]] = {}
    disagreements: list[dict[str, Any]] = []
    empty = {
        "entity_types": defaultdict(set),
        "entity_evidence": defaultdict(set),
        "relations": set(),
        "relation_claims": defaultdict(set),
        "relation_evidence": defaultdict(set),
    }
    for document_id in sorted(selected_documents):
        counts, rows = compare_document(
            document_id,
            reviewer_a.get(document_id, empty),
            reviewer_b.get(document_id, empty),
        )
        per_document[document_id] = counts
        disagreements.extend(rows)
    metric_names = tuple(next(iter(per_document.values())))
    metrics = {
        metric: summarize_metric(
            per_document, metric, args.bootstrap_resamples, args.bootstrap_seed
        )
        for metric in metric_names
    }
    adjudication = adjudication_summary(disagreements, args.adjudication_log)
    summary = {
        "protocol_id": "annotation-agreement-v1",
        "status": "pre_adjudication_complete",
        "formal_test_read": False,
        "validation_read": False,
        "reviewers": [reviewer_a_id, reviewer_b_id],
        "documents": len(selected_documents),
        "source_jobs": len(jobs),
        "bootstrap": {
            "unit": "document",
            "resamples": args.bootstrap_resamples,
            "seed": args.bootstrap_seed,
            "confidence_level": 0.95,
        },
        "metrics": metrics,
        "resolution_issues": {
            "reviewer_a": len(issues_a),
            "reviewer_b": len(issues_b),
            "items": issues_a + issues_b,
        },
        "disagreements": len(disagreements),
        "adjudication": adjudication,
        "inputs": {
            "selection_manifest": sha256_file(args.selection_manifest),
            "source_jobs": sha256_file(args.source_jobs),
            "reviewer_a": sha256_file(args.reviewer_a),
            "reviewer_b": sha256_file(args.reviewer_b),
        },
        "per_document_counts": per_document,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(markdown_table(metrics, adjudication), encoding="utf-8")
    write_jsonl(args.disagreements, disagreements)
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "per_document_counts"},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("data/processed/reviewed/annotation_agreement_v1")
    parser.add_argument(
        "--selection-manifest", type=Path, default=root / "selection_manifest.jsonl"
    )
    parser.add_argument("--source-jobs", type=Path, default=root / "source_queue.jsonl")
    parser.add_argument(
        "--reviewer-a", type=Path, default=root / "reviewer_a_annotations.jsonl"
    )
    parser.add_argument(
        "--reviewer-b", type=Path, default=root / "reviewer_b_annotations.jsonl"
    )
    parser.add_argument(
        "--output", type=Path, default=root / "annotation_agreement_summary.json"
    )
    parser.add_argument(
        "--table", type=Path, default=root / "annotation_agreement_table.md"
    )
    parser.add_argument(
        "--disagreements", type=Path, default=root / "disagreement_queue.jsonl"
    )
    parser.add_argument("--adjudication-log", type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260830)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
