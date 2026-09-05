#!/usr/bin/env python3
"""Train and apply a local-pair semantic relation classifier for KG V3.

The classifier sees only ordered entity pairs that co-occur in a source segment
and relation labels allowed by the ontology signature.  Gold relations are
positive; unobserved legal pairs are low-weight PU-style negatives, while
wrong-type and reversed-pair negatives receive higher weights.  At inference,
the best relation type per ordered pair is retained only above an explicit
threshold; otherwise the pair is treated as NONE.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def annotation_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("annotation", row)


def indexed_annotations(
    annotations: list[dict[str, Any]], index: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    return {
        row["job_id"]: annotations[row["record_index"]]
        for row in index
    }


def evidence_items(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def entity_segment_ids(entity: dict[str, Any], job: dict[str, Any]) -> set[str]:
    available = {
        str(segment.get("segment_id"))
        for segment in job.get("segments", [])
        if segment.get("segment_id") is not None
    }
    supplied = {
        str(item.get("segment_id"))
        for item in evidence_items(entity.get("evidence"))
        if item.get("segment_id") is not None
    } & available
    if supplied:
        return supplied
    text = normalize(entity.get("text"))
    if not text:
        return set()
    return {
        str(segment.get("segment_id"))
        for segment in job.get("segments", [])
        if text in normalize(segment.get("text"))
    }


def legal_relation_types(
    source: dict[str, Any], target: dict[str, Any], signatures: dict[str, Any]
) -> list[str]:
    return [
        relation_type
        for relation_type, signature in signatures.items()
        if source.get("type") in signature.get("source", [])
        and target.get("type") in signature.get("target", [])
    ]


def kg_features(job: dict[str, Any], source: dict[str, Any], target: dict[str, Any], relation_type: str) -> tuple[bool, bool]:
    context = job.get("kg_v2_context", {})
    source_text = normalize(source.get("text"))
    target_text = normalize(target.get("text"))
    edge_prior = any(
        normalize(edge.get("source")) == source_text
        and normalize(edge.get("target")) == target_text
        and edge.get("relation") == relation_type
        for edge in context.get("edge_priors", [])
    )
    semantic_pattern = any(
        pattern.get("relation") == relation_type
        and pattern.get("source_type") == source.get("type")
        and pattern.get("target_type") == target.get("type")
        for pattern in context.get("semantic_relation_patterns", [])
    )
    return edge_prior, semantic_pattern


def candidate_feature(candidate: dict[str, Any], use_kg_features: bool) -> str:
    fields = [
        f"language={candidate['language']}",
        f"source_type={candidate['source']['type']}",
        f"source={normalize(candidate['source']['text'])}",
        f"relation={candidate['relation_type']}",
        f"target_type={candidate['target']['type']}",
        f"target={normalize(candidate['target']['text'])}",
    ]
    if use_kg_features:
        fields.extend(
            (
                f"kg_edge_prior={'yes' if candidate['edge_prior'] else 'no'}",
                f"kg_semantic_pattern={'yes' if candidate['semantic_pattern'] else 'no'}",
            )
        )
    fields.append(f"context={normalize(candidate['segment'].get('text'))[:1600]}")
    return " | ".join(fields)


def enumerate_candidates(
    job_id: str,
    annotation: dict[str, Any],
    job: dict[str, Any],
    signatures: dict[str, Any],
) -> list[dict[str, Any]]:
    entities = [entity for entity in annotation.get("entities", []) if entity.get("id")]
    segments = {
        str(segment.get("segment_id")): segment for segment in job.get("segments", [])
    }
    occurrences = {
        entity["id"]: entity_segment_ids(entity, job) for entity in entities
    }
    candidates = []
    for source in entities:
        for target in entities:
            if source["id"] == target["id"]:
                continue
            shared = occurrences[source["id"]] & occurrences[target["id"]]
            if not shared:
                continue
            segment_id = next(
                (
                    str(segment.get("segment_id"))
                    for segment in job.get("segments", [])
                    if str(segment.get("segment_id")) in shared
                ),
                sorted(shared)[0],
            )
            for relation_type in legal_relation_types(source, target, signatures):
                edge_prior, semantic_pattern = kg_features(
                    job, source, target, relation_type
                )
                candidates.append(
                    {
                        "job_id": job_id,
                        "document_id": annotation.get("document_id", job.get("document_id")),
                        "language": annotation.get("language", job.get("language", "unknown")),
                        "source": source,
                        "target": target,
                        "relation_type": relation_type,
                        "segment": segments[segment_id],
                        "edge_prior": edge_prior,
                        "semantic_pattern": semantic_pattern,
                    }
                )
    return candidates


def relation_keys(annotation: dict[str, Any]) -> dict[tuple[str, str, str], str]:
    return {
        (
            str(relation.get("source_id")),
            str(relation.get("type")),
            str(relation.get("target_id")),
        ): str(relation.get("claim_status", "uncertain"))
        for relation in annotation.get("relations", [])
    }


def training_rows(
    annotations: dict[str, dict[str, Any]],
    jobs: dict[str, dict[str, Any]],
    signatures: dict[str, Any],
    negatives_per_positive: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[int], list[float], Counter[str]]:
    rng = random.Random(seed)
    selected_candidates = []
    labels = []
    weights = []
    kinds: Counter[str] = Counter()
    for job_id, annotation in annotations.items():
        candidates = enumerate_candidates(job_id, annotation, jobs[job_id], signatures)
        positives = relation_keys(annotation)
        directed_gold = {(source, target) for source, _, target in positives}
        reverse_gold = {(target, source) for source, _, target in positives}
        positive_rows = []
        hard_rows = []
        random_rows = []
        for candidate in candidates:
            key = (
                str(candidate["source"]["id"]),
                candidate["relation_type"],
                str(candidate["target"]["id"]),
            )
            pair = (key[0], key[2])
            if key in positives:
                positive_rows.append(candidate)
            elif pair in directed_gold:
                hard_rows.append((candidate, "wrong_relation_type", 0.70))
            elif pair in reverse_gold:
                hard_rows.append((candidate, "reversed_pair", 0.70))
            else:
                random_rows.append((candidate, "unobserved_legal_pair", 0.20))

        for candidate in positive_rows:
            selected_candidates.append(candidate)
            labels.append(1)
            weights.append(1.0)
            kinds["positive"] += 1

        negative_limit = max(len(positive_rows) * negatives_per_positive, 2)
        rng.shuffle(hard_rows)
        rng.shuffle(random_rows)
        chosen = hard_rows[:negative_limit]
        if len(chosen) < negative_limit:
            chosen.extend(random_rows[: negative_limit - len(chosen)])
        for candidate, kind, weight in chosen:
            selected_candidates.append(candidate)
            labels.append(0)
            weights.append(weight)
            kinds[kind] += 1
    if not selected_candidates or len(set(labels)) < 2:
        raise ValueError("relation classifier requires positive and negative examples")
    return selected_candidates, labels, weights, kinds


def relation_claims(annotation: dict[str, Any]) -> dict[tuple[str, str, str], str]:
    return relation_keys(annotation)


def evidence_from_segment(segment: dict[str, Any]) -> dict[str, Any]:
    return {
        key: segment.get(key)
        for key in ("text", "segment_id", "page", "start", "end")
        if key in segment
    }


def predictions_for_threshold(
    entity_rows: list[dict[str, Any]],
    candidates_by_job: dict[str, list[dict[str, Any]]],
    probabilities_by_job: dict[str, list[float]],
    raw_claims: dict[str, dict[tuple[str, str, str], str]],
    threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    output = []
    audit = []
    counts: Counter[str] = Counter()
    annotations = {row["job_id"]: annotation_from_row(row) for row in entity_rows}
    for job_id, annotation in annotations.items():
        candidates = candidates_by_job[job_id]
        probabilities = probabilities_by_job[job_id]
        best_by_pair: dict[tuple[str, str], tuple[dict[str, Any], float]] = {}
        for candidate, probability in zip(candidates, probabilities):
            pair = (str(candidate["source"]["id"]), str(candidate["target"]["id"]))
            current = best_by_pair.get(pair)
            if current is None or probability > current[1]:
                best_by_pair[pair] = (candidate, probability)

        relations = []
        for candidate, probability in best_by_pair.values():
            accepted = probability >= threshold
            key = (
                str(candidate["source"]["id"]),
                candidate["relation_type"],
                str(candidate["target"]["id"]),
            )
            audit.append(
                {
                    "job_id": job_id,
                    "source_id": key[0],
                    "target_id": key[2],
                    "relation_type": key[1],
                    "probability": round(float(probability), 6),
                    "accepted": accepted,
                    "edge_prior": candidate["edge_prior"],
                    "semantic_pattern": candidate["semantic_pattern"],
                    "segment_id": candidate["segment"].get("segment_id"),
                }
            )
            if not accepted:
                counts["pairs_rejected_as_none"] += 1
                continue
            counts["relations_accepted"] += 1
            relations.append(
                {
                    "id": f"R{len(relations) + 1}",
                    "source_id": key[0],
                    "type": key[1],
                    "target_id": key[2],
                    "claim_status": raw_claims.get(job_id, {}).get(key, "uncertain"),
                    "evidence": [evidence_from_segment(candidate["segment"])],
                    "confidence": round(float(probability), 6),
                    "review_status": "pending",
                    "created_by": "kg-v3-relation-classifier",
                }
            )
        output.append(
            {
                "job_id": job_id,
                "annotation": {
                    **annotation,
                    "relations": relations,
                    "review": {
                        "status": "unreviewed",
                        "reviewers": [],
                        "notes": f"KG V3 relation classifier threshold={threshold}",
                    },
                },
            }
        )
    return output, audit, counts


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def threshold_name(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def run(args: argparse.Namespace) -> int:
    ontology = yaml.safe_load(args.ontology.read_text(encoding="utf-8"))
    signatures = ontology.get("allowed_relation_signatures", {})
    train_annotations = indexed_annotations(
        load_jsonl(args.train_gold), load_jsonl(args.train_index)
    )
    train_jobs = {row["job_id"]: row for row in load_jsonl(args.train_jobs)}
    candidates, labels, weights, kinds = training_rows(
        train_annotations,
        train_jobs,
        signatures,
        args.negatives_per_positive,
        args.seed,
    )
    features = [candidate_feature(row, args.use_kg_features) for row in candidates]

    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression

    encoder = SentenceTransformer(
        str(args.embedding_model), device=args.device, local_files_only=True
    )
    matrix = encoder.encode(
        features,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=args.seed,
        C=args.regularization,
    )
    classifier.fit(matrix, labels, sample_weight=weights)

    entity_rows = load_jsonl(args.validation_entities)
    entity_annotations = {
        row["job_id"]: annotation_from_row(row) for row in entity_rows
    }
    validation_jobs = {row["job_id"]: row for row in load_jsonl(args.validation_jobs)}
    missing = sorted(set(validation_jobs) - set(entity_annotations))
    if missing:
        raise ValueError(f"missing validation entity rows: {missing[:5]}")
    raw_annotations = (
        {
            row["job_id"]: annotation_from_row(row)
            for row in load_jsonl(args.raw_predictions)
        }
        if args.raw_predictions
        else {}
    )
    raw_claims = {
        job_id: relation_claims(annotation)
        for job_id, annotation in raw_annotations.items()
    }

    candidates_by_job = {
        job_id: enumerate_candidates(
            job_id, entity_annotations[job_id], validation_jobs[job_id], signatures
        )
        for job_id in validation_jobs
    }
    flat_candidates = [
        candidate
        for job_id in validation_jobs
        for candidate in candidates_by_job[job_id]
    ]
    validation_features = [
        candidate_feature(candidate, args.use_kg_features)
        for candidate in flat_candidates
    ]
    validation_matrix = encoder.encode(
        validation_features,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    flat_probabilities = (
        classifier.predict_proba(validation_matrix)[:, 1].tolist()
        if validation_features
        else []
    )
    probabilities_by_job = {}
    offset = 0
    for job_id in validation_jobs:
        size = len(candidates_by_job[job_id])
        probabilities_by_job[job_id] = flat_probabilities[offset : offset + size]
        offset += size

    args.output.mkdir(parents=True, exist_ok=True)
    threshold_summaries = {}
    for threshold in args.thresholds:
        predictions, audit, counts = predictions_for_threshold(
            entity_rows,
            candidates_by_job,
            probabilities_by_job,
            raw_claims,
            threshold,
        )
        name = threshold_name(threshold)
        prediction_path = args.output / f"predictions_t{name}.jsonl"
        audit_path = args.output / f"audit_t{name}.jsonl"
        write_jsonl(prediction_path, predictions)
        write_jsonl(audit_path, audit)
        threshold_summaries[str(threshold)] = {
            **dict(counts),
            "predictions": str(prediction_path),
            "audit": str(audit_path),
        }

    model_path = args.output / "classifier.pkl"
    with model_path.open("wb") as stream:
        pickle.dump(
            {
                "classifier": classifier,
                "embedding_model": str(args.embedding_model),
                "use_kg_features": args.use_kg_features,
            },
            stream,
        )
    summary = {
        "version": "kg-v3-local-pair-classifier-1.0",
        "classifier": "BGE-M3 normalized embedding + weighted logistic regression",
        "use_kg_features": args.use_kg_features,
        "train_candidates": len(candidates),
        "train_positive": sum(labels),
        "train_negative": len(labels) - sum(labels),
        "training_kinds": dict(kinds),
        "validation_typed_candidates": len(flat_candidates),
        "validation_ordered_pairs": sum(
            len(
                {
                    (candidate["source"]["id"], candidate["target"]["id"])
                    for candidate in rows
                }
            )
            for rows in candidates_by_job.values()
        ),
        "thresholds": threshold_summaries,
        "model": str(model_path),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-gold", type=Path, required=True)
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--train-jobs", type=Path, required=True)
    parser.add_argument("--validation-entities", type=Path, required=True)
    parser.add_argument("--validation-jobs", type=Path, required=True)
    parser.add_argument("--raw-predictions", type=Path)
    parser.add_argument("--ontology", type=Path, default=Path("configs/risk_ontology.yaml"))
    parser.add_argument("--embedding-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--negatives-per-positive", type=int, default=8)
    parser.add_argument("--regularization", type=float, default=1.0)
    parser.add_argument("--use-kg-features", action="store_true")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
