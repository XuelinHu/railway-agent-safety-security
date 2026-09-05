#!/usr/bin/env python3
"""Train a lightweight relation verifier with weighted diversified negatives."""

from __future__ import annotations

import argparse
import json
import pickle
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def annotation_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("annotation", row)


def normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def evidence_items(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def relation_key(relation: dict[str, Any], entities: dict[str, dict[str, Any]]) -> tuple[str, str, str]:
    source = entities.get(relation.get("source_id"), {})
    target = entities.get(relation.get("target_id"), {})
    return normalize(source.get("text")), str(relation.get("type", "")), normalize(target.get("text"))


def legal_relation_types(source: dict[str, Any], target: dict[str, Any], signatures: dict[str, Any]) -> list[str]:
    return [
        relation_type
        for relation_type, signature in signatures.items()
        if source.get("type") in signature.get("source", [])
        and target.get("type") in signature.get("target", [])
    ]


def text_feature(
    source: dict[str, Any], target: dict[str, Any], relation_type: str, evidence_text: str
) -> str:
    return " | ".join(
        (
            f"source_type={source.get('type', '')}",
            f"source={normalize(source.get('text'))}",
            f"relation={relation_type}",
            f"target_type={target.get('type', '')}",
            f"target={normalize(target.get('text'))}",
            f"evidence={normalize(evidence_text)}",
        )
    )


def evidence_for_pair(source: dict[str, Any], target: dict[str, Any], segments: list[dict[str, Any]]) -> str:
    source_items = evidence_items(source.get("evidence"))
    target_items = evidence_items(target.get("evidence"))
    for source_evidence in source_items:
        for target_evidence in target_items:
            if source_evidence.get("segment_id") == target_evidence.get("segment_id"):
                segment_id = source_evidence.get("segment_id")
                segment = next((item for item in segments if item.get("segment_id") == segment_id), None)
                if segment:
                    return str(segment.get("text", ""))
    return " ".join(
        [str(item.get("text", "")) for item in source_items[:1] + target_items[:1]]
    )


def build_training_examples(
    annotations: list[dict[str, Any]],
    jobs_by_document: dict[str, dict[str, Any]],
    ontology: dict[str, Any],
    seed: int,
    negatives_per_positive: int,
) -> tuple[list[str], list[int], list[float], Counter[str]]:
    rng = random.Random(seed)
    signatures = ontology.get("allowed_relation_signatures", {})
    features: list[str] = []
    labels: list[int] = []
    weights: list[float] = []
    negative_kinds: Counter[str] = Counter()
    for annotation in annotations:
        entities = annotation.get("entities", [])
        entity_by_id = {entity.get("id"): entity for entity in entities}
        relations = annotation.get("relations", [])
        positive_keys = {relation_key(relation, entity_by_id) for relation in relations}
        job = jobs_by_document.get(annotation.get("document_id"), {})
        segments = job.get("segments", [])
        for relation in relations:
            source = entity_by_id.get(relation.get("source_id"), {})
            target = entity_by_id.get(relation.get("target_id"), {})
            evidence = evidence_items(relation.get("evidence"))
            evidence_text = evidence[0].get("text", "") if evidence else evidence_for_pair(source, target, segments)
            features.append(text_feature(source, target, relation.get("type", ""), evidence_text))
            labels.append(1)
            weights.append(1.0)

        candidates: list[tuple[dict[str, Any], dict[str, Any], str, str, float]] = []
        for source in entities:
            for target in entities:
                if source.get("id") == target.get("id"):
                    continue
                legal_types = legal_relation_types(source, target, signatures)
                for relation_type in legal_types:
                    key = (normalize(source.get("text")), relation_type, normalize(target.get("text")))
                    if key not in positive_keys:
                        candidates.append((source, target, relation_type, "random_type_valid", 0.35))

        for relation in relations:
            source = entity_by_id.get(relation.get("source_id"), {})
            target = entity_by_id.get(relation.get("target_id"), {})
            reverse_types = legal_relation_types(target, source, signatures)
            if relation.get("type") in reverse_types:
                reverse_key = (normalize(target.get("text")), relation.get("type"), normalize(source.get("text")))
                if reverse_key not in positive_keys:
                    candidates.append((target, source, relation.get("type", ""), "reversed", 0.6))
            alternatives = [relation_type for relation_type in legal_relation_types(source, target, signatures) if relation_type != relation.get("type")]
            if alternatives:
                candidates.append((source, target, alternatives[0], "wrong_relation_type", 0.6))

        rng.shuffle(candidates)
        selected = candidates[: max(len(relations) * negatives_per_positive, 1)] if candidates else []
        for source, target, relation_type, kind, weight in selected:
            features.append(text_feature(source, target, relation_type, evidence_for_pair(source, target, segments)))
            labels.append(0)
            weights.append(weight)
            negative_kinds[kind] += 1
    if not features or len(set(labels)) < 2:
        raise ValueError("training data must contain both positive and negative relation examples")
    return features, labels, weights, negative_kinds


def prediction_rows(
    predictions: list[dict[str, Any]],
    gold_by_job: dict[str, dict[str, Any]],
    jobs_by_id: dict[str, dict[str, Any]],
    model: Any,
    vectorizer: Any,
    threshold: float,
) -> dict[str, Any]:
    gold_relation_keys = {
        job_id: {
            relation_key(relation, {entity.get("id"): entity for entity in annotation.get("entities", [])})
            for relation in annotation.get("relations", [])
        }
        for job_id, annotation in gold_by_job.items()
    }
    before = Counter()
    after = Counter()
    by_language: dict[str, Counter[str]] = {}
    for row in predictions:
        job_id = row["job_id"]
        annotation = annotation_from_row(row)
        gold_annotation = gold_by_job[job_id]
        gold_keys = gold_relation_keys[job_id]
        predicted_entities = {entity.get("id"): entity for entity in annotation.get("entities", [])}
        job = jobs_by_id.get(job_id, {})
        segments = job.get("segments", [])
        candidate_features = []
        relations = annotation.get("relations", [])
        for relation in relations:
            source = predicted_entities.get(relation.get("source_id"), {})
            target = predicted_entities.get(relation.get("target_id"), {})
            evidence = evidence_items(relation.get("evidence"))
            evidence_text = evidence[0].get("text", "") if evidence else evidence_for_pair(source, target, segments)
            candidate_features.append(text_feature(source, target, relation.get("type", ""), evidence_text))
        probabilities = model.predict_proba(vectorizer.transform(candidate_features))[:, 1] if candidate_features else []
        language = gold_annotation.get("language", "unknown")
        bucket = by_language.setdefault(language, Counter())
        before["predicted"] += len(relations)
        before["correct"] += sum(relation_key(relation, predicted_entities) in gold_keys for relation in relations)
        for relation, probability in zip(relations, probabilities):
            if probability < threshold:
                continue
            after["predicted"] += 1
            correct = relation_key(relation, predicted_entities) in gold_keys
            after["correct"] += correct
            bucket["predicted"] += 1
            bucket["correct"] += correct
    for counter in (before, after):
        counter["gold"] = sum(len(keys) for keys in gold_relation_keys.values())
        counter["precision"] = counter["correct"] / counter["predicted"] if counter["predicted"] else 0.0
        counter["recall"] = counter["correct"] / counter["gold"] if counter["gold"] else 0.0
        counter["f1"] = 2 * counter["precision"] * counter["recall"] / (counter["precision"] + counter["recall"]) if counter["precision"] + counter["recall"] else 0.0
    return {
        "threshold": threshold,
        "before": dict(before),
        "after": dict(after),
        "by_language": {language: dict(counter) for language, counter in sorted(by_language.items())},
    }


def run(args: argparse.Namespace) -> int:
    ontology = yaml.safe_load(args.ontology.read_text(encoding="utf-8"))
    train_annotations = load_jsonl(args.train_gold)
    train_index = load_jsonl(args.train_index)
    train_jobs = {row["job_id"]: row for row in load_jsonl(args.train_jobs)}
    train_jobs_by_document = {row["document_id"]: row for row in train_jobs.values()}
    features, labels, weights, negative_kinds = build_training_examples(
        train_annotations, train_jobs_by_document, ontology, args.seed, args.negatives_per_positive
    )
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1, sublinear_tf=True)
    matrix = vectorizer.fit_transform(features)
    model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=args.seed)
    model.fit(matrix, labels, sample_weight=weights)

    validation_annotations = load_jsonl(args.validation_gold)
    validation_index = load_jsonl(args.validation_index)
    validation_gold_by_job = {
        row["job_id"]: validation_annotations[row["record_index"]] for row in validation_index
    }
    validation_jobs = {row["job_id"]: row for row in load_jsonl(args.validation_jobs)}
    predictions = load_jsonl(args.validation_predictions)
    if set(validation_gold_by_job) != {row["job_id"] for row in predictions}:
        raise ValueError("validation gold and predictions do not cover the same job IDs")
    result = {
        "train_examples": len(features),
        "positive_examples": sum(labels),
        "negative_examples": len(labels) - sum(labels),
        "negative_kinds": dict(negative_kinds),
        "classifier": {"type": "char_tfidf_logistic_regression", "negative_weighting": "0.35 random/type-valid; 0.60 hard reversed/wrong-type"},
        "validation": {
            str(threshold): prediction_rows(
                predictions,
                validation_gold_by_job,
                validation_jobs,
                model,
                vectorizer,
                threshold,
            )
            for threshold in args.thresholds
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output / "model.pkl").open("wb") as stream:
        pickle.dump({"model": model, "vectorizer": vectorizer}, stream)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-gold", type=Path, required=True)
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--train-jobs", type=Path, required=True)
    parser.add_argument("--validation-gold", type=Path, required=True)
    parser.add_argument("--validation-index", type=Path, required=True)
    parser.add_argument("--validation-jobs", type=Path, required=True)
    parser.add_argument("--validation-predictions", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, default=Path("configs/risk_ontology.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--negatives-per-positive", type=int, default=3)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.65, 0.8])
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
