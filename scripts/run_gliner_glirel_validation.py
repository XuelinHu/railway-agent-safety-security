#!/usr/bin/env python3
"""Run a gold-independent GLiNER + GLiREL public benchmark baseline.

Inference reads only ``validation_baseline_jobs.jsonl``. Entity mentions are
predicted by GLiNER, converted to GLiREL token spans, and then used as the only
relation endpoints. Relation labels and directions are filtered through the
training-derived ontology embedded in the jobs. Model loading is local-only.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


SUPPORTED_DATASETS = ("conll04", "scierc", "ade")
RELATION_LABEL_MODES = ("canonical", "naturalized")
NATURALIZED_RELATION_LABELS = {
    "conll04": {
        "Work_For": "work for",
        "Kill": "kill",
        "OrgBased_In": "organization based in",
        "Live_In": "live in",
        "Located_In": "located in",
    },
    "scierc": {
        "Used-for": "used for",
        "Feature-of": "feature of",
        "Hyponym-of": "hyponym of",
        "Evaluate-for": "evaluate for",
        "Part-of": "part of",
        "Compare": "compare",
        "Conjunction": "conjunction",
    },
    "ade": {
        "Adverse-Effect": "adverse effect",
    },
}
TOKEN_PATTERN = re.compile(r"\w+(?:[-_]\w+)*|\S")
FORBIDDEN_JOB_KEYS = {
    "annotation",
    "annotations",
    "entities",
    "gold",
    "gold_entities",
    "gold_relations",
    "labels",
    "relations",
}
MODEL_WEIGHT_NAMES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.fp16.safetensors",
    "model.bf16.safetensors",
)
TOKENIZER_NAMES = ("tokenizer.json", "spm.model", "vocab.txt", "vocab.json")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_ontology(ontology: dict[str, Any]) -> None:
    entity_types = ontology.get("entity_types")
    relation_types = ontology.get("relation_types")
    signatures = ontology.get("allowed_relation_signatures")
    if not isinstance(entity_types, dict) or not entity_types:
        raise ValueError("embedded ontology has no entity_types mapping")
    if not isinstance(relation_types, dict) or not relation_types:
        raise ValueError("embedded ontology has no relation_types mapping")
    if not isinstance(signatures, dict):
        raise ValueError("embedded ontology has no allowed_relation_signatures mapping")
    if set(relation_types) != set(signatures):
        raise ValueError("every relation label must have exactly one ontology signature")
    if "explicit" not in ontology.get("claim_statuses", {}):
        raise ValueError("embedded ontology does not permit claim_status=explicit")

    known_entities = set(entity_types)
    for label, signature in signatures.items():
        if not isinstance(signature, dict):
            raise ValueError(f"relation {label!r} has a malformed signature")
        source = signature.get("source")
        target = signature.get("target")
        if not isinstance(source, list) or not source:
            raise ValueError(f"relation {label!r} has no allowed source types")
        if not isinstance(target, list) or not target:
            raise ValueError(f"relation {label!r} has no allowed target types")
        unknown = (set(source) | set(target)) - known_entities
        if unknown:
            raise ValueError(f"relation {label!r} references unknown entity types: {sorted(unknown)}")


def validate_job(
    job: dict[str, Any], dataset: str, line_number: int, split: str = "validation"
) -> None:
    prefix = f"line {line_number}"
    leaked = sorted(FORBIDDEN_JOB_KEYS & set(job))
    if leaked:
        raise ValueError(f"{prefix}: forbidden prediction/gold fields in inference job: {leaked}")
    if job.get("category") != dataset:
        raise ValueError(f"{prefix}: category must be {dataset!r}")
    if job.get("experiment_mode") != "baseline":
        raise ValueError(f"{prefix}: experiment_mode must be 'baseline'")
    if split not in {"validation", "test"}:
        raise ValueError(f"{prefix}: unsupported split {split!r}")
    if f":{dataset}:{split}:" not in str(job.get("source_path", "")):
        raise ValueError(f"{prefix}: source_path is not the {dataset} {split} split")
    document_id = str(job.get("document_id", ""))
    job_id = str(job.get("job_id", ""))
    if not document_id.startswith(f"{dataset}_{split}_"):
        raise ValueError(f"{prefix}: document_id is not a {split} identifier")
    if not job_id.startswith(f"{document_id}_C"):
        raise ValueError(f"{prefix}: job_id does not belong to document_id")

    ontology = job.get("ontology")
    if not isinstance(ontology, dict):
        raise ValueError(f"{prefix}: baseline job has no embedded ontology")
    validate_ontology(ontology)

    segments = job.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"{prefix}: job has no source segments")
    seen_segments: set[str] = set()
    for segment in segments:
        segment_id = segment.get("segment_id")
        text = segment.get("text")
        start = segment.get("start")
        end = segment.get("end")
        if not isinstance(segment_id, str) or not segment_id or segment_id in seen_segments:
            raise ValueError(f"{prefix}: segment IDs must be non-empty and unique")
        seen_segments.add(segment_id)
        if not isinstance(text, str) or not text:
            raise ValueError(f"{prefix}: segment {segment_id} has no source text")
        if not isinstance(start, int) or not isinstance(end, int) or start < 0:
            raise ValueError(f"{prefix}: segment {segment_id} has invalid offsets")
        if end != start + len(text):
            raise ValueError(f"{prefix}: segment {segment_id} offsets do not match its text")


def load_validation_jobs(
    path: Path,
    dataset: str,
    offset: int = 0,
    limit: int = 0,
    split: str = "validation",
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"unsupported dataset {dataset!r}")
    expected_name = f"{split}_baseline_jobs.jsonl"
    if path.name != expected_name:
        raise ValueError(f"inference input must be named {expected_name}")
    if offset < 0 or limit < 0:
        raise ValueError("offset and limit must be non-negative")

    rows: list[dict[str, Any]] = []
    seen_jobs: set[str] = set()
    ontology: dict[str, Any] | None = None
    ontology_hash: str | None = None
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: job must be a JSON object")
            validate_job(row, dataset, line_number, split)
            job_id = row["job_id"]
            if job_id in seen_jobs:
                raise ValueError(f"line {line_number}: duplicate job_id {job_id!r}")
            seen_jobs.add(job_id)
            row_hash = canonical_digest(row["ontology"])
            if ontology_hash is None:
                ontology = row["ontology"]
                ontology_hash = row_hash
            elif row_hash != ontology_hash:
                raise ValueError(f"line {line_number}: embedded ontology changed within the job file")
            rows.append(row)

    selected = rows[offset : offset + limit if limit else None]
    if not selected:
        raise ValueError(f"the selected {split} job range is empty")
    assert ontology is not None and ontology_hash is not None
    return selected, ontology, ontology_hash


def tokenize_with_offsets(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    matches = list(TOKEN_PATTERN.finditer(text))
    return [match.group() for match in matches], [(match.start(), match.end()) for match in matches]


def token_span_for_chars(
    token_offsets: list[tuple[int, int]], start: int, end: int
) -> tuple[int, int] | None:
    indices = [
        index
        for index, (token_start, token_end) in enumerate(token_offsets)
        if token_start >= start and token_end <= end
    ]
    if not indices:
        return None
    first, last = indices[0], indices[-1]
    if token_offsets[first][0] != start or token_offsets[last][1] != end:
        return None
    if indices != list(range(first, last + 1)):
        return None
    return first, last + 1


def confidence(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(score):
        return 0.0
    return min(1.0, max(0.0, score))


def predict_segment_entities(
    segment: dict[str, Any],
    ontology: dict[str, Any],
    gliner_model: Any,
    threshold: float,
    first_entity_number: int,
) -> tuple[
    list[dict[str, Any]],
    list[list[Any]],
    dict[tuple[int, int], dict[str, Any]],
    list[str],
    dict[str, int],
]:
    text = segment["text"]
    tokens, token_offsets = tokenize_with_offsets(text)
    entity_labels = list(ontology["entity_types"])
    raw_predictions = gliner_model.predict_entities(
        text,
        entity_labels,
        flat_ner=True,
        threshold=threshold,
    )
    if not isinstance(raw_predictions, list):
        raise TypeError("GLiNER predict_entities must return a list")

    diagnostics = {
        "raw_entities": len(raw_predictions),
        "dropped_entity_label": 0,
        "dropped_entity_span": 0,
        "deduplicated_entities": 0,
        "kept_entities": 0,
    }
    candidates: dict[tuple[int, int], dict[str, Any]] = {}
    for raw in raw_predictions:
        if not isinstance(raw, dict) or raw.get("label") not in ontology["entity_types"]:
            diagnostics["dropped_entity_label"] += 1
            continue
        start, end = raw.get("start"), raw.get("end")
        if not isinstance(start, int) or not isinstance(end, int) or not (0 <= start < end <= len(text)):
            diagnostics["dropped_entity_span"] += 1
            continue
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        predicted_text = raw.get("text")
        if predicted_text is not None and predicted_text != text[start:end]:
            diagnostics["dropped_entity_span"] += 1
            continue
        token_span = token_span_for_chars(token_offsets, start, end)
        if token_span is None:
            diagnostics["dropped_entity_span"] += 1
            continue
        candidate = {
            "start": start,
            "end": end,
            "text": text[start:end],
            "type": raw["label"],
            "score": confidence(raw.get("score")),
            "token_span": token_span,
        }
        previous = candidates.get(token_span)
        if previous is None:
            candidates[token_span] = candidate
        else:
            diagnostics["deduplicated_entities"] += 1
            if (candidate["score"], candidate["type"]) > (previous["score"], previous["type"]):
                candidates[token_span] = candidate

    ordered = sorted(candidates.values(), key=lambda item: (item["start"], item["end"], item["type"]))
    entities: list[dict[str, Any]] = []
    glirel_ner: list[list[Any]] = []
    by_token_span: dict[tuple[int, int], dict[str, Any]] = {}
    segment_start = segment["start"]
    for local_index, item in enumerate(ordered):
        entity = {
            "id": f"E{first_entity_number + local_index}",
            "text": item["text"],
            "normalized_name": None,
            "type": item["type"],
            "evidence": {
                "text": item["text"],
                "segment_id": segment["segment_id"],
                "page": segment.get("page"),
                "start": segment_start + item["start"],
                "end": segment_start + item["end"],
            },
            "confidence": item["score"],
            "review_status": "pending",
            "created_by": "gliner_small-v2.1",
        }
        token_start, token_end = item["token_span"]
        entities.append(entity)
        glirel_ner.append([token_start, token_end - 1, entity["type"], entity["text"]])
        by_token_span[(token_start, token_end)] = entity
    diagnostics["kept_entities"] = len(entities)
    return entities, glirel_ner, by_token_span, tokens, diagnostics


def viable_relation_labels(
    entities: list[dict[str, Any]], ontology: dict[str, Any]
) -> list[str]:
    signatures = ontology["allowed_relation_signatures"]
    labels: list[str] = []
    for label in ontology["relation_types"]:
        signature = signatures[label]
        if any(
            source is not target
            and source["type"] in signature["source"]
            and target["type"] in signature["target"]
            for source in entities
            for target in entities
        ):
            labels.append(label)
    return labels


def relation_prompt_labels(
    dataset: str, canonical_labels: list[str], mode: str
) -> tuple[list[str], dict[str, str]]:
    if mode not in RELATION_LABEL_MODES:
        raise ValueError(f"unsupported relation label mode {mode!r}")
    if mode == "canonical":
        aliases = {label: label for label in canonical_labels}
    else:
        declared = NATURALIZED_RELATION_LABELS.get(dataset, {})
        missing = sorted(set(canonical_labels) - set(declared))
        if missing:
            raise ValueError(
                f"naturalized relation labels are not declared for {dataset}: {missing}"
            )
        aliases = {label: declared[label] for label in canonical_labels}
    reverse = {alias: canonical for canonical, alias in aliases.items()}
    if len(reverse) != len(aliases):
        raise ValueError(f"relation label aliases are not unique for {dataset}")
    return [aliases[label] for label in canonical_labels], reverse


def _relation_position(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    start, end = value
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
        return None
    return start, end


def predict_segment_relations(
    segment: dict[str, Any],
    entities: list[dict[str, Any]],
    glirel_ner: list[list[Any]],
    entities_by_token_span: dict[tuple[int, int], dict[str, Any]],
    tokens: list[str],
    ontology: dict[str, Any],
    glirel_model: Any,
    threshold: float,
    top_k: int,
    relation_label_mode: str = "canonical",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    diagnostics = {
        "raw_relations": 0,
        "dropped_relation_label": 0,
        "dropped_relation_endpoint": 0,
        "dropped_relation_signature": 0,
        "deduplicated_relations": 0,
        "kept_relations": 0,
    }
    canonical_labels = viable_relation_labels(entities, ontology)
    if len(entities) < 2 or not canonical_labels:
        return [], diagnostics
    prompt_labels, canonical_by_prompt = relation_prompt_labels(
        str(segment.get("dataset", "")), canonical_labels, relation_label_mode
    )

    # Request every above-threshold label. Applying top-k before signature
    # filtering can hide the best legal label behind an illegal direction.
    raw_predictions = glirel_model.predict_relations(
        tokens,
        prompt_labels,
        flat_ner=True,
        threshold=threshold,
        ner=glirel_ner,
        top_k=-1,
    )
    if not isinstance(raw_predictions, list):
        raise TypeError("GLiREL predict_relations must return a list")
    diagnostics["raw_relations"] = len(raw_predictions)

    signatures = ontology["allowed_relation_signatures"]
    deduplicated: dict[tuple[str, str, str], dict[str, Any]] = {}
    for raw in raw_predictions:
        if not isinstance(raw, dict) or raw.get("label") not in canonical_by_prompt:
            diagnostics["dropped_relation_label"] += 1
            continue
        head_pos = _relation_position(raw.get("head_pos"))
        tail_pos = _relation_position(raw.get("tail_pos"))
        source = entities_by_token_span.get(head_pos) if head_pos else None
        target = entities_by_token_span.get(tail_pos) if tail_pos else None
        if source is None or target is None or source["id"] == target["id"]:
            diagnostics["dropped_relation_endpoint"] += 1
            continue
        label = canonical_by_prompt[raw["label"]]
        signature = signatures[label]
        if source["type"] not in signature["source"] or target["type"] not in signature["target"]:
            diagnostics["dropped_relation_signature"] += 1
            continue
        key = source["id"], label, target["id"]
        candidate = {
            "source_id": source["id"],
            "type": label,
            "target_id": target["id"],
            "claim_status": "explicit",
            "evidence": [
                {
                    "text": segment["text"],
                    "segment_id": segment["segment_id"],
                    "page": segment.get("page"),
                    "start": segment["start"],
                    "end": segment["end"],
                }
            ],
            "confidence": confidence(raw.get("score")),
            "review_status": "pending",
            "created_by": "glirel-large-v0",
        }
        previous = deduplicated.get(key)
        if previous is None or candidate["confidence"] > previous["confidence"]:
            if previous is not None:
                diagnostics["deduplicated_relations"] += 1
            deduplicated[key] = candidate
        else:
            diagnostics["deduplicated_relations"] += 1

    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for relation in deduplicated.values():
        by_pair[(relation["source_id"], relation["target_id"])].append(relation)
    kept: list[dict[str, Any]] = []
    for pair in sorted(by_pair):
        ranked = sorted(by_pair[pair], key=lambda item: (-item["confidence"], item["type"]))
        kept.extend(ranked[:top_k])
    diagnostics["kept_relations"] = len(kept)
    return kept, diagnostics


def _add_counts(target: dict[str, int], values: dict[str, int]) -> None:
    for key, value in values.items():
        target[key] = target.get(key, 0) + value


def predict_job(
    job: dict[str, Any],
    gliner_model: Any,
    glirel_model: Any,
    entity_threshold: float,
    relation_threshold: float,
    top_k: int,
    relation_label_mode: str = "canonical",
) -> tuple[dict[str, Any], dict[str, int]]:
    ontology = job["ontology"]
    all_entities: list[dict[str, Any]] = []
    all_relations: list[dict[str, Any]] = []
    diagnostics: dict[str, int] = {}
    for segment in job["segments"]:
        segment = {**segment, "dataset": job["category"]}
        entities, ner, by_span, tokens, entity_diagnostics = predict_segment_entities(
            segment,
            ontology,
            gliner_model,
            entity_threshold,
            len(all_entities) + 1,
        )
        relations, relation_diagnostics = predict_segment_relations(
            segment,
            entities,
            ner,
            by_span,
            tokens,
            ontology,
            glirel_model,
            relation_threshold,
            top_k,
            relation_label_mode,
        )
        all_entities.extend(entities)
        all_relations.extend(relations)
        _add_counts(diagnostics, entity_diagnostics)
        _add_counts(diagnostics, relation_diagnostics)

    for index, relation in enumerate(all_relations, 1):
        relation["id"] = f"R{index}"
    annotation = {
        "schema_version": ontology.get("annotation_schema_version", "0.1.0"),
        "document_id": job["document_id"],
        "language": job.get("language", "unknown"),
        "entities": all_entities,
        "relations": all_relations,
        "review": {
            "status": "unreviewed",
            "reviewers": [],
            "notes": "Gold-independent local-only GLiNER + GLiREL zero-shot baseline.",
        },
    }
    return {"job_id": job["job_id"], "annotation": annotation}, diagnostics


def _hub_dir(hf_home: Path) -> Path:
    return hf_home if hf_home.name == "hub" else hf_home / "hub"


def resolve_local_model(reference: str, hf_home: Path, revision: str = "main") -> Path:
    direct = Path(reference).expanduser()
    if direct.is_dir():
        return direct.resolve()
    if direct.exists():
        raise ValueError(f"model reference is not a directory: {direct}")
    if "/" not in reference:
        raise FileNotFoundError(f"local model directory does not exist: {reference}")

    repository = _hub_dir(hf_home) / f"models--{reference.replace('/', '--')}"
    revision_file = repository / "refs" / revision
    commit = revision_file.read_text(encoding="utf-8").strip() if revision_file.is_file() else revision
    snapshot = repository / "snapshots" / commit
    if snapshot.is_dir():
        return snapshot.resolve()
    snapshots = sorted(path for path in (repository / "snapshots").glob("*") if path.is_dir())
    if revision == "main" and len(snapshots) == 1:
        return snapshots[0].resolve()
    raise FileNotFoundError(
        f"model {reference!r} is not available in local cache {_hub_dir(hf_home)}; "
        "download it before starting this offline baseline"
    )


def require_files(directory: Path, exact: tuple[str, ...], any_of: tuple[str, ...] = ()) -> None:
    missing = [name for name in exact if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{directory} is missing required files: {missing}")
    if any_of and not any((directory / name).is_file() for name in any_of):
        raise FileNotFoundError(f"{directory} needs one of: {list(any_of)}")


def ensure_python_dependencies(glirel_source: Path) -> None:
    package = glirel_source / "glirel" / "__init__.py"
    if not package.is_file():
        raise FileNotFoundError(f"GLiREL source checkout is incomplete: {package}")
    missing = [name for name in ("gliner", "torch", "transformers", "seqeval") if importlib.util.find_spec(name) is None]
    if missing:
        raise ModuleNotFoundError(f"missing Python dependencies: {', '.join(missing)}")


def model_preflight(args: argparse.Namespace) -> dict[str, Path]:
    hf_home = args.hf_home.expanduser().resolve()
    gliner_checkpoint = resolve_local_model(args.gliner_model, hf_home)
    glirel_checkpoint = resolve_local_model(args.glirel_model, hf_home)
    require_files(gliner_checkpoint, ("gliner_config.json",), MODEL_WEIGHT_NAMES)
    require_files(glirel_checkpoint, ("glirel_config.json",), MODEL_WEIGHT_NAMES)

    gliner_config = json.loads((gliner_checkpoint / "gliner_config.json").read_text(encoding="utf-8"))
    glirel_config = json.loads((glirel_checkpoint / "glirel_config.json").read_text(encoding="utf-8"))
    gliner_backbone_ref = args.gliner_backbone or gliner_config.get("model_name")
    glirel_backbone_ref = args.glirel_backbone or glirel_config.get("model_name")
    if not gliner_backbone_ref or not glirel_backbone_ref:
        raise ValueError("checkpoint config does not identify its transformer backbone")
    gliner_backbone = resolve_local_model(str(gliner_backbone_ref), hf_home)
    glirel_backbone = resolve_local_model(str(glirel_backbone_ref), hf_home)
    require_files(gliner_backbone, ("config.json",), TOKENIZER_NAMES)
    require_files(glirel_backbone, ("config.json",), TOKENIZER_NAMES)
    if not args.glirel_init_from_checkpoint:
        require_files(glirel_backbone, (), MODEL_WEIGHT_NAMES)
    ensure_python_dependencies(args.glirel_source)
    return {
        "gliner_checkpoint": gliner_checkpoint,
        "gliner_backbone": gliner_backbone,
        "glirel_checkpoint": glirel_checkpoint,
        "glirel_backbone": glirel_backbone,
    }


@contextlib.contextmanager
def patched_checkpoint(
    checkpoint: Path, config_name: str, backbone: Path
) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="local-ie-checkpoint-") as directory:
        target = Path(directory)
        config = json.loads((checkpoint / config_name).read_text(encoding="utf-8"))
        config["model_name"] = str(backbone)
        (target / config_name).write_text(json.dumps(config), encoding="utf-8")
        for source in checkpoint.iterdir():
            if source.name == config_name or not source.is_file():
                continue
            (target / source.name).symlink_to(source.resolve())
        yield target


@contextlib.contextmanager
def glirel_config_only_backbone(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    from glirel.modules import transformer_embeddings
    from transformers import AutoConfig

    real_auto_model = transformer_embeddings.AutoModel

    class ConfigOnlyAutoModel:
        @classmethod
        def from_pretrained(cls, model_name: str, *_args: Any, **_kwargs: Any) -> Any:
            config = AutoConfig.from_pretrained(model_name, local_files_only=True)
            return real_auto_model.from_config(config, trust_remote_code=True)

    transformer_embeddings.AutoModel = ConfigOnlyAutoModel
    try:
        yield
    finally:
        transformer_embeddings.AutoModel = real_auto_model


def _torch_dtype_name(device: str, requested: str) -> str:
    if requested != "auto":
        return requested
    return "float16" if device.startswith("cuda") else "float32"


def load_models(args: argparse.Namespace, paths: dict[str, Path]) -> tuple[Any, Any, str]:
    os.environ["HF_HOME"] = str(args.hf_home.expanduser().resolve())
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    source = str(args.glirel_source.resolve())
    if source not in sys.path:
        sys.path.insert(0, source)

    import torch
    from gliner import GLiNER
    from glirel import GLiREL

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    dtype_name = _torch_dtype_name(args.device, args.dtype)
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]
    if args.device == "cpu" and dtype == torch.float16:
        raise ValueError("float16 CPU inference is unsupported; use --dtype float32 or bfloat16")

    with patched_checkpoint(
        paths["gliner_checkpoint"], "gliner_config.json", paths["gliner_backbone"]
    ) as local_gliner:
        gliner_model = GLiNER.from_pretrained(
            str(local_gliner),
            local_files_only=True,
            map_location="cpu",
            strict=True,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
    gliner_model.to(args.device)
    gliner_model.eval()

    with patched_checkpoint(
        paths["glirel_checkpoint"], "glirel_config.json", paths["glirel_backbone"]
    ) as local_glirel:
        with glirel_config_only_backbone(args.glirel_init_from_checkpoint):
            glirel_model = GLiREL.from_pretrained(
                str(local_glirel),
                local_files_only=True,
                map_location="cpu",
                strict=True,
            )
    if dtype == torch.float16:
        glirel_model.half()
    elif dtype == torch.bfloat16:
        glirel_model.bfloat16()
    glirel_model.to(args.device)
    glirel_model.eval()
    return gliner_model, glirel_model, dtype_name


def existing_job_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    result: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("job_id"), str) or "annotation" not in row:
                raise ValueError(f"existing output line {line_number} is not a wrapped annotation")
            if row["job_id"] in result:
                raise ValueError(f"existing output contains duplicate job_id {row['job_id']!r}")
            result.add(row["job_id"])
    return result


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def summary_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}.summary.json")


def error_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}.errors.jsonl")


def run(args: argparse.Namespace) -> int:
    if not (0.0 <= args.entity_threshold <= 1.0 and 0.0 <= args.relation_threshold <= 1.0):
        raise ValueError("entity and relation thresholds must be between 0 and 1")
    if args.top_k < 1:
        raise ValueError("top-k must be at least 1")
    jobs_path = args.jobs or (
        Path("data/processed/public_benchmarks_full")
        / args.dataset
        / f"{args.split}_baseline_jobs.jsonl"
    )
    jobs, ontology, ontology_hash = load_validation_jobs(
        jobs_path, args.dataset, args.offset, args.limit, args.split
    )
    paths = model_preflight(args)
    preflight_report = {
        "status": "ready",
        "dataset": args.dataset,
        "split": args.split,
        "formal_test_read": args.split == "test",
        "jobs": len(jobs),
        "jobs_path": str(jobs_path),
        "ontology_sha256": ontology_hash,
        "local_only": True,
        "models": {key: str(value) for key, value in paths.items()},
        "glirel_backbone_initialized_from_checkpoint": args.glirel_init_from_checkpoint,
        "relation_label_mode": args.relation_label_mode,
        "relation_label_aliases": (
            NATURALIZED_RELATION_LABELS[args.dataset]
            if args.relation_label_mode == "naturalized"
            else {label: label for label in ontology["relation_types"]}
        ),
    }
    if args.preflight_only:
        print(json.dumps(preflight_report, ensure_ascii=False, indent=2))
        return 0

    if args.output is None:
        raise ValueError("--output is required unless --preflight-only is used")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    errors = args.errors or error_path(args.output)
    summary = args.summary or summary_path(args.output)
    if args.overwrite:
        args.output.write_text("", encoding="utf-8")
        errors.write_text("", encoding="utf-8")
    elif args.output.exists() and not args.resume:
        raise FileExistsError(f"output exists; pass --resume or --overwrite: {args.output}")

    completed = existing_job_ids(args.output)
    selected_ids = {job["job_id"] for job in jobs}
    unknown = completed - selected_ids
    if unknown:
        raise ValueError(f"existing output contains jobs outside the selected range: {sorted(unknown)[:3]}")
    pending = [job for job in jobs if job["job_id"] not in completed]
    if not pending:
        result = {**preflight_report, "status": "complete", "completed": len(completed), "failed": 0}
        write_json_atomic(summary, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    gliner_model, glirel_model, dtype_name = load_models(args, paths)
    started = time.monotonic()
    processed = 0
    failed = 0
    diagnostics: dict[str, int] = {}
    mode = "a" if args.output.exists() else "w"
    with args.output.open(mode, encoding="utf-8") as output_stream, errors.open(
        "a", encoding="utf-8"
    ) as error_stream:
        for job in pending:
            try:
                prediction, job_diagnostics = predict_job(
                    job,
                    gliner_model,
                    glirel_model,
                    args.entity_threshold,
                    args.relation_threshold,
                    args.top_k,
                    args.relation_label_mode,
                )
                output_stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                output_stream.flush()
                processed += 1
                _add_counts(diagnostics, job_diagnostics)
                if args.fsync_every and processed % args.fsync_every == 0:
                    os.fsync(output_stream.fileno())
            except Exception as exc:
                failed += 1
                error_stream.write(
                    json.dumps(
                        {
                            "job_id": job.get("job_id"),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                            "timestamp": utc_now(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                error_stream.flush()
                if not args.continue_on_error:
                    raise
            if (processed + failed) % args.log_every == 0:
                print(
                    json.dumps(
                        {
                            "dataset": args.dataset,
                            "attempted": processed + failed,
                            "remaining": len(pending) - processed - failed,
                            "failed": failed,
                            "elapsed_seconds": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )

    result = {
        **preflight_report,
        "status": "complete" if failed == 0 else "completed_with_errors",
        "dtype": dtype_name,
        "device": args.device,
        "entity_threshold": args.entity_threshold,
        "relation_threshold": args.relation_threshold,
        "relation_label_mode": args.relation_label_mode,
        "top_k_after_signature_filter": args.top_k,
        "completed_before_run": len(completed),
        "completed_this_run": processed,
        "failed": failed,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "diagnostics": diagnostics,
        "output": str(args.output),
        "errors": str(errors),
        "finished_at": utc_now(),
    }
    write_json_atomic(summary, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if failed else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--jobs",
        type=Path,
        help="Must be the selected dataset's SPLIT_baseline_jobs.jsonl",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--errors", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 processes the full selected split")
    parser.add_argument("--entity-threshold", type=float, default=0.5)
    parser.add_argument("--relation-threshold", type=float, default=0.5)
    parser.add_argument(
        "--relation-label-mode",
        choices=RELATION_LABEL_MODES,
        default="canonical",
        help="Prompt-label representation; outputs always use canonical dataset labels",
    )
    parser.add_argument("--top-k", type=int, default=1, help="Legal labels retained per directed entity pair")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=Path(os.environ.get("HF_HOME", "/ds2/xuelin/cache/huggingface")),
    )
    parser.add_argument("--gliner-model", default="urchade/gliner_small-v2.1")
    parser.add_argument("--gliner-backbone")
    parser.add_argument("--glirel-model", default="jackboyla/glirel-large-v0")
    parser.add_argument("--glirel-backbone")
    parser.add_argument(
        "--glirel-source",
        type=Path,
        default=Path("tools/external-baselines/glirel"),
    )
    parser.add_argument(
        "--glirel-init-from-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Initialize the backbone architecture locally, then load all weights from the full GLiREL checkpoint",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--fsync-every", type=int, default=25)
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    if args.log_every < 1 or args.fsync_every < 0:
        parser.error("--log-every must be positive and --fsync-every non-negative")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
