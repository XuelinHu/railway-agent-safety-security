#!/usr/bin/env python3
"""Reproduce the frozen KG V3 reject-option pipeline without reading gold.

The runner first applies the deterministic evidence/signature verifier to the
expanded V2 predictions.  It then retains V2 entities supported by at least one
of the three frozen V1-style acceptance signals and keeps only verified
relations whose endpoints survive the entity gate.  Input hashes make the
validation checkpoint immutable; a non-validation config requires an explicit
opt-in so the frozen test split cannot be opened accidentally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fuse_kg_v1_v2_predictions import (  # noqa: E402
    fuse_job,
    keyed_annotations,
    load_jsonl,
)
from verify_relations import verify_annotation  # noqa: E402


FROZEN_ACCEPTANCE_RULES = (
    "v1_v2_exact_agreement",
    "source_gated_anchor_type_match",
    "verified_relation_endpoint",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def checked_input(root: Path, item: dict[str, Any], label: str) -> Path:
    path = resolve(root, str(item.get("path", "")))
    expected = str(item.get("sha256", ""))
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    actual = sha256(path)
    if not expected or actual != expected:
        raise ValueError(
            f"{label} hash mismatch: expected {expected or '<missing>'}, got {actual}"
        )
    return path


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_config(config: dict[str, Any], allow_non_validation: bool) -> None:
    if config.get("status") != "frozen-validation-checkpoint":
        raise ValueError("config is not a frozen validation checkpoint")
    split = config.get("selection_split")
    if split != "validation" and not allow_non_validation:
        raise ValueError(
            f"refusing selection_split={split!r}; pass --allow-non-validation explicitly"
        )
    pipeline = config.get("pipeline", {})
    gate = pipeline.get("entity_gate", {})
    rules = tuple(gate.get("acceptance_any_of", []))
    if rules != FROZEN_ACCEPTANCE_RULES:
        raise ValueError(f"frozen entity-gate rules changed: {rules}")
    if gate.get("relation_mode") != "verified":
        raise ValueError("frozen pipeline requires relation_mode=verified")
    verifier = pipeline.get("relation_verifier", {})
    if verifier.get("require_local_cooccurrence") is not True:
        raise ValueError("frozen pipeline requires local relation co-occurrence")


def run(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_config(config, args.allow_non_validation)

    ontology_path = checked_input(root, config["ontology"], "ontology")
    input_paths = {
        label: checked_input(root, item, label)
        for label, item in config["inputs"].items()
    }
    output_paths = {
        label: resolve(root, value)
        for label, value in config["outputs"].items()
    }
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "frozen outputs already exist; use --overwrite for these exact targets: "
            + ", ".join(existing)
        )

    ontology = yaml.safe_load(ontology_path.read_text(encoding="utf-8"))
    v1 = keyed_annotations(load_jsonl(input_paths["v1_predictions"]))
    v2 = keyed_annotations(load_jsonl(input_paths["v2_predictions"]))
    jobs = {
        row["job_id"]: row for row in load_jsonl(input_paths["jobs"])
    }
    requested = list(jobs)
    missing = {
        "v1_predictions": sorted(set(requested) - set(v1)),
        "v2_predictions": sorted(set(requested) - set(v2)),
    }
    if any(missing.values()):
        raise ValueError(f"missing requested predictions: {missing}")

    verified_by_job: dict[str, dict[str, Any]] = {}
    verified_rows: list[dict[str, Any]] = []
    relation_audit_rows: list[dict[str, Any]] = []
    relation_counts: Counter[str] = Counter()
    for job_id in requested:
        annotation = v2[job_id]
        verified, audit = verify_annotation(
            annotation, ontology, require_local_cooccurrence=True
        )
        verified_by_job[job_id] = verified
        verified_rows.append({"job_id": job_id, "annotation": verified})
        relation_counts["input"] += len(annotation.get("relations", []))
        for item in audit:
            relation_counts["accepted" if item["accepted"] else "rejected"] += 1
            relation_audit_rows.append(
                {
                    "job_id": job_id,
                    "document_id": annotation.get("document_id"),
                    **item,
                }
            )

    final_rows: list[dict[str, Any]] = []
    entity_audit_rows: list[dict[str, Any]] = []
    fusion_counts: Counter[str] = Counter()
    for job_id in requested:
        fused, audit, counts = fuse_job(
            job_id,
            v1[job_id],
            v2[job_id],
            verified_by_job[job_id],
            jobs[job_id],
            "verified",
        )
        final_rows.append({"job_id": job_id, "annotation": fused})
        entity_audit_rows.extend(audit)
        fusion_counts.update(counts)

    write_jsonl(output_paths["v2_verified_predictions"], verified_rows)
    write_jsonl(output_paths["relation_verifier_audit"], relation_audit_rows)
    write_jsonl(output_paths["final_predictions"], final_rows)
    write_jsonl(output_paths["entity_gate_audit"], entity_audit_rows)

    output_hashes = {
        label: sha256(path)
        for label, path in output_paths.items()
        if label != "run_manifest"
    }
    manifest = {
        "pipeline_id": config["pipeline_id"],
        "status": config["status"],
        "selection_split": config["selection_split"],
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "gold_read": False,
        "jobs": len(requested),
        "input_sha256": {
            "ontology": sha256(ontology_path),
            **{label: sha256(path) for label, path in input_paths.items()},
        },
        "relation_verifier": dict(sorted(relation_counts.items())),
        "entity_gate": dict(sorted(fusion_counts.items())),
        "outputs": {
            label: str(path) for label, path in output_paths.items()
        },
        "output_sha256": output_hashes,
        "selection_record": config.get("selection_record", {}),
    }
    output_paths["run_manifest"].parent.mkdir(parents=True, exist_ok=True)
    output_paths["run_manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/kg_v3_frozen_validation.yaml"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-non-validation",
        action="store_true",
        help="Explicitly allow a separately reviewed non-validation config.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

