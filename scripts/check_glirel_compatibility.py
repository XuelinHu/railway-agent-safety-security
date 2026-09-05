#!/usr/bin/env python3
"""Gate GLiREL validation on a frozen large-v0 runtime snapshot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_gliner_glirel_validation import (  # noqa: E402
    MODEL_WEIGHT_NAMES,
    glirel_config_only_backbone,
    patched_checkpoint,
    require_files,
    resolve_local_model,
)


CANARY_TEXT = (
    'Derren Nesbitt had a history of being cast in "Doctor Who", having played '
    'villainous warlord Tegana in the 1964 First Doctor serial "Marco Polo".'
)
CANARY_TOKENS = [
    "Derren",
    "Nesbitt",
    "had",
    "a",
    "history",
    "of",
    "being",
    "cast",
    "in",
    '"',
    "Doctor",
    "Who",
    '"',
    ",",
    "having",
    "played",
    "villainous",
    "warlord",
    "Tegana",
    "in",
    "the",
    "1964",
    "First",
    "Doctor",
    "serial",
    '"',
    "Marco",
    "Polo",
    '"',
    ".",
]
CANARY_LABELS = [
    "country of origin",
    "licensed to broadcast to",
    "father",
    "followed by",
    "characters",
]
CANARY_NER = [
    [26, 27, "PERSON", "Marco Polo"],
    [22, 23, "Q2989412", "First Doctor"],
]
EXPECTED_RELATIONS = {
    ((22, 24), (26, 28)): {
        "label": "followed by",
        "score": 0.0028011202812194824,
    },
    ((26, 28), (22, 24)): {
        "label": "followed by",
        "score": 0.0027414096985012293,
    },
}
REFERENCE_CHECKPOINT_REVISION = "40a523e12a8432d6da364cf2a195a28755ff04d3"
REFERENCE_SOURCE_REVISION = "e3fd8fe637d679d133cebdf7a7007b359ae2dae8"
README_MODEL_SWAP_REVISION = "23921bc95360b4a7ba9a512bd0a19a01c053b248"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_revision(source: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def normalize_prediction(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "head_pos": list(row.get("head_pos", [])),
        "tail_pos": list(row.get("tail_pos", [])),
        "head_text": row.get("head_text"),
        "tail_text": row.get("tail_text"),
        "label": row.get("label"),
        "score": float(row.get("score", 0.0)),
    }


def canary_passes(predictions: list[dict[str, Any]], score_atol: float) -> bool:
    if len(predictions) != len(EXPECTED_RELATIONS):
        return False
    for row in predictions:
        pair = (tuple(row.get("head_pos", ())), tuple(row.get("tail_pos", ())))
        expected = EXPECTED_RELATIONS.get(pair)
        if expected is None or row.get("label") != expected["label"]:
            return False
        if abs(float(row.get("score", 0.0)) - float(expected["score"])) > score_atol:
            return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/public_horizontal_validation/gliner_glirel_t0/compatibility_canary.json"
        ),
    )
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=Path(os.environ.get("HF_HOME", "/ds2/xuelin/cache/huggingface")),
    )
    parser.add_argument("--glirel-model", default="jackboyla/glirel-large-v0")
    parser.add_argument("--glirel-backbone")
    parser.add_argument(
        "--glirel-source", type=Path, default=Path("tools/external-baselines/glirel")
    )
    parser.add_argument("--score-atol", type=float, default=1e-6)
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    if not 0.0 < args.score_atol < 1.0:
        raise ValueError("--score-atol must be between 0 and 1")

    os.environ["HF_HOME"] = str(args.hf_home.expanduser().resolve())
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    source = args.glirel_source.expanduser().resolve()
    source_package = source / "glirel" / "__init__.py"
    if not source_package.is_file():
        raise FileNotFoundError(f"GLiREL source checkout is incomplete: {source_package}")
    sys.path.insert(0, str(source))

    checkpoint = resolve_local_model(args.glirel_model, args.hf_home)
    require_files(checkpoint, ("glirel_config.json",), MODEL_WEIGHT_NAMES)
    config = json.loads((checkpoint / "glirel_config.json").read_text(encoding="utf-8"))
    backbone_reference = args.glirel_backbone or config.get("model_name")
    if not backbone_reference:
        raise ValueError("GLiREL checkpoint config does not identify its transformer backbone")
    backbone = resolve_local_model(str(backbone_reference), args.hf_home)
    require_files(backbone, ("config.json",))
    weight = next(path for name in MODEL_WEIGHT_NAMES if (path := checkpoint / name).is_file())

    import torch
    import transformers
    from glirel import GLiREL, __version__ as glirel_version

    with patched_checkpoint(checkpoint, "glirel_config.json", backbone) as local_checkpoint:
        with glirel_config_only_backbone(True):
            model = GLiREL.from_pretrained(
                str(local_checkpoint),
                local_files_only=True,
                map_location="cpu",
                strict=True,
            )
    model.to("cpu")
    model.eval()
    with torch.inference_mode():
        raw_predictions = model.predict_relations(
            CANARY_TOKENS,
            CANARY_LABELS,
            flat_ner=True,
            threshold=0.0,
            ner=CANARY_NER,
            top_k=1,
        )
    predictions = [normalize_prediction(row) for row in raw_predictions]
    predictions.sort(key=lambda row: (-row["score"], row["label"]))
    compatible = canary_passes(predictions, args.score_atol)
    payload = {
        "schema_version": "glirel-compatibility-canary-v1",
        "status": "passed" if compatible else "failed",
        "runtime_compatible": compatible,
        "checked_at": utc_now(),
        "device": "cpu",
        "offline": True,
        "contract": {
            "source": "project-frozen large-v0 reference-runtime snapshot",
            "text": CANARY_TEXT,
            "labels": CANARY_LABELS,
            "ner": CANARY_NER,
            "threshold": 0.0,
            "top_k": 1,
            "expected_relations": [
                {
                    "head_pos": list(head),
                    "tail_pos": list(tail),
                    "label": expected["label"],
                    "score": expected["score"],
                }
                for (head, tail), expected in sorted(EXPECTED_RELATIONS.items())
            ],
            "score_absolute_tolerance": args.score_atol,
            "reference_environment": {
                "glirel_source_revision": REFERENCE_SOURCE_REVISION,
                "glirel_release": "1.0.1",
                "flair": "0.15.0",
                "transformers": "4.47.1",
            },
            "semantic_accuracy_claim": False,
            "upstream_readme_warning": {
                "status": "stale_beta_expected_output",
                "model_swap_revision": README_MODEL_SWAP_REVISION,
                "detail": (
                    "The README changed glirel_beta to glirel-large-v0 without "
                    "updating the beta model's characters~=0.992 expected output."
                ),
            },
        },
        "observed": {
            "relations": predictions,
            "relation_count": len(predictions),
        },
        "provenance": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "huggingface_hub": package_version("huggingface_hub"),
            "glirel": glirel_version,
            "glirel_source": str(source),
            "glirel_source_revision": git_revision(source),
            "checkpoint": str(checkpoint),
            "checkpoint_revision": checkpoint.name,
            "reference_checkpoint_revision": REFERENCE_CHECKPOINT_REVISION,
            "checkpoint_weight": weight.name,
            "checkpoint_weight_bytes": weight.stat().st_size,
            "checkpoint_weight_sha256": file_sha256(weight),
            "backbone": str(backbone),
            "config_only_backbone_initialization": True,
        },
    }
    write_json_atomic(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if compatible else 1


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except Exception as exc:  # noqa: BLE001
        payload = {
            "schema_version": "glirel-compatibility-canary-v1",
            "status": "error",
            "runtime_compatible": False,
            "checked_at": utc_now(),
            "device": "cpu",
            "offline": True,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }
        write_json_atomic(args.output, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
