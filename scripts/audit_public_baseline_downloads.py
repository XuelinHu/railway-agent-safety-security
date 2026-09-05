#!/usr/bin/env python3
"""Build an offline integrity inventory for downloaded public baselines."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPOSITORIES = {
    "spert": ("tools/external-baselines/spert", "a53f468bebfa9de6d66456dcfbf4b62aef237bf7"),
    "instructuie": (
        "tools/external-baselines/instructuie",
        "052a536abf9a01aa6bce1982fac2e803395e5f5c",
    ),
    "mirror": ("tools/external-baselines/mirror", "8a6b5109aa2ad66b63c3a5bb3432b39b66eb36e3"),
    "oneke": ("tools/external-baselines/oneke", "55f701a5d99d76bbad092bd2552d45a82199e516"),
    "pl_marker": (
        "tools/external-baselines/pl-marker",
        "dd19a854abec018fd38b0b42312a0c233aef64e2",
    ),
}

SPERT_RELEASE_BYTES = {
    "conll04": 433_380_286,
    "scierc": 439_924_262,
    "ade": 433_330_110,
}

HF_MODELS = {
    "instructuie": {
        "cache_name": "models--ZWK--InstructUIE",
        "revision": "48f45b25a01df1798f8c1c31751a973adb7e8647",
        "required": ("config.json", "pytorch_model.bin.index.json", "tokenizer.json", "spiece.model"),
        "index": "pytorch_model.bin.index.json",
        "shards": 5,
        "weight_bytes": 45_066_534_897,
    },
    "oneke": {
        "cache_name": "models--zjunlp--OneKE",
        "revision": "696148c0581b29f530af738ddab500deaa8fe8f2",
        "required": ("config.json", "pytorch_model.bin.index.json", "tokenizer.model"),
        "index": "pytorch_model.bin.index.json",
        "shards": 3,
        "weight_bytes": 26_508_955_138,
    },
    "bert_base_cased": {
        "cache_name": "models--google-bert--bert-base-cased",
        "revision": "cd5ef92a9fb2f889e972770a36d4ed042daf221e",
        "required": ("config.json", "pytorch_model.bin", "vocab.txt"),
        "weight": "pytorch_model.bin",
    },
    "scibert": {
        "cache_name": "models--allenai--scibert_scivocab_uncased",
        "revision": "24f92d32b1bfb0bcaf9ab193ff3ad01e87732fc1",
        "required": ("config.json", "pytorch_model.bin", "vocab.txt"),
        "weight": "pytorch_model.bin",
    },
    "roberta_large": {
        "cache_name": "models--FacebookAI--roberta-large",
        "revision": "722cf37b1afa9454edce342e7895e588b6ff1d59",
        "required": ("config.json", "pytorch_model.bin", "tokenizer.json", "vocab.json", "merges.txt"),
        "weight": "pytorch_model.bin",
    },
}

MIRROR_ARCHIVES = {
    "mirror_outputs.zip": {
        "bytes": 4_357_712_150,
        "required_suffix": "SchemaGuidedInstructBertModel.best.pth",
    },
    "resources.zip": {"bytes": 306_218_565, "required_suffix": None},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_git(directory: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", str(directory), *arguments),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def inspect_repository(workspace: Path, relative: str, expected_commit: str) -> dict[str, Any]:
    directory = workspace / relative
    errors: list[str] = []
    commit = None
    clean = False
    fsck = False
    try:
        commit = run_git(directory, "rev-parse", "HEAD")
        clean = not run_git(directory, "status", "--porcelain")
        subprocess.run(
            ("git", "-C", str(directory), "fsck", "--no-progress", "--connectivity-only"),
            check=True,
            capture_output=True,
            text=True,
        )
        fsck = True
    except (OSError, subprocess.CalledProcessError) as exc:
        errors.append(f"git verification failed: {exc}")
    if commit != expected_commit:
        errors.append(f"expected commit {expected_commit}, found {commit}")
    if not clean:
        errors.append("repository has local changes")
    if not fsck:
        errors.append("git connectivity check failed")
    return {
        "status": "ready" if not errors else "invalid",
        "path": str(directory),
        "commit": commit,
        "expected_commit": expected_commit,
        "clean": clean,
        "git_connectivity_ok": fsck,
        "errors": errors,
    }


def inspect_hf_model(hf_hub: Path, specification: dict[str, Any]) -> dict[str, Any]:
    repository = hf_hub / specification["cache_name"]
    ref = repository / "refs" / "main"
    revision = ref.read_text(encoding="utf-8").strip() if ref.is_file() else None
    snapshot = repository / "snapshots" / str(revision)
    errors: list[str] = []
    if revision != specification["revision"]:
        errors.append(f"expected revision {specification['revision']}, found {revision}")
    missing = [name for name in specification["required"] if not (snapshot / name).is_file()]
    if missing:
        errors.append(f"missing required files: {missing}")
    broken_links = [str(path) for path in snapshot.rglob("*") if path.is_symlink() and not path.exists()]
    incomplete = [str(path) for path in repository.rglob("*.incomplete")] if repository.is_dir() else []
    if broken_links:
        errors.append(f"broken snapshot links: {broken_links[:3]}")
    if incomplete:
        errors.append(f"incomplete downloads: {incomplete[:3]}")

    shards: list[str] = []
    weight_bytes = 0
    index_name = specification.get("index")
    if index_name and (snapshot / index_name).is_file():
        try:
            index = json.loads((snapshot / index_name).read_text(encoding="utf-8"))
            shards = sorted(set(index["weight_map"].values()))
        except (KeyError, json.JSONDecodeError, OSError) as exc:
            errors.append(f"invalid weight index: {exc}")
        missing_shards = [name for name in shards if not (snapshot / name).is_file()]
        if missing_shards:
            errors.append(f"missing weight shards: {missing_shards}")
        weight_bytes = sum((snapshot / name).stat().st_size for name in shards if (snapshot / name).is_file())
        if len(shards) != specification["shards"]:
            errors.append(f"expected {specification['shards']} shards, found {len(shards)}")
        if weight_bytes != specification["weight_bytes"]:
            errors.append(
                f"expected {specification['weight_bytes']} weight bytes, found {weight_bytes}"
            )
    else:
        weight = snapshot / str(specification.get("weight", ""))
        if weight.is_file():
            weight_bytes = weight.stat().st_size
            shards = [weight.name]
        elif specification.get("weight"):
            errors.append(f"missing weight file: {weight.name}")

    return {
        "status": "ready" if not errors else "invalid",
        "repository": specification["cache_name"],
        "revision": revision,
        "expected_revision": specification["revision"],
        "snapshot": str(snapshot),
        "weight_files": shards,
        "weight_bytes": weight_bytes,
        "broken_links": len(broken_links),
        "incomplete_files": len(incomplete),
        "errors": errors,
    }


def inspect_spert_releases(workspace: Path) -> dict[str, Any]:
    root = workspace / "tools/external-baselines/spert/data/models"
    datasets: dict[str, Any] = {}
    all_ready = True
    for dataset, expected_bytes in SPERT_RELEASE_BYTES.items():
        directory = root / dataset
        required = (directory / "config.json", directory / "pytorch_model.bin", directory / "vocab.txt")
        missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
        actual_bytes = (directory / "pytorch_model.bin").stat().st_size if not missing else 0
        errors = []
        if missing:
            errors.append(f"missing or empty files: {missing}")
        if actual_bytes != expected_bytes:
            errors.append(f"expected {expected_bytes} weight bytes, found {actual_bytes}")
        if (directory / "config.json").is_file():
            try:
                json.loads((directory / "config.json").read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                errors.append(f"invalid config: {exc}")
        datasets[dataset] = {
            "status": "ready" if not errors else "invalid",
            "weight_bytes": actual_bytes,
            "expected_weight_bytes": expected_bytes,
            "errors": errors,
        }
        all_ready &= not errors
    return {
        "status": "ready" if all_ready else "invalid",
        "validation_safe": False,
        "usage": "download reproduction only; fresh-train for current validation",
        "datasets": datasets,
    }


def inspect_archive(path: Path, expected_bytes: int, required_suffix: str | None) -> dict[str, Any]:
    actual_bytes = path.stat().st_size if path.is_file() else 0
    errors: list[str] = []
    member_present: bool | None = None
    if actual_bytes < expected_bytes:
        status = "downloading" if actual_bytes else "missing"
    elif actual_bytes > expected_bytes:
        status = "invalid"
        errors.append(f"file exceeds expected size: {actual_bytes} > {expected_bytes}")
    else:
        status = "ready"
        try:
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
            if required_suffix:
                member_present = any(name.endswith(required_suffix) for name in names)
                if not member_present:
                    errors.append(f"required archive member is missing: {required_suffix}")
        except (OSError, zipfile.BadZipFile) as exc:
            errors.append(f"invalid ZIP central directory: {exc}")
        if errors:
            status = "invalid"
    return {
        "status": status,
        "path": str(path),
        "bytes": actual_bytes,
        "expected_bytes": expected_bytes,
        "progress_percent": round(100 * actual_bytes / expected_bytes, 2) if expected_bytes else None,
        "required_member_present": member_present,
        "errors": errors,
    }


def build_inventory(workspace: Path, hf_hub: Path, mirror_cache: Path) -> dict[str, Any]:
    repositories = {
        name: inspect_repository(workspace, relative, commit)
        for name, (relative, commit) in REPOSITORIES.items()
    }
    models = {name: inspect_hf_model(hf_hub, spec) for name, spec in HF_MODELS.items()}
    spert_releases = inspect_spert_releases(workspace)
    mirror_archives = {
        name: inspect_archive(mirror_cache / name, spec["bytes"], spec["required_suffix"])
        for name, spec in MIRROR_ARCHIVES.items()
    }
    task_checkpoints = sorted(
        str(path)
        for pattern in ("*.bin", "*.safetensors", "*.ckpt", "*.pth")
        for path in (workspace / "tools/external-baselines/pl-marker").rglob(pattern)
    )

    components = {
        "spert": {
            "status": "fresh_train_ready"
            if repositories["spert"]["status"] == "ready"
            and spert_releases["status"] == "ready"
            and models["bert_base_cased"]["status"] == "ready"
            and models["scibert"]["status"] == "ready"
            else "invalid",
            "code": repositories["spert"],
            "released_checkpoints": spert_releases,
            "fresh_train_backbones": {
                "bert_base_cased": models["bert_base_cased"],
                "scibert": models["scibert"],
            },
        },
        "instructuie": {
            "status": "ready"
            if repositories["instructuie"]["status"] == models["instructuie"]["status"] == "ready"
            else "invalid",
            "code": repositories["instructuie"],
            "model": models["instructuie"],
        },
        "mirror": {
            "status": "ready"
            if repositories["mirror"]["status"] == "ready"
            and all(item["status"] == "ready" for item in mirror_archives.values())
            else (
                "downloading"
                if any(item["status"] == "downloading" for item in mirror_archives.values())
                else "invalid"
            ),
            "code": repositories["mirror"],
            "archives": mirror_archives,
        },
        "oneke": {
            "status": "ready"
            if repositories["oneke"]["status"] == models["oneke"]["status"] == "ready"
            else "invalid",
            "code": repositories["oneke"],
            "model": models["oneke"],
            "upstream_all_md5": "stale; not used as the inference-artifact integrity authority",
        },
        "pl_marker": {
            "status": "fresh_train_ready"
            if repositories["pl_marker"]["status"] == "ready"
            and models["scibert"]["status"] == "ready"
            and models["roberta_large"]["status"] == "ready"
            else "invalid",
            "code": repositories["pl_marker"],
            "fresh_train_backbones": {
                "scibert": models["scibert"],
                "roberta_large": models["roberta_large"],
            },
            "published_task_checkpoints_present": bool(task_checkpoints),
            "published_task_checkpoints": task_checkpoints,
            "usage": "fresh train only; no downloaded task checkpoint is available for direct evaluation",
        },
    }
    invalid = [name for name, item in components.items() if item["status"] == "invalid"]
    downloading = [name for name, item in components.items() if item["status"] == "downloading"]
    return {
        "schema_version": "public-baseline-integrity-v1",
        "status": "invalid" if invalid else ("downloading" if downloading else "ready"),
        "generated_at": utc_now(),
        "workspace": str(workspace),
        "hf_hub": str(hf_hub),
        "invalid_components": invalid,
        "downloading_components": downloading,
        "components": components,
    }


def write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=workspace)
    parser.add_argument(
        "--hf-hub",
        type=Path,
        default=Path(os.environ.get("HF_HUB_ROOT", "/ds2/xuelin/cache/huggingface/hub")),
    )
    parser.add_argument(
        "--mirror-cache",
        type=Path,
        default=Path("/ds2/xuelin/cache/external-baselines/mirror"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=workspace / "outputs/public_baseline_downloads/integrity_status.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_inventory(args.workspace.resolve(), args.hf_hub.resolve(), args.mirror_cache.resolve())
    write_atomic(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1 if payload["status"] == "invalid" else 0


if __name__ == "__main__":
    raise SystemExit(main())
