#!/usr/bin/env python3
"""CPU-only preflight and resumable queue inventory for public baselines.

This script never imports a baseline model, opens a CUDA device, starts a GPU
worker, or reads a public test-gold file.  It answers the narrower question:
can InstructUIE, OneKE, Mirror, and PL-Marker be placed on the *validation*
queue for the complete CoNLL04, SciERC, and ADE assets while retaining the
project's canonical exact-character-span evaluator?

The result is deliberately fail-closed.  A downloaded repository/checkpoint is
not treated as an executable baseline.  A baseline also needs an isolated and
compatible Python environment, a canonical-schema adapter, a formal runner,
and (unless a hard 24 GiB incompatibility is already known) a recorded GPU
canary.  Existing ``complete`` markers are accepted only when their prediction,
metric, and manifest artifacts satisfy the contract implemented below.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

try:
    from packaging.requirements import InvalidRequirement, Requirement
except ImportError:  # pragma: no cover - packaging is present in project envs
    InvalidRequirement = ValueError  # type: ignore[assignment,misc]
    Requirement = None  # type: ignore[assignment]


SCHEMA_VERSION = "public-external-formal-preflight-v1"
RUN_MANIFEST_VERSION = "public-external-formal-run-v1"
CANARY_VERSION = "public-external-gpu-canary-v1"
EVALUATOR = "scripts/evaluate_public_validation_spans.py"
SPLIT = "validation"
SEED = 42
GIB = 1024**3

PUBLIC_DATASETS: dict[str, dict[str, int]] = {
    "conll04": {"train": 922, "validation": 231},
    "scierc": {"train": 1861, "validation": 275},
    "ade": {"train": 3461, "validation": 384},
}

# Paths here are an explicit executable contract.  Merely having upstream code
# and weights is insufficient: each formal runner must translate the canonical
# public records into the upstream schema and translate predictions back into
# evidence-bearing global character spans consumed by EVALUATOR.
BASELINES: dict[str, dict[str, Any]] = {
    "instructuie": {
        "label": "InstructUIE",
        "mode": "published_checkpoint_inference",
        "code_files": (
            "tools/external-baselines/instructuie/src/run_uie.py",
            "tools/external-baselines/instructuie/requirements.txt",
        ),
        "requirements": "tools/external-baselines/instructuie/requirements.txt",
        "manual_modules": (),
        "adapter": "scripts/public_external_adapters/instructuie.py",
        "runner": "scripts/run_public_instructuie_formal.sh",
        "inventory_component": "instructuie",
        "checkpoint_kind": "hf_model",
        "memory_policy": "upstream_full_precision",
        "memory_note": (
            "The upstream run_uie.py loader does not request quantization, a reduced "
            "dtype, or device_map sharding for this float32 T5 checkpoint."
        ),
    },
    "oneke": {
        "label": "OneKE",
        "mode": "published_checkpoint_inference",
        "code_files": (
            "tools/external-baselines/oneke/src/run.py",
            "tools/external-baselines/oneke/src/pipeline.py",
            "tools/external-baselines/oneke/src/models/llm_def.py",
            "scripts/run_public_oneke_formal.py",
            "requirements/oneke-formal.txt",
        ),
        "requirements": "requirements/oneke-formal.txt",
        "manual_modules": (),
        "adapter": "scripts/public_external_adapters/oneke.py",
        "runner": "scripts/run_public_oneke_formal.sh",
        "inventory_component": "oneke",
        "checkpoint_kind": "hf_model",
        "memory_policy": "upstream_bnb_nf4",
        "memory_note": (
            "The upstream OneKE class requests bitsandbytes NF4 4-bit loading; "
            "a real single-GPU canary is still required before formal inference."
        ),
    },
    "mirror": {
        "label": "Mirror",
        "mode": "published_checkpoint_inference",
        "code_files": (
            "tools/external-baselines/mirror/src/model.py",
            "tools/external-baselines/mirror/src/task.py",
            "tools/external-baselines/mirror/requirements.txt",
        ),
        "requirements": "tools/external-baselines/mirror/requirements.txt",
        "manual_modules": ("rex",),
        "adapter": "scripts/public_external_adapters/mirror.py",
        "runner": "scripts/run_public_mirror_formal.sh",
        "inventory_component": "mirror",
        "checkpoint_kind": "mirror_archive",
        "memory_policy": "checkpoint_static_unknown",
        "memory_note": (
            "The released checkpoint is archived locally.  Its serialized size "
            "does not prove peak inference memory; extraction and a canary are required."
        ),
    },
    "pl_marker": {
        "label": "PL-Marker",
        "mode": "fresh_train_then_validation",
        "code_files": (
            "tools/external-baselines/pl-marker/run_acener.py",
            "tools/external-baselines/pl-marker/run_re.py",
            "tools/external-baselines/pl-marker/requirement.txt",
            "tools/external-baselines/pl-marker/transformers/setup.py",
        ),
        "requirements": "tools/external-baselines/pl-marker/requirement.txt",
        "manual_modules": ("apex",),
        "adapter": "scripts/public_external_adapters/pl_marker.py",
        "runner": "scripts/run_public_pl_marker_formal.sh",
        "inventory_component": "pl_marker",
        "checkpoint_kind": "fresh_train_backbones",
        "memory_policy": "encoder_fresh_train",
        "memory_note": (
            "The downloaded SciBERT/RoBERTa backbones are small enough statically, "
            "but the official fp16 path uses Apex and still needs a training canary."
        ),
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_preflight_artifacts(
    summary_path: Path, output_root: Path, payload: dict[str, Any]
) -> None:
    """Publish one summary, one queue, and one marker per baseline atomically."""

    write_atomic(summary_path, payload)
    write_jsonl_atomic(output_root / "queue.jsonl", payload["queue"])
    for baseline, record in payload["baselines"].items():
        jobs = [item for item in payload["queue"] if item["baseline"] == baseline]
        marker = {
            "schema_version": SCHEMA_VERSION,
            "status": record["status"],
            "generated_at": payload["generated_at"],
            "scope": payload["scope"],
            "safety": payload["safety"],
            "evaluator": payload["evaluator"],
            "baseline": baseline,
            "preflight": record,
            "jobs": jobs,
        }
        write_atomic(output_root / baseline / "preflight_status.json", marker)


def read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, str(exc)
    if not isinstance(value, dict):
        return None, "top-level JSON value is not an object"
    return value, None


def nonblank_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON: {exc}")
                continue
            if not isinstance(value, dict):
                errors.append(f"line {line_number}: row is not an object")
                continue
            rows.append(value)
    except OSError as exc:
        errors.append(str(exc))
    return rows, errors


def inspect_public_dataset(
    workspace: Path, dataset: str, expected: dict[str, int]
) -> dict[str, Any]:
    """Inspect only train/validation canonical assets; never touch test files."""

    root = workspace / "data/processed/public_benchmarks_full" / dataset
    files: dict[str, Any] = {}
    blockers: list[str] = []
    split_job_ids: dict[str, set[str]] = {}

    for split in ("train", "validation"):
        expected_rows = expected[split]
        split_job_ids[split] = set()
        for role, suffix in (
            ("gold", "gold.jsonl"),
            ("index", "index.jsonl"),
            ("jobs", "baseline_jobs.jsonl"),
        ):
            relative = f"data/processed/public_benchmarks_full/{dataset}/{split}_{suffix}"
            path = workspace / relative
            item: dict[str, Any] = {"path": relative, "expected_rows": expected_rows}
            if not path.is_file():
                item.update({"status": "missing", "rows": 0, "sha256": None})
                blockers.append(f"dataset_asset_missing:{relative}")
                files[f"{split}_{role}"] = item
                continue
            rows, errors = nonblank_jsonl(path)
            item.update(
                {
                    "status": "ready" if not errors and len(rows) == expected_rows else "invalid",
                    "rows": len(rows),
                    "sha256": sha256(path),
                    "errors": errors,
                }
            )
            if len(rows) != expected_rows:
                blockers.append(
                    f"dataset_row_count_mismatch:{relative}:expected={expected_rows}:found={len(rows)}"
                )
            if errors:
                blockers.append(f"dataset_jsonl_invalid:{relative}")
            if role in {"index", "jobs"}:
                job_ids = [row.get("job_id") for row in rows]
                if any(not isinstance(value, str) or not value for value in job_ids):
                    blockers.append(f"dataset_job_id_invalid:{relative}")
                elif len(job_ids) != len(set(job_ids)):
                    blockers.append(f"dataset_job_id_duplicate:{relative}")
                else:
                    if role == "index":
                        split_job_ids[split] = set(job_ids)
                    elif split_job_ids[split] and set(job_ids) != split_job_ids[split]:
                        blockers.append(f"dataset_index_jobs_mismatch:{dataset}:{split}")
            files[f"{split}_{role}"] = item

    overlap = sorted(split_job_ids["train"] & split_job_ids["validation"])
    if overlap:
        blockers.append(f"train_validation_job_overlap:{dataset}:{len(overlap)}")
    return {
        "status": "ready" if not blockers else "blocked_with_reason",
        "root": str(root.relative_to(workspace)),
        "expected": expected,
        "files": files,
        "train_validation_job_overlap": len(overlap),
        "test_namespace": "sealed_not_inspected",
        "test_gold_read": False,
        "blocking_reasons": sorted(set(blockers)),
    }


def load_integrity_inventory(path: Path) -> tuple[dict[str, Any], list[str]]:
    if not path.is_file():
        return {}, [f"download_integrity_inventory_missing:{path}"]
    value, error = read_json(path)
    if error or value is None:
        return {}, [f"download_integrity_inventory_invalid:{path}:{error}"]
    blockers: list[str] = []
    if value.get("schema_version") != "public-baseline-integrity-v1":
        blockers.append("download_integrity_schema_mismatch")
    if value.get("status") != "ready":
        blockers.append(f"download_integrity_not_ready:{value.get('status')}")
    return value, blockers


def existing_nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def inspect_checkpoint(
    baseline: str,
    specification: dict[str, Any],
    integrity: dict[str, Any],
    workspace: Path,
) -> dict[str, Any]:
    component = integrity.get("components", {}).get(
        specification["inventory_component"], {}
    )
    blockers: list[str] = []
    if not isinstance(component, dict) or not component:
        return {
            "status": "blocked_with_reason",
            "weight_bytes": 0,
            "blocking_reasons": [f"download_component_missing:{baseline}"],
        }
    component_status = component.get("status")
    if component_status not in {"ready", "fresh_train_ready"}:
        blockers.append(f"download_component_not_ready:{baseline}:{component_status}")

    kind = specification["checkpoint_kind"]
    weight_bytes = 0
    paths: list[str] = []
    detail: dict[str, Any] = {}
    if kind == "hf_model":
        model = component.get("model", {})
        snapshot = Path(str(model.get("snapshot", "")))
        weight_files = model.get("weight_files", [])
        weight_bytes = int(model.get("weight_bytes") or 0)
        if model.get("status") != "ready":
            blockers.append(f"checkpoint_integrity_not_ready:{baseline}")
        if not snapshot.is_dir():
            blockers.append(f"checkpoint_snapshot_missing:{baseline}:{snapshot}")
        for name in weight_files if isinstance(weight_files, list) else []:
            path = snapshot / str(name)
            paths.append(str(path))
            if not existing_nonempty(path):
                blockers.append(f"checkpoint_weight_missing:{baseline}:{path}")
        if not weight_files:
            blockers.append(f"checkpoint_weight_list_empty:{baseline}")
        detail = {
            "revision": model.get("revision"),
            "snapshot": str(snapshot),
            "weight_files": paths,
        }
    elif kind == "mirror_archive":
        archives = component.get("archives", {})
        release = archives.get("mirror_outputs.zip", {})
        archive_path = Path(str(release.get("path", "")))
        weight_bytes = int(release.get("bytes") or 0)
        paths.append(str(archive_path))
        if release.get("status") != "ready" or not existing_nonempty(archive_path):
            blockers.append("mirror_checkpoint_archive_not_ready")
        extracted_candidates = (
            archive_path.parent
            / "mirror_outputs/Mirror_Pretrain_AllExcluded_2/ckpt/SchemaGuidedInstructBertModel.best.pth",
            workspace
            / "tools/external-baselines/mirror/outputs/Mirror_Pretrain_AllExcluded_2/ckpt/SchemaGuidedInstructBertModel.best.pth",
        )
        extracted = next((path for path in extracted_candidates if existing_nonempty(path)), None)
        if extracted is None:
            blockers.append("mirror_checkpoint_not_extracted")
        else:
            weight_bytes = extracted.stat().st_size
            paths.append(str(extracted))
        detail = {
            "archive": str(archive_path),
            "extracted_checkpoint": str(extracted) if extracted else None,
        }
    elif kind == "fresh_train_backbones":
        backbones = component.get("fresh_train_backbones", {})
        for name in ("scibert", "roberta_large"):
            model = backbones.get(name, {})
            if model.get("status") != "ready":
                blockers.append(f"fresh_train_backbone_not_ready:{baseline}:{name}")
                continue
            snapshot = Path(str(model.get("snapshot", "")))
            model_bytes = int(model.get("weight_bytes") or 0)
            weight_bytes = max(weight_bytes, model_bytes)
            paths.append(str(snapshot))
            if not snapshot.is_dir():
                blockers.append(f"fresh_train_backbone_missing:{baseline}:{name}:{snapshot}")
        detail = {"backbones": paths, "largest_backbone_bytes": weight_bytes}
    else:  # pragma: no cover - specifications are frozen above
        blockers.append(f"unknown_checkpoint_kind:{kind}")

    return {
        "status": "ready" if not blockers else "blocked_with_reason",
        "kind": kind,
        "weight_bytes": weight_bytes,
        "detail": detail,
        "blocking_reasons": sorted(set(blockers)),
    }


def assess_3090_memory(
    policy: str, weight_bytes: int, gpu_memory_gib: float = 24.0
) -> dict[str, Any]:
    """Conservative static assessment; never allocate CUDA memory."""

    capacity = int(gpu_memory_gib * GIB)
    reserve = 4 * GIB
    if policy == "upstream_full_precision":
        estimated = weight_bytes + reserve
        state = "incompatible_24gb" if estimated > capacity else "static_fit_requires_gpu_canary"
    elif policy == "upstream_bnb_nf4":
        # Inventory bytes are the published bf16 checkpoint.  NF4 weights use
        # roughly one quarter of that storage; reserve 6 GiB for quantization
        # metadata, activations, KV cache, and generation workspace.
        reserve = 6 * GIB
        estimated = (weight_bytes + 3) // 4 + reserve
        state = "static_fit_requires_gpu_canary" if estimated <= capacity else "incompatible_24gb"
    elif policy == "encoder_fresh_train":
        # A conservative 8x multiplier covers weights, gradients, optimizer
        # state, and mixed-precision copies; activation peaks remain empirical.
        reserve = 3 * GIB
        estimated = weight_bytes * 8 + reserve
        state = "static_fit_requires_gpu_canary" if estimated <= capacity else "unknown_requires_gpu_canary"
    else:
        estimated = weight_bytes + reserve
        state = "unknown_requires_gpu_canary"
    return {
        "assessment": state,
        "gpu_model": "RTX 3090",
        "capacity_gib": gpu_memory_gib,
        "checkpoint_or_backbone_bytes": weight_bytes,
        "conservative_estimated_bytes": estimated,
        "cuda_queried": False,
        "gpu_process_started": False,
    }


def parse_requirements(path: Path) -> tuple[list[Any], list[str]]:
    parsed: list[Any] = []
    errors: list[str] = []
    if Requirement is None:
        return parsed, ["packaging_module_missing_in_preflight_environment"]
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return parsed, [f"requirements_unreadable:{path}:{exc}"]
    for line_number, raw in enumerate(lines, start=1):
        value = raw.split("#", 1)[0].strip()
        if not value or value.startswith(("-r", "--")):
            continue
        if value.startswith("git+"):
            # Mirror's REx direct URL is checked through manual_modules.
            continue
        try:
            parsed.append(Requirement(value))
        except InvalidRequirement as exc:
            errors.append(f"requirements_invalid:{path}:line={line_number}:{exc}")
    return parsed, errors


def query_python_environment(
    python: Path, distributions: Iterable[str], modules: Iterable[str]
) -> tuple[dict[str, Any], str | None]:
    """Query metadata/find_spec only; do not import torch or baseline packages."""

    request = {
        "distributions": sorted(set(distributions)),
        "modules": sorted(set(modules)),
    }
    program = r"""
import importlib.metadata
import importlib.util
import json
import sys

request = json.loads(sys.argv[1])
versions = {}
for name in request["distributions"]:
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None
modules = {name: importlib.util.find_spec(name) is not None for name in request["modules"]}
print(json.dumps({"python": sys.version, "versions": versions, "modules": modules}))
"""
    if not python.is_file():
        return {}, f"runtime_python_missing:{python}"
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    try:
        completed = subprocess.run(
            (str(python), "-I", "-c", program, json.dumps(request)),
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
            env=environment,
        )
        value = json.loads(completed.stdout)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        return {}, f"runtime_metadata_probe_failed:{python}:{exc}"
    return value, None


def inspect_dependencies(
    workspace: Path,
    specification: dict[str, Any],
    python: Path,
    query: Callable[[Path, Iterable[str], Iterable[str]], tuple[dict[str, Any], str | None]] = query_python_environment,
) -> dict[str, Any]:
    requirements_path = workspace / specification["requirements"]
    requirements, blockers = parse_requirements(requirements_path)
    distributions = [requirement.name for requirement in requirements]
    manual_modules = tuple(specification.get("manual_modules", ()))
    environment, error = query(python, distributions, manual_modules)
    if error:
        blockers.append(error)
        return {
            "status": "blocked_with_reason",
            "python": str(python),
            "requirements": specification["requirements"],
            "packages": {},
            "modules": {},
            "blocking_reasons": sorted(set(blockers)),
        }

    versions = environment.get("versions", {})
    package_results: dict[str, Any] = {}
    for requirement in requirements:
        installed = versions.get(requirement.name)
        satisfied = installed is not None and (
            not requirement.specifier or requirement.specifier.contains(installed, prereleases=True)
        )
        package_results[requirement.name] = {
            "required": str(requirement.specifier) or "installed",
            "installed": installed,
            "satisfied": satisfied,
        }
        if installed is None:
            blockers.append(f"dependency_missing:{requirement.name}")
        elif not satisfied:
            blockers.append(
                f"dependency_version_mismatch:{requirement.name}:required={requirement.specifier}:found={installed}"
            )

    module_results = environment.get("modules", {})
    for module in manual_modules:
        if module_results.get(module) is not True:
            blockers.append(f"dependency_module_missing:{module}")
    return {
        "status": "ready" if not blockers else "blocked_with_reason",
        "python": str(python),
        "python_version": environment.get("python"),
        "metadata_only_probe": True,
        "cuda_visible_devices": "",
        "requirements": specification["requirements"],
        "packages": package_results,
        "modules": module_results,
        "blocking_reasons": sorted(set(blockers)),
    }


def inspect_code_and_adapters(
    workspace: Path, baseline: str, specification: dict[str, Any]
) -> dict[str, Any]:
    blockers: list[str] = []
    upstream: dict[str, str] = {}
    for relative in specification["code_files"]:
        path = workspace / relative
        upstream[relative] = "ready" if existing_nonempty(path) else "missing"
        if not existing_nonempty(path):
            blockers.append(f"upstream_code_missing:{relative}")

    adapter = workspace / specification["adapter"]
    runner = workspace / specification["runner"]
    if not existing_nonempty(adapter):
        blockers.append(f"schema_adapter_missing:{specification['adapter']}")
    if not existing_nonempty(runner):
        blockers.append(f"formal_runner_missing:{specification['runner']}")

    syntax_errors: list[str] = []
    for relative in (*specification["code_files"], specification["adapter"]):
        path = workspace / relative
        if path.suffix != ".py" or not existing_nonempty(path):
            continue
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except (OSError, SyntaxError) as exc:
            syntax_errors.append(f"python_syntax_invalid:{relative}:{exc}")
    blockers.extend(syntax_errors)
    return {
        "status": "ready" if not blockers else "blocked_with_reason",
        "upstream": upstream,
        "schema_adapter": specification["adapter"],
        "formal_runner": specification["runner"],
        "prediction_contract": (
            "one canonical JSONL row per validation job with evidence.start/end "
            "global character spans"
        ),
        "blocking_reasons": sorted(set(blockers)),
    }


def inspect_gpu_canary(
    path: Path, baseline: str, gpu_memory_gib: float
) -> dict[str, Any]:
    if not path.is_file():
        return {
            "status": "not_run",
            "path": str(path),
            "passed": False,
            "blocking_reasons": [],
        }
    value, error = read_json(path)
    blockers: list[str] = []
    if error or value is None:
        blockers.append(f"gpu_canary_invalid:{path}:{error}")
        value = {}
    checks = {
        "schema_version": value.get("schema_version") == CANARY_VERSION,
        "baseline": value.get("baseline") == baseline,
        "gpu": value.get("gpu") == "RTX 3090",
        "capacity": value.get("capacity_gib") == gpu_memory_gib,
        "status": value.get("status") == "passed",
        "terminal": value.get("terminal") is True,
        "peak_memory": isinstance(value.get("peak_allocated_bytes"), int)
        and 0 < value.get("peak_allocated_bytes", 0) <= int(gpu_memory_gib * GIB),
        "exit_code": value.get("exit_code") == 0,
        "actual_gpu": isinstance(value.get("actual_gpu_name"), str)
        and "3090" in value.get("actual_gpu_name", ""),
        "actual_capacity": isinstance(value.get("actual_total_memory_bytes"), int)
        and 20 * GIB
        <= value.get("actual_total_memory_bytes", 0)
        <= int(gpu_memory_gib * GIB),
        "no_test_gold": value.get("test_gold_read") is False,
    }
    if baseline == "oneke":
        signature = value.get("synthetic_relation_signature")
        checks.update(
            {
                "runtime_compatible": value.get("runtime_compatible") is True,
                "oneke_model_revision": value.get("model_revision")
                == "696148c0581b29f530af738ddab500deaa8fe8f2",
                "oneke_quantization": value.get("quantization")
                == "bitsandbytes-nf4-double-quantization",
                "oneke_prompt_version": value.get("prompt_version")
                == "oneke-upstream-jsonlike-batched-v3",
                "oneke_relation_signature": isinstance(signature, dict)
                and signature.get("source") == "Adverse-Effect"
                and signature.get("target") == "Drug",
            }
        )
    for name, passed in checks.items():
        if not passed:
            blockers.append(f"gpu_canary_check_failed:{name}")
    return {
        "status": "passed" if not blockers else "failed",
        "path": str(path),
        "passed": not blockers,
        "checks": checks,
        "blocking_reasons": sorted(set(blockers)),
    }


def prediction_job_ids(path: Path) -> tuple[list[str], list[str]]:
    rows, errors = nonblank_jsonl(path)
    ids = [row.get("job_id") for row in rows]
    if any(not isinstance(value, str) or not value for value in ids):
        errors.append("prediction row has missing or invalid job_id")
    text_ids = [value for value in ids if isinstance(value, str) and value]
    if len(text_ids) != len(set(text_ids)):
        errors.append("prediction job_id values are not unique")
    return text_ids, errors


def inspect_progress(
    output_dir: Path,
    baseline: str,
    dataset: str,
    expected_job_ids: set[str],
    evaluator_sha256: str,
    input_hashes: dict[str, str],
) -> dict[str, Any]:
    status_path = output_dir / "status.json"
    predictions_path = output_dir / "validation_predictions.jsonl"
    partial_path = output_dir / "validation_predictions.partial.jsonl"
    metrics_path = output_dir / "validation_character_span_metrics.json"
    manifest_path = output_dir / "run_manifest.json"

    prior_status: str | None = None
    marker_errors: list[str] = []
    if status_path.is_file():
        marker, error = read_json(status_path)
        if error or marker is None:
            marker_errors.append(f"status_marker_invalid:{error}")
        else:
            prior_status = str(marker.get("status")) if marker.get("status") is not None else None

    artifact_presence = {
        "status": status_path.is_file(),
        "predictions": predictions_path.is_file(),
        "partial_predictions": partial_path.is_file(),
        "metrics": metrics_path.is_file(),
        "manifest": manifest_path.is_file(),
    }
    any_artifact = any(artifact_presence.values())
    if prior_status != "complete":
        return {
            "state": "resume_required" if any_artifact else "not_started",
            "output_dir": str(output_dir),
            "prior_status": prior_status,
            "artifact_presence": artifact_presence,
            "verified_prediction_rows": 0,
            "blocking_reasons": marker_errors,
        }

    completion_errors = list(marker_errors)
    if not predictions_path.is_file():
        completion_errors.append("invalid_completion_marker:predictions_missing")
        prediction_ids: list[str] = []
    else:
        prediction_ids, errors = prediction_job_ids(predictions_path)
        completion_errors.extend(f"invalid_completion_marker:{error}" for error in errors)
        if set(prediction_ids) != expected_job_ids:
            completion_errors.append(
                "invalid_completion_marker:prediction_job_ids_do_not_match_frozen_validation"
            )

    metrics, metrics_error = read_json(metrics_path) if metrics_path.is_file() else (None, "missing")
    if metrics_error or metrics is None:
        completion_errors.append(f"invalid_completion_marker:metrics:{metrics_error}")
    elif metrics.get("metric") != "strict-source-character-span":
        completion_errors.append("invalid_completion_marker:wrong_metric")

    manifest, manifest_error = (
        read_json(manifest_path) if manifest_path.is_file() else (None, "missing")
    )
    if manifest_error or manifest is None:
        completion_errors.append(f"invalid_completion_marker:manifest:{manifest_error}")
    else:
        expected_fields = {
            "schema_version": RUN_MANIFEST_VERSION,
            "status": "complete",
            "baseline": baseline,
            "dataset": dataset,
            "split": SPLIT,
            "seed": SEED,
            "evaluator": EVALUATOR,
            "evaluator_sha256": evaluator_sha256,
            "test_gold_read": False,
        }
        for field, expected_value in expected_fields.items():
            if manifest.get(field) != expected_value:
                completion_errors.append(f"invalid_completion_marker:manifest_field:{field}")
        if manifest.get("input_sha256") != input_hashes:
            completion_errors.append("invalid_completion_marker:input_hashes")
        if manifest.get("prediction_sha256") != (
            sha256(predictions_path) if predictions_path.is_file() else None
        ):
            completion_errors.append("invalid_completion_marker:prediction_hash")
        if manifest.get("metric_sha256") != (
            sha256(metrics_path) if metrics_path.is_file() else None
        ):
            completion_errors.append("invalid_completion_marker:metric_hash")

    return {
        "state": "complete" if not completion_errors else "invalid_completion_marker",
        "output_dir": str(output_dir),
        "prior_status": prior_status,
        "artifact_presence": artifact_presence,
        "verified_prediction_rows": len(prediction_ids),
        "blocking_reasons": sorted(set(completion_errors)),
    }


def load_runtime_map(path: Path | None, baseline_ids: Iterable[str]) -> tuple[dict[str, Path], list[str]]:
    identifiers = list(baseline_ids)
    if path is None:
        return {name: Path(sys.executable).resolve() for name in identifiers}, []
    value, error = read_json(path)
    if error or value is None:
        return {}, [f"runtime_map_invalid:{path}:{error}"]
    runtimes: dict[str, Path] = {}
    blockers: list[str] = []
    for name in identifiers:
        raw = value.get(name)
        if isinstance(raw, dict):
            raw = raw.get("python")
        if not isinstance(raw, str) or not raw:
            blockers.append(f"runtime_not_declared:{name}")
            continue
        # Preserve a venv's interpreter symlink.  A standard venv points
        # ``bin/python`` at its bootstrap interpreter and relies on the invoked
        # path plus ``pyvenv.cfg`` to select the isolated site-packages.  Calling
        # the resolved target would silently probe the bootstrap environment.
        runtimes[name] = Path(raw).expanduser().absolute()
    extras = sorted(set(value) - set(identifiers))
    if extras:
        blockers.append(f"runtime_map_unknown_baselines:{','.join(extras)}")
    return runtimes, blockers


def validation_job_ids(dataset_record: dict[str, Any], workspace: Path) -> set[str]:
    relative = dataset_record["files"]["validation_jobs"]["path"]
    path = workspace / relative
    if not path.is_file():
        return set()
    rows, errors = nonblank_jsonl(path)
    if errors:
        return set()
    return {
        row["job_id"]
        for row in rows
        if isinstance(row.get("job_id"), str) and row["job_id"]
    }


def frozen_input_hashes(dataset_record: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, item in dataset_record["files"].items():
        if isinstance(item.get("sha256"), str):
            result[name] = item["sha256"]
    return result


def build_preflight(
    workspace: Path,
    output_root: Path,
    integrity_path: Path,
    runtime_map_path: Path | None = None,
    gpu_memory_gib: float = 24.0,
    dataset_specs: dict[str, dict[str, int]] = PUBLIC_DATASETS,
    baseline_specs: dict[str, dict[str, Any]] = BASELINES,
    dependency_query: Callable[[Path, Iterable[str], Iterable[str]], tuple[dict[str, Any], str | None]] = query_python_environment,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    output_root = output_root.resolve()
    evaluator_path = workspace / EVALUATOR
    evaluator_blockers: list[str] = []
    if not existing_nonempty(evaluator_path):
        evaluator_blockers.append(f"canonical_evaluator_missing:{EVALUATOR}")
        evaluator_hash = ""
    else:
        evaluator_hash = sha256(evaluator_path)

    datasets = {
        name: inspect_public_dataset(workspace, name, expected)
        for name, expected in dataset_specs.items()
    }
    integrity, integrity_blockers = load_integrity_inventory(integrity_path)
    runtimes, runtime_map_blockers = load_runtime_map(runtime_map_path, baseline_specs)

    baselines: dict[str, Any] = {}
    queue: list[dict[str, Any]] = []
    for baseline, specification in baseline_specs.items():
        code = inspect_code_and_adapters(workspace, baseline, specification)
        checkpoint = inspect_checkpoint(baseline, specification, integrity, workspace)
        memory = assess_3090_memory(
            specification["memory_policy"], checkpoint["weight_bytes"], gpu_memory_gib
        )
        runtime = runtimes.get(baseline)
        if runtime is None:
            dependencies = {
                "status": "blocked_with_reason",
                "python": None,
                "blocking_reasons": [f"runtime_not_declared:{baseline}"],
            }
        else:
            dependencies = inspect_dependencies(
                workspace, specification, runtime, query=dependency_query
            )
        canary_path = output_root / baseline / "gpu_canary.json"
        canary = inspect_gpu_canary(canary_path, baseline, gpu_memory_gib)

        hard_blockers = sorted(
            set(
                integrity_blockers
                + [
                    reason
                    for reason in runtime_map_blockers
                    if reason == f"runtime_not_declared:{baseline}"
                    or reason.startswith("runtime_map_")
                ]
                + code["blocking_reasons"]
                + checkpoint["blocking_reasons"]
                + dependencies["blocking_reasons"]
                + evaluator_blockers
            )
        )
        if memory["assessment"] == "incompatible_24gb":
            hard_blockers.append(
                f"gpu_memory_incompatible:RTX3090:{memory['conservative_estimated_bytes']}>{int(gpu_memory_gib * GIB)}"
            )
        if canary["status"] == "failed":
            hard_blockers.extend(canary["blocking_reasons"])
        hard_blockers = sorted(set(hard_blockers))

        if hard_blockers:
            static_status = "blocked_with_reason"
        elif canary["passed"]:
            static_status = "ready_to_queue"
        else:
            static_status = "ready_for_gpu_canary"

        baselines[baseline] = {
            "label": specification["label"],
            "status": static_status,
            "mode": specification["mode"],
            "code_and_adapter": code,
            "checkpoint": checkpoint,
            "dependencies": dependencies,
            "gpu_compatibility": {**memory, "note": specification["memory_note"]},
            "gpu_canary": canary,
            "blocking_reasons": hard_blockers,
        }

        for dataset, dataset_record in datasets.items():
            output_dir = output_root / baseline / dataset / f"seed{SEED}"
            input_hashes = frozen_input_hashes(dataset_record)
            progress = inspect_progress(
                output_dir,
                baseline,
                dataset,
                validation_job_ids(dataset_record, workspace),
                evaluator_hash,
                input_hashes,
            )
            job_blockers = sorted(
                set(
                    hard_blockers
                    + dataset_record["blocking_reasons"]
                    + progress["blocking_reasons"]
                )
            )
            if progress["state"] == "complete" and not job_blockers:
                status = "complete"
            elif job_blockers:
                status = "blocked_with_reason"
            elif progress["state"] in {"resume_required", "invalid_completion_marker"}:
                status = "resume_required"
            else:
                status = static_status
            queue.append(
                {
                    "job_id": f"{baseline}-{dataset}-{SPLIT}-seed{SEED}",
                    "baseline": baseline,
                    "label": specification["label"],
                    "dataset": dataset,
                    "split": SPLIT,
                    "seed": SEED,
                    "status": status,
                    "runner": specification["runner"] if status in {"ready_to_queue", "resume_required"} else None,
                    "evaluator": EVALUATOR,
                    "expected_prediction_rows": dataset_specs[dataset]["validation"],
                    "input_sha256": input_hashes,
                    "progress": progress,
                    "blocking_reasons": job_blockers,
                }
            )

    counts: dict[str, int] = {}
    for item in queue:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    runnable = counts.get("ready_to_queue", 0) + counts.get("resume_required", 0)
    if counts.get("blocked_with_reason", 0) == len(queue):
        overall = "blocked_with_reason"
    elif counts.get("blocked_with_reason", 0):
        overall = "partially_ready"
    elif runnable:
        overall = "ready_to_queue"
    elif counts.get("ready_for_gpu_canary", 0):
        overall = "ready_for_gpu_canary"
    else:
        overall = "complete"

    return {
        "schema_version": SCHEMA_VERSION,
        "status": overall,
        "generated_at": utc_now(),
        "workspace": str(workspace),
        "mode": "cpu_only_static_preflight",
        "scope": {
            "datasets": list(dataset_specs),
            "split": SPLIT,
            "full_public_assets": True,
            "railway_data_included": False,
            "formal_test_status": "sealed",
            "test_gold_read": False,
        },
        "safety": {
            "cuda_queried": False,
            "gpu_process_started": False,
            "model_imported": False,
            "predictions_generated": False,
            "metrics_generated": False,
        },
        "evaluator": {
            "path": EVALUATOR,
            "sha256": evaluator_hash or None,
            "protocol": "strict-source-character-span",
            "status": "ready" if not evaluator_blockers else "blocked_with_reason",
            "blocking_reasons": evaluator_blockers,
        },
        "datasets": datasets,
        "baselines": baselines,
        "queue_counts": counts,
        "queue": queue,
    }


def parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=workspace)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=workspace / "outputs/public_external_formal",
    )
    parser.add_argument(
        "--integrity-status",
        type=Path,
        default=workspace / "outputs/public_baseline_downloads/integrity_status.json",
    )
    parser.add_argument(
        "--runtime-map",
        type=Path,
        default=workspace / "outputs/public_external_formal/runtime_map.json",
        help=(
            "Optional JSON mapping each baseline ID to its isolated Python interpreter; "
            "defaults to the environment registry written by setup scripts."
        ),
    )
    parser.add_argument("--gpu-memory-gib", type=float, default=24.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=workspace / "outputs/public_external_formal/preflight_status.json",
    )
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Return exit code 2 unless at least one job is ready/resumable or all are complete.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.gpu_memory_gib <= 0:
        raise SystemExit("--gpu-memory-gib must be positive")
    payload = build_preflight(
        args.workspace,
        args.output_root,
        args.integrity_status,
        args.runtime_map,
        args.gpu_memory_gib,
    )
    write_preflight_artifacts(args.output, args.output_root, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.require_ready:
        runnable = payload["queue_counts"].get("ready_to_queue", 0) + payload[
            "queue_counts"
        ].get("resume_required", 0)
        complete = payload["queue_counts"].get("complete", 0)
        if runnable == 0 and complete != len(payload["queue"]):
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
