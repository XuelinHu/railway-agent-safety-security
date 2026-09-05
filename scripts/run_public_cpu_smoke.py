#!/usr/bin/env python3
"""Run the public-data CPU-only smoke pipeline without model training or inference."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("conll04", "scierc", "ade")
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
GPU_RUNTIME_STATUS = "gpu_runtime_not_covered"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def base_status(run_id: str, artifact_root: Path) -> dict[str, Any]:
    return {
        "schema_version": "public-cpu-smoke-v1",
        "status": "running",
        "mode": "cpu_only_single_thread_smoke",
        "run_id": run_id,
        "pid": os.getpid(),
        "systemd_invocation_id": os.environ.get("INVOCATION_ID"),
        "artifact_root": str(artifact_root),
        "active_stage": "initializing",
        "current_check": "initializing",
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "finished_at": None,
        "cpu_only": True,
        "execution": {
            "thread_limit": 1,
            "systemd_cpu_quota_requested": "100%",
        },
        "counts": {"passed": 0, "failed": 0, "warnings": 0},
        "checks": [],
        "gpu_runtime": {
            "status": GPU_RUNTIME_STATUS,
            "covered": False,
            "not_executed": [
                "QLoRA training",
                "bitsandbytes quantized model loading",
                "Qwen generation",
                "GLiNER/GLiREL model inference",
            ],
            "allowed_checks_only": [
                "module import",
                "CLI parameter parsing",
                "fixture construction",
                "mock/unit tests",
                "GLiNER/GLiREL local preflight",
            ],
        },
        "environment": {},
        "datasets": {},
        "gliner_glirel": {
            "status": "pending",
            "scope": "preflight_only",
            "runtime": GPU_RUNTIME_STATUS,
        },
        "qlora_bitsandbytes": {"status": GPU_RUNTIME_STATUS},
        "pytest": {"status": "pending", "scope": "tests/"},
    }


class StatusTracker:
    def __init__(self, path: Path, run_id: str, artifact_root: Path) -> None:
        self.path = path
        self.value = base_status(run_id, artifact_root)
        self.write()

    def write(self) -> None:
        self.value["updated_at"] = utc_now()
        write_json(self.path, self.value)

    def stage(self, name: str) -> None:
        self.value["active_stage"] = name
        self.value["current_check"] = name
        self.write()

    def check(self, name: str, result: str, severity: str = "pass") -> None:
        self.value["checks"].append(
            {
                "check": name,
                "severity": severity,
                "result": result,
                "checked_at": utc_now(),
            }
        )
        if severity == "warning":
            self.value["counts"]["warnings"] += 1
        elif severity == "error":
            self.value["counts"]["failed"] += 1
        else:
            self.value["counts"]["passed"] += 1
        self.write()

    def fail(self, error: BaseException) -> None:
        failed_stage = str(self.value.get("active_stage", "unknown"))
        self.check(failed_stage, f"failed: {type(error).__name__}: {error}", "error")
        self.value["status"] = "failed"
        self.value["active_stage"] = "failed"
        self.value["current_check"] = failed_stage
        self.value["finished_at"] = utc_now()
        self.value["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        self.write()

    def complete(self) -> None:
        self.value["status"] = "complete"
        self.value["active_stage"] = "complete"
        self.value["current_check"] = None
        self.value["finished_at"] = utc_now()
        self.write()


class CommandFailure(RuntimeError):
    def __init__(self, command: Sequence[str], returncode: int, output: str) -> None:
        self.command = list(command)
        self.returncode = returncode
        self.output = output
        super().__init__(f"command exited {returncode}: {' '.join(command)}")


def run_command(command: Sequence[str], environment: Mapping[str, str]) -> str:
    printable = " ".join(command)
    print(f"$ {printable}", flush=True)
    completed = subprocess.run(
        list(command),
        cwd=ROOT,
        env=dict(environment),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n", flush=True)
    if completed.returncode:
        raise CommandFailure(command, completed.returncode, completed.stdout)
    return completed.stdout


def validate_cpu_environment(environment: Mapping[str, str], inspect_torch: bool = True) -> dict[str, Any]:
    wrong_threads = {
        name: environment.get(name) for name in THREAD_VARIABLES if environment.get(name) != "1"
    }
    if wrong_threads:
        raise RuntimeError(f"single-thread environment is not enforced: {wrong_threads}")
    if environment.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be the empty string")
    if environment.get("NVIDIA_VISIBLE_DEVICES") not in {"", "void", "none"}:
        raise RuntimeError("NVIDIA_VISIBLE_DEVICES must hide every GPU")

    result: dict[str, Any] = {
        "CUDA_VISIBLE_DEVICES": environment.get("CUDA_VISIBLE_DEVICES"),
        "NVIDIA_VISIBLE_DEVICES": environment.get("NVIDIA_VISIBLE_DEVICES"),
        "thread_limits": {name: environment.get(name) for name in THREAD_VARIABLES},
        "python": sys.executable,
        "python_version": sys.version.split()[0],
    }
    if inspect_torch:
        import torch

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        cuda_available = bool(torch.cuda.is_available())
        cuda_devices = int(torch.cuda.device_count())
        if cuda_available or cuda_devices:
            raise RuntimeError(
                f"CUDA is visible inside CPU smoke: available={cuda_available}, devices={cuda_devices}"
            )
        result.update(
            {
                "torch": torch.__version__,
                "torch_threads": torch.get_num_threads(),
                "torch_cuda_available": cuda_available,
                "torch_cuda_device_count": cuda_devices,
            }
        )
    return result


def select_public_fixture(
    train_jobs: Sequence[dict[str, Any]],
    validation_jobs: Sequence[dict[str, Any]],
    mentions: Sequence[dict[str, Any]],
    edges: Sequence[dict[str, Any]],
    split_manifest: Sequence[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    train_by_document: dict[str, dict[str, Any]] = {}
    for row in train_jobs:
        train_by_document.setdefault(str(row["document_id"]), row)
    selected_edge = next(
        (row for row in edges if str(row.get("document_id")) in train_by_document),
        None,
    )
    if selected_edge is None:
        raise ValueError("no training edge has a matching training job")
    train_document = str(selected_edge["document_id"])
    train_job = train_by_document[train_document]
    if not validation_jobs:
        raise ValueError("validation jobs are empty")
    validation_job = validation_jobs[0]
    validation_document = str(validation_job["document_id"])
    selected_documents = {train_document, validation_document}
    selected_manifest = [
        row for row in split_manifest if str(row.get("document_id")) in selected_documents
    ]
    assignments = {
        str(row["document_id"]): str(row["split"]) for row in selected_manifest
    }
    if assignments.get(train_document) != "train" or assignments.get(validation_document) != "validation":
        raise ValueError("fixture split manifest does not match selected jobs")
    selected_mentions = [
        row for row in mentions if str(row.get("document_id")) in selected_documents
    ]
    selected_edges = [row for row in edges if str(row.get("document_id")) == train_document]
    if not selected_mentions or not selected_edges:
        raise ValueError("fixture needs source mentions and at least one training edge")
    return {
        "train_jobs": [train_job],
        "validation_jobs": [validation_job],
        "mentions": selected_mentions,
        "edges": selected_edges,
        "split_manifest": selected_manifest,
    }


def build_source_fixture(dataset: str, source_root: Path, target_root: Path) -> dict[str, Any]:
    source = source_root / dataset
    selected = select_public_fixture(
        load_jsonl(source / "train_baseline_jobs.jsonl"),
        load_jsonl(source / "validation_baseline_jobs.jsonl"),
        load_jsonl(source / "knowledge_graph" / "mentions.jsonl"),
        load_jsonl(source / "knowledge_graph" / "training_edges.jsonl"),
        load_jsonl(source / "split_manifest.jsonl"),
    )
    target = target_root / dataset
    write_jsonl(target / "train_baseline_jobs.jsonl", selected["train_jobs"])
    write_jsonl(target / "validation_baseline_jobs.jsonl", selected["validation_jobs"])
    write_jsonl(target / "split_manifest.jsonl", selected["split_manifest"])
    write_jsonl(target / "knowledge_graph" / "mentions.jsonl", selected["mentions"])
    write_jsonl(target / "knowledge_graph" / "training_edges.jsonl", selected["edges"])
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / "ontology.yaml", target / "ontology.yaml")

    validation_job = selected["validation_jobs"][0]
    job_id = str(validation_job["job_id"])
    source_index = load_jsonl(source / "validation_index.jsonl")
    index_row = next((row for row in source_index if str(row.get("job_id")) == job_id), None)
    if index_row is None:
        raise ValueError(f"{dataset}: validation index is missing {job_id}")
    gold_rows = load_jsonl(source / "validation_gold.jsonl")
    record_index = int(index_row["record_index"])
    if record_index < 0 or record_index >= len(gold_rows):
        raise ValueError(f"{dataset}: invalid validation record_index {record_index}")
    write_jsonl(target / "validation_gold.jsonl", [gold_rows[record_index]])
    write_jsonl(target / "validation_index.jsonl", [{**index_row, "record_index": 0}])
    return {
        "train_job_id": selected["train_jobs"][0]["job_id"],
        "validation_job_id": job_id,
        "mentions": len(selected["mentions"]),
        "training_edges": len(selected["edges"]),
        "test_job_file_read": False,
        "test_gold_read": False,
    }


def _entity(
    entity_id: str,
    text: str,
    entity_type: str,
    segment: Mapping[str, Any],
    local_start: int,
) -> dict[str, Any]:
    global_start = int(segment.get("start", 0)) + local_start
    return {
        "id": entity_id,
        "text": text,
        "normalized_name": None,
        "type": entity_type,
        "evidence": {
            "text": text,
            "segment_id": segment["segment_id"],
            "page": segment.get("page"),
            "start": global_start,
            "end": global_start + len(text),
        },
        "confidence": 1.0,
        "review_status": "pending",
        "created_by": "public-cpu-smoke-fixture",
    }


def build_gate_fixture(job: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    segment = next(
        (item for item in job.get("segments", []) if str(item.get("text", "")).strip()),
        None,
    )
    if segment is None:
        raise ValueError(f"job {job.get('job_id')!r} has no non-empty segment")
    source_text = str(segment["text"])
    entity_types = list(job["ontology"]["entity_types"])
    if not entity_types:
        raise ValueError("job ontology has no entity types")
    anchors = job.get("kg_v2_context", {}).get("anchors", [])
    accepted_match: tuple[str, str, int] | None = None
    for anchor in anchors:
        match = re.search(re.escape(str(anchor.get("text", ""))), source_text, re.IGNORECASE)
        if match and anchor.get("type") in entity_types:
            accepted_match = (match.group(0), str(anchor["type"]), match.start())
            break
    if accepted_match is None:
        match = re.search(r"\w+(?:[-_]\w+)*", source_text)
        if not match:
            raise ValueError("cannot construct a source-grounded fixture entity")
        accepted_match = (match.group(0), entity_types[0], match.start())
    accepted_text, accepted_type, accepted_start = accepted_match
    accepted_key = (accepted_text.casefold(), accepted_type)
    anchor_keys = {
        (str(anchor.get("text", "")).casefold(), str(anchor.get("type", "")))
        for anchor in anchors
    }

    rejected_match: tuple[str, str, int] | None = None
    for match in re.finditer(r"\w+(?:[-_]\w+)*", source_text):
        candidate = (match.group(0).casefold(), accepted_type)
        if candidate != accepted_key and candidate not in anchor_keys:
            rejected_match = (match.group(0), accepted_type, match.start())
            break
    if rejected_match is None:
        rejected_type = next((value for value in entity_types if value != accepted_type), accepted_type)
        rejected_match = (accepted_text, rejected_type, accepted_start)
        if (rejected_match[0].casefold(), rejected_match[1]) in anchor_keys:
            rejected_match = (source_text, rejected_type, 0)
    rejected_text, rejected_type, rejected_start = rejected_match

    accepted = _entity("E1", accepted_text, accepted_type, segment, accepted_start)
    rejected = _entity("E2", rejected_text, rejected_type, segment, rejected_start)
    common = {
        "schema_version": job["ontology"].get("annotation_schema_version", "0.1.0"),
        "document_id": job["document_id"],
        "language": job.get("language", "unknown"),
        "relations": [],
        "review": {"status": "unreviewed", "reviewers": [], "notes": "CPU smoke fixture"},
    }
    v1 = {**common, "entities": [{**accepted, "id": "V1"}]}
    v2 = {**common, "entities": [accepted, rejected]}
    return v1, v2


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def qlora_static_checks(fixture_source: Path) -> dict[str, Any]:
    trainer = load_module(ROOT / "scripts" / "train_qlora.py", "cpu_smoke_train_qlora")
    inference = load_module(
        ROOT / "scripts" / "run_qlora_inference.py", "cpu_smoke_run_qlora_inference"
    )
    dataset = DATASETS[0]
    source = fixture_source / dataset
    examples = trainer.build_examples(
        source / "validation_gold.jsonl",
        source / "validation_index.jsonl",
        source / "validation_baseline_jobs.jsonl",
        compact_target=True,
        use_job_instruction=False,
    )
    if len(examples) != 1:
        raise ValueError(f"QLoRA fixture check expected one example, got {len(examples)}")
    parsed = inference.parse_json('{"entities": [], "relations": []}')
    if parsed.get("entities") != [] or parsed.get("relations") != []:
        raise ValueError("QLoRA parser fixture check returned an unexpected annotation")
    bitsandbytes = importlib.import_module("bitsandbytes")
    return {
        "status": GPU_RUNTIME_STATUS,
        "gpu_runtime_executed": False,
        "module_imports": ["train_qlora", "run_qlora_inference", "bitsandbytes"],
        "bitsandbytes_version": getattr(bitsandbytes, "__version__", "unknown"),
        "fixture_examples_built": len(examples),
        "mock_parser_checked": True,
    }


def pytest_command(python: str, junit_path: Path) -> list[str]:
    bootstrap = (
        "import sys; "
        "sys.path.append('/usr/lib/python3/dist-packages'); "
        "import pytest; "
        "raise SystemExit(pytest.main(sys.argv[1:]))"
    )
    return [
        python,
        "-c",
        bootstrap,
        "-q",
        "tests",
        f"--junitxml={junit_path}",
    ]


def junit_summary(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    totals: Counter[str] = Counter()
    for suite in suites:
        for key in ("tests", "failures", "errors", "skipped"):
            totals[key] += int(suite.attrib.get(key, 0))
    return dict(totals)


def run(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}"
    artifact_root = output_root / "runs" / run_id
    fixture_source = artifact_root / "source_fixture"
    prepared_root = artifact_root / "prepared"
    status = StatusTracker(output_root / "status.json", run_id, artifact_root)
    environment = dict(os.environ)
    python = sys.executable
    try:
        status.stage("cpu_environment_guard")
        status.value["environment"] = validate_cpu_environment(environment)
        status.check("cpu_environment_guard", "passed")

        status.stage("public_input_fixture")
        for dataset in DATASETS:
            status.value["datasets"][dataset] = {
                "status": "building_input_fixture",
                **build_source_fixture(dataset, args.source_root, fixture_source),
            }
            status.check(f"{dataset}_public_input_fixture", "passed")

        status.stage("public_hrge_preprocessing")
        run_command(
            [
                python,
                "scripts/prepare_public_hrge_cpu.py",
                "--datasets",
                *DATASETS,
                "--source-root",
                str(fixture_source),
                "--output-root",
                str(prepared_root),
                "--disable-semantic",
                "--threads",
                "1",
                "--batch-size",
                "1",
                "--force",
            ],
            environment,
        )
        for dataset in DATASETS:
            manifest_path = prepared_root / dataset / "preparation_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not (
                manifest.get("status") == "prepared_train_and_validation"
                and manifest.get("semantic_device") == "cpu"
                and manifest.get("test_job_file_read") is False
                and manifest.get("test_gold_read") is False
                and manifest.get("validation_gold_read") is False
            ):
                raise ValueError(f"{dataset}: CPU preparation manifest failed its leakage/runtime gate")
            status.value["datasets"][dataset].update(
                {
                    "preprocessing": "complete",
                    "preparation_manifest": str(manifest_path),
                    "preparation_manifest_sha256": sha256(manifest_path),
                }
            )
            status.check(f"{dataset}_hrge_preprocessing", "passed")

        status.stage("hrge_gate_fusion_and_evaluation")
        for dataset in DATASETS:
            dataset_artifacts = artifact_root / "pipeline" / dataset
            prepared_job = load_jsonl(
                prepared_root / dataset / "jobs" / "validation_hrge_jobs.jsonl"
            )[0]
            jobs_path = dataset_artifacts / "validation_baseline_jobs.jsonl"
            v1_path = dataset_artifacts / "eae_fixture.jsonl"
            v2_path = dataset_artifacts / "hrge_fixture.jsonl"
            verified_path = dataset_artifacts / "hrge_verified.jsonl"
            verified_audit = dataset_artifacts / "hrge_verified_audit.jsonl"
            fused_path = dataset_artifacts / "pge_fused.jsonl"
            fusion_audit = dataset_artifacts / "pge_fusion_audit.jsonl"
            write_jsonl(jobs_path, [prepared_job])
            v1, v2 = build_gate_fixture(prepared_job)
            write_jsonl(v1_path, [{"job_id": prepared_job["job_id"], "annotation": v1}])
            write_jsonl(v2_path, [{"job_id": prepared_job["job_id"], "annotation": v2}])
            ontology = fixture_source / dataset / "ontology.yaml"
            run_command(
                [
                    python,
                    "scripts/verify_relations.py",
                    "--annotations",
                    str(v2_path),
                    "--ontology",
                    str(ontology),
                    "--output",
                    str(verified_path),
                    "--audit",
                    str(verified_audit),
                ],
                environment,
            )
            run_command(
                [
                    python,
                    "scripts/fuse_kg_v1_v2_predictions.py",
                    "--v1",
                    str(v1_path),
                    "--v2",
                    str(v2_path),
                    "--verified",
                    str(verified_path),
                    "--jobs",
                    str(jobs_path),
                    "--relation-mode",
                    "verified",
                    "--output",
                    str(fused_path),
                    "--audit",
                    str(fusion_audit),
                ],
                environment,
            )
            audit_rows = load_jsonl(fusion_audit)
            accepted = sum(bool(row.get("accepted")) for row in audit_rows)
            rejected = sum(not bool(row.get("accepted")) for row in audit_rows)
            if accepted < 1 or rejected < 1:
                raise ValueError(
                    f"{dataset}: gate fixture did not exercise acceptance and rejection"
                )

            gold = fixture_source / dataset / "validation_gold.jsonl"
            gold_index = fixture_source / dataset / "validation_index.jsonl"
            normalized_metrics = dataset_artifacts / "normalized_text_metrics.json"
            span_metrics = dataset_artifacts / "character_span_metrics.json"
            evidence_metrics = dataset_artifacts / "evidence_graph_metrics.json"
            run_command(
                [
                    python,
                    "scripts/evaluate_annotations.py",
                    "--gold",
                    str(gold),
                    "--gold-index",
                    str(gold_index),
                    "--predictions",
                    str(fused_path),
                    "--jobs",
                    str(jobs_path),
                    "--include-missing-as-empty",
                    "--output",
                    str(normalized_metrics),
                ],
                environment,
            )
            run_command(
                [
                    python,
                    "scripts/evaluate_public_validation_spans.py",
                    "--gold",
                    str(gold),
                    "--gold-index",
                    str(gold_index),
                    "--predictions",
                    str(fused_path),
                    "--jobs",
                    str(jobs_path),
                    "--output",
                    str(span_metrics),
                ],
                environment,
            )
            run_command(
                [
                    python,
                    "scripts/evaluate_evidence_graph.py",
                    "--gold",
                    str(gold),
                    "--gold-index",
                    str(gold_index),
                    "--predictions",
                    str(fused_path),
                    "--jobs",
                    str(jobs_path),
                    "--ontology",
                    str(ontology),
                    "--output",
                    str(evidence_metrics),
                ],
                environment,
            )
            metrics = {
                "normalized_text": json.loads(normalized_metrics.read_text(encoding="utf-8")),
                "character_span": json.loads(span_metrics.read_text(encoding="utf-8")),
                "evidence_graph": json.loads(evidence_metrics.read_text(encoding="utf-8")),
            }
            if metrics["normalized_text"].get("jobs_evaluated") != 1:
                raise ValueError(f"{dataset}: normalized evaluator did not process one job")
            if metrics["character_span"].get("jobs_evaluated") != 1:
                raise ValueError(f"{dataset}: span evaluator did not process one job")
            status.value["datasets"][dataset].update(
                {
                    "status": "complete",
                    "hrge_gate": {
                        "accepted_entities": accepted,
                        "rejected_entities": rejected,
                        "audit": str(fusion_audit),
                    },
                    "evaluation": {
                        "status": "complete",
                        "jobs": 1,
                        "normalized_text": str(normalized_metrics),
                        "character_span": str(span_metrics),
                        "evidence_graph": str(evidence_metrics),
                    },
                }
            )
            status.check(f"{dataset}_hrge_gate_and_evaluation", "passed")

        status.stage("gliner_glirel_preflight")
        preflights = {}
        for dataset in DATASETS:
            output = run_command(
                [
                    python,
                    "scripts/run_gliner_glirel_validation.py",
                    "--dataset",
                    dataset,
                    "--jobs",
                    str(
                        args.source_root
                        / dataset
                        / "validation_baseline_jobs.jsonl"
                    ),
                    "--limit",
                    "1",
                    "--device",
                    "cpu",
                    "--dtype",
                    "float32",
                    "--preflight-only",
                ],
                environment,
            )
            report = json.loads(output)
            if report.get("status") != "ready" or report.get("local_only") is not True:
                raise ValueError(f"{dataset}: GLiNER/GLiREL preflight was not ready/local-only")
            preflight_path = artifact_root / "preflight" / f"{dataset}.json"
            write_json(preflight_path, report)
            preflights[dataset] = {
                "status": "ready",
                "jobs_checked": report["jobs"],
                "report": str(preflight_path),
            }
        status.value["gliner_glirel"] = {
            "status": "preflight_complete",
            "scope": "preflight_only",
            "runtime": GPU_RUNTIME_STATUS,
            "model_inference_executed": False,
            "datasets": preflights,
        }
        status.check("gliner_glirel_preflight_all_datasets", "passed")
        status.check("gliner_glirel_model_runtime", GPU_RUNTIME_STATUS, "warning")

        status.stage("qlora_import_parameter_fixture_checks")
        qlora = qlora_static_checks(fixture_source)
        for script, required in (
            ("scripts/train_qlora.py", ("--model-path", "--compact-target")),
            ("scripts/run_qlora_inference.py", ("--model-path", "--max-new-tokens")),
        ):
            help_output = run_command([python, script, "--help"], environment)
            if any(flag not in help_output for flag in required):
                raise ValueError(f"{script}: required CLI parameters are missing")
        qlora["cli_parameter_help_checked"] = True
        status.value["qlora_bitsandbytes"] = qlora
        status.check("qlora_bitsandbytes_static_and_fixture_checks", "passed")
        status.check("qlora_bitsandbytes_gpu_runtime", GPU_RUNTIME_STATUS, "warning")

        status.stage("project_pytest")
        junit = artifact_root / "pytest" / "junit.xml"
        junit.parent.mkdir(parents=True, exist_ok=True)
        test_files = sorted(str(path.relative_to(ROOT)) for path in (ROOT / "tests").glob("test_*.py"))
        run_command(pytest_command(python, junit), environment)
        summary = junit_summary(junit)
        if summary.get("failures", 0) or summary.get("errors", 0):
            raise ValueError(f"project pytest reported failures: {summary}")
        status.value["pytest"] = {
            "status": "passed",
            "scope": "tests/",
            "vendored_external_tests_excluded": True,
            "test_files": test_files,
            "test_file_count": len(test_files),
            **summary,
            "junit_xml": str(junit),
        }
        status.check("project_pytest", f"passed: {summary.get('tests', 0)} tests")
        status.complete()
        print(json.dumps(status.value, ensure_ascii=False, indent=2), flush=True)
        return 0
    except BaseException as error:
        status.fail(error)
        traceback.print_exc()
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=ROOT / "data" / "processed" / "public_benchmarks_full",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs" / "public_cpu_smoke",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
