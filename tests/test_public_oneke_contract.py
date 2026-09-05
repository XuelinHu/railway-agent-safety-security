import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_module(name: str, relative: str):
    scripts_text = str(SCRIPTS)
    if scripts_text not in sys.path:
        sys.path.insert(0, scripts_text)
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def validation_job() -> dict:
    text = "Aspirin caused nausea."
    return {
        "job_id": "ade_validation_1_C1",
        "document_id": "ade_validation_1",
        "category": "ade",
        "source_path": "public:ade:validation:1",
        "segments": [
            {"segment_id": "S1", "start": 10, "end": 10 + len(text), "text": text}
        ],
        "ontology": {
            "entity_types": {"Drug": {}, "Adverse-Effect": {}},
            "relation_types": {"Adverse-Effect": {}},
            "allowed_relation_signatures": {
                "Adverse-Effect": [["Drug", "Adverse-Effect"]]
            },
        },
    }


def test_prepare_and_convert_are_cpu_only_and_preserve_global_spans(tmp_path, monkeypatch):
    adapter = load_module("public_oneke_adapter_test", "scripts/public_external_adapters/oneke.py")
    monkeypatch.chdir(tmp_path)
    jobs = Path("validation_jobs.jsonl")
    write_jsonl(jobs, [validation_job()])

    requests, summary = adapter.prepare_requests(jobs, "ade", 1)

    assert summary["test_gold_read"] is False
    assert requests[0]["provenance"] == {
        "source_path": "public:ade:validation:1",
        "prompt_uses_gold": False,
        "test_gold_read": False,
    }
    raw = {
        "schema_version": adapter.RAW_SCHEMA,
        "job_id": requests[0]["job_id"],
        "dataset": "ade",
        "split": "validation",
        "seed": 42,
        "status": "complete",
        "tasks": {
            "ner": {
                "parsed": {
                    "entity_list": [
                        {"name": "Aspirin", "type": "Drug"},
                        {"name": "nausea", "type": "Adverse-Effect"},
                    ]
                }
            },
            "re": {
                "parsed": {
                    "relation_list": [
                        {
                            "head": "Aspirin",
                            "tail": "nausea",
                            "relation": "Adverse-Effect",
                        }
                    ]
                }
            },
        },
    }
    prediction, audit = adapter.convert_one(requests[0], raw)

    entities = prediction["annotation"]["entities"]
    assert [(item["evidence"]["start"], item["evidence"]["end"]) for item in entities] == [
        (10, 17),
        (25, 31),
    ]
    assert prediction["annotation"]["relations"][0]["evidence"]["start"] == 10
    assert audit["test_gold_read"] is False
    assert "torch" not in adapter.__dict__


def test_nested_test_namespaces_are_rejected_even_with_validation_basename():
    adapter = load_module("public_oneke_adapter_path_test", "scripts/public_external_adapters/oneke.py")

    with pytest.raises(ValueError, match="test namespace"):
        adapter.reject_test_path(
            Path("public_benchmarks_test_v1") / "validation_jobs.jsonl", "jobs"
        )
    # Do not reject incidental substrings such as pytest or contest.
    adapter.reject_test_path(Path("pytest-cache") / "contest.jsonl", "jobs")
    assert adapter.canonical_label("adverse effect", ["Adverse-Effect"], "unknown") == (
        "Adverse-Effect",
        True,
    )


def test_worker_rejects_gold_conditioning_and_untrusted_resume_rows(tmp_path, monkeypatch):
    adapter = load_module("public_oneke_adapter_worker_test", "scripts/public_external_adapters/oneke.py")
    worker = load_module("public_oneke_worker_contract_test", "scripts/run_public_oneke_formal.py")
    monkeypatch.chdir(tmp_path)
    jobs = Path("validation_jobs.jsonl")
    write_jsonl(jobs, [validation_job()])
    requests, _ = adapter.prepare_requests(jobs, "ade", 1)
    request_path = Path("validation_requests.jsonl")
    write_jsonl(request_path, requests)

    worker.validate_requests(requests, request_path)
    conditioned = json.loads(json.dumps(requests))
    conditioned[0]["provenance"]["prompt_uses_gold"] = True
    with pytest.raises(ValueError, match="gold-conditioned"):
        worker.validate_requests(conditioned, request_path)

    resumable = {
        "schema_version": adapter.RAW_SCHEMA,
        "job_id": requests[0]["job_id"],
        "dataset": "ade",
        "split": "validation",
        "seed": 42,
        "status": "complete",
        "model_revision": worker.MODEL_REVISION,
        "request_sha256": worker.request_sha256(requests[0]),
        "test_gold_read": False,
    }
    assert worker.validate_resumable_raw([resumable], requests, Path("raw.jsonl")) == {
        requests[0]["job_id"]
    }
    resumable["test_gold_read"] = True
    with pytest.raises(ValueError, match="test_gold_read"):
        worker.validate_resumable_raw([resumable], requests, Path("raw.jsonl"))


def test_worker_reproduces_upstream_prompt_and_normalizes_documented_native_output():
    worker = load_module("public_oneke_worker_prompt_test", "scripts/run_public_oneke_formal.py")

    prompt = worker.build_prompt("NER", ["Drug", "Adverse-Effect"], "Aspirin caused nausea.")
    expected_body = (
        "\n{\n"
        f'    "instruction": {worker.INSTRUCTIONS["NER"]},\n'
        "    \"schema\": ['Drug', 'Adverse-Effect'],\n"
        "    \"input\": Aspirin caused nausea.,\n"
        "}\n"
    )
    assert prompt == (
        "[INST] <<SYS>>\nYou are a helpful assistant. "
        "你是一个乐于助人的助手。\n<</SYS>>\n\n"
        f"{expected_body}[/INST]"
    )
    assert worker.schema_batches("RE", [str(i) for i in range(7)]) == [
        ["0", "1", "2", "3"],
        ["4", "5", "6"],
    ]

    ner, error, response_format = worker.normalize_task_payload(
        "NER",
        {"Drug": ["Aspirin"], "Adverse Effect": ["nausea"]},
        ["Drug", "Adverse-Effect"],
    )
    assert error is None
    assert response_format == "native_schema_map"
    assert ner == {
        "entity_list": [
            {"name": "Aspirin", "type": "Drug"},
            {"name": "nausea", "type": "Adverse-Effect"},
        ]
    }
    relation, error, _ = worker.normalize_task_payload(
        "RE",
        {"Adverse Effect": [{"subject": "Aspirin", "object": "nausea"}]},
        ["Adverse-Effect"],
    )
    assert error is None
    assert relation == {
        "relation_list": [
            {"head": "Aspirin", "tail": "nausea", "relation": "Adverse-Effect"}
        ]
    }

    valid_canary = {
        "tasks": {
            "ner": {"parsed": ner},
            "re": {
                "parsed": {
                    "relation_list": [
                        {
                            "head": "nausea",
                            "tail": "Aspirin",
                            "relation": "Adverse-Effect",
                        }
                    ]
                }
            },
        }
    }
    worker.validate_canary_extraction(valid_canary)
    valid_canary["tasks"]["re"]["parsed"]["relation_list"][0].update(
        {"head": "Aspirin", "tail": "nausea"}
    )
    with pytest.raises(RuntimeError, match="frozen ADE relation signature"):
        worker.validate_canary_extraction(valid_canary)


def test_failed_direction_assertion_can_be_hash_revalidated_without_gpu(tmp_path):
    worker = load_module(
        "public_oneke_worker_revalidate_test", "scripts/run_public_oneke_formal.py"
    )
    model = tmp_path / worker.MODEL_REVISION
    model.mkdir()
    for name in ("config.json", "tokenizer.model"):
        (model / name).write_text("{}", encoding="utf-8")
    (model / "weights.bin").write_bytes(b"weight")
    (model / "pytorch_model.bin.index.json").write_text(
        json.dumps({"weight_map": {"layer": "weights.bin"}}), encoding="utf-8"
    )
    ner_raw = '{"Drug": ["Aspirin"], "Adverse-Effect": ["nausea"], "Person": ["Alice"]}'
    re_raw = '{"Adverse-Effect": [{"subject": "nausea", "object": "Aspirin"}]}'

    def diagnostic(raw: str, constraint: list[str]):
        return {
            "batches": [
                {
                    "batch": 1,
                    "constraint": constraint,
                    "raw_text_sha256": __import__("hashlib").sha256(
                        raw.encode("utf-8")
                    ).hexdigest(),
                    "raw_text_preview": raw,
                    "raw_text_truncated": False,
                    "telemetry": {"input_tokens": 100, "output_tokens": 20},
                }
            ]
        }

    marker = tmp_path / "gpu_canary.json"
    marker.write_text(
        json.dumps(
            {
                "schema_version": "public-external-gpu-canary-v1",
                "baseline": "oneke",
                "gpu": "RTX 3090",
                "capacity_gib": 24.0,
                "status": "failed",
                "exit_code": 1,
                "model_revision": worker.MODEL_REVISION,
                "quantization": "bitsandbytes-nf4-double-quantization",
                "prompt_version": worker.PROMPT_VERSION,
                "test_gold_read": False,
                "terminal": True,
                "actual_gpu_name": "NVIDIA GeForce RTX 3090",
                "actual_total_memory_bytes": 23 * 1024**3,
                "peak_allocated_bytes": 8 * 1024**3,
                "peak_reserved_bytes": 9 * 1024**3,
                "error": "RuntimeError: canary RE did not recover the directed synthetic relation",
                "canary_task_diagnostics": {
                    "ner": diagnostic(ner_raw, ["Drug", "Adverse-Effect", "Person"]),
                    "re": diagnostic(re_raw, ["Adverse-Effect"]),
                },
            }
        ),
        encoding="utf-8",
    )

    assert worker.revalidate_canary(
        argparse.Namespace(model=model, marker=marker)
    ) == 0
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["runtime_compatible"] is True
    assert payload["revalidated_without_gpu"] is True
    assert payload["synthetic_relation_signature"] == {
        "type": "Adverse-Effect",
        "source": "Adverse-Effect",
        "target": "Drug",
        "observed": "nausea->Aspirin",
    }
    assert "directed synthetic relation" in payload["revalidation"][
        "prior_assertion_error"
    ]


def test_canary_preflight_failure_always_writes_terminal_failure_marker(tmp_path):
    worker = load_module("public_oneke_worker_marker_test", "scripts/run_public_oneke_formal.py")
    marker = tmp_path / "gpu_canary.json"
    args = argparse.Namespace(
        model=tmp_path / worker.MODEL_REVISION,
        marker=marker,
        max_input_tokens=32,
        max_new_tokens=8,
    )

    assert worker.run_canary(args) == 1
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["exit_code"] == 1
    assert payload["terminal"] is True
    assert payload["test_gold_read"] is False
    assert "snapshot is incomplete" in payload["error"]


def test_runtime_map_preserves_venv_interpreter_symlink(tmp_path):
    preflight = load_module(
        "public_external_preflight_runtime_test",
        "scripts/preflight_public_external_formal.py",
    )
    bootstrap = tmp_path / "bootstrap-python"
    bootstrap.write_text("#!/bin/sh\n", encoding="utf-8")
    bootstrap.chmod(0o755)
    venv_python = tmp_path / "venv/bin/python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(bootstrap)
    runtime_map = tmp_path / "runtime_map.json"
    runtime_map.write_text(
        json.dumps({"oneke": {"python": str(venv_python)}}), encoding="utf-8"
    )

    runtimes, blockers = preflight.load_runtime_map(runtime_map, ["oneke"])

    assert blockers == []
    assert runtimes["oneke"] == venv_python.absolute()
    assert runtimes["oneke"] != bootstrap.resolve()


def test_formal_runner_holds_shared_gpu_lock_only_during_inference():
    source = (ROOT / "scripts/run_public_oneke_formal.sh").read_text(encoding="utf-8")
    inference = source.index('CUDA_VISIBLE_DEVICES=0 "$runtime" scripts/run_public_oneke_formal.py')
    unlock = source.index("release_gpu_lock", inference)
    conversion = source.index("scripts/public_external_adapters/oneke.py convert", inference)
    evaluation = source.index("scripts/evaluate_public_validation_spans.py", conversion)

    assert source.index("while ! flock -n 8", source.index("for dataset in conll04")) < inference
    assert inference < unlock < conversion < evaluation
    on_exit = source.split("on_exit() {", 1)[1].split("\n}", 1)[0]
    assert "release_gpu_lock" in on_exit
    assert 'runner_status="$run_root/status.json"' in source


def test_watchdog_tracks_oneke_canary_and_only_expects_gpu_in_real_gpu_phases():
    watchdog = (ROOT / "scripts/monitor_public_experiments.sh").read_text(
        encoding="utf-8"
    )

    assert "public-formal-oneke-canary.service" in watchdog
    assert "public-formal-oneke-validation.service" in watchdog
    assert '*"run_public_oneke_formal.py"*' in watchdog
    assert '"$oneke_canary_stage" == gpu_canary' in watchdog
    assert '"$oneke_formal_stage" == inference' in watchdog
    assert "outputs/public_external_formal/oneke/gpu_canary.json passed" in watchdog
    assert "outputs/public_external_formal/oneke/status.json complete" in watchdog
