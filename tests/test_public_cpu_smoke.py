import importlib.util
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_public_cpu_smoke",
    ROOT / "scripts" / "run_public_cpu_smoke.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_status_never_claims_gpu_runtime_coverage(tmp_path):
    status = MODULE.base_status("run-1", tmp_path)
    assert status["cpu_only"] is True
    assert status["gpu_runtime"]["status"] == "gpu_runtime_not_covered"
    assert status["gpu_runtime"]["covered"] is False
    assert status["qlora_bitsandbytes"]["status"] == "gpu_runtime_not_covered"
    assert status["gliner_glirel"]["runtime"] == "gpu_runtime_not_covered"
    assert status["execution"]["thread_limit"] == 1
    assert status["counts"] == {"passed": 0, "failed": 0, "warnings": 0}
    assert status["checks"] == []


def test_cpu_guard_requires_hidden_cuda_and_every_thread_limit():
    environment = {
        "CUDA_VISIBLE_DEVICES": "",
        "NVIDIA_VISIBLE_DEVICES": "void",
        **{name: "1" for name in MODULE.THREAD_VARIABLES},
    }
    result = MODULE.validate_cpu_environment(environment, inspect_torch=False)
    assert result["CUDA_VISIBLE_DEVICES"] == ""
    broken = {**environment, "OPENBLAS_NUM_THREADS": "4"}
    try:
        MODULE.validate_cpu_environment(broken, inspect_torch=False)
    except RuntimeError as error:
        assert "single-thread" in str(error)
    else:
        raise AssertionError("multi-thread environment should be rejected")


def test_fixture_selection_excludes_unselected_and_test_documents():
    train = [{"job_id": "T_C1", "document_id": "T"}]
    validation = [{"job_id": "V_C1", "document_id": "V"}]
    mentions = [
        {"document_id": "T", "text": "train"},
        {"document_id": "V", "text": "validation"},
        {"document_id": "X", "text": "test"},
    ]
    edges = [{"document_id": "T", "relation_type": "R"}]
    manifest = [
        {"document_id": "T", "split": "train"},
        {"document_id": "V", "split": "validation"},
        {"document_id": "X", "split": "test"},
    ]
    selected = MODULE.select_public_fixture(train, validation, mentions, edges, manifest)
    assert {row["document_id"] for row in selected["mentions"]} == {"T", "V"}
    assert {row["split"] for row in selected["split_manifest"]} == {
        "train",
        "validation",
    }


def test_gate_fixture_contains_one_agreed_and_one_rejected_candidate():
    job = {
        "job_id": "demo_C1",
        "document_id": "demo",
        "language": "en",
        "ontology": {
            "annotation_schema_version": "0.1.0",
            "entity_types": {"Peop": {}, "Org": {}},
        },
        "segments": [
            {
                "segment_id": "S1",
                "start": 10,
                "end": 29,
                "text": "Alice joined Acme .",
            }
        ],
        "kg_v2_context": {
            "anchors": [{"text": "Alice", "type": "Peop"}]
        },
    }
    v1, v2 = MODULE.build_gate_fixture(job)
    assert len(v1["entities"]) == 1
    assert len(v2["entities"]) == 2
    assert (v1["entities"][0]["text"], v1["entities"][0]["type"]) == (
        v2["entities"][0]["text"],
        v2["entities"][0]["type"],
    )
    assert v2["entities"][0]["evidence"]["start"] == 10


def test_shell_wrapper_hides_cuda_and_sets_all_thread_limits():
    wrapper = (ROOT / "scripts" / "run_public_cpu_smoke.sh").read_text(encoding="utf-8")
    assert 'export CUDA_VISIBLE_DEVICES=""' in wrapper
    assert 'export NVIDIA_VISIBLE_DEVICES="void"' in wrapper
    for name in MODULE.THREAD_VARIABLES:
        assert f"export {name}=1" in wrapper


def test_pytest_command_is_limited_to_project_tests(tmp_path):
    command = MODULE.pytest_command("python", tmp_path / "junit.xml")
    assert "tests" in command
    assert not any("tools/external-baselines" in value for value in command)
