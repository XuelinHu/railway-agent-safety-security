import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_gliner_glirel_formal_test.sh"
INFERENCE = ROOT / "scripts/run_gliner_glirel_validation.py"
CONTRACT_PATH = ROOT / "scripts/gliner_glirel_formal_contract.py"
PROTOCOL = (
    ROOT
    / "outputs"
    / "public_horizontal_validation"
    / "gliner_glirel_calibrated"
    / "protocol.json"
)


def load_contract():
    spec = importlib.util.spec_from_file_location("formal_glirel_contract", CONTRACT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def load_inference():
    spec = importlib.util.spec_from_file_location("formal_glirel_inference", INFERENCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_formal_runner_is_release_gated_resumable_and_uses_shared_lock():
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    source = RUNNER.read_text(encoding="utf-8")
    assert 'release_status="outputs/public_formal_matrix/release_status.json"' in source
    assert "while ! jq -e '.status == \"complete\"'" in source
    assert 'gpu_lock="${PUBLIC_GPU_LOCK_FILE:-outputs/.locks/public-validation-gpu.lock}"' in source
    assert "--split test" in source
    assert "test_baseline_jobs.jsonl" in source
    assert "--resume" in source
    assert "--allow-non-validation" in source
    assert "canonical_fingerprint" in source
    assert 'release_contract="scripts/qwen_zeroshot_formal_contract.py"' in source
    assert "verify-release" in source
    assert "current_release_sha256" in source
    assert "current_release_fingerprint" in source
    assert '[[ ! -f "$span_metrics" ]]' not in source
    resume_validation = source.index('"$formal_contract" validate --data-root')
    acquire = source.index("  acquire_gpu_lock\n", resume_validation)
    inference = source.index("scripts/run_gliner_glirel_validation.py", acquire)
    release = source.index("  release_gpu_lock\n", inference)
    merge = source.index("scripts/merge_gliner_glirel_shards.py", release)
    evaluation = source.index("scripts/evaluate_annotations.py", release)
    assert resume_validation < acquire < inference < release < merge < evaluation

    canonical = source.index('"$formal_contract" verify-release')
    formal_read = source.index("formal_test_read=true")
    test_preflight = source.index('"$release_contract" preflight')
    assert canonical < formal_read < test_preflight


def test_test_job_contract_accepts_only_exact_test_namespace(tmp_path: Path):
    module = load_inference()
    path = tmp_path / "test_baseline_jobs.jsonl"
    job = {
        "job_id": "conll04_test_17_C1",
        "document_id": "conll04_test_17",
        "category": "conll04",
        "source_path": "public:conll04:test:17",
        "experiment_mode": "baseline",
        "segments": [{"segment_id": "S1", "start": 0, "end": 4, "text": "Demo"}],
        "ontology": {
            "entity_types": {"Peop": {}},
            "relation_types": {"Live_In": {}},
            "allowed_relation_signatures": {
                "Live_In": {"source": ["Peop"], "target": ["Peop"]}
            },
            "claim_statuses": {"explicit": "direct"},
        },
    }
    path.write_text(json.dumps(job) + "\n", encoding="utf-8")
    rows, _, _ = module.load_validation_jobs(path, "conll04", split="test")
    assert rows[0]["job_id"] == job["job_id"]

    job["source_path"] = "public:conll04:validation:17"
    path.write_text(json.dumps(job) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="test split"):
        module.load_validation_jobs(path, "conll04", split="test")


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_resume_contract_rejects_existing_but_incomplete_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    contract = load_contract()
    monkeypatch.setitem(contract.EXPECTED_JOBS, "conll04", 2)
    data_root = tmp_path / "data"
    run_root = tmp_path / "run"
    base = data_root / "conll04"
    jobs = []
    gold = []
    index = []
    predictions = []
    for position in range(2):
        job_id = f"conll04_test_{position + 1}_C1"
        document_id = f"conll04_test_{position + 1}"
        text = f"Sentence {position}."
        jobs.append(
            {
                "job_id": job_id,
                "document_id": document_id,
                "language": "en",
                "category": "conll04",
                "source_path": f"public:conll04:test:{position + 1}",
                "experiment_mode": "baseline",
                "segments": [
                    {
                        "segment_id": "S1",
                        "start": 0,
                        "end": len(text),
                        "text": text,
                    }
                ],
            }
        )
        value = {
            "document_id": document_id,
            "language": "en",
            "entities": [],
            "relations": [],
        }
        gold.append(value)
        index.append({"job_id": job_id, "record_index": position})
        predictions.append({"job_id": job_id, "annotation": value})
    _write_jsonl(base / "test_baseline_jobs.jsonl", jobs)
    _write_jsonl(base / "test_gold.jsonl", gold)
    _write_jsonl(base / "test_index.jsonl", index)
    (base / "ontology.yaml").write_text("entity_types: {}\n", encoding="utf-8")
    paths = contract.paths_for(data_root, run_root, "conll04")
    _write_jsonl(paths["predictions"], predictions)
    _write_json(
        paths["normalized_text_metrics"],
        {
            "jobs_gold": 2,
            "jobs_predicted": 2,
            "jobs_evaluated": 2,
            "jobs_missing_predictions": 0,
            "generation_success_rate": 1.0,
        },
    )
    _write_json(
        paths["character_span_metrics"],
        {
            "metric": "strict-source-character-span",
            "selection_split": "test",
            "formal_test_read": True,
            "jobs_gold": 2,
            "jobs_predicted": 2,
            "jobs_evaluated": 2,
            "jobs_missing_predictions": 0,
            "generation_success_rate": 1.0,
            "per_job": {row["job_id"]: {} for row in jobs},
        },
    )

    result = contract.validate_dataset(data_root, run_root, "conll04", 2)
    assert result["prediction_rows"] == 2

    broken = json.loads(paths["character_span_metrics"].read_text(encoding="utf-8"))
    broken["jobs_predicted"] = 1
    _write_json(paths["character_span_metrics"], broken)
    with pytest.raises(contract.ContractError, match="metric coverage"):
        contract.validate_dataset(data_root, run_root, "conll04", 2)


def test_calibrated_protocol_is_hash_bound_to_train_only_calibration():
    contract = load_contract()
    result = contract.validate_protocol(PROTOCOL.relative_to(ROOT))
    assert result["status"] == "valid"
    assert len(result["sha256"]) == 64
    assert result["test_gold_used_for_selection"] is False
    assert result["datasets"]["conll04"]["relation_threshold"] == 0.1
