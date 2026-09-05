import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_qwen_zeroshot_formal_test.sh"
CONTRACT_PATH = ROOT / "scripts" / "qwen_zeroshot_formal_contract.py"
SPEC = importlib.util.spec_from_file_location(
    "qwen_zeroshot_formal_contract", CONTRACT_PATH
)
CONTRACT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(CONTRACT)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def annotation(document_id: str) -> dict:
    return {
        "schema_version": "0.1.0",
        "document_id": document_id,
        "language": "en",
        "entities": [],
        "relations": [],
    }


def prepare_complete_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, list[str]]:
    monkeypatch.setitem(CONTRACT.EXPECTED_JOBS, "conll04", 2)
    data_root = tmp_path / "data"
    run_root = tmp_path / "run"
    base = data_root / "conll04"
    job_ids = ["conll04_test_1_C1", "conll04_test_2_C1"]
    document_ids = ["conll04_test_1", "conll04_test_2"]
    jobs = []
    gold = []
    index = []
    for position, (job_id, document_id) in enumerate(zip(job_ids, document_ids)):
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
        gold.append(annotation(document_id))
        index.append({"job_id": job_id, "record_index": position})
    write_jsonl(base / "test_baseline_jobs.jsonl", jobs)
    write_jsonl(base / "test_gold.jsonl", gold)
    write_jsonl(base / "test_index.jsonl", index)
    (base / "ontology.yaml").write_text("entity_types: {}\n", encoding="utf-8")

    paths = CONTRACT.artifact_paths(data_root, run_root, "conll04")
    prediction_rows = [
        {"job_id": job_id, "annotation": annotation(document_id)}
        for job_id, document_id in zip(job_ids, document_ids)
    ]
    write_jsonl(paths["partial_predictions"], prediction_rows[:1])
    write_jsonl(
        paths["terminal_log"],
        [
            {"job_id": job_ids[0], "status": "success"},
            {"job_id": job_ids[1], "status": "failed"},
        ],
    )
    write_json(
        paths["materialization"],
        {
            "status": "complete",
            "jobs": 2,
            "successful_prediction_rows": 1,
            "failures_materialized_as_empty": 1,
            "missing_job_ids": [job_ids[1]],
            "gold_read": False,
        },
    )
    for name in (
        "complete_predictions",
        "expanded_predictions",
        "verified_predictions",
    ):
        write_jsonl(paths[name], prediction_rows)
    write_jsonl(paths["expansion_errors"], [])
    write_jsonl(paths["verification_audit"], [])
    write_jsonl(
        paths["span_index"],
        [
            {**row, "parent_job_id": row["job_id"], "split": "test"}
            for row in index
        ],
    )
    write_json(
        paths["normalized_text_metrics"],
        {
            "jobs_gold": 2,
            "jobs_predicted": 2,
            "jobs_evaluated": 2,
            "jobs_missing_predictions": 0,
            "generation_success_rate": 1.0,
        },
    )
    write_json(
        paths["character_span_metrics"],
        {
            "metric": "strict-global-character-span-one-to-one",
            "selection_split": "explicit-non-validation-opt-in",
            "formal_test_read": True,
            "jobs": 2,
            "generation_success_rate": 1.0,
            "per_job": {job_id: {} for job_id in job_ids},
        },
    )
    return data_root, run_root, job_ids


def test_runner_has_release_lock_inference_and_metric_contracts() -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    runner = RUNNER.read_text(encoding="utf-8")

    assert 'release_status="outputs/public_formal_matrix/release_status.json"' in runner
    assert (
        "outputs/public_formal_matrix/horizontal/qwen3_4b_zero_shot" in runner
    )
    assert "while ! jq -e '.status == \"complete\"'" in runner
    assert 'gpu_lock="${PUBLIC_GPU_LOCK_FILE:-outputs/.locks/public-validation-gpu.lock}"' in runner
    assert 'exec 8>"$gpu_lock"' in runner
    assert "flock -n 8" in runner
    assert "test_baseline_jobs.jsonl" in runner
    assert "--workers 2" in runner
    assert "--compact-target" in runner
    assert "--resume" in runner
    assert "--max-input-tokens 4096" in runner
    assert "--max-new-tokens 1024" in runner
    assert "--max-seconds-per-job 90" in runner
    assert "--retry-failed" not in runner
    assert "--adapter" not in runner
    assert "train_qlora.py" not in runner

    release_wait = runner.index("while ! jq -e")
    formal_read = runner.index("formal_test_read=true")
    preflight = runner.index('"$contract" preflight')
    lock = runner.index("  acquire_gpu_lock\n", preflight)
    inference_stage = runner.index('  current_stage="test_inference"\n', lock)
    inference_status = runner.index("  write_status running\n", inference_stage)
    inference = runner.index("scripts/run_qlora_inference_sharded.sh")
    unlock = runner.index("  release_gpu_lock\n", inference)
    assert (
        release_wait
        < formal_read
        < preflight
        < lock
        < inference_stage
        < inference_status
        < inference
        < unlock
    )

    materialize = runner.index("scripts/materialize_missing_predictions.py")
    expand = runner.index("scripts/expand_compact_predictions.py")
    verify = runner.index("scripts/verify_relations.py")
    normalized = runner.index("scripts/evaluate_annotations.py")
    strict_span = runner.index("scripts/evaluate_span_aware.py")
    assert unlock < materialize < expand < verify < normalized < strict_span
    assert "--allow-non-validation" in runner


def test_runner_freezes_counts_artifacts_and_comprehensive_status() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    contract = CONTRACT_PATH.read_text(encoding="utf-8")
    for dataset, count in (("conll04", 288), ("scierc", 551), ("ade", 427)):
        assert f"[{dataset}]={count}" in runner
    for suffix in (
        "test_partial.jsonl",
        "test_complete.jsonl",
        "test_materialization.json",
        "test_expanded.jsonl",
        "test_expand_errors.jsonl",
        "test_verified.jsonl",
        "test_verification_audit.jsonl",
        "test_normalized_text_metrics.json",
        "test_character_span_metrics.json",
    ):
        assert suffix in runner
    for field in (
        '"formal_test_read"',
        '"active_dataset"',
        '"captured_sha256"',
        '"captured_canonical_fingerprint"',
        '"successful_raw_rows"',
        '"materialized_failures"',
        '"complete_rows"',
        '"completion_valid"',
        '"terminal_failure_policy"',
        '"canonical_release_verification"',
    ):
        assert field in contract
    assert "promotion_value = canonical_validate_promotion(promotion)" in contract
    assert "from prepare_public_test_inputs import validate_promotion" in contract
    assert "write_status failed" in runner
    assert 'write_status complete' in runner
    assert "scripts/monitor_public_experiments.sh" not in runner
    assert "scripts/run_public_formal_internal_matrix.sh" not in runner


def test_dataset_resume_validation_checks_contents_not_existence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, run_root, job_ids = prepare_complete_dataset(tmp_path, monkeypatch)

    result = CONTRACT.validate_dataset(data_root, run_root, "conll04", 2)
    assert result["status"] == "complete"
    assert result["successful_raw_rows"] == 1
    assert result["materialized_failures"] == 1
    assert result["complete_rows"] == 2

    paths = CONTRACT.artifact_paths(data_root, run_root, "conll04")
    rows = CONTRACT.load_jsonl(paths["verified_predictions"])
    rows[1]["job_id"] = job_ids[0]
    write_jsonl(paths["verified_predictions"], rows)
    with pytest.raises(CONTRACT.ContractError, match="duplicate|exactly match"):
        CONTRACT.validate_dataset(data_root, run_root, "conll04", 2)


def test_status_keeps_test_sources_sealed_until_formal_read(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    malformed = data_root / "conll04" / "test_baseline_jobs.jsonl"
    malformed.parent.mkdir(parents=True)
    malformed.write_text("{not-json\n", encoding="utf-8")

    result = CONTRACT.build_status(
        state="waiting_for_formal_release",
        stage="waiting_for_formal_release",
        active_dataset=None,
        formal_test_read=False,
        data_root=data_root,
        run_root=tmp_path / "run",
        release_status=tmp_path / "release_status.json",
        release_sha256=None,
        promotion=tmp_path / "promotion.json",
        prepared_root=tmp_path / "prepared",
        model_path=tmp_path / "model",
        gpu_lock=tmp_path / "gpu.lock",
    )

    source = result["datasets"]["conll04"]["artifacts"]["jobs"]
    assert source == {
        "path": str(malformed),
        "access": "sealed_until_formal_release",
    }
    assert result["formal_test_read"] is False


def test_release_verification_requires_all_preparation_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release_status = tmp_path / "release_status.json"
    promotion = tmp_path / "promotion.json"
    prepared = tmp_path / "prepared"
    write_json(
        promotion,
        {
            "status": "promoted",
            "schema_version": "public-formal-test-promotion-v2",
            "promoted_systems": ["soe", "pge"],
            "attestation_sha256": "a" * 64,
        },
    )
    canonical_calls = []

    def canonical_verifier(path: Path) -> dict:
        canonical_calls.append(path)
        return json.loads(path.read_text(encoding="utf-8"))

    monkeypatch.setattr(CONTRACT, "canonical_validate_promotion", canonical_verifier)
    for dataset in CONTRACT.DATASETS:
        write_json(
            prepared / dataset / "preparation_manifest.json",
            {"status": "prepared_test", "test_gold_read": False},
        )

    canonical_calls_prepared = []
    canonical_release = {
        "schema_version": "public-formal-test-release-v2",
        "status": "verified_release",
        "promotion": {
            "path": str(promotion),
            "bytes": promotion.stat().st_size,
            "sha256": CONTRACT.sha256(promotion),
        },
        "promotion_attestation_sha256": "a" * 64,
        "datasets": {
            dataset: {
                "dataset": dataset,
                "manifest": {
                    "path": str(prepared / dataset / "preparation_manifest.json"),
                    "bytes": (prepared / dataset / "preparation_manifest.json").stat().st_size,
                    "sha256": CONTRACT.sha256(
                        prepared / dataset / "preparation_manifest.json"
                    ),
                },
                "status": "verified_prepared_test",
            }
            for dataset in CONTRACT.DATASETS
        },
        "release_sha256": "c" * 64,
    }

    def prepared_verifier(promotion_path: Path, prepared_root: Path) -> dict:
        canonical_calls_prepared.append((promotion_path, prepared_root))
        return canonical_release

    monkeypatch.setattr(
        CONTRACT, "canonical_validate_prepared_release", prepared_verifier
    )
    attestation = tmp_path / "prepared_release_attestation.json"
    write_json(attestation, canonical_release)
    release_payload = {
        "status": "complete",
        "stage": "complete",
        "promotion_status": "promoted",
        "gate_review_status": "passed",
        "updated_at": "2026-09-04T00:00:00+00:00",
        "datasets": {
            dataset: {
                "preparation_status": "prepared_test",
                "manifest": str(prepared / dataset / "preparation_manifest.json"),
            }
            for dataset in CONTRACT.DATASETS
        },
        "canonical_prepared_release": {
            "path": str(attestation),
            "bytes": attestation.stat().st_size,
            "sha256": CONTRACT.sha256(attestation),
            "status": "verified_release",
            "schema_version": "public-formal-test-release-v2",
            "release_sha256": "c" * 64,
        },
    }
    write_json(release_status, release_payload)

    result = CONTRACT.verify_release(release_status, promotion, prepared)
    assert result["status"] == "complete"
    assert result["release_status"]["sha256"] == CONTRACT.sha256(release_status)
    assert result["promotion"]["attestation_sha256"] == "a" * 64
    assert result["prepared_release_sha256"] == "c" * 64
    assert len(result["canonical_fingerprint"]) == 64
    assert canonical_calls == [promotion]
    assert canonical_calls_prepared == [(promotion, prepared)]

    write_json(
        prepared / "ade" / "preparation_manifest.json",
        {"status": "prepared_test", "test_gold_read": True},
    )
    with pytest.raises(CONTRACT.ContractError, match="manifest identity changed"):
        CONTRACT.verify_release(release_status, promotion, prepared)


def test_complete_release_status_cannot_bypass_canonical_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release_status = tmp_path / "release_status.json"
    promotion = tmp_path / "promotion.json"
    write_json(
        release_status,
        {
            "status": "complete",
            "stage": "complete",
            "promotion_status": "promoted",
            "gate_review_status": "passed",
            "canonical_prepared_release": {},
            "updated_at": "2026-09-04T00:00:00+00:00",
            "datasets": {
                dataset: {"preparation_status": "prepared_test"}
                for dataset in CONTRACT.DATASETS
            },
        },
    )
    write_json(promotion, {"status": "promoted"})

    def reject_forged_marker(_path: Path) -> dict:
        raise CONTRACT.ContractError("canonical fingerprint mismatch")

    monkeypatch.setattr(
        CONTRACT, "canonical_validate_promotion", reject_forged_marker
    )
    with pytest.raises(CONTRACT.ContractError, match="canonical fingerprint"):
        CONTRACT.verify_release(release_status, promotion, tmp_path / "prepared")
