import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "preflight_public_external_formal.py"


def load_script():
    spec = importlib.util.spec_from_file_location(
        "preflight_public_external_formal", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def make_dataset(workspace: Path, dataset: str = "tiny"):
    root = workspace / "data/processed/public_benchmarks_full" / dataset
    train_ids = ["train-1", "train-2"]
    validation_ids = ["val-1"]
    for split, ids in (("train", train_ids), ("validation", validation_ids)):
        write_jsonl(root / f"{split}_gold.jsonl", [{"annotation": {}} for _ in ids])
        write_jsonl(
            root / f"{split}_index.jsonl",
            [{"job_id": job_id, "record_index": index} for index, job_id in enumerate(ids)],
        )
        write_jsonl(
            root / f"{split}_baseline_jobs.jsonl",
            [{"job_id": job_id, "segments": []} for job_id in ids],
        )
    # A malformed sentinel demonstrates that the validation preflight never
    # opens or parses test gold.
    (root / "test_gold.jsonl").write_text("not-json\n", encoding="utf-8")


def test_dataset_preflight_is_full_train_validation_and_never_reads_test(tmp_path):
    preflight = load_script()
    make_dataset(tmp_path)

    result = preflight.inspect_public_dataset(
        tmp_path, "tiny", {"train": 2, "validation": 1}
    )

    assert result["status"] == "ready"
    assert result["test_namespace"] == "sealed_not_inspected"
    assert result["test_gold_read"] is False
    assert set(result["files"]) == {
        "train_gold",
        "train_index",
        "train_jobs",
        "validation_gold",
        "validation_index",
        "validation_jobs",
    }


def test_static_3090_assessment_rejects_full_precision_45gb_but_not_nf4():
    preflight = load_script()
    serialized = 45_592_264_704

    full = preflight.assess_3090_memory(
        "upstream_full_precision", serialized, 24.0
    )
    nf4 = preflight.assess_3090_memory("upstream_bnb_nf4", 26_508_830_720, 24.0)

    assert full["assessment"] == "incompatible_24gb"
    assert nf4["assessment"] == "static_fit_requires_gpu_canary"
    assert full["cuda_queried"] is False
    assert nf4["gpu_process_started"] is False


def test_complete_marker_is_not_trusted_without_frozen_artifacts(tmp_path):
    preflight = load_script()
    output = tmp_path / "oneke" / "tiny" / "seed42"
    output.mkdir(parents=True)
    (output / "status.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )

    invalid = preflight.inspect_progress(
        output,
        "oneke",
        "tiny",
        {"val-1"},
        "evaluator-hash",
        {"validation_jobs": "jobs-hash"},
    )
    assert invalid["state"] == "invalid_completion_marker"
    assert any("predictions_missing" in item for item in invalid["blocking_reasons"])

    predictions = output / "validation_predictions.jsonl"
    metrics = output / "validation_character_span_metrics.json"
    write_jsonl(predictions, [{"job_id": "val-1", "annotation": {}}])
    metrics.write_text(
        json.dumps({"metric": "strict-source-character-span"}), encoding="utf-8"
    )
    manifest = {
        "schema_version": preflight.RUN_MANIFEST_VERSION,
        "status": "complete",
        "baseline": "oneke",
        "dataset": "tiny",
        "split": preflight.SPLIT,
        "seed": preflight.SEED,
        "evaluator": preflight.EVALUATOR,
        "evaluator_sha256": "evaluator-hash",
        "test_gold_read": False,
        "input_sha256": {"validation_jobs": "jobs-hash"},
        "prediction_sha256": preflight.sha256(predictions),
        "metric_sha256": preflight.sha256(metrics),
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    complete = preflight.inspect_progress(
        output,
        "oneke",
        "tiny",
        {"val-1"},
        "evaluator-hash",
        {"validation_jobs": "jobs-hash"},
    )
    assert complete["state"] == "complete"
    assert complete["verified_prediction_rows"] == 1


def test_missing_formal_adapter_and_runner_are_explicit_queue_blockers(tmp_path):
    preflight = load_script()
    make_dataset(tmp_path)
    evaluator = tmp_path / preflight.EVALUATOR
    evaluator.parent.mkdir(parents=True, exist_ok=True)
    evaluator.write_text("# canonical evaluator\n", encoding="utf-8")

    code = tmp_path / "vendor/oneke/run.py"
    requirements = tmp_path / "vendor/oneke/requirements.txt"
    code.parent.mkdir(parents=True)
    code.write_text("pass\n", encoding="utf-8")
    requirements.write_text("demo==1.0\n", encoding="utf-8")
    snapshot = tmp_path / "cache/oneke"
    snapshot.mkdir(parents=True)
    (snapshot / "weights.bin").write_bytes(b"weights")
    integrity = {
        "schema_version": "public-baseline-integrity-v1",
        "status": "ready",
        "components": {
            "oneke": {
                "status": "ready",
                "model": {
                    "status": "ready",
                    "snapshot": str(snapshot),
                    "revision": "abc",
                    "weight_files": ["weights.bin"],
                    "weight_bytes": 1024,
                },
            }
        },
    }
    integrity_path = tmp_path / "integrity.json"
    integrity_path.write_text(json.dumps(integrity), encoding="utf-8")
    baseline_specs = {
        "oneke": {
            "label": "OneKE",
            "mode": "published_checkpoint_inference",
            "code_files": ("vendor/oneke/run.py", "vendor/oneke/requirements.txt"),
            "requirements": "vendor/oneke/requirements.txt",
            "manual_modules": (),
            "adapter": "scripts/public_external_adapters/oneke.py",
            "runner": "scripts/run_public_oneke_formal.sh",
            "inventory_component": "oneke",
            "checkpoint_kind": "hf_model",
            "memory_policy": "upstream_bnb_nf4",
            "memory_note": "test",
        }
    }

    def dependency_query(_python, distributions, modules):
        assert list(distributions) == ["demo"]
        assert list(modules) == []
        return {
            "python": "test-python",
            "versions": {"demo": "1.0"},
            "modules": {},
        }, None

    result = preflight.build_preflight(
        tmp_path,
        tmp_path / "outputs/public_external_formal",
        integrity_path,
        dataset_specs={"tiny": {"train": 2, "validation": 1}},
        baseline_specs=baseline_specs,
        dependency_query=dependency_query,
    )

    assert result["scope"]["datasets"] == ["tiny"]
    assert result["scope"]["railway_data_included"] is False
    assert result["safety"]["gpu_process_started"] is False
    assert result["queue_counts"] == {"blocked_with_reason": 1}
    reasons = result["queue"][0]["blocking_reasons"]
    assert "schema_adapter_missing:scripts/public_external_adapters/oneke.py" in reasons
    assert "formal_runner_missing:scripts/run_public_oneke_formal.sh" in reasons
    assert result["queue"][0]["runner"] is None


def test_dependency_versions_are_checked_without_importing_packages(tmp_path):
    preflight = load_script()
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("demo==1.0\n", encoding="utf-8")
    specification = {
        "requirements": "requirements.txt",
        "manual_modules": ("optional_runtime",),
    }

    def query(_python, _distributions, _modules):
        return {
            "python": "test-python",
            "versions": {"demo": "2.0"},
            "modules": {"optional_runtime": False},
        }, None

    result = preflight.inspect_dependencies(
        tmp_path, specification, Path("/fake/python"), query=query
    )
    assert result["status"] == "blocked_with_reason"
    assert any("dependency_version_mismatch:demo" in item for item in result["blocking_reasons"])
    assert "dependency_module_missing:optional_runtime" in result["blocking_reasons"]
    assert result["metadata_only_probe"] is True


def test_preflight_artifacts_expose_one_unified_queue_and_per_model_markers(tmp_path):
    preflight = load_script()
    payload = {
        "schema_version": preflight.SCHEMA_VERSION,
        "status": "blocked_with_reason",
        "generated_at": "2026-09-04T00:00:00+00:00",
        "scope": {"datasets": ["conll04"], "test_gold_read": False},
        "safety": {"gpu_process_started": False},
        "evaluator": {"path": preflight.EVALUATOR},
        "baselines": {
            "oneke": {"status": "blocked_with_reason", "blocking_reasons": ["reason"]}
        },
        "queue": [
            {
                "job_id": "oneke-conll04-validation-seed42",
                "baseline": "oneke",
                "status": "blocked_with_reason",
            }
        ],
    }
    output_root = tmp_path / "formal"
    summary = output_root / "preflight_status.json"

    preflight.write_preflight_artifacts(summary, output_root, payload)

    assert json.loads(summary.read_text(encoding="utf-8"))["status"] == "blocked_with_reason"
    queue = [json.loads(line) for line in (output_root / "queue.jsonl").read_text().splitlines()]
    assert queue == payload["queue"]
    marker = json.loads(
        (output_root / "oneke/preflight_status.json").read_text(encoding="utf-8")
    )
    assert marker["baseline"] == "oneke"
    assert marker["safety"]["gpu_process_started"] is False
