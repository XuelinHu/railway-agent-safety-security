import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts" / "run_public_post_pge_bootstrap.py"
LAUNCHER = ROOT / "scripts" / "launch_public_post_pge_bootstrap.sh"
DATASETS = ("conll04", "scierc", "ade")


def load_worker_module():
    spec = importlib.util.spec_from_file_location(WORKER.stem, WORKER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


WORKER_MODULE = load_worker_module()


def span_metric(dataset: str, correct: int) -> dict:
    fields = {
        "entity": {"gold": 2, "predicted": 2, "correct": correct},
        "relation": {"gold": 1, "predicted": 1, "correct": min(correct, 1)},
        "relation_with_claim_status": {
            "gold": 1,
            "predicted": 1,
            "correct": min(correct, 1),
        },
    }
    return {
        "metric": "strict-global-character-span-one-to-one",
        "selection_split": "validation",
        "formal_test_read": False,
        "jobs": 2,
        "documents": 2,
        "per_document": {
            f"{dataset}_validation_doc-1": {
                "document_id": f"{dataset}_validation_doc-1",
                **fields,
            },
            f"{dataset}_validation_doc-2": {
                "document_id": f"{dataset}_validation_doc-2",
                **fields,
            },
        },
    }


def write_manifest(pge_root: Path) -> None:
    files = {}
    for path in sorted(pge_root.glob("*/metrics/*_span.json")):
        files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (pge_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "selection_split": "validation",
                "formal_test_read": False,
                "seed": 42,
                "files": files,
            }
        ),
        encoding="utf-8",
    )


def write_inputs(pge_root: Path) -> None:
    for dataset in DATASETS:
        metrics = pge_root / dataset / "metrics"
        metrics.mkdir(parents=True)
        (metrics / "soe_span.json").write_text(
            json.dumps(span_metric(dataset, 1)), encoding="utf-8"
        )
        (metrics / "pge_span.json").write_text(
            json.dumps(span_metric(dataset, 2)), encoding="utf-8"
        )
    write_manifest(pge_root)


def complete_marker() -> dict:
    return {
        "status": "complete",
        "selection_split": "validation",
        "formal_test_read": False,
        "seed": 42,
    }


def complete_comparison_payload(tmp_path: Path):
    left_path = tmp_path / "soe_span.json"
    right_path = tmp_path / "pge_span.json"
    manifest_path = tmp_path / "run_manifest.json"
    left_path.write_text("left", encoding="utf-8")
    right_path.write_text("right", encoding="utf-8")
    manifest_path.write_text("manifest", encoding="utf-8")
    left = WORKER_MODULE.source_record(left_path)
    right = WORKER_MODULE.source_record(right_path)
    manifest = WORKER_MODULE.source_record(manifest_path)

    observed_left = {
        "pooled_f1": 0.5,
        "macro_f1": 0.4,
        "gold": 10,
        "predicted": 8,
        "correct": 5,
        "pooled_f1_ci95": {"lower": 0.3, "upper": 0.7},
        "macro_f1_ci95": {"lower": 0.2, "upper": 0.6},
    }
    observed_right = {
        "pooled_f1": 0.6,
        "macro_f1": 0.45,
        "gold": 10,
        "predicted": 9,
        "correct": 6,
        "pooled_f1_ci95": {"lower": 0.4, "upper": 0.8},
        "macro_f1_ci95": {"lower": 0.25, "upper": 0.65},
    }
    difference = {
        "pooled_f1": 0.1,
        "macro_f1": 0.05,
        "pooled_f1_ci95": {"lower": -0.1, "upper": 0.3},
        "macro_f1_ci95": {"lower": -0.15, "upper": 0.25},
        "paired_permutation_p_pooled_f1": 0.2,
        "paired_permutation_p_macro_f1": 0.3,
    }
    field = {
        "left": observed_left,
        "right": observed_right,
        "right_minus_left": difference,
    }
    payload = {
        "schema_version": WORKER_MODULE.SCHEMA_VERSION,
        "dataset": "conll04",
        "selection_split": "validation",
        "formal_test_read": False,
        "iterations": WORKER_MODULE.ITERATIONS,
        "seed": WORKER_MODULE.BOOTSTRAP_SEED,
        "metric": WORKER_MODULE.BOOTSTRAP_METRIC,
        "unit": "document",
        "documents": 2,
        "comparison": "SOE_vs_PGE",
        "interpretation": "right_minus_left",
        "system_roles": {"left": "SOE", "right": "PGE"},
        "systems": {"left": "soe_span.json", "right": "pge_span.json"},
        "source_artifacts": {"soe": left, "pge": right},
        "upstream_manifest": manifest,
        "statistical_implementation": WORKER_MODULE.implementation_records(),
        "cpu_only": True,
        "generated_at": "2026-09-04T00:00:00+00:00",
        "fields": {
            name: copy.deepcopy(field) for name in WORKER_MODULE.FIELDS
        },
    }
    return payload, left, right, manifest


def wait_for_status(path: Path, expected: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
            else:
                if value.get("status") == expected:
                    return value
        time.sleep(0.02)
    raise AssertionError(f"did not observe status={expected} in {path}")


def test_worker_waits_then_runs_all_frozen_validation_comparisons(tmp_path):
    pge_root = tmp_path / "pge"
    output_root = tmp_path / "post"
    write_inputs(pge_root)

    # A malformed, unread test-shaped canary must be irrelevant: the worker has
    # no test path argument and only opens the two fixed validation metrics.
    test_canary = pge_root / "test" / "must_not_be_read.json"
    test_canary.parent.mkdir()
    test_canary.write_text("this is deliberately not JSON", encoding="utf-8")

    args = SimpleNamespace(
        pge_root=pge_root,
        output_root=output_root,
        poll_seconds=0.05,
        expected_validation_jobs={dataset: 2 for dataset in DATASETS},
    )
    worker = WORKER_MODULE.Worker(args)
    errors = []

    def run_worker():
        try:
            worker.run()
        except Exception as error:  # pragma: no cover - asserted below
            errors.append(error)

    thread = threading.Thread(target=run_worker)
    thread.start()
    waiting = wait_for_status(output_root / "status.json", "waiting_for_pge")
    assert waiting["formal_test_read"] is False
    assert waiting["test_artifacts_opened"] == []

    (pge_root / "status.json").write_text(
        json.dumps(complete_marker()), encoding="utf-8"
    )
    thread.join(timeout=30)
    assert not thread.is_alive()
    assert not errors

    status = json.loads((output_root / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete"
    assert status["selection_split"] == "validation"
    assert status["formal_test_read"] is False
    assert status["test_artifacts_opened"] == []
    assert status["iterations"] == 20_000
    assert status["completed_datasets"] == 3

    hashes = {}
    for dataset in DATASETS:
        path = output_root / "comparisons" / f"{dataset}_soe_vs_pge.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        assert result["schema_version"] == "public-post-pge-soe-vs-pge-v1"
        assert result["dataset"] == dataset
        assert result["comparison"] == "SOE_vs_PGE"
        assert result["system_roles"] == {"left": "SOE", "right": "PGE"}
        assert result["systems"] == {"left": "soe_span.json", "right": "pge_span.json"}
        assert result["metric"] == "strict-global-character-span-one-to-one"
        assert result["interpretation"] == "right_minus_left"
        assert result["iterations"] == 20_000
        assert result["selection_split"] == "validation"
        assert result["formal_test_read"] is False
        assert set(result["source_artifacts"]) == {"soe", "pge"}
        assert set(result["fields"]) == {
            "entity",
            "relation",
            "relation_with_claim_status",
        }
        assert set(result["statistical_implementation"]) == {
            "bootstrap_span_compare",
            "bootstrap_compare",
        }
        for record in result["statistical_implementation"].values():
            implementation = ROOT / record["path"]
            assert hashlib.sha256(implementation.read_bytes()).hexdigest() == record["sha256"]
        hashes[dataset] = path.read_bytes()

    # A restart is resumable: frozen results with matching source hashes are
    # retained instead of spending another 20,000 iterations per dataset.
    WORKER_MODULE.Worker(args).run()
    for dataset in DATASETS:
        path = output_root / "comparisons" / f"{dataset}_soe_vs_pge.json"
        assert path.read_bytes() == hashes[dataset]


def test_worker_rejects_a_completed_marker_that_read_formal_test(tmp_path):
    pge_root = tmp_path / "pge"
    output_root = tmp_path / "post"
    pge_root.mkdir()
    marker = complete_marker()
    marker["formal_test_read"] = True
    (pge_root / "status.json").write_text(json.dumps(marker), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(WORKER),
            "--pge-root",
            str(pge_root),
            "--output-root",
            str(output_root),
            "--poll-seconds",
            "0.01",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    status = json.loads((output_root / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert "formal_test_read=false" in status["error"]
    assert not (output_root / "comparisons").exists()


def test_worker_rejects_a_non_validation_strict_metric(tmp_path):
    pge_root = tmp_path / "pge"
    output_root = tmp_path / "post"
    write_inputs(pge_root)
    invalid = span_metric("conll04", 1)
    invalid["selection_split"] = "test"
    (pge_root / "conll04" / "metrics" / "soe_span.json").write_text(
        json.dumps(invalid), encoding="utf-8"
    )
    write_manifest(pge_root)
    (pge_root / "status.json").write_text(
        json.dumps(complete_marker()), encoding="utf-8"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(WORKER),
            "--pge-root",
            str(pge_root),
            "--output-root",
            str(output_root),
            "--poll-seconds",
            "0.01",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    status = json.loads((output_root / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert "refusing non-validation artifact" in status["error"]
    assert not (output_root / "comparisons").exists()


def test_currentness_rejects_shell_results_and_wrong_statistical_direction(tmp_path):
    payload, left, right, manifest = complete_comparison_payload(tmp_path)
    target = tmp_path / "comparison.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    assert WORKER_MODULE.comparison_is_current(
        target, "conll04", left, right, manifest, 2
    )

    corruptions = {}
    corruptions["missing-fields"] = copy.deepcopy(payload)
    corruptions["missing-fields"].pop("fields")
    corruptions["missing-one-required-field"] = copy.deepcopy(payload)
    corruptions["missing-one-required-field"]["fields"].pop("relation")
    corruptions["partial-field-body"] = copy.deepcopy(payload)
    corruptions["partial-field-body"]["fields"]["entity"].pop("right_minus_left")
    corruptions["wrong-metric"] = copy.deepcopy(payload)
    corruptions["wrong-metric"]["metric"] = "not-the-frozen-metric"
    corruptions["wrong-interpretation"] = copy.deepcopy(payload)
    corruptions["wrong-interpretation"]["interpretation"] = "left_minus_right"
    corruptions["swapped-roles"] = copy.deepcopy(payload)
    corruptions["swapped-roles"]["system_roles"] = {"left": "PGE", "right": "SOE"}
    corruptions["wrong-source-filename"] = copy.deepcopy(payload)
    corruptions["wrong-source-filename"]["systems"]["left"] = "pge_span.json"
    corruptions["wrong-delta-sign"] = copy.deepcopy(payload)
    corruptions["wrong-delta-sign"]["fields"]["relation"]["right_minus_left"][
        "pooled_f1"
    ] = -0.1
    corruptions["missing-p-value"] = copy.deepcopy(payload)
    corruptions["missing-p-value"]["fields"]["relation_with_claim_status"][
        "right_minus_left"
    ].pop("paired_permutation_p_macro_f1")
    corruptions["stale-statistics-code"] = copy.deepcopy(payload)
    corruptions["stale-statistics-code"]["statistical_implementation"][
        "bootstrap_compare"
    ]["sha256"] = "0" * 64

    for label, corrupted in corruptions.items():
        target.write_text(json.dumps(corrupted), encoding="utf-8")
        assert not WORKER_MODULE.comparison_is_current(
            target, "conll04", left, right, manifest, 2
        ), label


def test_metric_pair_requires_manifest_hashes_matching_units_and_gold(tmp_path):
    pge_root = tmp_path / "pge"
    write_inputs(pge_root)
    manifest_path = pge_root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    left = pge_root / "conll04" / "metrics" / "soe_span.json"
    right = pge_root / "conll04" / "metrics" / "pge_span.json"
    with pytest.raises(WORKER_MODULE.ProtocolError, match="evaluation units are incomplete"):
        WORKER_MODULE.validate_metric_pair("conll04", left, right, manifest)
    WORKER_MODULE.validate_metric_pair("conll04", left, right, manifest, 2)

    missing_hash = copy.deepcopy(manifest)
    missing_hash["files"].pop(str(left))
    with pytest.raises(WORKER_MODULE.ProtocolError, match="exactly one hash"):
        WORKER_MODULE.validate_metric_pair("conll04", left, right, missing_hash, 2)

    wrong_units = span_metric("conll04", 2)
    wrong_units["per_document"]["conll04_validation_extra"] = {
        "document_id": "conll04_validation_extra",
        "entity": {"gold": 0, "predicted": 0, "correct": 0},
        "relation": {"gold": 0, "predicted": 0, "correct": 0},
        "relation_with_claim_status": {"gold": 0, "predicted": 0, "correct": 0},
    }
    right.write_text(json.dumps(wrong_units), encoding="utf-8")
    write_manifest(pge_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with pytest.raises(WORKER_MODULE.ProtocolError, match="evaluation unit IDs differ"):
        WORKER_MODULE.validate_metric_pair("conll04", left, right, manifest, 2)

    wrong_gold = span_metric("conll04", 2)
    first = next(iter(wrong_gold["per_document"].values()))
    first["relation"]["gold"] = 2
    right.write_text(json.dumps(wrong_gold), encoding="utf-8")
    write_manifest(pge_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with pytest.raises(WORKER_MODULE.ProtocolError, match="gold counts differ"):
        WORKER_MODULE.validate_metric_pair("conll04", left, right, manifest, 2)


def test_worker_rejects_manifest_without_frozen_metric_hash(tmp_path):
    pge_root = tmp_path / "pge"
    output_root = tmp_path / "post"
    write_inputs(pge_root)
    manifest_path = pge_root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    omitted = pge_root / "conll04" / "metrics" / "soe_span.json"
    manifest["files"].pop(str(omitted))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (pge_root / "status.json").write_text(json.dumps(complete_marker()), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(WORKER),
            "--pge-root",
            str(pge_root),
            "--output-root",
            str(output_root),
            "--poll-seconds",
            "0.01",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    status = json.loads((output_root / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert "must contain exactly one hash" in status["error"]


def test_nonblocking_launcher_has_cpu_only_systemd_contract():
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "systemd-run --user" in source
    assert "--collect --no-block" in source
    assert "--property=Nice=19" in source
    assert "--property=CPUWeight=10" in source
    assert "--property=CPUQuota=100%" in source
    assert "--setenv=CUDA_VISIBLE_DEVICES=" in source
    assert "--setenv=OMP_NUM_THREADS=1" in source
    assert "scripts/run_public_post_pge_bootstrap.py" in source
    assert '--pge-root "$pge_root"' in source
    assert '--output-root "$run_root"' in source
    assert "jq -r '.status" not in source


def test_bootstrap_implementation_cannot_be_replaced_from_cli():
    result = subprocess.run(
        [sys.executable, str(WORKER), "--bootstrap-script", "/tmp/untrusted.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "unrecognized arguments: --bootstrap-script" in result.stderr


def test_worker_protocol_is_fixed_to_three_validation_metric_pairs():
    source = WORKER.read_text(encoding="utf-8")
    assert 'DATASETS = ("conll04", "scierc", "ade")' in source
    assert "ITERATIONS = 20_000" in source
    assert '"metrics" / "soe_span.json"' in source
    assert '"metrics" / "pge_span.json"' in source
    assert 'value.get("selection_split") != "validation"' in source
    assert 'value.get("formal_test_read") is not False' in source
