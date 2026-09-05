import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_public_validation_results.py"


def load_script():
    spec = importlib.util.spec_from_file_location("public_validation_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def roots(module, root):
    return module.Roots(
        data=root / "public",
        stage1_run=root / "stage1",
        stage1_analysis=root / "stage1" / "validation_analysis",
        pge=root / "pge_validation_seed42",
        qwen=root / "horizontal" / "qwen_validation",
        spert=root / "horizontal" / "spert_validation",
        glirel=root / "horizontal" / "glirel_validation",
        glirel_t0=root / "horizontal" / "glirel_t0_validation",
        glirel_calibrated=root / "horizontal" / "glirel_calibrated_validation",
        glirel_calibration=root / "horizontal" / "glirel_train_calibration",
        gliner_entity=root / "horizontal" / "gliner_entity_validation",
        post_pge=root / "post_pge_validation",
        pge_config=root / "public_pge_seed42.yaml",
    )


def headline():
    return {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "gold": 1,
        "predicted": 1,
        "correct": 1,
    }


def strict_metric(split="validation"):
    return {
        "metric": "strict-source-character-span",
        "selection_split": split,
        "jobs_gold": 1,
        "jobs_predicted": 1,
        "jobs_evaluated": 1,
        "jobs_missing_predictions": 0,
        "generation_success_rate": 1.0,
        "entity_strict": headline(),
        "relation_strict": headline(),
        "relation_with_claim_status": headline(),
        "per_job": {"conll04_validation_demo_C1": {}},
    }


def test_closed_registry_has_every_declared_system_dataset_row(tmp_path):
    module = load_script()
    systems = module.build_systems(roots(module, tmp_path))

    assert len(systems) == 54
    assert len({system.system_id for system in systems}) == 18
    assert {system.role for system in systems} >= {
        "formal_internal_baseline",
        "formal_internal_system",
        "formal_zero_shot_baseline",
        "formal_trained_baseline",
        "uncalibrated_diagnostic",
        "threshold_sensitivity",
        "formal_train_calibrated_baseline",
        "formal_entity_only_baseline",
    }


def test_test_named_input_is_rejected_before_invalid_contents_are_parsed(tmp_path):
    module = load_script()
    forbidden = tmp_path / "test_gold.jsonl"
    forbidden.write_text("this is deliberately not JSON", encoding="utf-8")
    tracker = module.InputTracker()

    with pytest.raises(module.AuditError, match="test-named"):
        tracker.jsonl(forbidden, "forbidden")
    assert tracker.manifest() == []


def test_duplicate_and_non_validation_prediction_ids_are_rejected(tmp_path):
    module = load_script()
    path = tmp_path / "demo_validation.jsonl"
    row = {"job_id": "conll04_validation_demo_C1", "annotation": {}}

    with pytest.raises(module.AuditError, match="duplicate job_id"):
        module.index_rows([row, row], path, require_annotation=True)
    with pytest.raises(module.AuditError, match="non-validation job ID"):
        module.index_rows(
            [{"job_id": "conll04_holdout_demo_C1", "annotation": {}}],
            path,
            require_annotation=True,
        )


def test_job_registry_requires_validation_identity_and_no_gold_payload(tmp_path):
    module = load_script()
    path = tmp_path / "validation_baseline_jobs.jsonl"
    valid = {
        "job_id": "conll04_validation_demo_C1",
        "document_id": "conll04_validation_demo",
        "category": "conll04",
        "source_path": "public:conll04:validation:demo",
    }
    assert module.validate_validation_jobs([valid], "conll04", path)

    leaked = {**valid, "gold": {}}
    with pytest.raises(module.AuditError, match="forbidden"):
        module.validate_validation_jobs([leaked], "conll04", path)
    wrong_split = {**valid, "source_path": "public:conll04:holdout:demo"}
    with pytest.raises(module.AuditError, match="validation split"):
        module.validate_validation_jobs([wrong_split], "conll04", path)


def test_full_public_validation_counts_are_frozen(tmp_path):
    module = load_script()
    path = tmp_path / "validation_baseline_jobs.jsonl"
    truncated = {
        "conll04_validation_demo_C1": {
            "job_id": "conll04_validation_demo_C1",
            "document_id": "conll04_validation_demo",
        }
    }

    assert module.EXPECTED_VALIDATION_JOBS == {
        "conll04": 231,
        "scierc": 275,
        "ade": 384,
    }
    with pytest.raises(module.AuditError, match="full validation job count mismatch"):
        module.require_full_validation_job_count(truncated, "conll04", path)


def test_metric_audit_requires_validation_denominator_and_finite_values():
    module = load_script()
    expected = {"conll04_validation_demo_C1"}
    protocol, normalized = module.audit_metric(
        strict_metric(), "strict_source_span", expected, 1, 0
    )

    assert protocol == "strict-source-character-span"
    assert normalized["entity"]["f1"] == 1.0

    with pytest.raises(module.AuditError, match="selection_split"):
        module.audit_metric(
            strict_metric(split="holdout"), "strict_source_span", expected, 1, 0
        )
    invalid = strict_metric()
    invalid["entity_strict"]["f1"] = math.nan
    with pytest.raises(module.AuditError, match="non-finite"):
        module.audit_metric(invalid, "strict_source_span", expected, 1, 0)
    invalid = strict_metric()
    invalid["relation_strict"]["gold"] = -1
    with pytest.raises(module.AuditError, match="non-negative integer"):
        module.audit_metric(invalid, "strict_source_span", expected, 1, 0)


def test_comparison_source_accepts_absolute_path_but_rejects_wrong_hash(tmp_path):
    module = load_script()
    source = tmp_path / "metrics" / "soe_validation_span.json"
    record = {"path": str(source.resolve()), "sha256": "a" * 64}

    module.validate_source_artifact_record(record, source, "a" * 64)
    with pytest.raises(module.AuditError, match="SHA-256"):
        module.validate_source_artifact_record(record, source, "b" * 64)


def test_materialized_pge_rate_is_not_conflated_with_generator_failures():
    module = load_script()
    metric = {"generation_success_rate": 1.0}

    # One upstream failure is transparently retained in lineage, while the
    # fully materialized PGE input still has complete metric-row coverage.
    module.audit_generation_success_rate("public_pge_seed42", metric, 231, 1)
    with pytest.raises(module.AuditError, match="generation success rate"):
        module.audit_generation_success_rate("public_stage1", metric, 231, 1)


def test_summary_status_and_manifest_explicitly_seal_test_namespace(tmp_path):
    module = load_script()
    status = {
        "results": [
            {
                "system_id": "demo",
                "label": "Demo",
                "family": "demo",
                "role": "formal",
                "dataset": "conll04",
                "audit_status": "failed",
                "seed": 42,
                "parameters": {"threshold": 0.5},
                "relation_baseline_valid": True,
                "error": "missing registered validation input",
            }
        ]
    }
    tsv = module.summary_tsv(status)

    assert "system_id\tlabel" in tsv
    assert "missing registered validation input" in tsv
    assert module.TEST_ACCESS_SEAL == "sealed_not_read"
    assert "rglob(" not in SCRIPT.read_text(encoding="utf-8")
    assert "os.walk(" not in SCRIPT.read_text(encoding="utf-8")
