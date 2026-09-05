import copy
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prepare_public_test_inputs as preparation  # noqa: E402
import promote_public_validation_to_test as promoter  # noqa: E402
import verify_public_formal_internal_state as internal_verifier  # noqa: E402


SUMMARY_JSON = ROOT / "outputs/public_validation_audit/summary.json"
SUMMARY_TSV = ROOT / "outputs/public_validation_audit/summary.tsv"


@pytest.fixture(scope="module")
def canonical_promotion(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("promotion") / "promotion.json"
    promoter.promote(SUMMARY_JSON, SUMMARY_TSV, path)
    return path


def first_test_job(dataset: str = "conll04") -> dict:
    path = ROOT / "data/processed/public_benchmarks_full" / dataset / "test_baseline_jobs.jsonl"
    return json.loads(path.read_text(encoding="utf-8").splitlines()[0])


def test_real_promotion_is_closed_hash_bound_and_evidence_safe(
    canonical_promotion: Path,
) -> None:
    stored = preparation.validate_promotion(canonical_promotion)
    assert stored["schema_version"] == "public-formal-test-promotion-v2"
    assert stored["status"] == "promoted"
    assert stored["promoted_systems"] == ["soe", "pge"]
    assert stored["validation_audit_counts"]["passed_system_dataset_rows"] == 54
    assert stored["validation_audit_counts"]["passed_markers"] == 14
    assert stored["validation_audit_counts"]["passed_comparisons"] == 3
    assert len(stored["validated_marker_ids"]) == 14
    assert stored["gate"]["comparison_iterations"] == 20_000
    assert stored["gate"]["comparison_seed"] == 20_260_830
    assert stored["gate"]["confidence_intervals"] == "exploratory_non_gating"
    assert stored["gate"]["preregistered_non_inferiority_claim"] is False
    assert set(stored["inputs"]) == {
        "summary_json", "summary_tsv", "input_manifest_json", "promotion_policy_document"
    }
    for decision in stored["datasets"].values():
        safety = decision["pge_evidence_safety"]
        assert safety["status"] == "passed"
        assert safety["micro_counts"]["entity_count"] > 0
        assert safety["micro_counts"]["relation_count"] > 0
        assert safety["micro_counts"]["unsupported_claim_count"] == 0
        assert safety["micro_counts"]["invalid_relation_count"] == 0
        assert safety["micro_rates"]["relation_evidence_coverage"] == 1.0
        assert safety["micro_rates"]["relation_evidence_correctness"] == 1.0


@pytest.mark.parametrize(
    ("soe", "pge", "passes"),
    [(0.6, 0.59, True), (0.6, 0.589999, False)],
)
def test_point_estimate_margin_boundary(soe: float, pge: float, passes: bool) -> None:
    if passes:
        assert promoter.validate_point_estimate_gate(soe, pge, "conll04") == pytest.approx(-0.01)
    else:
        with pytest.raises(ValueError, match="exceeding 0.01"):
            promoter.validate_point_estimate_gate(soe, pge, "conll04")


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, "nan"])
def test_point_estimate_gate_rejects_nonfinite_values(bad: object) -> None:
    with pytest.raises(ValueError, match="not finite"):
        promoter.validate_point_estimate_gate(0.6, bad, "conll04")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda audit: audit["counts"].update(passed_system_dataset_rows=53), "count mismatch"),
        (lambda audit: audit["counts"].update(failed_markers=1), "count mismatch"),
        (lambda audit: audit["counts"].update(failed_comparisons=1), "count mismatch"),
        (lambda audit: audit.update(failures=None), "failures"),
        (lambda audit: audit.update(test_artifacts_opened=None), "opened test artifacts"),
        (lambda audit: audit["registry"].update(system_dataset_rows=53), "dimensions"),
    ],
)
def test_closed_audit_header_rejects_fail_open_mutations(mutation, message: str) -> None:
    audit = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    mutation(audit)
    with pytest.raises(ValueError, match=message):
        promoter._validate_closed_audit_header(audit)


def audit_components():
    tracker = promoter.InputTracker()
    audit = tracker.json(SUMMARY_JSON)
    _, files = promoter._load_input_manifest(SUMMARY_JSON.parent / "input_manifest.json", tracker)
    results = promoter._validate_results(audit, files, tracker)
    return tracker, audit, files, results


@pytest.mark.parametrize(("field", "value"), [("seed", 1), ("iterations", 19_999)])
def test_comparison_revalidation_rejects_seed_or_iteration_drift(field: str, value: int) -> None:
    tracker, audit, files, results = audit_components()
    mutated = copy.deepcopy(audit)
    mutated["comparisons"][0][field] = value
    with pytest.raises(ValueError, match="comparison contract failed"):
        promoter._validate_comparisons(mutated, results, files, tracker)


def test_marker_revalidation_rejects_summary_only_forgery() -> None:
    tracker, audit, files, _ = audit_components()
    mutated = copy.deepcopy(audit)
    mutated["markers"][0]["status"] = "complete_but_forged"
    with pytest.raises(ValueError, match="revalidation disagrees"):
        promoter._validate_markers(mutated, files, tracker)


def test_consumer_recomputes_and_rejects_forged_promotion(
    canonical_promotion: Path, tmp_path: Path
) -> None:
    forged = json.loads(canonical_promotion.read_text(encoding="utf-8"))
    forged["datasets"]["conll04"]["pge_minus_soe_relation_f1"] = 0.9
    path = tmp_path / "promotion.json"
    path.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(ValueError, match="fresh closed-world recomputation"):
        preparation.validate_promotion(path)


def test_pge_evidence_mutation_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    tracker, _, _, results = audit_components()
    original_json = tracker.json

    def mutated_json(path: Path):
        value = original_json(path)
        if str(path).endswith("conll04/metrics/pge_evidence.json"):
            value = copy.deepcopy(value)
            value["overall"]["unsupported_claim_rate"] = 0.01
        return value

    monkeypatch.setattr(tracker, "json", mutated_json)
    with pytest.raises(ValueError, match="unsupported_claim_rate"):
        promoter._validate_pge_safety(results, tracker)


def test_gold_path_and_symlink_guards_run_before_read() -> None:
    with tempfile.TemporaryDirectory(prefix="release-guard-", dir="/tmp") as directory:
        root = Path(directory)
        gold = root / "formal-test" / "test-gold.jsonl"
        gold.parent.mkdir()
        gold.write_text("{not-json\n", encoding="utf-8")
        disguised = root / "test_baseline_jobs.jsonl"
        disguised.symlink_to(gold)
        tracker = preparation.SnapshotTracker()
        with pytest.raises(ValueError, match="test gold"):
            tracker.bytes(gold)
        with pytest.raises(ValueError, match="symlinked"):
            tracker.bytes(disguised)


def test_test_job_schema_rejects_truncation_mode_id_offset_and_labels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "baseline.jsonl"
    row = first_test_job()
    monkeypatch.setitem(preparation.EXPECTED_TEST_JOBS, "conll04", 1)
    preparation.validate_source_jobs(path, [row], "conll04", "test")
    with pytest.raises(ValueError, match="job count mismatch"):
        preparation.validate_source_jobs(path, [], "conll04", "test")
    for mutation, message in (
        (lambda x: x.update(experiment_mode="eae"), "baseline contract"),
        (lambda x: x.update(category="ade"), "category/language"),
        (lambda x: x.update(job_id="wrong_C1"), "identity"),
        (lambda x: x["segments"][0].update(end=1), "character offsets"),
        (lambda x: x.update(annotation={}), "label-bearing"),
    ):
        changed = copy.deepcopy(row)
        mutation(changed)
        with pytest.raises(ValueError, match=message):
            preparation.validate_source_jobs(path, [changed], "conll04", "test")


def test_physical_graph_rejects_orphan_and_nontrain_relation() -> None:
    concepts = [{"concept_id": "C1", "source_documents": ["D1"]}]
    mentions = [{"mention_id": "M1", "concept_id": "C1", "document_id": "D1", "split": "train"}]
    relations = [{
        "relation_id": "R1", "document_id": "D1", "split": "train",
        "source_concept_id": "C1", "target_concept_id": "C1",
        "provenance": {"source_split": "train"},
    }]
    audit = {"concepts": 1, "mentions": 1, "relations": 1, "train_documents_in_manifest": 1}
    assert preparation.validate_physical_train_graph(
        concepts, mentions, relations, {"D1"}, audit
    )["relations"] == 1
    orphaned = concepts + [{"concept_id": "C2", "source_documents": ["D1"]}]
    with pytest.raises(ValueError, match="orphan concept"):
        preparation.validate_physical_train_graph(orphaned, mentions, relations, {"D1"}, audit)
    bad_relation = copy.deepcopy(relations)
    bad_relation[0]["split"] = "test"
    with pytest.raises(ValueError, match="physically train-only"):
        preparation.validate_physical_train_graph(concepts, mentions, bad_relation, {"D1"}, audit)


def test_real_graph_manifest_rejects_settings_output_hash_and_model_drift(
    canonical_promotion: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = ROOT / "data/processed/public_benchmarks_hrge_v1/conll04/preparation_manifest.json"
    promotion = preparation.validate_promotion(canonical_promotion)
    for mutate, message in (
        (
            lambda x: x["settings"].update(hrge_edge_limit=99),
            "settings drifted",
        ),
        (
            lambda x: x["outputs"]["train_only_concepts"].update(sha256="0" * 64),
            "hash mismatch",
        ),
        (
            lambda x: x["fingerprint_inputs"]["semantic_model"]["files"].pop("pytorch_model.bin"),
            "semantic model file identity is incomplete",
        ),
    ):
        tracker = preparation.SnapshotTracker()
        original_json = tracker.json
        bad = copy.deepcopy(original_json(manifest_path))
        mutate(bad)
        if "semantic_model" in bad.get("fingerprint_inputs", {}):
            bad["fingerprint"] = preparation.stable_digest(bad["fingerprint_inputs"])

        def patched_json(path: Path, *, _bad=bad, _original=original_json):
            return _bad if preparation._resolve(path) == preparation._resolve(manifest_path) else _original(path)

        monkeypatch.setattr(tracker, "json", patched_json)
        with pytest.raises(ValueError, match=message):
            preparation._validate_graph_manifest(
                "conll04", manifest_path,
                ROOT / "data/processed/public_benchmarks_full",
                ROOT / "data/processed/public_benchmarks_hrge_v1",
                promotion, tracker,
            )
        monkeypatch.undo()


def test_real_three_dataset_dry_run_is_read_only_and_fully_frozen(
    canonical_promotion: Path, tmp_path: Path
) -> None:
    before = set(tmp_path.iterdir())
    rows = [
        preparation.prepare_dataset(
            dataset,
            promotion_path=canonical_promotion,
            source_root=ROOT / "data/processed/public_benchmarks_full",
            graph_root=ROOT / "data/processed/public_benchmarks_hrge_v1",
            output_root=tmp_path / "unused",
            dry_run=True,
        )
        for dataset in promoter.DATASETS
    ]
    assert set(tmp_path.iterdir()) == before
    assert [row["job_count"] for row in rows] == [288, 551, 427]
    assert all(row["status"] == "ready_no_writes" for row in rows)
    assert all(row["semantic_model_revision"] == "5617a9f61b028005a4858fdac845db406aefb181" for row in rows)


def test_formal_input_overrides_and_force_are_rejected_before_access(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="semantic-model override"):
        preparation.prepare_dataset("conll04", semantic_model=tmp_path)
    with pytest.raises(ValueError, match="batch size is frozen"):
        preparation.prepare_dataset("conll04", batch_size=1)
    with pytest.raises(ValueError, match="force/overwrite"):
        preparation.prepare_dataset("conll04", force=True)


def test_seed42_adapter_reuse_is_parameter_and_hash_validated() -> None:
    result = internal_verifier.validate_adapter(
        ROOT / "outputs/public_pge_validation_seed42/conll04/eae_adapter",
        42,
        "conll04",
        "eae",
    )
    assert result["seed"] == 42
    assert set(result["files"]) == internal_verifier.ADAPTER_FILES
    assert len(result["files"]["adapter_model.safetensors"]["sha256"]) == 64


def test_internal_resume_prediction_and_materialization_require_full_coverage(
    tmp_path: Path,
) -> None:
    ids = ["conll04_test_1_C1", "conll04_test_2_C1"]
    documents = {
        "conll04_test_1_C1": "conll04_test_1",
        "conll04_test_2_C1": "conll04_test_2",
    }
    prediction = tmp_path / "complete.jsonl"
    prediction.write_text(
        "".join(
            json.dumps({
                "job_id": job_id,
                "annotation": {
                    "document_id": documents[job_id], "entities": [], "relations": []
                },
            }) + "\n"
            for job_id in ids
        ),
        encoding="utf-8",
    )
    assert internal_verifier.validate_prediction(prediction, ids, documents)["bytes"] > 0
    truncated = prediction.read_text(encoding="utf-8").splitlines()[0] + "\n"
    prediction.write_text(truncated, encoding="utf-8")
    with pytest.raises(ValueError, match="full job order"):
        internal_verifier.validate_prediction(prediction, ids, documents)

    jobs = tmp_path / "jobs.jsonl"
    complete = tmp_path / "materialized.jsonl"
    jobs.write_text("{}\n{}\n", encoding="utf-8")
    complete.write_text("{}\n{}\n", encoding="utf-8")
    manifest = tmp_path / "materialization.json"
    manifest.write_text(json.dumps({
        "status": "complete",
        "jobs": 2,
        "successful_prediction_rows": 1,
        "failures_materialized_as_empty": 1,
        "missing_job_ids": [ids[1]],
        "jobs_path": str(jobs),
        "output": str(complete),
        "gold_read": False,
    }), encoding="utf-8")
    internal_verifier.validate_materialization(manifest, ids, complete, jobs)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["jobs"] = 1
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="materialization contract"):
        internal_verifier.validate_materialization(manifest, ids, complete, jobs)


def test_internal_launcher_uses_hardened_pre_during_post_release_guards() -> None:
    launcher = (ROOT / "scripts/run_public_formal_internal_after_release.sh").read_text(
        encoding="utf-8"
    )
    assert "verify_public_formal_internal_state.py" in launcher
    assert "--mode preflight --quarantine-invalid" in launcher
    assert "--mode release-only" in launcher
    assert "--mode postflight" in launcher
    for binding in (
        "--expected-release-status-sha256",
        "--expected-canonical-fingerprint",
        "--expected-prepared-release-sha256",
    ):
        assert launcher.count(binding) == 3
    assert "prepare_public_test_inputs.py --verify-release" not in launcher
    assert launcher.index(
        "while ! python3 scripts/verify_public_formal_internal_state.py"
    ) < launcher.index("setsid scripts/run_public_formal_internal_matrix.sh")
    matrix = ROOT / "scripts/run_public_formal_internal_matrix.sh"
    assert matrix.exists()


def test_internal_release_guard_binds_all_completed_release_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release_status = tmp_path / "release_status.json"
    promotion = tmp_path / "promotion.json"
    prepared_root = tmp_path / "prepared"
    release = {
        "status": "complete",
        "release_status": {"path": str(release_status), "sha256": "a" * 64},
        "canonical_fingerprint": "b" * 64,
        "prepared_release_sha256": "c" * 64,
    }
    calls = []

    def verify(status: Path, marker: Path, prepared: Path) -> dict:
        calls.append((status, marker, prepared))
        return copy.deepcopy(release)

    monkeypatch.setattr(internal_verifier, "verify_completed_release", verify)
    result = internal_verifier.validate_state(
        "release-only",
        tmp_path / "run",
        "c" * 64,
        False,
        release_status=release_status,
        promotion=promotion,
        prepared_root=prepared_root,
        expected_release_status_sha256="a" * 64,
        expected_canonical_fingerprint="b" * 64,
    )
    assert result == {"status": "release_unchanged", "release": release}
    assert calls == [(release_status, promotion, prepared_root)]

    calls.clear()
    monkeypatch.setattr(internal_verifier, "DATASETS", ())
    preflight = internal_verifier.validate_state(
        "preflight",
        tmp_path / "run",
        "c" * 64,
        False,
        release_status=release_status,
        promotion=promotion,
        prepared_root=prepared_root,
        expected_release_status_sha256="a" * 64,
        expected_canonical_fingerprint="b" * 64,
    )
    assert preflight["status"] == "preflight_ready"
    assert calls == [
        (release_status, promotion, prepared_root),
        (release_status, promotion, prepared_root),
    ]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"expected_release_status_sha256": "0" * 64}, "release-status identity"),
        ({"expected_canonical_fingerprint": "0" * 64}, "canonical release fingerprint"),
        ({"expected_prepared_release_sha256": "0" * 64}, "prepared-release fingerprint"),
    ],
)
def test_internal_release_guard_rejects_each_changed_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, str],
    message: str,
) -> None:
    release = {
        "status": "complete",
        "release_status": {"path": "release_status.json", "sha256": "a" * 64},
        "canonical_fingerprint": "b" * 64,
        "prepared_release_sha256": "c" * 64,
    }
    monkeypatch.setattr(
        internal_verifier,
        "verify_completed_release",
        lambda *_args: copy.deepcopy(release),
    )
    expected = {
        "expected_release_status_sha256": "a" * 64,
        "expected_canonical_fingerprint": "b" * 64,
        "expected_prepared_release_sha256": "c" * 64,
    }
    expected.update(overrides)
    prepared_sha = expected.pop("expected_prepared_release_sha256")
    with pytest.raises(ValueError, match=message):
        internal_verifier.validate_state(
            "release-only",
            tmp_path / "run",
            prepared_sha,
            False,
            **expected,
        )
