import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import gliner_glirel_formal_contract as gliner_contract  # noqa: E402
import prepare_public_test_inputs as preparation  # noqa: E402
import qwen_zeroshot_formal_contract as qwen_contract  # noqa: E402
import verify_public_formal_internal_state as internal_verifier  # noqa: E402


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def identity(path: Path) -> dict:
    payload = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def baseline_job(dataset: str) -> dict:
    text = "Alice joined Acme"
    return {
        "job_id": f"{dataset}_test_1_C1",
        "document_id": f"{dataset}_test_1",
        "language": "en",
        "category": dataset,
        "source_path": f"public:{dataset}:test:1",
        "chunk_number": 1,
        "chunk_count": 1,
        "teacher_model": "benchmark_model",
        "prompt_version": "public-benchmark-v1",
        "system_instruction": "Extract exact source spans.",
        "segments": [
            {
                "segment_id": "S1",
                "segment_type": "sentence",
                "page": None,
                "start": 0,
                "end": len(text),
                "text": text,
            }
        ],
        "status": "benchmark",
        "ontology": {
            "entity_types": {"Peop": {}, "Org": {}},
            "relation_types": {"Work_For": {}},
            "claim_statuses": {"explicit": "direct"},
            "allowed_relation_signatures": {
                "Work_For": {"source": ["Peop"], "target": ["Org"]}
            },
        },
        "experiment_mode": "baseline",
    }


def build_context_factory(
    tmp_path: Path, promotion_path: Path, promotion: dict
):
    source_root = tmp_path / "source"
    graph_root = tmp_path / "graph"
    static: dict[str, dict] = {}
    for dataset in preparation.DATASETS:
        test_path = source_root / dataset / "test_baseline_jobs.jsonl"
        write_jsonl(test_path, [baseline_job(dataset)])
        graph = graph_root / dataset
        graph_manifest = graph / "preparation_manifest.json"
        concepts = graph / "knowledge_graph" / "concepts.jsonl"
        mentions = graph / "knowledge_graph" / "mentions.jsonl"
        relations = graph / "knowledge_graph" / "relations.jsonl"
        write_json(graph_manifest, {"dataset": dataset, "status": "prepared"})
        write_jsonl(concepts, [])
        write_jsonl(mentions, [])
        write_jsonl(relations, [])
        static[dataset] = {
            "test_path": test_path,
            "graph_paths": {
                "train_only_concepts": concepts,
                "train_only_mentions": mentions,
                "train_only_relations": relations,
            },
            "graph_manifest": graph_manifest,
        }

    def factory(dataset: str, *_args) -> dict:
        item = static[dataset]
        tracker = preparation.SnapshotTracker()
        graph_outputs = {
            name: tracker.identity(path)
            for name, path in item["graph_paths"].items()
        }
        builder_inputs = {
            "eae_builder": {"path": "eae.py", "bytes": 1, "sha256": "1" * 64},
            "hrge_builder": {"path": "hrge.py", "bytes": 1, "sha256": "2" * 64},
        }
        fingerprint_payload = {
            "schema_version": preparation.PREPARATION_SCHEMA_VERSION,
            "dataset": dataset,
            "promotion": tracker.identity(promotion_path),
            "promotion_attestation_sha256": promotion["attestation_sha256"],
            "test_baseline_jobs": tracker.identity(item["test_path"]),
            "train_only_graph_manifest": tracker.identity(item["graph_manifest"]),
            "train_only_graph_outputs": graph_outputs,
            "validation_frozen_builders": builder_inputs,
            "preparation_runner": {"path": "prepare.py", "bytes": 1, "sha256": "3" * 64},
            "settings": preparation.SETTINGS,
            "batch_size": preparation.FROZEN_BATCH_SIZE,
            "semantic_model": {"path": "frozen-model", "revision": "revision", "files": {}},
        }
        return {
            "promotion": promotion,
            "promotion_identity": tracker.identity(promotion_path),
            "tracker": tracker,
            "test_path": item["test_path"],
            "graph_paths": item["graph_paths"],
            "test_jobs": [baseline_job(dataset)],
            "train_jobs": [],
            "concepts": [],
            "mentions": [],
            "relations": [],
            "graph_counts": {"concepts": 0, "mentions": 0, "relations": 0, "training_documents": 0},
            "graph_proof": {
                "graph_manifest": tracker.identity(item["graph_manifest"]),
                "outputs": graph_outputs,
                "inputs": builder_inputs,
            },
            "model_path": Path("frozen-model"),
            "semantic_identity": fingerprint_payload["semantic_model"],
            "quarantine": {},
            "fingerprint_payload": fingerprint_payload,
            "fingerprint": preparation.stable_digest(fingerprint_payload),
        }

    return source_root, graph_root, factory


def stub_hrge_run(args: SimpleNamespace) -> int:
    rows = preparation.SnapshotTracker().jsonl(args.jobs)
    for row in rows:
        row["kg_v2_context"] = {
            "train_graph_only": True,
            "leave_current_document_out": True,
            "anchors": [],
            "edge_priors": [],
            "semantic_relation_patterns": [],
        }
    write_jsonl(args.output, rows)
    write_json(args.audit, {"jobs": len(rows)})
    return 0


@pytest.fixture
def generated_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    promotion_path = tmp_path / "promotion.json"
    promotion = {
        "schema_version": "public-formal-test-promotion-v2",
        "status": "promoted",
        "promoted_systems": ["soe", "pge"],
        "attestation_sha256": "a" * 64,
        "datasets": {
            dataset: {"status": "passed"} for dataset in preparation.DATASETS
        },
    }
    write_json(promotion_path, promotion)
    source_root, graph_root, context_factory = build_context_factory(
        tmp_path, promotion_path, promotion
    )
    monkeypatch.setattr(preparation, "validate_promotion", lambda _path: promotion)
    monkeypatch.setattr(preparation, "_verified_context", context_factory)
    monkeypatch.setattr(preparation.hrge_builder, "run", stub_hrge_run)
    for dataset in preparation.DATASETS:
        monkeypatch.setitem(preparation.EXPECTED_TEST_JOBS, dataset, 1)

    prepared_root = tmp_path / "prepared"
    for dataset in preparation.DATASETS:
        result = preparation.prepare_dataset(
            dataset,
            promotion_path=promotion_path,
            source_root=source_root,
            graph_root=graph_root,
            output_root=prepared_root,
        )
        assert result["status"] == "prepared_test"

    canonical = preparation.validate_prepared_release(
        promotion_path=promotion_path,
        output_root=prepared_root,
        source_root=source_root,
        graph_root=graph_root,
    )
    release_status = tmp_path / "release_status.json"
    attestation = tmp_path / "prepared_release_attestation.json"
    write_json(attestation, canonical)
    release = {
        "status": "complete",
        "stage": "complete",
        "promotion_status": "promoted",
        "gate_review_status": "passed",
        "updated_at": "2026-09-04T00:00:00+00:00",
        "datasets": {
            dataset: {
                "preparation_status": "prepared_test",
                "manifest": str(prepared_root / dataset / "preparation_manifest.json"),
            }
            for dataset in preparation.DATASETS
        },
        "canonical_prepared_release": {
            **identity(attestation),
            "status": "verified_release",
            "schema_version": "public-formal-test-release-v2",
            "release_sha256": canonical["release_sha256"],
        },
    }
    write_json(release_status, release)
    monkeypatch.setattr(
        qwen_contract, "canonical_validate_promotion", lambda _path: promotion
    )
    # Both consumers still call the real validate_prepared_release.  Its
    # imported defaults are bypassed only for source/graph roots by the real
    # context factory above.
    return {
        "promotion_path": promotion_path,
        "promotion": promotion,
        "prepared_root": prepared_root,
        "source_root": source_root,
        "graph_root": graph_root,
        "canonical": canonical,
        "release_status": release_status,
        "attestation": attestation,
        "release": release,
    }


def test_three_dataset_writer_canonical_release_qwen_and_gliner_chain(
    generated_release: dict,
) -> None:
    fixture = generated_release
    for dataset in preparation.DATASETS:
        hrge_path = fixture["prepared_root"] / dataset / "jobs/test_hrge_jobs.jsonl"
        row = preparation.SnapshotTracker().jsonl(hrge_path)[0]
        assert row["method_name"] == "HRGE"
        assert row["train_graph_only"] is True
        assert row["kg_v2_context"]["train_graph_only"] is True

    qwen = qwen_contract.verify_release(
        fixture["release_status"], fixture["promotion_path"], fixture["prepared_root"]
    )
    gliner = gliner_contract.verify_release(
        fixture["release_status"], fixture["promotion_path"], fixture["prepared_root"]
    )
    internal = internal_verifier.validate_state(
        "release-only",
        fixture["prepared_root"] / "unused-run-root",
        qwen["prepared_release_sha256"],
        False,
        release_status=fixture["release_status"],
        promotion=fixture["promotion_path"],
        prepared_root=fixture["prepared_root"],
        expected_release_status_sha256=qwen["release_status"]["sha256"],
        expected_canonical_fingerprint=qwen["canonical_fingerprint"],
    )
    assert qwen["prepared_release_sha256"] == fixture["canonical"]["release_sha256"]
    assert gliner["canonical_fingerprint"] == qwen["canonical_fingerprint"]
    assert internal["release"]["canonical_fingerprint"] == qwen["canonical_fingerprint"]


def test_internal_consumer_rejects_release_before_publisher_complete(
    generated_release: dict,
) -> None:
    fixture = generated_release
    release = copy.deepcopy(fixture["release"])
    release["status"] = "running"
    release["stage"] = "contract_verification"
    write_json(fixture["release_status"], release)
    with pytest.raises(qwen_contract.ContractError, match="not complete"):
        internal_verifier.validate_state(
            "release-only",
            fixture["prepared_root"] / "unused-run-root",
            None,
            False,
            release_status=fixture["release_status"],
            promotion=fixture["promotion_path"],
            prepared_root=fixture["prepared_root"],
        )


@pytest.mark.parametrize("field", ["top_level", "nested"])
def test_canonical_release_rejects_hrge_provenance_even_after_hash_refresh(
    generated_release: dict, field: str
) -> None:
    fixture = generated_release
    dataset = "conll04"
    hrge_path = fixture["prepared_root"] / dataset / "jobs/test_hrge_jobs.jsonl"
    rows = preparation.SnapshotTracker().jsonl(hrge_path)
    if field == "top_level":
        rows[0].pop("train_graph_only")
        message = "HRGE provenance"
    else:
        rows[0]["kg_v2_context"]["train_graph_only"] = False
        message = "HRGE leakage guard"
    write_jsonl(hrge_path, rows)
    manifest_path = fixture["prepared_root"] / dataset / "preparation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"]["test_hrge_jobs"] = identity(hrge_path)
    write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match=message):
        preparation.validate_prepared_release(
            promotion_path=fixture["promotion_path"],
            output_root=fixture["prepared_root"],
            source_root=fixture["source_root"],
            graph_root=fixture["graph_root"],
        )


def refresh_nested_hrge_tamper(fixture: dict) -> None:
    dataset = "conll04"
    hrge_path = fixture["prepared_root"] / dataset / "jobs/test_hrge_jobs.jsonl"
    rows = preparation.SnapshotTracker().jsonl(hrge_path)
    rows[0]["kg_v2_context"]["train_graph_only"] = False
    write_jsonl(hrge_path, rows)
    manifest_path = fixture["prepared_root"] / dataset / "preparation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"]["test_hrge_jobs"] = identity(hrge_path)
    write_json(manifest_path, manifest)


def test_qwen_rejects_real_nested_provenance_tamper(
    generated_release: dict,
) -> None:
    fixture = generated_release
    refresh_nested_hrge_tamper(fixture)
    with pytest.raises(
        qwen_contract.ContractError,
        match="canonical prepared-release verification failed.*HRGE leakage guard",
    ):
        qwen_contract.verify_release(
            fixture["release_status"],
            fixture["promotion_path"],
            fixture["prepared_root"],
        )


def test_gliner_independent_replay_rejects_real_nested_provenance_tamper(
    generated_release: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = generated_release
    shared_result = qwen_contract.verify_release(
        fixture["release_status"], fixture["promotion_path"], fixture["prepared_root"]
    )
    monkeypatch.setattr(
        gliner_contract, "qwen_verify_release", lambda *_args: shared_result
    )
    refresh_nested_hrge_tamper(fixture)
    with pytest.raises(
        gliner_contract.ContractError,
        match="GLiNER/GLiREL canonical prepared-release verification failed.*HRGE leakage guard",
    ):
        gliner_contract.verify_release(
            fixture["release_status"],
            fixture["promotion_path"],
            fixture["prepared_root"],
        )


def test_consumer_fingerprints_rotate_with_canonical_prepared_release(
    generated_release: dict,
) -> None:
    fixture = generated_release
    first_qwen = qwen_contract.verify_release(
        fixture["release_status"], fixture["promotion_path"], fixture["prepared_root"]
    )
    first_gliner = gliner_contract.verify_release(
        fixture["release_status"], fixture["promotion_path"], fixture["prepared_root"]
    )

    # Change only the serialized bytes, not row semantics, then refresh the
    # superficial output identity.  The canonical release fingerprint must
    # still rotate and both consumers must bind their fingerprints to it.
    dataset = "conll04"
    hrge_path = fixture["prepared_root"] / dataset / "jobs/test_hrge_jobs.jsonl"
    hrge_path.write_text(hrge_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    manifest_path = fixture["prepared_root"] / dataset / "preparation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"]["test_hrge_jobs"] = identity(hrge_path)
    write_json(manifest_path, manifest)

    changed = preparation.validate_prepared_release(
        promotion_path=fixture["promotion_path"],
        output_root=fixture["prepared_root"],
        source_root=fixture["source_root"],
        graph_root=fixture["graph_root"],
    )
    assert changed["release_sha256"] != fixture["canonical"]["release_sha256"]
    write_json(fixture["attestation"], changed)
    release = copy.deepcopy(fixture["release"])
    release["canonical_prepared_release"] = {
        **identity(fixture["attestation"]),
        "status": "verified_release",
        "schema_version": "public-formal-test-release-v2",
        "release_sha256": changed["release_sha256"],
    }
    write_json(fixture["release_status"], release)

    changed_qwen = qwen_contract.verify_release(
        fixture["release_status"], fixture["promotion_path"], fixture["prepared_root"]
    )
    changed_gliner = gliner_contract.verify_release(
        fixture["release_status"], fixture["promotion_path"], fixture["prepared_root"]
    )
    assert changed_qwen["prepared_release_sha256"] == changed["release_sha256"]
    assert changed_qwen["canonical_fingerprint"] != first_qwen["canonical_fingerprint"]
    assert changed_gliner["canonical_fingerprint"] != first_gliner["canonical_fingerprint"]


def test_qwen_rejects_changed_attestation_even_with_refreshed_status_hash(
    generated_release: dict,
) -> None:
    fixture = generated_release
    changed = copy.deepcopy(fixture["canonical"])
    changed["release_sha256"] = "f" * 64
    write_json(fixture["attestation"], changed)
    release = copy.deepcopy(fixture["release"])
    release["canonical_prepared_release"] = {
        **identity(fixture["attestation"]),
        "status": "verified_release",
        "schema_version": "public-formal-test-release-v2",
        "release_sha256": "f" * 64,
    }
    write_json(fixture["release_status"], release)
    with pytest.raises(qwen_contract.ContractError, match="attestation differs"):
        qwen_contract.verify_release(
            fixture["release_status"], fixture["promotion_path"], fixture["prepared_root"]
        )
    with pytest.raises(qwen_contract.ContractError, match="attestation differs"):
        internal_verifier.validate_state(
            "release-only",
            fixture["prepared_root"] / "unused-run-root",
            None,
            False,
            release_status=fixture["release_status"],
            promotion=fixture["promotion_path"],
            prepared_root=fixture["prepared_root"],
        )


def test_qwen_rejects_changed_release_attestation_record(
    generated_release: dict,
) -> None:
    fixture = generated_release
    release = copy.deepcopy(fixture["release"])
    release["canonical_prepared_release"]["release_sha256"] = "0" * 64
    write_json(fixture["release_status"], release)
    with pytest.raises(qwen_contract.ContractError, match="not bound"):
        qwen_contract.verify_release(
            fixture["release_status"], fixture["promotion_path"], fixture["prepared_root"]
        )
    with pytest.raises(qwen_contract.ContractError, match="not bound"):
        internal_verifier.validate_state(
            "release-only",
            fixture["prepared_root"] / "unused-run-root",
            None,
            False,
            release_status=fixture["release_status"],
            promotion=fixture["promotion_path"],
            prepared_root=fixture["prepared_root"],
        )


def test_release_wrapper_cannot_complete_before_canonical_replay() -> None:
    runner = (ROOT / "scripts/run_public_formal_release_and_prepare.sh").read_text(
        encoding="utf-8"
    )
    verification = runner.index(
        'scripts/prepare_public_test_inputs.py --verify-release >"$attestation_tmp"'
    )
    persisted = runner.index('mv "$attestation_tmp" "$release_attestation"')
    replay = runner.index(
        'scripts/prepare_public_test_inputs.py --verify-release |', persisted
    )
    complete = runner.index('current_stage="complete"')
    assert verification < persisted < replay < complete
    assert 'cmp -s - "$release_attestation"' in runner
