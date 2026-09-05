import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_spert_fresh_test.sh"


def load_script(name: str):
    path = ROOT / "scripts" / name
    module_name = f"test_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def types_payload() -> dict:
    return {
        "entities": {
            "Person": {"short": "Per", "verbose": "Person"},
            "Org": {"short": "Org", "verbose": "Organization"},
        },
        "relations": {
            "Works_For": {
                "short": "Work",
                "verbose": "Works for",
                "symmetric": False,
            }
        },
    }


def source_row(identifier: str = "doc-1") -> dict:
    return {
        "orig_id": identifier,
        "tokens": ["Alice", "joined", "Acme"],
        "entities": [],
        "relations": [],
    }


def public_job(identifier: str = "doc-1") -> dict:
    text = "Alice joined Acme"
    return {
        "job_id": f"conll04_test_{identifier}_C1",
        "document_id": f"conll04_test_{identifier}",
        "language": "en",
        "category": "conll04",
        "source_path": f"public:conll04:test:{identifier}",
        "experiment_mode": "baseline",
        "segments": [
            {
                "segment_id": "S1",
                "start": 0,
                "end": len(text),
                "text": text,
                "page": None,
            }
        ],
        "ontology": {
            "entity_types": {"Person": {}, "Org": {}},
            "relation_types": {"Works_For": {}},
        },
    }


def prediction_row() -> dict:
    return {
        "tokens": ["Alice", "joined", "Acme"],
        "entities": [
            {"type": "Person", "start": 0, "end": 1},
            {"type": "Org", "start": 2, "end": 3},
        ],
        "relations": [{"type": "Works_For", "head": 0, "tail": 1}],
    }


def fake_release() -> dict:
    return {
        "release_status": {"path": "release.json", "sha256": "a" * 64},
        "promotion": {
            "path": "promotion.json",
            "sha256": "b" * 64,
            "attestation_sha256": "c" * 64,
        },
        "canonical_fingerprint": "d" * 64,
    }


def test_preparation_checks_complete_release_before_promotion_or_test_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script("prepare_spert_fresh_test.py")
    monkeypatch.chdir(ROOT)
    calls = []

    def reject_release(*_args):
        calls.append("release")
        raise ValueError("release incomplete")

    monkeypatch.setattr(module.formal_release, "verify_release", reject_release)
    monkeypatch.setattr(
        module.release,
        "validate_promotion",
        lambda *_args: calls.append("promotion"),
    )
    monkeypatch.setattr(
        module.Tracker,
        "bytes",
        lambda *_args: pytest.fail("no input may be opened before complete release"),
    )
    with pytest.raises(ValueError, match="canonical formal release is not complete"):
        module.verified_inputs(
            "conll04",
            Path("outputs/public_formal_matrix/promotion.json"),
            Path("data/external/spert"),
            Path("data/processed/spert_fresh_train_v1"),
            Path("data/processed/public_benchmarks_full"),
            Path("outputs/public_horizontal_validation/spert_fresh/conll04/seed42"),
        )
    assert calls == ["release"]


def test_preparation_requires_canonical_project_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_script("prepare_spert_fresh_test.py")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="canonical project root"):
        module.verified_inputs(
            "conll04", Path("promotion.json"), Path("native"), Path("train"),
            Path("reference"), Path("validation"),
        )


def test_prepared_manifest_recomputes_orig_ids_and_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_script("prepare_spert_fresh_test.py")
    monkeypatch.chdir(ROOT)
    monkeypatch.setitem(module.EXPECTED_TEST_ROWS, "conll04", 1)
    target = tmp_path / "conll04"
    test_path, types_path = target / "test.json", target / "types.json"
    write_json(test_path, [source_row()])
    write_json(types_path, types_payload())
    tracker = module.Tracker()
    release_value = {
        "release_status": {"path": "release", "sha256": "a" * 64},
        "promotion": {"path": "promotion", "sha256": "b" * 64},
        "canonical_fingerprint": "c" * 64,
    }
    context = {
        "tracker": tracker,
        "native_rows": [source_row()],
        "types": types_payload(),
        "inputs": {},
        "lineage": {},
        "promotion": {"attestation_sha256": "d" * 64},
        "release": release_value,
        "fingerprint": "e" * 64,
    }
    monkeypatch.setattr(
        module,
        "verified_inputs",
        lambda *_args, **_kwargs: {**context, "tracker": module.Tracker()},
    )
    orig_digest = module.training.sha256_bytes(
        module.training.canonical_bytes(["doc-1"])
    )
    manifest = {
        "schema_version": module.SCHEMA_VERSION,
        "status": "prepared_test",
        "dataset": "conll04",
        "split": "test",
        "formal_test_read": True,
        "test_gold_read_for_alignment": True,
        "test_gold_used_for_selection": False,
        "rows": 1,
        "fingerprint": "e" * 64,
        "promotion_attestation_sha256": "d" * 64,
        "inputs": {},
        "validation_lineage": {},
        "canonical_release": release_value,
        "orig_id_sha256": orig_digest,
        "alignment": {
            "native_test_equals_public_test_gold": True,
            "native_test_equals_public_test_job_order_and_text": True,
            "train_validation_test_orig_ids_disjoint": True,
        },
        "outputs": {
            "test.json": tracker.identity(test_path),
            "types.json": tracker.identity(types_path),
        },
    }
    write_json(target / "manifest.json", manifest)
    assert module.validate_manifest("conll04", target / "manifest.json")["status"] == "verified_prepared_test"

    manifest["orig_id_sha256"] = "0" * 64
    write_json(target / "manifest.json", manifest)
    with pytest.raises(ValueError, match="orig_id_sha256"):
        module.validate_manifest("conll04", target / "manifest.json")


def test_converter_emits_exact_test_ids_character_spans_types_and_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script("convert_spert_test_predictions.py")
    monkeypatch.setitem(module.EXPECTED_ROWS, "conll04", 1)
    converted = module.build_conversion(
        "conll04", [source_row()], [prediction_row()], [public_job()], types_payload()
    )
    assert [row["job_id"] for row in converted] == ["conll04_test_doc-1_C1"]
    annotation = converted[0]["annotation"]
    assert annotation["document_id"] == "conll04_test_doc-1"
    assert [entity["type"] for entity in annotation["entities"]] == ["Person", "Org"]
    assert [
        (entity["evidence"]["start"], entity["evidence"]["end"], entity["text"])
        for entity in annotation["entities"]
    ] == [(0, 5, "Alice"), (13, 17, "Acme")]
    assert annotation["relations"][0]["evidence"][0] == {
        "text": "Alice joined Acme",
        "segment_id": "S1",
        "page": None,
        "start": 0,
        "end": 17,
    }

    wrong = prediction_row()
    wrong["tokens"] = ["Acme", "joined", "Alice"]
    with pytest.raises(ValueError, match="tokens do not match"):
        module.build_conversion(
            "conll04", [source_row()], [wrong], [public_job()], types_payload()
        )


def test_converter_snapshots_hash_bound_jobs_and_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_script("convert_spert_test_predictions.py")
    monkeypatch.chdir(ROOT)
    monkeypatch.setitem(module.EXPECTED_ROWS, "conll04", 1)
    monkeypatch.setattr(module, "FORMAL_ROOT", tmp_path / "formal")
    prepared_root = tmp_path / "prepared" / "conll04"
    test_path = prepared_root / "test.json"
    types_path = prepared_root / "types.json"
    jobs_path = tmp_path / "test_baseline_jobs.jsonl"
    raw_path = tmp_path / "raw.json"
    inference_path = tmp_path / "capture.json"
    prepared_manifest_path = prepared_root / "manifest.json"
    write_json(test_path, [source_row()])
    write_json(types_path, types_payload())
    write_jsonl(jobs_path, [public_job()])
    write_json(raw_path, [prediction_row()])
    prepared_manifest = {
        "promotion_attestation_sha256": "a" * 64,
        "fingerprint": "b" * 64,
        "orig_id_sha256": "c" * 64,
        "inputs": {"test_jobs": module.identity(jobs_path)},
    }
    write_json(prepared_manifest_path, prepared_manifest)
    inference = {
        "status": "captured_exact_new_eval_output",
        "dataset": "conll04",
        "split": "test",
        "formal_test_read": True,
        "seed": 42,
        "prepared_manifest": module.identity(prepared_manifest_path),
        "captured": {"predictions": module.identity(raw_path)},
    }
    write_json(inference_path, inference)

    def verified(*_args, **_kwargs):
        return {
            "manifest": module.identity(prepared_manifest_path),
            "outputs": {
                "test.json": module.identity(test_path),
                "types.json": module.identity(types_path),
            },
        }

    monkeypatch.setattr(module.prepared, "validate_manifest", verified)
    monkeypatch.setattr(module.prepared.release, "validate_source_jobs", lambda *_args: None)
    output = tmp_path / "formal" / "conll04" / "test_predictions.jsonl"
    conversion_manifest = tmp_path / "formal" / "conll04" / "conversion_manifest.json"
    args = SimpleNamespace(
        dataset="conll04",
        prepared_manifest=prepared_manifest_path,
        predictions=raw_path,
        jobs=jobs_path,
        inference_manifest=inference_path,
        output=output,
        manifest=conversion_manifest,
    )
    result = module.convert(args)
    assert result["rows"] == 1
    assert result["formal_test_read"] is True
    assert result["inputs"]["inference_manifest"] == module.identity(inference_path)

    tampered_jobs = public_job()
    tampered_jobs["language"] = "fr"
    write_jsonl(jobs_path, [tampered_jobs])
    with pytest.raises(ValueError, match="hash-bound"):
        module.convert(args)


def test_frozen_checkpoint_contract_rejects_any_weight_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_script("spert_fresh_formal_contract.py")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    state = {"epoch": 20, "epoch_iteration": 0, "iteration": 9, "updates_epoch": 3}
    (checkpoint / "extra.state").write_bytes(b"frozen-extra-state")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(load=lambda *_args, **_kwargs: state),
    )
    for name, payload in {
        "config.json": b"{}\n",
        "pytorch_model.bin": b"frozen-weight",
        "special_tokens_map.json": b"{}\n",
        "tokenizer_config.json": b"{}\n",
        "vocab.txt": b"token\n",
    }.items():
        (checkpoint / name).write_bytes(payload)
    pointer = tmp_path / "pointer.txt"
    pointer.write_text(f"{checkpoint}\n", encoding="utf-8")
    train_args = tmp_path / "train_args.json"
    training = {
        "model_type": "spert", "train_batch_size": 2, "eval_batch_size": 1,
        "neg_entity_count": 100, "neg_relation_count": 100, "epochs": 20,
        "lr": 5e-5, "lr_warmup": 0.1, "weight_decay": 0.01,
        "max_grad_norm": 1.0, "rel_filter_threshold": 0.4,
        "size_embedding": 25, "prop_drop": 0.1, "max_span_size": 10,
        "sampling_processes": 4, "max_pairs": 1000, "seed": 42,
        "final_eval": True, "store_predictions": True,
        "train_path": str(tmp_path / "conll04/train.json"),
        "valid_path": str(tmp_path / "conll04/validation.json"),
        "types_path": str(tmp_path / "conll04/types.json"),
    }
    write_json(train_args, training)
    status_path = tmp_path / "validation" / "status.json"
    write_json(
        status_path,
        {
            "status": "complete", "split": "validation", "seed": 42,
            "test_split_access": "forbidden-and-not-read",
            "datasets": {
                "conll04": {"checkpoint": str(checkpoint), "prediction_rows": 231}
            },
        },
    )
    files = {
        path.name: (path.stat().st_size, sha256(path)) for path in checkpoint.iterdir()
    }
    spec = {
        "pointer": str(pointer),
        "pointer_sha256": sha256(pointer),
        "checkpoint": str(checkpoint),
        "train_args": str(train_args),
        "train_args_sha256": sha256(train_args),
        "extra_state": state,
        "files": files,
    }
    monkeypatch.setitem(module.CHECKPOINT_SPECS, "conll04", spec)
    monkeypatch.setattr(module, "VALIDATION_ROOT", tmp_path / "validation")
    verified = module.verify_checkpoint("conll04")
    assert verified["checkpoint"]["files"]["pytorch_model.bin"]["sha256"] == sha256(
        checkpoint / "pytorch_model.bin"
    )

    (checkpoint / "pytorch_model.bin").write_bytes(b"changed-weight")
    with pytest.raises(module.ContractError, match="checkpoint file changed"):
        module.verify_checkpoint("conll04")


def test_exact_output_discovery_requires_one_new_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_script("spert_fresh_formal_contract.py")
    label_root = tmp_path / "logs" / "label"
    (label_root / "old").mkdir(parents=True)
    new = label_root / "new"
    new.mkdir()
    write_json(new / "args.json", {})
    write_json(new / "predictions_test_epoch_0.json", [])
    prepared_manifest = tmp_path / "prepared" / "manifest.json"
    write_json(prepared_manifest, {})
    write_json(prepared_manifest.parent / "test.json", [])
    write_json(prepared_manifest.parent / "types.json", {})
    jobs = tmp_path / "jobs.jsonl"
    write_jsonl(jobs, [])
    snapshot_path = tmp_path / "snapshot.json"
    write_json(snapshot_path, {"placeholder": True})
    snapshot = {
        "label_root": str(label_root),
        "before_directories": ["old"],
        "eval_args": {},
        "invocation_id": "00000000-0000-4000-8000-000000000000",
    }
    prepared_value = {"identity": module.identity(prepared_manifest)}
    monkeypatch.setattr(
        module, "_validate_snapshot", lambda *_args, **_kwargs: (snapshot, prepared_value)
    )
    monkeypatch.setattr(module.converter, "build_conversion", lambda *_args: [])
    monkeypatch.setitem(module.EXPECTED_ROWS, "conll04", 0)
    capture = module.capture_eval(
        "conll04", prepared_manifest, jobs, tmp_path / "logs", snapshot_path,
        tmp_path / "raw.json", tmp_path / "args.json", tmp_path / "capture.json",
        fake_release(),
    )
    assert capture["new_directories"] == ["new"]

    second = label_root / "newer"
    second.mkdir()
    write_json(second / "args.json", {})
    write_json(second / "predictions_test_epoch_0.json", [])
    with pytest.raises(module.ContractError, match="exactly one newly created"):
        module.capture_eval(
            "conll04", prepared_manifest, jobs, tmp_path / "logs", snapshot_path,
            tmp_path / "raw2.json", tmp_path / "args2.json", tmp_path / "capture2.json",
            fake_release(),
        )


def test_completion_resume_is_exact_content_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_script("spert_fresh_formal_contract.py")
    payload = {
        "schema_version": module.SCHEMA_VERSION,
        "status": "complete",
        "dataset": "conll04",
        "artifact_sha256": "a" * 64,
    }
    monkeypatch.setattr(module, "_validated_payload", lambda *_args: payload)
    module.finalize_dataset(tmp_path / "data", tmp_path / "prepared", tmp_path / "run", "conll04", fake_release())
    assert module.validate_dataset(
        tmp_path / "data", tmp_path / "prepared", tmp_path / "run", "conll04", fake_release()
    ) == payload
    completion = tmp_path / "run" / "conll04" / "completion_manifest.json"
    changed = dict(payload)
    changed["artifact_sha256"] = "b" * 64
    write_json(completion, changed)
    with pytest.raises(module.ContractError, match="no longer matches"):
        module.validate_dataset(
            tmp_path / "data", tmp_path / "prepared", tmp_path / "run", "conll04", fake_release()
        )


def test_runner_is_eval_only_locked_release_gated_and_strict() -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    text = RUNNER.read_text(encoding="utf-8")
    assert 'gpu_lock="${PUBLIC_GPU_LOCK_FILE:-$root/outputs/.locks/public-validation-gpu.lock}"' in text
    assert 'exec 8>"$gpu_lock"' in text
    assert "flock -n 8" in text
    assert "while ! jq -e" in text
    assert "prepare_spert_fresh_test.py" in text
    assert "run_spert_compat.py" in text
    assert "run_spert_fresh_baseline.sh" not in text
    assert "-- train" not in text
    assert "bert-base-cased" not in text
    assert "scibert" not in text
    assert "--eval_batch_size 1" in text
    assert "--rel_filter_threshold 0.4" in text
    assert "--size_embedding 25" in text
    assert "--prop_drop 0.1" in text
    assert "--max_span_size 10" in text
    assert "--sampling_processes 4" in text
    assert "--max_pairs 1000" in text
    assert "--seed 42" in text
    assert "--store_predictions" in text
    assert "--allow-non-validation" in text
    assert "find " not in text
    assert "sort -nr" not in text
    release = text.index("release_verification=\"$(verify_release)\"")
    formal_read = text.index("formal_test_read=true")
    preparation = text.index("scripts/prepare_spert_fresh_test.py")
    snapshot = text.index("begin-eval")
    evaluation = text.index('"$compat" --repo')
    capture = text.index("capture-eval")
    conversion = text.index("convert_spert_test_predictions.py")
    normalized = text.index("evaluate_annotations.py")
    strict = text.index("evaluate_public_validation_spans.py")
    finalize = text.index(" finalize ")
    assert release < formal_read < preparation < snapshot < evaluation < capture
    assert capture < conversion < normalized < strict < finalize
