import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "convert_spert_predictions", ROOT / "scripts" / "convert_spert_predictions.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def fixture(tmp_path: Path) -> Namespace:
    validation = tmp_path / "validation.json"
    predictions = tmp_path / "predictions.json"
    jobs = tmp_path / "validation_baseline_jobs.jsonl"
    output = tmp_path / "converted.jsonl"
    manifest = tmp_path / "manifest.json"
    source = {
        "orig_id": 7,
        "tokens": ["Alice", "works", "at", "Acme", "."],
        "entities": [],
        "relations": [],
    }
    prediction = {
        "tokens": source["tokens"],
        "entities": [
            {"type": "Peop", "start": 0, "end": 1},
            {"type": "Org", "start": 3, "end": 4},
        ],
        "relations": [{"type": "Work_For", "head": 0, "tail": 1}],
    }
    job = {
        "job_id": "conll04_validation_7_C1",
        "document_id": "conll04_validation_7",
        "language": "en",
        "ontology": {
            "entity_types": {"Peop": {}, "Org": {}},
            "relation_types": {"Work_For": {}},
        },
        "segments": [
            {
                "segment_id": "S1",
                "page": None,
                "start": 0,
                "end": 21,
                "text": "Alice works at Acme .",
            }
        ],
    }
    write_json(validation, [source])
    write_json(predictions, [prediction])
    write_jsonl(jobs, [job])
    return Namespace(
        dataset="conll04",
        validation_data=validation,
        predictions=predictions,
        jobs=jobs,
        output=output,
        manifest=manifest,
    )


def test_converts_token_spans_relations_and_provenance(tmp_path):
    args = fixture(tmp_path)
    summary = MODULE.convert(args)
    converted = MODULE.read_jsonl(args.output)[0]

    assert summary["status"] == "complete"
    assert summary["test_split_access"] == "forbidden-and-not-read"
    assert [entity["text"] for entity in converted["annotation"]["entities"]] == [
        "Alice",
        "Acme",
    ]
    assert converted["annotation"]["entities"][1]["evidence"]["start"] == 15
    assert converted["annotation"]["relations"][0]["source_id"] == "E1"
    assert converted["annotation"]["relations"][0]["target_id"] == "E2"
    assert converted["annotation"]["relations"][0]["evidence"][0]["text"] == "Alice works at Acme"
    assert json.loads(args.manifest.read_text())["output"]["sha256"] == MODULE.sha256_file(args.output)


def test_rejects_reordered_or_changed_prediction_tokens(tmp_path):
    args = fixture(tmp_path)
    prediction = json.loads(args.predictions.read_text())
    prediction[0]["tokens"][0] = "Bob"
    write_json(args.predictions, prediction)

    with pytest.raises(ValueError, match="tokens do not match"):
        MODULE.convert(args)


def test_rejects_invalid_relation_endpoint(tmp_path):
    args = fixture(tmp_path)
    prediction = json.loads(args.predictions.read_text())
    prediction[0]["relations"][0]["tail"] = 2
    write_json(args.predictions, prediction)

    with pytest.raises(ValueError, match="endpoint is outside"):
        MODULE.convert(args)


def test_rejects_row_count_mismatch(tmp_path):
    args = fixture(tmp_path)
    write_json(args.predictions, [])

    with pytest.raises(ValueError, match="row-count mismatch"):
        MODULE.convert(args)
