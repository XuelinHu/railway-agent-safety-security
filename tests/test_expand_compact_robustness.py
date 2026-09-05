import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def load_expander():
    path = ROOT / "scripts" / "expand_compact_predictions.py"
    spec = importlib.util.spec_from_file_location("expand_compact_robustness", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_entity_without_text_is_audited_and_skipped(tmp_path):
    expander = load_expander()
    jobs = tmp_path / "jobs.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "expanded.jsonl"
    errors = tmp_path / "errors.jsonl"
    jobs.write_text(
        json.dumps(
            {
                "job_id": "J1",
                "document_id": "D1",
                "language": "en",
                "segments": [
                    {"segment_id": "S1", "text": "Alice works.", "start": 0, "page": 1}
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    predictions.write_text(
        json.dumps(
            {
                "job_id": "J1",
                "annotation": {
                    "entities": [{"id": "E1", "type": "Person"}],
                    "relations": [],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    expander.run(
        SimpleNamespace(jobs=jobs, predictions=predictions, output=output, errors=errors)
    )

    expanded = json.loads(output.read_text(encoding="utf-8"))
    audit = json.loads(errors.read_text(encoding="utf-8"))
    assert expanded["annotation"]["entities"] == []
    assert "missing or invalid text" in audit["errors"][0]
