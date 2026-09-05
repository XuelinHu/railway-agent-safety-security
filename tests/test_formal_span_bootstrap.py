import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/bootstrap_span_compare_formal_test.py"


def load_script():
    spec = importlib.util.spec_from_file_location("formal_span_bootstrap", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def metric(formal_test_read: bool = True):
    item = {
        "document_id": "D",
        "entity": {"gold": 1, "predicted": 1, "correct": 1},
        "relation": {"gold": 1, "predicted": 1, "correct": 1},
        "relation_with_claim_status": {"gold": 1, "predicted": 1, "correct": 1},
    }
    return {
        "metric": "strict-global-character-span-one-to-one",
        "selection_split": "explicit-non-validation-opt-in",
        "formal_test_read": formal_test_read,
        "per_job": {"D_C1": item},
    }


def test_formal_bootstrap_requires_and_preserves_test_proof(tmp_path: Path):
    module = load_script()
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    output = tmp_path / "result.json"
    left.write_text(json.dumps(metric()), encoding="utf-8")
    right.write_text(json.dumps(metric()), encoding="utf-8")
    module.run(SimpleNamespace(left=left, right=right, output=output, iterations=10, seed=7))
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["formal_test_read"] is True
    assert result["selection_split"] == "explicit-non-validation-opt-in"
    assert result["iterations"] == 10


def test_formal_bootstrap_rejects_validation_or_unproven_artifacts(tmp_path: Path):
    module = load_script()
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(metric(formal_test_read=False)), encoding="utf-8")
    with pytest.raises(ValueError, match="formal-test metric proof"):
        module.load_metrics(path)
