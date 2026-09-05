import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "evaluate_public_validation_spans.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def entity(entity_id, text, entity_type, start, end):
    return {
        "id": entity_id,
        "text": text,
        "type": entity_type,
        "evidence": {"text": text, "segment_id": "S1", "start": start, "end": end},
    }


class PublicValidationSpanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator = load_script()

    def test_exact_span_and_relation_metrics(self):
        job = {
            "job_id": "conll04_validation_demo_C1",
            "segments": [
                {"segment_id": "S1", "start": 0, "end": 21, "text": "Alice works at Acme ."}
            ],
        }
        gold = {
            "document_id": "conll04_validation_demo",
            "language": "en",
            "entities": [entity("E1", "Alice", "Peop", 0, 5), entity("E2", "Acme", "Org", 15, 19)],
            "relations": [
                {
                    "id": "R1",
                    "source_id": "E1",
                    "type": "Work_For",
                    "target_id": "E2",
                    "claim_status": "explicit",
                }
            ],
        }
        predicted = json.loads(json.dumps(gold))
        predicted["entities"].append(entity("E3", "Alice", "Peop", 1, 5))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = root / "validation_baseline_jobs.jsonl"
            gold_path = root / "gold.jsonl"
            index = root / "index.jsonl"
            predictions = root / "predictions.jsonl"
            output = root / "metrics.json"
            jobs.write_text(json.dumps(job) + "\n", encoding="utf-8")
            gold_path.write_text(json.dumps(gold) + "\n", encoding="utf-8")
            index.write_text(json.dumps({"job_id": job["job_id"], "record_index": 0}) + "\n", encoding="utf-8")
            predictions.write_text(
                json.dumps({"job_id": job["job_id"], "annotation": predicted}) + "\n",
                encoding="utf-8",
            )
            self.evaluator.run(
                SimpleNamespace(
                    gold=gold_path,
                    gold_index=index,
                    predictions=predictions,
                    jobs=jobs,
                    output=output,
                )
            )
            result = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(result["entity_strict"]["gold"], 2)
        self.assertEqual(result["entity_strict"]["predicted"], 3)
        self.assertEqual(result["entity_strict"]["correct"], 2)
        self.assertEqual(result["relation_strict"]["correct"], 1)
        self.assertEqual(result["relation_with_claim_status"]["correct"], 1)
        self.assertEqual(result["resolution"]["unresolved_predicted_entities"], 1)

    def test_missing_prediction_is_scored_as_empty(self):
        job = {
            "job_id": "ade_validation_demo_C1",
            "segments": [{"segment_id": "S1", "start": 0, "end": 4, "text": "Drug"}],
        }
        gold = {
            "document_id": "ade_validation_demo",
            "language": "en",
            "entities": [entity("E1", "Drug", "Drug", 0, 4)],
            "relations": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = root / "validation_baseline_jobs.jsonl"
            gold_path = root / "gold.jsonl"
            index = root / "index.jsonl"
            predictions = root / "predictions.jsonl"
            output = root / "metrics.json"
            jobs.write_text(json.dumps(job) + "\n", encoding="utf-8")
            gold_path.write_text(json.dumps(gold) + "\n", encoding="utf-8")
            index.write_text(json.dumps({"job_id": job["job_id"], "record_index": 0}) + "\n", encoding="utf-8")
            predictions.write_text("", encoding="utf-8")
            self.evaluator.run(
                SimpleNamespace(
                    gold=gold_path,
                    gold_index=index,
                    predictions=predictions,
                    jobs=jobs,
                    output=output,
                )
            )
            result = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(result["jobs_missing_predictions"], 1)
        self.assertEqual(result["entity_strict"]["recall"], 0.0)


if __name__ == "__main__":
    unittest.main()
