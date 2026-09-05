import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "merge_qlora_shards.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class MergeQLoRAShardsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.merger = load_script()

    def paths(self, root):
        return root / "jobs.jsonl", root / "predictions.jsonl", root / "inference.log"

    def test_merge_orders_successes_and_preserves_terminal_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs, output, log = self.paths(root)
            write_jsonl(jobs, [{"job_id": value} for value in ("a", "b", "c")])
            write_jsonl(Path(f"{output}.part0"), [{"job_id": "a", "annotation": {}}])
            write_jsonl(Path(f"{log}.part0"), [
                {"job_id": "a", "status": "success"},
                {"job_id": "b", "status": "failed"},
            ])
            write_jsonl(Path(f"{output}.part1"), [{"job_id": "c", "annotation": {}}])
            write_jsonl(Path(f"{log}.part1"), [
                {"job_id": "c", "status": "failed"},
                {"job_id": "c", "status": "success"},
            ])

            result = self.merger.merge(jobs, output, log, workers=2)
            predictions = [json.loads(line) for line in output.read_text().splitlines()]
            terminal = [json.loads(line) for line in log.read_text().splitlines()]

        self.assertEqual([row["job_id"] for row in predictions], ["a", "c"])
        self.assertEqual([row["job_id"] for row in terminal], ["a", "b", "c"])
        self.assertEqual(result["terminal_failures"], 1)

    def test_merge_rejects_missing_terminal_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs, output, log = self.paths(root)
            write_jsonl(jobs, [{"job_id": "a"}, {"job_id": "b"}])
            write_jsonl(Path(f"{output}.part0"), [{"job_id": "a"}])
            write_jsonl(Path(f"{log}.part0"), [{"job_id": "a", "status": "success"}])
            with self.assertRaisesRegex(ValueError, "terminal logs are missing"):
                self.merger.merge(jobs, output, log, workers=1)

    def test_merge_rejects_prediction_without_final_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs, output, log = self.paths(root)
            write_jsonl(jobs, [{"job_id": "a"}])
            write_jsonl(Path(f"{output}.part0"), [{"job_id": "a"}])
            write_jsonl(Path(f"{log}.part0"), [{"job_id": "a", "status": "failed"}])
            with self.assertRaisesRegex(ValueError, "prediction/log mismatch"):
                self.merger.merge(jobs, output, log, workers=1)

    def test_merge_rejects_unknown_job(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs, output, log = self.paths(root)
            write_jsonl(jobs, [{"job_id": "a"}])
            write_jsonl(Path(f"{output}.part0"), [{"job_id": "other"}])
            write_jsonl(Path(f"{log}.part0"), [{"job_id": "a", "status": "failed"}])
            with self.assertRaisesRegex(ValueError, "unknown job_id"):
                self.merger.merge(jobs, output, log, workers=1)


if __name__ == "__main__":
    unittest.main()
