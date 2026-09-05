import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "run_qlora_inference.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


class QLoRAResumeFilesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inference = load_script()

    def test_prepare_jsonl_drops_only_truncated_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text(
                '{"job_id":"complete"}\n{"job_id":',
                encoding="utf-8",
            )
            self.inference.prepare_jsonl_for_append(path)
            with path.open("a", encoding="utf-8") as stream:
                stream.write('{"job_id":"resumed"}\n')
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([row["job_id"] for row in rows], ["complete", "resumed"])

    def test_prepare_jsonl_adds_missing_newline_after_valid_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inference.log"
            path.write_text('{"job_id":"complete"}', encoding="utf-8")
            self.inference.prepare_jsonl_for_append(path)
            self.assertTrue(path.read_text(encoding="utf-8").endswith("\n"))

    def test_prepare_jsonl_rejects_corruption_before_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inference.log"
            path.write_text(
                '{"job_id":\n{"job_id":"later"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "before the final row"):
                self.inference.prepare_jsonl_for_append(path)


if __name__ == "__main__":
    unittest.main()
