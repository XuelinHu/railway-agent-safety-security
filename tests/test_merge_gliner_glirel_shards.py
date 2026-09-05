import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "merge_gliner_glirel_shards.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def job(index):
    return {"job_id": f"conll04_validation_{index}_C1", "document_id": f"conll04_validation_{index}"}


def prediction(index):
    source = job(index)
    return {
        "job_id": source["job_id"],
        "annotation": {
            "document_id": source["document_id"],
            "entities": [],
            "relations": [],
        },
    }


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class MergeGLiNERGLiRELShardsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.merger = load_script()

    def test_merge_restores_job_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = root / "validation_baseline_jobs.jsonl"
            shard_a, shard_b, output = root / "a.jsonl", root / "b.jsonl", root / "merged.jsonl"
            write_jsonl(jobs, [job(1), job(2), job(3), job(4)])
            write_jsonl(shard_a, [prediction(3), prediction(4)])
            write_jsonl(shard_b, [prediction(1), prediction(2)])
            result = self.merger.merge(jobs, [shard_a, shard_b], output)
            merged = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        self.assertTrue(result["complete"])
        self.assertEqual([row["job_id"] for row in merged], [job(i)["job_id"] for i in range(1, 5)])

    def test_merge_rejects_duplicate_and_missing_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = root / "validation_baseline_jobs.jsonl"
            shard_a, shard_b, output = root / "a.jsonl", root / "b.jsonl", root / "merged.jsonl"
            write_jsonl(jobs, [job(1), job(2)])
            write_jsonl(shard_a, [prediction(1)])
            write_jsonl(shard_b, [prediction(1)])
            with self.assertRaisesRegex(ValueError, "duplicate prediction"):
                self.merger.merge(jobs, [shard_a, shard_b], output)
            write_jsonl(shard_b, [])
            with self.assertRaisesRegex(ValueError, "incomplete merge"):
                self.merger.merge(jobs, [shard_a, shard_b], output)
            result = self.merger.merge(jobs, [shard_a, shard_b], output, allow_incomplete=True)
            self.assertFalse(result["complete"])


if __name__ == "__main__":
    unittest.main()
