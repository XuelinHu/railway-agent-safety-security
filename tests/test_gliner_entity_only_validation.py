import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "derive_gliner_entity_only_validation.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def entity(entity_id, text, entity_type, start, end):
    return {
        "id": entity_id,
        "text": text,
        "type": entity_type,
        "evidence": {"text": text, "segment_id": "S1", "start": start, "end": end},
        "confidence": 0.75,
        "created_by": "gliner_small-v2.1",
    }


def fixture(root, dataset="conll04"):
    data_root = root / "public"
    source_root = root / "combined"
    output_root = root / "entity-only"
    dataset_root = data_root / dataset
    job_id = f"{dataset}_validation_demo_C1"
    document_id = f"{dataset}_validation_demo"
    jobs = [
        {
            "job_id": job_id,
            "document_id": document_id,
            "category": dataset,
            "segments": [
                {"segment_id": "S1", "start": 0, "end": 19, "text": "Alice works at Acme"}
            ],
        }
    ]
    entities = [entity("E1", "Alice", "Peop", 0, 5), entity("E2", "Acme", "Org", 15, 19)]
    relation = {
        "id": "R1",
        "source_id": "E1",
        "type": "Work_For",
        "target_id": "E2",
        "claim_status": "explicit",
    }
    annotation = {
        "schema_version": "0.1.0",
        "document_id": document_id,
        "language": "en",
        "entities": entities,
        "relations": [relation],
        "review": {"status": "unreviewed", "notes": "combined model"},
    }
    write_jsonl(dataset_root / "validation_baseline_jobs.jsonl", jobs)
    write_jsonl(dataset_root / "validation_gold.jsonl", [annotation])
    write_jsonl(
        dataset_root / "validation_index.jsonl", [{"job_id": job_id, "record_index": 0}]
    )
    write_jsonl(
        source_root / f"{dataset}_validation.jsonl",
        [{"job_id": job_id, "annotation": annotation}],
    )
    return data_root, source_root, output_root, annotation


class GLiNEREntityOnlyValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = load_script()

    def test_formal_projection_preserves_entities_and_discards_relations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root, source_root, output_root, source_annotation = fixture(root)
            lineage = self.script.derive_dataset(
                "conll04", data_root, source_root, output_root
            )
            prediction = json.loads(
                (output_root / "conll04_validation.jsonl").read_text(encoding="utf-8")
            )
            metrics = json.loads(
                (output_root / "conll04_validation.character_span_metrics.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(prediction["annotation"]["entities"], source_annotation["entities"])
        self.assertEqual(prediction["annotation"]["relations"], [])
        self.assertIn("explicitly discarded", prediction["annotation"]["review"]["notes"])
        self.assertTrue(lineage["derivation"]["entities_unchanged"])
        self.assertTrue(lineage["derivation"]["relations_explicitly_discarded"])
        self.assertEqual(lineage["derivation"]["relations_discarded"], 1)
        self.assertEqual(
            lineage["derivation"]["validation_alignment"]["status"],
            "proved-one-to-one",
        )
        self.assertEqual(lineage["derivation"]["validation_alignment"]["extra_gold_rows"], 0)
        self.assertFalse(lineage["derivation"]["gpu_used"])
        self.assertEqual(lineage["derivation"]["test_split_access"], "forbidden-and-not-read")
        self.assertEqual(metrics["entity_strict"]["f1"], 1.0)
        self.assertEqual(metrics["relation_strict"]["predicted"], 0)
        self.assertEqual(metrics["relation_strict"]["f1"], 0.0)

    def test_rejects_incomplete_source_before_writing_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root, source_root, output_root, _ = fixture(root)
            (source_root / "conll04_validation.jsonl").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "refusing incomplete source"):
                self.script.derive_dataset("conll04", data_root, source_root, output_root)
            self.assertFalse((output_root / "conll04_validation.jsonl").exists())

    def test_rejects_test_named_inputs_without_reading_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_path = root / "test_baseline_jobs.jsonl"
            test_path.write_text("this must not be parsed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be named 'validation_baseline_jobs.jsonl'"):
                self.script.require_validation_file(
                    test_path, "validation_baseline_jobs.jsonl", "validation jobs"
                )

    def test_rejects_extra_test_canary_gold_before_writing_any_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root, source_root, output_root, _ = fixture(root)
            gold = data_root / "conll04" / "validation_gold.jsonl"
            with gold.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "document_id": "conll04_test_CANARY",
                            "language": "en",
                            "entities": [],
                            "relations": [],
                        }
                    )
                    + "\n"
                )
            with self.assertRaisesRegex(ValueError, "refusing extra or missing gold rows"):
                self.script.derive_dataset("conll04", data_root, source_root, output_root)
            self.assertFalse(output_root.exists())

    def test_rejects_duplicate_gold_documents_and_mismatched_frozen_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root, source_root, output_root, _ = fixture(root)
            dataset_root = data_root / "conll04"
            jobs = read_jsonl(dataset_root / "validation_baseline_jobs.jsonl")
            second_job = copy.deepcopy(jobs[0])
            second_job["job_id"] = "conll04_validation_demo2_C1"
            second_job["document_id"] = "conll04_validation_demo2"
            write_jsonl(dataset_root / "validation_baseline_jobs.jsonl", jobs + [second_job])
            gold = read_jsonl(dataset_root / "validation_gold.jsonl")
            write_jsonl(dataset_root / "validation_gold.jsonl", gold + [copy.deepcopy(gold[0])])
            write_jsonl(
                dataset_root / "validation_index.jsonl",
                [
                    {"job_id": jobs[0]["job_id"], "record_index": 0},
                    {"job_id": second_job["job_id"], "record_index": 1},
                ],
            )
            with self.assertRaisesRegex(ValueError, "gold contains duplicate documents"):
                self.script.derive_dataset("conll04", data_root, source_root, output_root)
            self.assertFalse(output_root.exists())

            # A same-length index still fails if it names anything outside the
            # frozen validation job set/order.
            write_jsonl(dataset_root / "validation_gold.jsonl", [gold[0], {**gold[0], "document_id": second_job["document_id"]}])
            write_jsonl(
                dataset_root / "validation_index.jsonl",
                [
                    {"job_id": jobs[0]["job_id"], "record_index": 0},
                    {"job_id": "conll04_test_CANARY_C1", "record_index": 1},
                ],
            )
            with self.assertRaisesRegex(ValueError, "not in frozen validation job order"):
                self.script.derive_dataset("conll04", data_root, source_root, output_root)
            self.assertFalse(output_root.exists())

    def test_rejects_dataset_parent_symlink_that_escapes_data_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actual_data, source_root, output_root, _ = fixture(root / "actual")
            declared_data = root / "declared-public"
            declared_data.mkdir()
            (declared_data / "conll04").symlink_to(
                actual_data / "conll04", target_is_directory=True
            )
            with self.assertRaisesRegex(ValueError, "parent escapes its expected root"):
                self.script.derive_dataset(
                    "conll04", declared_data, source_root, output_root
                )
            self.assertFalse(output_root.exists())

            declared_source = root / "declared-source"
            declared_source.mkdir()
            (declared_source / "conll04_validation.jsonl").symlink_to(
                source_root / "conll04_validation.jsonl"
            )
            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                self.script.derive_dataset(
                    "conll04", actual_data, declared_source, output_root
                )
            self.assertFalse(output_root.exists())

    def test_staging_failure_does_not_replace_previous_complete_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root, source_root, output_root, _ = fixture(root)
            self.script.derive_dataset("conll04", data_root, source_root, output_root)
            paths = (
                output_root / "conll04_validation.jsonl",
                output_root / "conll04_validation.character_span_metrics.json",
                output_root / "conll04_validation.lineage.json",
            )
            before = {path: path.read_bytes() for path in paths}
            source_path = source_root / "conll04_validation.jsonl"
            source = read_jsonl(source_path)
            source[0]["annotation"]["entities"][0]["confidence"] = 0.25
            write_jsonl(source_path, source)

            evaluator = self.script.evaluate_strict_spans

            def fail_evaluation(*_args, **_kwargs):
                raise RuntimeError("injected evaluator failure")

            self.script.evaluate_strict_spans = fail_evaluation
            try:
                with self.assertRaisesRegex(RuntimeError, "injected evaluator failure"):
                    self.script.derive_dataset(
                        "conll04", data_root, source_root, output_root, overwrite=True
                    )
            finally:
                self.script.evaluate_strict_spans = evaluator
            self.assertEqual({path: path.read_bytes() for path in paths}, before)

    def test_same_source_complete_derivation_is_idempotent_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root, source_root, output_root, _ = fixture(root)
            first = self.script.derive_all(
                data_root, source_root, output_root, datasets=("conll04",)
            )
            artifact_paths = (
                output_root / "conll04_validation.jsonl",
                output_root / "conll04_validation.character_span_metrics.json",
                output_root / "conll04_validation.lineage.json",
            )
            before = {path: path.read_bytes() for path in artifact_paths}
            second = self.script.derive_all(
                data_root, source_root, output_root, datasets=("conll04",)
            )
            persisted = json.loads((output_root / "status.json").read_text(encoding="utf-8"))
            self.assertEqual({path: path.read_bytes() for path in artifact_paths}, before)
            self.assertEqual(first["status"], "complete")
            self.assertEqual(second["status"], "complete")
            self.assertEqual(persisted["status"], "complete")
            expected_hash = hashlib.sha256(artifact_paths[-1].read_bytes()).hexdigest()
            self.assertEqual(
                persisted["datasets"]["conll04"]["lineage_sha256"], expected_hash
            )

    def test_partial_derivation_requires_explicit_overwrite_and_is_repaired(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root, source_root, output_root, _ = fixture(root)
            self.script.derive_dataset("conll04", data_root, source_root, output_root)
            lineage = output_root / "conll04_validation.lineage.json"
            lineage.unlink()
            with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                self.script.derive_dataset("conll04", data_root, source_root, output_root)
            self.script.derive_dataset(
                "conll04", data_root, source_root, output_root, overwrite=True
            )
            self.assertTrue(lineage.is_file())

    def test_aggregate_status_covers_requested_datasets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "public"
            source_root = root / "combined"
            output_root = root / "entity-only"
            for dataset in self.script.DATASETS:
                fixture_root = root / f"fixture-{dataset}"
                fixture_data, fixture_source, _, _ = fixture(fixture_root, dataset)
                dataset_source = fixture_source / f"{dataset}_validation.jsonl"
                target_source = source_root / dataset_source.name
                target_source.parent.mkdir(parents=True, exist_ok=True)
                target_source.write_bytes(dataset_source.read_bytes())
                target_dataset = data_root / dataset
                target_dataset.mkdir(parents=True, exist_ok=True)
                for name in (
                    "validation_baseline_jobs.jsonl",
                    "validation_gold.jsonl",
                    "validation_index.jsonl",
                ):
                    (target_dataset / name).write_bytes((fixture_data / dataset / name).read_bytes())

            status = self.script.derive_all(data_root, source_root, output_root)
            persisted = json.loads((output_root / "status.json").read_text(encoding="utf-8"))

        self.assertEqual(status["status"], "complete")
        self.assertEqual(set(status["datasets"]), set(self.script.DATASETS))
        self.assertEqual(persisted["status"], "complete")
        self.assertTrue(persisted["execution"]["postprocessing_only"])
        self.assertFalse(persisted["execution"]["gpu_used"])


if __name__ == "__main__":
    unittest.main()
