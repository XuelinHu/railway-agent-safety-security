import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_gliner_glirel_validation.py"


def load_script():
    spec = importlib.util.spec_from_file_location("run_gliner_glirel_validation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


class FakeGLiNER:
    def __init__(self):
        self.calls = []

    def predict_entities(self, text, labels, flat_ner, threshold):
        self.calls.append((text, labels, flat_ner, threshold))
        return [
            {"text": "Alice", "start": 0, "end": 5, "label": "Peop", "score": 0.95},
            {"text": "Alice", "start": 0, "end": 5, "label": "Org", "score": 0.10},
            {"text": "Acme", "start": 15, "end": 19, "label": "Org", "score": 0.90},
            {"text": "works", "start": 6, "end": 11, "label": "NotInOntology", "score": 0.99},
        ]


class FakeGLiREL:
    def __init__(self):
        self.calls = []

    def predict_relations(self, tokens, labels, flat_ner, threshold, ner, top_k):
        self.calls.append((tokens, labels, flat_ner, threshold, ner, top_k))
        return [
            {"head_pos": [0, 1], "tail_pos": [3, 4], "label": "Work_For", "score": 0.83},
            {"head_pos": [0, 1], "tail_pos": [3, 4], "label": "Work_For", "score": 0.71},
            {"head_pos": [3, 4], "tail_pos": [0, 1], "label": "Work_For", "score": 0.99},
            {"head_pos": [0, 1], "tail_pos": [3, 4], "label": "Unknown", "score": 0.99},
        ]


class FakeNaturalizedGLiREL(FakeGLiREL):
    def predict_relations(self, tokens, labels, flat_ner, threshold, ner, top_k):
        self.calls.append((tokens, labels, flat_ner, threshold, ner, top_k))
        return [
            {"head_pos": [0, 1], "tail_pos": [3, 4], "label": "work for", "score": 0.83}
        ]


def ontology():
    return {
        "annotation_schema_version": "0.1.0",
        "entity_types": {"Peop": {}, "Org": {}, "Loc": {}},
        "relation_types": {"Work_For": {}, "OrgBased_In": {}},
        "claim_statuses": {"explicit": "direct"},
        "allowed_relation_signatures": {
            "Work_For": {"source": ["Peop"], "target": ["Org"]},
            "OrgBased_In": {"source": ["Org"], "target": ["Loc"]},
        },
    }


def job():
    return {
        "job_id": "conll04_validation_demo_C1",
        "document_id": "conll04_validation_demo",
        "language": "en",
        "category": "conll04",
        "source_path": "public:conll04:validation:demo",
        "experiment_mode": "baseline",
        "segments": [
            {
                "segment_id": "S1",
                "page": None,
                "start": 0,
                "end": 21,
                "text": "Alice works at Acme .",
            }
        ],
        "ontology": ontology(),
    }


class GLiNERGLiRELValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_script()

    def test_mock_end_to_end_uses_only_gliner_entities_and_enforces_signature(self):
        gliner = FakeGLiNER()
        glirel = FakeGLiREL()
        prediction, diagnostics = self.runner.predict_job(
            job(), gliner, glirel, entity_threshold=0.5, relation_threshold=0.5, top_k=1
        )

        annotation = prediction["annotation"]
        self.assertEqual([entity["text"] for entity in annotation["entities"]], ["Alice", "Acme"])
        self.assertEqual([entity["type"] for entity in annotation["entities"]], ["Peop", "Org"])
        self.assertEqual(len(annotation["relations"]), 1)
        relation = annotation["relations"][0]
        self.assertEqual(
            (relation["source_id"], relation["type"], relation["target_id"]),
            ("E1", "Work_For", "E2"),
        )
        self.assertEqual(relation["claim_status"], "explicit")
        self.assertEqual(relation["evidence"][0]["text"], "Alice works at Acme .")
        self.assertEqual(diagnostics["dropped_relation_signature"], 1)
        self.assertEqual(glirel.calls[0][1], ["Work_For"])
        self.assertEqual(glirel.calls[0][4], [[0, 0, "Peop", "Alice"], [3, 3, "Org", "Acme"]])
        self.assertEqual(glirel.calls[0][5], -1)

    def test_output_wrapper_is_accepted_by_project_evaluator_unpacker(self):
        prediction, _ = self.runner.predict_job(
            job(), FakeGLiNER(), FakeGLiREL(), 0.5, 0.5, 1
        )
        evaluator_path = ROOT / "scripts" / "evaluate_annotations.py"
        spec = importlib.util.spec_from_file_location("evaluate_annotations_for_glirel_test", evaluator_path)
        evaluator = importlib.util.module_from_spec(spec)
        assert spec.loader
        spec.loader.exec_module(evaluator)
        unpacked = evaluator.unpack([prediction], None)
        self.assertEqual(set(unpacked), {"conll04_validation_demo_C1"})
        self.assertEqual(evaluator.entity_key(unpacked[prediction["job_id"]]["entities"][0]), ("alice", "Peop"))

    def test_naturalized_prompt_labels_map_back_to_canonical_output(self):
        glirel = FakeNaturalizedGLiREL()
        prediction, _ = self.runner.predict_job(
            job(),
            FakeGLiNER(),
            glirel,
            entity_threshold=0.5,
            relation_threshold=0.0,
            top_k=1,
            relation_label_mode="naturalized",
        )

        self.assertEqual(glirel.calls[0][1], ["work for"])
        self.assertEqual(prediction["annotation"]["relations"][0]["type"], "Work_For")

    def test_naturalized_aliases_cover_each_supported_dataset_ontology(self):
        for dataset in self.runner.SUPPORTED_DATASETS:
            jobs = (
                ROOT
                / "data"
                / "processed"
                / "public_benchmarks_full"
                / dataset
                / "validation_baseline_jobs.jsonl"
            )
            first = json.loads(jobs.read_text(encoding="utf-8").splitlines()[0])
            canonical = list(first["ontology"]["relation_types"])
            prompts, reverse = self.runner.relation_prompt_labels(
                dataset, canonical, "naturalized"
            )
            self.assertEqual(len(prompts), len(canonical))
            self.assertEqual({reverse[prompt] for prompt in prompts}, set(canonical))

    def test_loader_accepts_two_real_validation_jobs_for_each_dataset(self):
        for dataset in self.runner.SUPPORTED_DATASETS:
            with self.subTest(dataset=dataset):
                path = (
                    ROOT
                    / "data"
                    / "processed"
                    / "public_benchmarks_full"
                    / dataset
                    / "validation_baseline_jobs.jsonl"
                )
                jobs, embedded, digest = self.runner.load_validation_jobs(path, dataset, limit=2)
                self.assertEqual(len(jobs), 2)
                self.assertEqual(digest, self.runner.canonical_digest(embedded))
                self.assertTrue(all(not (set(row) & self.runner.FORBIDDEN_JOB_KEYS) for row in jobs))

    def test_loader_rejects_wrong_filename_and_gold_like_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wrong = root / "jobs.jsonl"
            wrong.write_text(json.dumps(job()) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "validation_baseline_jobs"):
                self.runner.load_validation_jobs(wrong, "conll04")

            guarded = root / "validation_baseline_jobs.jsonl"
            leaked = job()
            leaked["entities"] = [{"text": "gold leak"}]
            guarded.write_text(json.dumps(leaked) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "forbidden"):
                self.runner.load_validation_jobs(guarded, "conll04")

    def test_loader_rejects_nonbaseline_and_nonvalidation_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation_baseline_jobs.jsonl"
            invalid = job()
            invalid["experiment_mode"] = "kg_constrained"
            path.write_text(json.dumps(invalid) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "experiment_mode"):
                self.runner.load_validation_jobs(path, "conll04")

            invalid = job()
            invalid["source_path"] = "public:conll04:test:demo"
            path.write_text(json.dumps(invalid) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "validation split"):
                self.runner.load_validation_jobs(path, "conll04")

    def test_local_cache_resolution_never_contacts_hub(self):
        with tempfile.TemporaryDirectory() as directory:
            hf_home = Path(directory)
            repository = hf_home / "hub" / "models--org--model"
            snapshot = repository / "snapshots" / "abc123"
            snapshot.mkdir(parents=True)
            (repository / "refs").mkdir()
            (repository / "refs" / "main").write_text("abc123\n", encoding="utf-8")
            self.assertEqual(
                self.runner.resolve_local_model("org/model", hf_home), snapshot.resolve()
            )
            with self.assertRaises(FileNotFoundError):
                self.runner.resolve_local_model("org/missing", hf_home)

    def test_token_mapping_requires_exact_boundaries(self):
        tokens, offsets = self.runner.tokenize_with_offsets("Alice works at Acme .")
        self.assertEqual(tokens, ["Alice", "works", "at", "Acme", "."])
        self.assertEqual(self.runner.token_span_for_chars(offsets, 0, 5), (0, 1))
        self.assertEqual(self.runner.token_span_for_chars(offsets, 15, 19), (3, 4))
        self.assertIsNone(self.runner.token_span_for_chars(offsets, 1, 5))


if __name__ == "__main__":
    unittest.main()
