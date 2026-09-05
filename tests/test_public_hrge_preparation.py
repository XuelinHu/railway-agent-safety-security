import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load_script(name):
    path = SCRIPTS / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def mention(document_id, split, entity_id, text, entity_type, concept_id):
    job_id = f"{document_id}_C1"
    return {
        "mention_id": f"{job_id}:{entity_id}",
        "concept_id": concept_id,
        "document_id": document_id,
        "job_id": job_id,
        "split": split,
        "source_id": entity_id,
        "text": text,
        "type": entity_type,
        "normalized_name": None,
        "confidence": 1.0,
        "evidence": {"text": text, "segment_id": "S1", "start": 0, "end": len(text)},
    }


class PublicTrainGraphTest(unittest.TestCase):
    def setUp(self):
        self.converter = load_script("convert_public_train_graph.py")
        self.manifest = [
            {"document_id": "T1", "split": "train"},
            {"document_id": "T2", "split": "train"},
            {"document_id": "V1", "split": "validation"},
            {"document_id": "X1", "split": "test"},
        ]
        self.mentions = [
            mention("T1", "train", "E1", "Alice", "Peop", "C_ALICE"),
            mention("T1", "train", "E2", "Acme", "Org", "C_ACME"),
            mention("T2", "train", "E1", "Alice", "Peop", "C_ALICE"),
            mention("V1", "validation", "E1", "Validation Secret", "Peop", "C_VAL"),
            mention("X1", "test", "E1", "Test Secret", "Peop", "C_TEST"),
        ]
        self.edge = {
            "document_id": "T1",
            "source_text": "Alice",
            "source_type": "Peop",
            "relation_type": "Work_For",
            "target_text": "Acme",
            "target_type": "Org",
            "evidence": {
                "text": "Alice works at Acme.",
                "segment_id": "S1",
                "start": 0,
                "end": 20,
            },
            "provenance_split": "train",
        }

    def test_conversion_physically_excludes_validation_and_test_mentions(self):
        concepts, mentions, relations, audit = self.converter.build_train_graph(
            "conll04", self.mentions, [self.edge], self.manifest
        )
        self.assertEqual({row["split"] for row in mentions}, {"train"})
        self.assertEqual({row["concept_id"] for row in concepts}, {"C_ALICE", "C_ACME"})
        self.assertEqual(audit["excluded_mentions_by_split"], {"test": 1, "validation": 1})
        self.assertEqual(relations[0]["source_concept_id"], "C_ALICE")
        self.assertEqual(relations[0]["target_concept_id"], "C_ACME")
        self.assertEqual(relations[0]["split"], "train")
        self.assertEqual(relations[0]["claim_status"], "explicit")
        self.assertEqual(relations[0]["evidence"], [self.edge["evidence"]])

    def test_conversion_rejects_an_edge_from_a_non_training_document(self):
        bad_edge = {
            **self.edge,
            "document_id": "V1",
            "source_text": "Validation Secret",
            "target_text": "Validation Secret",
            "source_type": "Peop",
            "target_type": "Peop",
        }
        with self.assertRaisesRegex(ValueError, "non-training document"):
            self.converter.build_train_graph(
                "conll04", self.mentions, [bad_edge], self.manifest
            )


class PublicHrgePreparationTest(unittest.TestCase):
    def test_preparation_is_idempotent_and_never_opens_test_jobs(self):
        runner = load_script("prepare_public_hrge_cpu.py")
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            source_root = temporary / "source"
            dataset_root = source_root / "conll04"
            graph_root = dataset_root / "knowledge_graph"
            output_root = temporary / "prepared"
            manifests = [
                {"document_id": "T1", "split": "train"},
                {"document_id": "T2", "split": "train"},
                {"document_id": "V1", "split": "validation"},
                {"document_id": "X1", "split": "test"},
            ]
            source_mentions = [
                mention("T1", "train", "E1", "Alice", "Peop", "C_ALICE"),
                mention("T1", "train", "E2", "Acme", "Org", "C_ACME"),
                mention("T2", "train", "E1", "Alice", "Peop", "C_ALICE"),
                mention("V1", "validation", "E1", "Validation Secret", "Peop", "C_VAL"),
                mention("X1", "test", "E1", "Test Secret", "Peop", "C_TEST"),
            ]
            edge = {
                "document_id": "T1",
                "source_text": "Alice",
                "source_type": "Peop",
                "relation_type": "Work_For",
                "target_text": "Acme",
                "target_type": "Org",
                "evidence": {
                    "text": "Alice works at Acme.",
                    "segment_id": "S1",
                    "start": 0,
                    "end": 20,
                },
                "provenance_split": "train",
            }

            def job(document_id, split, text):
                return {
                    "job_id": f"{document_id}_C1",
                    "document_id": document_id,
                    "language": "en",
                    "source_path": f"public:conll04:{split}:{document_id}",
                    "system_instruction": "Extract exact source spans.",
                    "segments": [{"segment_id": "S1", "text": text, "start": 0}],
                    "experiment_mode": "baseline",
                }

            train_jobs = [
                job("T1", "train", "Alice works at Acme."),
                job("T2", "train", "Alice arrived."),
            ]
            validation_jobs = [job("V1", "validation", "Alice works at Acme.")]
            write_jsonl(dataset_root / "train_baseline_jobs.jsonl", train_jobs)
            write_jsonl(dataset_root / "validation_baseline_jobs.jsonl", validation_jobs)
            write_jsonl(dataset_root / "split_manifest.jsonl", manifests)
            write_jsonl(graph_root / "mentions.jsonl", source_mentions)
            write_jsonl(graph_root / "training_edges.jsonl", [edge])
            ontology = {
                "entity_types": {"Peop": {}, "Org": {}},
                "relation_types": {"Work_For": {}},
                "claim_statuses": {"explicit": {}},
                "allowed_relation_signatures": {
                    "Work_For": {"source": ["Peop"], "target": ["Org"]}
                },
            }
            (dataset_root / "ontology.yaml").write_text(
                yaml.safe_dump(ontology, sort_keys=False), encoding="utf-8"
            )

            first = runner.prepare_dataset(
                "conll04", source_root, output_root, None, batch_size=2
            )
            second = runner.prepare_dataset(
                "conll04", source_root, output_root, None, batch_size=2
            )
            self.assertEqual(first["status"], "prepared_train_and_validation")
            self.assertEqual(second["status"], "skipped_unchanged")

            target = output_root / "conll04"
            train_only_mentions = (target / "knowledge_graph" / "mentions.jsonl").read_text(
                encoding="utf-8"
            )
            self.assertNotIn("Validation Secret", train_only_mentions)
            self.assertNotIn("Test Secret", train_only_mentions)
            for name in (
                "train_eae_jobs.jsonl",
                "validation_eae_jobs.jsonl",
                "train_hrge_jobs.jsonl",
                "validation_hrge_jobs.jsonl",
            ):
                text = (target / "jobs" / name).read_text(encoding="utf-8")
                self.assertNotIn("Test Secret", text)
                self.assertNotIn("X1_C1", text)

            hrge = runner.load_jsonl(target / "jobs" / "validation_hrge_jobs.jsonl")
            self.assertEqual(hrge[0]["method_name"], "HRGE")
            self.assertTrue(hrge[0]["kg_v2_context"]["train_graph_only"])
            self.assertIn(
                "exact_source_overlap_quarantine", hrge[0]["kg_v2_context"]
            )
            self.assertEqual(hrge[0]["kg_v2_context"]["anchors"], [])
            self.assertEqual(hrge[0]["kg_v2_context"]["edge_priors"], [])
            self.assertEqual(
                hrge[0]["kg_v2_context"]["semantic_relation_patterns"], []
            )
            eae = runner.load_jsonl(target / "jobs" / "validation_eae_jobs.jsonl")
            self.assertIn("graph_context_quarantine", eae[0])
            self.assertNotIn("KG_RULES:", eae[0]["system_instruction"])
            manifest = json.loads(
                (target / "preparation_manifest.json").read_text(encoding="utf-8")
            )
            self.assertFalse(manifest["test_job_file_read"])
            self.assertFalse(manifest["test_gold_read"])
            self.assertEqual(
                manifest["pge_contract"]["required_prediction_streams"], ["EAE", "HRGE"]
            )
            self.assertEqual(
                manifest["exact_source_overlap_audit"]["quarantined_validation_jobs"], 1
            )


if __name__ == "__main__":
    unittest.main()
