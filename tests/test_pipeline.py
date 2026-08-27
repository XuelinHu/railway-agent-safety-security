import importlib.util
import json
import unittest
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


class PipelineTest(unittest.TestCase):
    def test_schema_matches_ontology(self):
        ontology = yaml.safe_load((ROOT / "configs/risk_ontology.yaml").read_text(encoding="utf-8"))
        schema = json.loads((ROOT / "schemas/risk_annotation.schema.json").read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        self.assertEqual(
            set(ontology["entity_types"]),
            set(schema["$defs"]["entity"]["properties"]["type"]["enum"]),
        )
        self.assertEqual(
            set(ontology["relation_types"]),
            set(schema["$defs"]["relation"]["properties"]["type"]["enum"]),
        )
        self.assertEqual(set(ontology["relation_types"]), set(ontology["allowed_relation_signatures"]))
        self.assertEqual(ontology["version"], "1.0.0")
        self.assertEqual(ontology["annotation_schema_version"], schema["properties"]["schema_version"]["const"])

    def test_segments_preserve_global_offsets(self):
        corpus = load_script("build_corpus.py")
        text, segments = corpus.make_segments(
            [("page", 1, "first page"), ("page", 2, "second page")]
        )
        for segment in segments:
            self.assertEqual(text[segment["start"] : segment["end"]], segment["text"])

    def test_flexible_span_only_normalizes_whitespace(self):
        normalizer = load_script("normalize_preannotations.py")
        source = "Collision at Shalesmoor,\nSheffield"
        start, end = normalizer.flexible_span(source, "Collision at Shalesmoor, Sheffield")
        self.assertEqual(source[start:end], source)
        with self.assertRaises(ValueError):
            normalizer.flexible_span(source, "Collision near Shalesmoor, Sheffield")

    def test_unique_entity_evidence_repair_rejects_ambiguity(self):
        normalizer = load_script("normalize_preannotations.py")
        self.assertEqual(normalizer.matching_spans("source", ""), [])
        segments = {
            "S1": {"segment_id": "S1", "text": "Signal failure occurred.", "start": 10, "page": 1},
            "S2": {"segment_id": "S2", "text": "The driver stopped.", "start": 40, "page": 2},
        }
        evidence = normalizer.locate_unique_entity_evidence("Signal failure", segments)
        self.assertEqual(evidence["segment_id"], "S1")
        self.assertEqual((evidence["start"], evidence["end"]), (10, 24))
        segments["S2"]["text"] = "A second Signal failure was recorded."
        with self.assertRaises(ValueError):
            normalizer.locate_unique_entity_evidence("Signal failure", segments)

    def test_inverse_relation_repair_requires_legal_swap(self):
        normalizer = load_script("normalize_preannotations.py")
        entities = {"E1": {"type": "ACTOR"}, "E2": {"type": "OPERATION"}}
        signatures = {"performed_by": {"source": ["OPERATION"], "target": ["ACTOR"]}}
        relation = {"id": "R1", "type": "performed_by", "source_id": "E1", "target_id": "E2"}
        repaired, message = normalizer.constrain_relation_direction(
            relation, entities, signatures, repair_inverse=True
        )
        self.assertEqual((repaired["source_id"], repaired["target_id"]), ("E2", "E1"))
        self.assertIn("swapped direction", message)
        rejected, _ = normalizer.constrain_relation_direction(
            relation, entities, signatures, repair_inverse=False
        )
        self.assertIsNone(rejected)
        unknown = {**relation, "type": "unknown_relation"}
        rejected, _ = normalizer.constrain_relation_direction(
            unknown, entities, signatures, repair_inverse=True
        )
        self.assertIsNone(rejected)

    def test_representative_chunks_keep_document_coverage(self):
        preparer = load_script("prepare_preannotation_jobs.py")
        self.assertEqual(preparer.representative_chunk_indices(10, 3), [0, 5, 9])
        self.assertEqual(preparer.representative_chunk_indices(2, 3), [0, 1])
        self.assertEqual(preparer.representative_chunk_indices(10, 1), [5])

    def test_qlora_parser_prefers_complete_annotation(self):
        inference = load_script("run_qlora_inference.py")
        parsed = inference.parse_json(
            '{"segment_id":"S1"}\n'
            '{"entities":[{"id":"E1"}],"relations":[],"document_id":"D1"}'
        )
        self.assertIn("entities", parsed)
        self.assertEqual(parsed["document_id"], "D1")
        self.assertTrue(inference.complete_annotation_generated(
            '{"entities": [], "relations": [], "document_id": "D1"}'
        ))
        self.assertFalse(inference.complete_annotation_generated('{"entities": ['))
        self.assertIn("COMPACT OUTPUT MODE", inference.COMPACT_INSTRUCTION)
        truncated = '{"entities": [], "relations": [{"id": "R1"}'
        self.assertEqual(inference.parse_json(truncated)["relations"][0]["id"], "R1")
        self.assertFalse(inference.complete_annotation_generated(truncated))
        missing_array = '{"entities": [{"id": "E1"}], "relations": [{"id": "R1"}}'
        self.assertEqual(inference.parse_json(missing_array)["entities"][0]["id"], "E1")
        missing_entities_array = '{"entities": [{"id": "E1"}, "relations": []}'
        repaired = inference.parse_json(missing_entities_array)
        self.assertEqual(repaired["entities"][0]["id"], "E1")
        self.assertEqual(repaired["relations"], [])
        trainer = load_script("train_qlora.py")
        self.assertIn("COMPACT OUTPUT MODE", trainer.COMPACT_INSTRUCTION)
        self.assertIn("COMPACT OUTPUT MODE", trainer.COMPACT_SYSTEM_INSTRUCTION)
        self.assertNotIn("preannotation_candidate", trainer.COMPACT_SYSTEM_INSTRUCTION)
        inference_system = inference.COMPACT_SYSTEM_INSTRUCTION
        self.assertEqual(inference_system, trainer.COMPACT_SYSTEM_INSTRUCTION)

    def test_window_ranges_preserve_coverage_with_overlap(self):
        windows = load_script("build_inference_windows.py")
        ranges = windows.window_ranges(10, lambda start, end: end - start <= 3, overlap=1)
        self.assertEqual(ranges, [(0, 3), (2, 5), (4, 7), (6, 9), (8, 10)])
        covered = {index for start, end in ranges for index in range(start, end)}
        self.assertEqual(covered, set(range(10)))

    def test_window_prediction_merge_deduplicates_entities_and_relations(self):
        merger = load_script("merge_window_predictions.py")
        rows = [
            {
                "job_id": "D_W001",
                "parent_job_id": "D",
                "window_index": 1,
                "annotation": {
                    "document_id": "D", "language": "zh",
                    "entities": [{"id": "E1", "text": "火灾", "type": "EVENT"}, {"id": "E2", "text": "灭火器", "type": "ASSET"}],
                    "relations": [{"id": "R1", "source_id": "E1", "type": "involves", "target_id": "E2", "claim_status": "explicit"}],
                },
            },
            {
                "job_id": "D_W002",
                "parent_job_id": "D",
                "window_index": 2,
                "annotation": {
                    "document_id": "D", "language": "zh",
                    "entities": [{"id": "E4", "text": "火灾", "type": "EVENT"}, {"id": "E5", "text": "灭火器", "type": "ASSET"}],
                    "relations": [{"id": "R2", "source_id": "E4", "type": "involves", "target_id": "E5", "claim_status": "explicit"}],
                },
            },
        ]
        merged = merger.merge_rows(rows)
        self.assertEqual(len(merged), 1)
        self.assertEqual(len(merged[0]["annotation"]["entities"]), 2)
        self.assertEqual(len(merged[0]["annotation"]["relations"]), 1)

    def test_gguf_runner_prompt_is_compact_and_model_tagged(self):
        runner = load_script("run_gguf_inference.py")
        job = {
            "document_id": "D", "language": "zh", "ontology": {},
            "segments": [{"segment_id": "S1", "text": "火灾", "start": 0}],
        }
        text = runner.prompt(job)
        self.assertIn("qwen3.8-27b-gguf", text)
        self.assertIn("language, entities, relations", text)
        self.assertIn("Do not repeat entities", text)
        normalized = runner.normalize_ids({
            "entities": [{"id": "e7", "text": "火灾", "type": "EVENT"}],
            "relations": [{"id": "r9", "source_id": "e7", "type": "part_of", "target_id": "e7", "claim_status": "explicit"}],
        })
        self.assertEqual(normalized["entities"][0]["id"], "E1")
        self.assertEqual(normalized["relations"][0]["id"], "R1")
        self.assertEqual(normalized["relations"][0]["source_id"], "E1")
        self.assertEqual(normalized["schema_version"], "0.1.0")


if __name__ == "__main__":
    unittest.main()
