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


if __name__ == "__main__":
    unittest.main()
