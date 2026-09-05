import importlib.util
import hashlib
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


def load_script(name: str):
    path = SCRIPTS / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def inventory_row(document_id, cluster_id, language="en", category="collision"):
    return {
        "document_id": document_id,
        "cluster_id": cluster_id,
        "language": language,
        "source_group": "raib" if language == "en" else "cn",
        "category": category,
        "record_count": 1,
        "window_job_count": 1,
        "entity_count": 1,
        "relation_count": 1,
        "entity_types": ["EVENT"],
        "relation_types": ["precedes"],
    }


class LowResourceProtocolTest(unittest.TestCase):
    def test_train_only_guard_rejects_validation_and_test_paths(self):
        builder = load_script("build_low_resource_manifests.py")
        builder.require_train_only_path(Path("formal_split/train.jsonl"), "train")
        with self.assertRaises(ValueError):
            builder.require_train_only_path(
                Path("formal_split/validation.jsonl"), "annotations"
            )
        with self.assertRaises(ValueError):
            builder.require_train_only_path(Path("formal_split/test.jsonl"), "annotations")

    def test_nested_selection_is_deterministic_exact_and_cluster_closed(self):
        builder = load_script("build_low_resource_manifests.py")
        inventory = [
            inventory_row("D1", "C1", "en"),
            inventory_row("D2", "C1", "en"),
            inventory_row("D3", "C2", "zh", "fire"),
            inventory_row("D4", "C3", "en", "derailment"),
            inventory_row("D5", "C4", "zh", "management"),
            inventory_row("D6", "C5", "en", "collision"),
            inventory_row("D7", "C6", "zh", "construction"),
            inventory_row("D8", "C7", "zh", "fire"),
        ]
        first = builder.select_nested_groups(inventory, [3, 5, 8], "fixed")
        second = builder.select_nested_groups(inventory, [3, 5, 8], "fixed")
        self.assertEqual(first, second)
        previous = set()
        for budget in (3, 5, 8):
            selected = {row["document_id"] for row in first[budget]}
            self.assertEqual(len(selected), budget)
            self.assertTrue(previous <= selected)
            self.assertEqual(
                builder.cluster_boundary_violations(selected, inventory), []
            )
            previous = selected

    def test_inventory_rejects_any_non_train_index_row(self):
        builder = load_script("build_low_resource_manifests.py")
        annotations = [
            {
                "document_id": "D1",
                "language": "en",
                "entities": [],
                "relations": [],
            }
        ]
        index = [
            {"document_id": "D1", "job_id": "J1", "record_index": 0, "split": "test"}
        ]
        with self.assertRaises(ValueError):
            builder.build_inventory(annotations, index, [], {})

    def test_subset_concepts_remove_unselected_document_provenance(self):
        assets = load_script("build_low_resource_assets.py")
        assets.require_no_test_path(
            Path("windowed_validation_v2/baseline_jobs.jsonl"), "validation"
        )
        with self.assertRaises(ValueError):
            assets.require_no_test_path(
                Path("windowed_test_v1/baseline_jobs.jsonl"), "validation"
            )
        concepts = [
            {
                "concept_id": "C1",
                "canonical_name": "collision",
                "language": "en",
                "type": "EVENT",
                "mention_count": 2,
                "source_documents": ["D1", "D2"],
            }
        ]
        mentions = [
            {"concept_id": "C1", "document_id": "D1"},
        ]
        subset = assets.subset_concepts(concepts, mentions)
        self.assertEqual(subset[0]["source_documents"], ["D1"])
        self.assertEqual(subset[0]["mention_count"], 1)

    def test_agreement_counts_use_span_and_document_units(self):
        evaluator = load_script("evaluate_annotation_agreement.py")
        s1, s2, s3, s4 = (0, 4), (10, 14), (20, 24), (30, 34)

        def review(spans, relation=True):
            entity_types = {span: {"EVENT"} for span in spans}
            entity_evidence = {span: {(("S1", span[0], span[1], "text"),)} for span in spans}
            relations = {(s1, s2, "precedes")} if relation else set()
            return {
                "entity_types": entity_types,
                "entity_evidence": entity_evidence,
                "relations": relations,
                "relation_claims": {(s1, s2, "precedes"): {"explicit"}},
                "relation_evidence": {
                    (s1, s2, "precedes"): {(('S1', 0, 14, 'evidence'),)}
                },
            }

        left = review({s1, s2, s3})
        right = review({s1, s2, s4})
        counts, disagreements = evaluator.compare_document("D1", left, right)
        self.assertEqual(
            counts["exact_entity_span_agreement"],
            {"numerator": 4, "denominator": 6, "left": 3, "right": 3, "matched": 2},
        )
        self.assertEqual(
            counts["relation_agreement_conditional_on_matched_endpoints"]["numerator"],
            2,
        )
        self.assertTrue(any(row["kind"] == "entity_presence" for row in disagreements))

    def test_reviewer_template_is_label_free_and_pending(self):
        manifest = load_script("build_annotation_agreement_manifest.py")
        evaluator = load_script("evaluate_annotation_agreement.py")
        row = manifest.reviewer_template(
            {"job_id": "J1", "document_id": "D1", "language": "en"},
            "reviewer_a",
        )
        self.assertEqual(row["review_status"], "pending_independent_review")
        self.assertEqual(row["annotation"]["entities"], [])
        self.assertEqual(row["annotation"]["relations"], [])
        with self.assertRaises(ValueError):
            evaluator.collect_review(
                [row],
                {"J1": {"job_id": "J1", "document_id": "D1", "segments": []}},
            )

    def test_machine_protocol_freezes_full_matrix_and_test_gate(self):
        protocol = yaml.safe_load(
            (ROOT / "configs/low_resource_protocol_v1.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(protocol["data"]["budgets_documents"], [10, 25, 50, 100])
        self.assertEqual(protocol["seeds"], [20260830, 20260831, 20260901])
        self.assertEqual(protocol["execution"]["trainable_runs"], 36)
        self.assertEqual(protocol["formal_test_status"], "sealed")
        self.assertTrue(
            protocol["execution"]["formal_test_requires_explicit_user_confirmation"]
        )
        for relative_path, expected in protocol["implementation_sha256"].items():
            actual = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, relative_path)

    def test_v2_protocol_freezes_semantic_windows_and_memory_gate(self):
        protocol = yaml.safe_load(
            (ROOT / "configs/low_resource_protocol_v2.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(protocol["amended_from"], "low-resource-provenance-study-v1")
        self.assertEqual(protocol["training"]["max_length"], 4096)
        self.assertFalse(protocol["training"]["skip_overlength"])
        self.assertEqual(
            protocol["runtime_environment"]["PYTORCH_CUDA_ALLOC_CONF"],
            "expandable_segments:True",
        )
        self.assertEqual(protocol["windowing"]["entity_coverage_required"], 1.0)
        self.assertEqual(protocol["windowing"]["relation_coverage_required"], 1.0)
        self.assertEqual(protocol["frozen_d100_assets"]["over_4096"], {
            "baseline": 0,
            "kg_v1": 0,
            "kg_v2": 0,
        })
        for relative_path, expected in protocol["implementation_sha256"].items():
            actual = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, relative_path)

        matrix = [
            json.loads(line)
            for line in (
                ROOT
                / "data/processed/experiments/formal/low_resource_manifests_v2/run_matrix.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(matrix), 36)
        self.assertEqual(len({row["run_id"] for row in matrix}), 36)
        self.assertTrue(all(row["run_id"].startswith("lr_v2_") for row in matrix))
        self.assertTrue(
            all("low_resource_v2" in row["output_directory"] for row in matrix)
        )

    def test_training_gate_classifies_pre_model_and_oom_failures(self):
        audit = load_script("audit_low_resource_training_gate.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "return_code": 1,
                        "attempt_id": "A1",
                    }
                ),
                encoding="utf-8",
            )
            (root / "training.log").write_text(
                "RuntimeError: Invalid device argument",
                encoding="utf-8",
            )
            result = audit.classify_attempt(root)
            self.assertEqual(
                result["failure_class"],
                "telemetry_incompatibility_before_model_output",
            )
            self.assertFalse(result["terminal_under_frozen_retry_policy"])

            (root / "training.log").write_text(
                "prepared_examples\ntorch.OutOfMemoryError: CUDA out of memory",
                encoding="utf-8",
            )
            result = audit.classify_attempt(root)
            self.assertEqual(
                result["failure_class"],
                "cuda_out_of_memory_after_training_started",
            )
            self.assertTrue(result["terminal_under_frozen_retry_policy"])

    def test_semantic_windowing_protects_entities_and_rescues_relations(self):
        builder = load_script("build_semantic_window_dataset.py")
        annotation = {
            "schema_version": "0.1.0",
            "document_id": "D1",
            "language": "en",
            "entities": [
                {
                    "id": "E1",
                    "text": "signal failure",
                    "type": "FAILURE",
                    "evidence": {"segment_id": "S1"},
                },
                {
                    "id": "E2",
                    "text": "train collision",
                    "type": "EVENT",
                    "evidence": {"segment_id": "S3"},
                },
            ],
            "relations": [
                {
                    "id": "R1",
                    "source_id": "E1",
                    "type": "causes",
                    "target_id": "E2",
                    "claim_status": "explicit",
                    "evidence": [{"segment_id": "S3", "text": "train collision"}],
                }
            ],
        }
        segments = [
            {"segment_id": "S1", "text": "Context. signal failure happened."},
            {"segment_id": "S2", "text": "Unrelated middle context. " * 20},
            {"segment_id": "S3", "text": "Later, a train collision occurred."},
        ]
        split = builder.split_semantic_segments(segments, annotation, 80, 16)
        self.assertTrue(
            any("signal failure" in row["text"] for row in split)
        )

        class CharacterTokenizer:
            def apply_chat_template(self, rows, **kwargs):
                return "".join(row["content"] for row in rows)

            def __call__(self, text, **kwargs):
                return {"input_ids": list(text)}

        base = {
            "job_id": "J1",
            "document_id": "D1",
            "language": "en",
            "teacher_model": "test",
            "system_instruction": "base",
            "ontology": {},
        }
        kg = {**base, "system_instruction": "base\n\nKG_RULES: local evidence"}
        selected, focused, compact, _ = builder.relation_rescue_segments(
            CharacterTokenizer(),
            base,
            kg,
            annotation,
            annotation["relations"][0],
            split,
            2500,
        )
        self.assertEqual([row["id"] for row in compact["relations"]], ["R1"])
        self.assertGreater(selected[-1] - selected[0], 1)
        self.assertTrue(all(row["text"] for row in focused))


if __name__ == "__main__":
    unittest.main()
