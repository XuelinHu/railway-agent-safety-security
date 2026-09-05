import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

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

    def test_spert_import_keeps_evidence_local_and_derives_rules_from_train(self):
        importer = load_script("import_spert_benchmarks.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source" / "conll04"
            source.mkdir(parents=True)
            train = [{"orig_id": 1, "tokens": ["Alice", "works", "at", "Acme", "."], "entities": [{"type": "Peop", "start": 0, "end": 1}, {"type": "Org", "start": 3, "end": 4}], "relations": [{"type": "Work_For", "head": 0, "tail": 1}]}]
            dev = [{"orig_id": 2, "tokens": ["Acme", "is", "in", "Paris", "."], "entities": [{"type": "Org", "start": 0, "end": 1}, {"type": "Loc", "start": 3, "end": 4}], "relations": [{"type": "Located_In", "head": 0, "tail": 1}]}]
            test = [{"orig_id": 3, "tokens": ["Bob", "works", "at", "Acme", "."], "entities": [{"type": "Peop", "start": 0, "end": 1}, {"type": "Org", "start": 3, "end": 4}], "relations": [{"type": "Work_For", "head": 0, "tail": 1}]}]
            for name, rows in (("train", train), ("dev", dev), ("test", test)):
                (source / f"conll04_{name}.json").write_text(json.dumps(rows), encoding="utf-8")
            importer.run(SimpleNamespace(dataset="conll04", input_root=root / "source", output_root=root / "output", train_limit=0, validation_limit=0, test_limit=0, seed="test"))
            target = root / "output" / "conll04"
            ontology = yaml.safe_load((target / "ontology.yaml").read_text(encoding="utf-8"))
            self.assertEqual(set(ontology["relation_types"]), {"Work_For"})
            self.assertNotIn("Located_In", ontology["relation_types"])
            gold = [json.loads(line) for line in (target / "gold.jsonl").read_text(encoding="utf-8").splitlines()]
            relation = gold[0]["annotation"]["relations"][0]
            self.assertEqual(relation["evidence"][0]["text"], "Alice works at Acme .")
            self.assertIn("Alice", relation["evidence"][0]["text"])
            self.assertIn("Acme", relation["evidence"][0]["text"])

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
        repeated = ",".join('{"text":"signal failure"}' for _ in range(6))
        self.assertEqual(inference.repeated_entity_text(repeated, 6), "signal failure")
        self.assertIsNone(inference.repeated_entity_text(repeated, 7))
        job = {"system_instruction": "Use retrieved concepts."}
        kg_prompt = inference.system_instruction(job, compact_target=True, use_job_instruction=True)
        self.assertIn("Use retrieved concepts.", kg_prompt)
        self.assertIn("COMPACT OUTPUT MODE", kg_prompt)

    def test_qlora_resume_recovers_terminal_jobs_and_ignores_truncated_lines(self):
        inference = load_script("run_qlora_inference.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "predictions.jsonl"
            log = root / "inference.log"
            output.write_text('{"job_id":"success-from-output"}\n', encoding="utf-8")
            log.write_text(
                '{"job_id":"failed-job","status":"failed"}\n'
                '{"job_id":"success-from-log","status":"success"}\n'
                '{"job_id":',
                encoding="utf-8",
            )
            self.assertEqual(
                inference.completed_job_ids(output, log, retry_failed=False),
                {"success-from-output", "failed-job", "success-from-log"},
            )
            self.assertEqual(
                inference.completed_job_ids(output, log, retry_failed=True),
                {"success-from-output", "success-from-log"},
            )

    def test_compact_expansion_audits_incomplete_relations(self):
        expansion = load_script("expand_compact_predictions.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = root / "jobs.jsonl"
            predictions = root / "predictions.jsonl"
            output = root / "expanded.jsonl"
            errors = root / "errors.jsonl"
            jobs.write_text(
                json.dumps(
                    {
                        "job_id": "D_W001",
                        "document_id": "D",
                        "language": "en",
                        "segments": [
                            {
                                "segment_id": "S1",
                                "text": "Signal failure occurred.",
                                "start": 0,
                                "page": 1,
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            predictions.write_text(
                json.dumps(
                    {
                        "job_id": "D_W001",
                        "annotation": {
                            "entities": [
                                {"id": "E1", "text": "Signal failure", "type": "FAILURE"}
                            ],
                            "relations": [
                                {"id": "R1", "source_id": "E1", "type": "causes"}
                            ],
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            expansion.run(
                SimpleNamespace(
                    jobs=jobs,
                    predictions=predictions,
                    output=output,
                    errors=errors,
                )
            )
            expanded = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(expanded["annotation"]["relations"], [])
            audit = json.loads(errors.read_text(encoding="utf-8"))
            self.assertIn("missing source_id or target_id", audit["errors"][0])

    def test_window_ranges_preserve_coverage_with_overlap(self):
        windows = load_script("build_inference_windows.py")
        ranges = windows.window_ranges(10, lambda start, end: end - start <= 3, overlap=1)
        self.assertEqual(ranges, [(0, 3), (2, 5), (4, 7), (6, 9), (8, 10)])
        covered = {index for start, end in ranges for index in range(start, end)}
        self.assertEqual(covered, set(range(10)))

    def test_v1_v2_fusion_uses_only_auditable_acceptance_signals(self):
        fusion = load_script("fuse_kg_v1_v2_predictions.py")
        v1 = {
            "entities": [{"id": "V1E1", "text": "signal failure", "type": "FAILURE"}],
            "relations": [],
        }
        v2 = {
            "schema_version": "0.1.0",
            "document_id": "D",
            "language": "en",
            "entities": [
                {"id": "E1", "text": "signal failure", "type": "FAILURE"},
                {"id": "E2", "text": "the driver", "type": "ACTOR"},
                {"id": "E3", "text": "the train", "type": "ASSET"},
                {"id": "E4", "text": "unsupported", "type": "HAZARD"},
            ],
            "relations": [
                {"id": "R1", "source_id": "E2", "type": "involves", "target_id": "E3"},
                {"id": "R2", "source_id": "E4", "type": "involves", "target_id": "E3"},
            ],
        }
        verified = {
            **v2,
            "relations": [
                {"id": "R1", "source_id": "E2", "type": "involves", "target_id": "E3"}
            ],
        }
        job = {
            "kg_v2_context": {
                "anchors": [{"text": "the train", "type": "ASSET"}]
            }
        }
        result, audit, counts = fusion.fuse_job(
            "D_W001", v1, v2, verified, job, "raw"
        )
        self.assertEqual(
            {entity["id"] for entity in result["entities"]}, {"E1", "E2", "E3"}
        )
        self.assertEqual([relation["id"] for relation in result["relations"]], ["R1"])
        accepted = {item["entity_id"]: item for item in audit if item["accepted"]}
        self.assertIn("v1_v2_exact_agreement", accepted["E1"]["reasons"])
        self.assertIn("verified_relation_endpoint", accepted["E2"]["reasons"])
        self.assertIn("source_gated_anchor_type_match", accepted["E3"]["reasons"])
        self.assertEqual(counts["entities_rejected"], 1)

    def test_v3_relation_candidates_require_local_legal_pairs(self):
        classifier = load_script("train_v3_relation_classifier.py")
        ontology = yaml.safe_load(
            (ROOT / "configs/risk_ontology.yaml").read_text(encoding="utf-8")
        )
        annotation = {
            "document_id": "D",
            "language": "en",
            "entities": [
                {"id": "E1", "text": "signal failure", "type": "FAILURE"},
                {"id": "E2", "text": "collision", "type": "EVENT"},
                {"id": "E3", "text": "driver", "type": "ACTOR"},
            ],
            "relations": [],
        }
        job = {
            "document_id": "D",
            "language": "en",
            "segments": [
                {
                    "segment_id": "S1",
                    "text": "A signal failure caused the collision.",
                    "start": 0,
                },
                {"segment_id": "S2", "text": "The driver responded.", "start": 40},
            ],
            "kg_v2_context": {
                "edge_priors": [
                    {
                        "source": "signal failure",
                        "relation": "causes",
                        "target": "collision",
                    }
                ],
                "semantic_relation_patterns": [],
            },
        }
        candidates = classifier.enumerate_candidates(
            "D_W001", annotation, job, ontology["allowed_relation_signatures"]
        )
        triples = {
            (
                row["source"]["id"],
                row["relation_type"],
                row["target"]["id"],
            )
            for row in candidates
        }
        self.assertIn(("E1", "causes", "E2"), triples)
        self.assertFalse(any(source == "E3" and target == "E1" for source, _, target in triples))
        causes = next(row for row in candidates if row["relation_type"] == "causes")
        self.assertTrue(causes["edge_prior"])

    def test_frozen_v3_config_locks_rules_and_split(self):
        runner = load_script("run_frozen_kg_v3.py")
        config = {
            "status": "frozen-validation-checkpoint",
            "selection_split": "validation",
            "pipeline": {
                "relation_verifier": {"require_local_cooccurrence": True},
                "entity_gate": {
                    "relation_mode": "verified",
                    "acceptance_any_of": list(runner.FROZEN_ACCEPTANCE_RULES),
                },
            },
        }
        runner.validate_config(config, allow_non_validation=False)
        config["selection_split"] = "test"
        with self.assertRaises(ValueError):
            runner.validate_config(config, allow_non_validation=False)
        config["selection_split"] = "validation"
        config["pipeline"]["entity_gate"]["acceptance_any_of"] = [
            "v1_v2_exact_agreement"
        ]
        with self.assertRaises(ValueError):
            runner.validate_config(config, allow_non_validation=False)

    def test_xlmr_ner_prefers_longer_non_overlapping_mentions(self):
        baseline = load_script("train_xlmr_ner_baseline.py")
        text = "signal failure caused a failure"
        self.assertEqual(
            baseline.find_all_spans(text, "failure"), [(7, 14), (24, 31)]
        )
        selected = baseline.select_non_overlapping(
            [(0, 14, "FAILURE"), (7, 14, "HAZARD"), (22, 31, "FAILURE")]
        )
        self.assertEqual(
            selected, [(0, 14, "FAILURE"), (22, 31, "FAILURE")]
        )
        weights = baseline.class_weights(
            [{"labels": [0] * 9 + [1]}], 3, "binary_sqrt"
        )
        self.assertAlmostEqual(float(weights[0]), 1 / 3)
        self.assertEqual(weights[1:].tolist(), [1.0, 1.0])
        candidates = [
            (0, 7, "ACTOR", 0.9),
            (0, 12, "ACTOR", 0.6),
        ]
        self.assertEqual(
            baseline.consolidate_predictions(candidates, "confidence"),
            [(0, 7, "ACTOR", 0.9)],
        )
        self.assertEqual(
            baseline.consolidate_predictions(candidates, "longest"),
            [(0, 12, "ACTOR", 0.6)],
        )

    def test_xlmr_bio_chunk_repair_converts_interrupted_inside_tags(self):
        baseline = load_script("train_xlmr_ner_baseline.py")
        label_to_id = {"O": 0, "B-ACTOR": 1, "I-ACTOR": 2, "B-EVENT": 3, "I-EVENT": 4}
        labels, count, by_type = baseline.repair_bio_labels(
            [-100, 2, 2, 0, 4, 4, -100], label_to_id
        )
        self.assertEqual(labels, [-100, 1, 2, 0, 3, 4, -100])
        self.assertEqual(count, 2)
        self.assertEqual(by_type, {"ACTOR": 1, "EVENT": 1})

    def test_xlmr_bio_viterbi_repairs_illegal_initial_inside_tag(self):
        import torch

        inference = load_script("run_xlmr_ner_inference.py")
        logits = torch.tensor([[[5.0, 4.0, 0.0], [0.0, 0.0, 6.0]]])
        paths = inference.bio_viterbi_paths(
            logits,
            [[(0, 3), (3, 7)]],
            {0: "O", 1: "B-ACTOR", 2: "I-ACTOR"},
        )
        self.assertEqual(paths.tolist(), [[1, 2]])

    def test_xlmr_learned_crf_masks_special_tokens_and_round_trips(self):
        import torch

        crf_module = load_script("xlmr_crf.py")
        crf = crf_module.LinearChainCRF(3)
        emissions = torch.tensor(
            [[[0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [0.0, 0.0, 0.0]]]
        )
        tags = torch.tensor([[0, 1, 2]])
        mask = torch.tensor([[False, True, True]])
        loss = crf.neg_log_likelihood(emissions, tags, mask)
        self.assertGreater(loss.item(), 0.0)
        path = crf.decode(emissions, mask)
        self.assertEqual(path.shape, (1, 3))
        self.assertEqual(path[0, 0].item(), 0)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "crf.pt"
            crf.save(target)
            restored = crf_module.LinearChainCRF.load(target)
            self.assertTrue(
                all(
                    torch.equal(left, right)
                    for left, right in zip(crf.parameters(), restored.parameters())
                )
            )

    def test_xlmr_span_split_alignment_and_nonoverlap(self):
        span = load_script("train_xlmr_span_ner.py")
        jobs = [
            {"document_id": "D1"},
            {"document_id": "D2"},
            {"document_id": "D3"},
            {"document_id": "D4"},
            {"document_id": "D5"},
        ]
        fit, dev = span.stable_document_split(jobs, 20260830, 0.2)
        self.assertEqual(len(fit & dev), 0)
        self.assertEqual(fit | dev, {f"D{i}" for i in range(1, 6)})
        self.assertEqual(
            span.align_span_to_tokens(
                [(0, 1), (1, 5), (5, 6)], 1, 5, "EVENT", {"EVENT": 1}, 64
            ),
            (1, 1, 1),
        )
        self.assertIsNone(
            span.align_span_to_tokens(
                [(0, 1), (1, 5), (5, 6)], 2, 5, "EVENT", {"EVENT": 1}, 64
            )
        )
        selected = span.consolidate_span_predictions(
            [(0, 10, "EVENT", 0.7), (0, 10, "EVENT", 0.8), (4, 12, "HAZARD", 0.9)]
        )
        self.assertEqual(selected, [(4, 12, "HAZARD", 0.9)])
        metrics = span.micro_span_metrics(
            {"D1": {(0, 2, "EVENT")}, "D2": {(0, 2, "EVENT")}},
            {"D1": {(0, 2, "EVENT")}},
        )
        self.assertEqual((metrics["gold"], metrics["predicted"], metrics["correct"]), (2, 1, 1))
        global_spans = span.document_span_sets(
            [
                {"document_id": "D", "segment": {"start": 0}},
                {"document_id": "D", "segment": {"start": 5}},
            ],
            {
                0: [(4, 10, "EVENT", 0.8)],
                1: [(0, 8, "HAZARD", 0.9)],
            },
        )
        self.assertEqual(global_spans, {"D": {(5, 13, "HAZARD")}})

    def test_xlmr_span_head_round_trips_without_encoder(self):
        import torch
        import torch.nn as nn

        span = load_script("train_xlmr_span_ner.py")

        class TinyConfig:
            hidden_size = 8

        class TinyEncoder(nn.Module):
            config = TinyConfig()

            def forward(self, input_ids, attention_mask):
                hidden = torch.nn.functional.one_hot(input_ids % 8, num_classes=8).float()
                return type("Output", (), {"last_hidden_state": hidden})()

        model = span.SpanBoundaryModel(TinyEncoder(), 2, 4, span_dim=3, dropout=0.0)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "span_head.pt"
            model.save_head(target)
            restored = span.SpanBoundaryModel(TinyEncoder(), 2, 4, span_dim=3, dropout=0.0)
            restored.load_head(target)
            for key in ("start_head.weight", "end_head.bias", "span_classifier.3.weight"):
                self.assertTrue(torch.equal(model.state_dict()[key], restored.state_dict()[key]))

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

    def test_bootstrap_comparison_aggregates_text_blocks_by_document(self):
        comparison = load_script("bootstrap_compare.py")
        gold_rows = [
            {"job_id": "D_C1", "annotation": {"document_id": "D", "language": "zh", "entities": [{"id": "E1", "text": "火灾", "type": "EVENT"}], "relations": []}},
            {"job_id": "D_C2", "annotation": {"document_id": "D", "language": "zh", "entities": [{"id": "E1", "text": "灭火器", "type": "ASSET"}], "relations": []}},
        ]
        baseline_rows = [
            {"job_id": "D_C1", "annotation": {"document_id": "D", "language": "zh", "entities": [{"id": "E1", "text": "火灾", "type": "EVENT"}], "relations": []}},
            {"job_id": "D_C2", "annotation": {"document_id": "D", "language": "zh", "entities": [], "relations": []}},
        ]
        kg_rows = [
            {"job_id": "D_C1", "annotation": {"document_id": "D", "language": "zh", "entities": [], "relations": []}},
            {"job_id": "D_C2", "annotation": {"document_id": "D", "language": "zh", "entities": [{"id": "E1", "text": "灭火器", "type": "ASSET"}], "relations": []}},
        ]
        jobs = [{"job_id": "D_C1"}, {"job_id": "D_C2"}]
        units = comparison.build_units(gold_rows, baseline_rows, kg_rows, jobs)
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0]["baseline"]["entity"]["correct"], 1)
        self.assertEqual(units[0]["kg"]["entity"]["correct"], 1)

    def test_span_bootstrap_aggregates_windows_by_document(self):
        comparison = load_script("bootstrap_span_compare.py")
        item = lambda document, correct: {
            "document_id": document,
            "entity": {"gold": 1, "predicted": 1, "correct": correct},
            "relation": {"gold": 0, "predicted": 0, "correct": 0},
            "relation_with_claim_status": {
                "gold": 0,
                "predicted": 0,
                "correct": 0,
            },
        }
        left = {"per_job": {"D_W1": item("D", 1), "D_W2": item("D", 0)}}
        right = {"per_job": {"D_W1": item("D", 0), "D_W2": item("D", 1)}}
        units = comparison.document_units(left, right)
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0]["left"]["entity"]["gold"], 2)
        self.assertEqual(units[0]["right"]["entity"]["correct"], 1)

    def test_relation_error_analysis_separates_endpoint_and_type_errors(self):
        analyzer = load_script("analyze_relation_errors.py")
        gold = {
            "document_id": "D", "language": "en",
            "entities": [
                {"id": "E1", "text": "signal failure", "type": "FAILURE"},
                {"id": "E2", "text": "derailment", "type": "EVENT"},
            ],
            "relations": [{"id": "R1", "source_id": "E1", "type": "causes", "target_id": "E2"}],
        }
        predicted = {
            "document_id": "D", "language": "en",
            "entities": [
                {"id": "E1", "text": "signal failure", "type": "FAILURE"},
                {"id": "E2", "text": "derailment", "type": "EVENT"},
            ],
            "relations": [{"id": "R1", "source_id": "E1", "type": "involves", "target_id": "E2"}],
        }
        result = analyzer.analyze_job(gold, predicted)
        self.assertEqual(result["gold_relations_with_both_endpoints_predicted"], 1)
        self.assertEqual(result["predicted_relation_categories"]["wrong_relation_type"], 1)
        self.assertEqual(result["missed_relation_categories"]["wrong_relation_type_or_claim"], 1)

    def test_relaxed_matching_distinguishes_type_and_boundary(self):
        evaluator = load_script("evaluate_relaxed_matching.py")
        gold = {"text": "signal failure", "type": "FAILURE"}
        same_text_wrong_type = {"text": "signal failure", "type": "HAZARD"}
        longer_same_type = {"text": "a signal failure", "type": "FAILURE"}
        self.assertFalse(evaluator.entity_matches(same_text_wrong_type, gold, "strict"))
        self.assertTrue(evaluator.entity_matches(same_text_wrong_type, gold, "text"))
        self.assertFalse(evaluator.entity_matches(same_text_wrong_type, gold, "boundary_type"))
        self.assertTrue(evaluator.entity_matches(longer_same_type, gold, "boundary_type"))

    def test_relaxed_matching_removes_only_outer_markup(self):
        evaluator = load_script("evaluate_relaxed_matching.py")
        self.assertEqual(evaluator.strip_outer_markup("《安全生产法》"), "安全生产法")
        self.assertEqual(evaluator.strip_outer_markup("危险与可操作性分析（HAZOP）"), "危险与可操作性分析（hazop）")
        self.assertEqual(evaluator.strip_outer_markup("（主要指粉尘太大）"), "主要指粉尘太大")

    def test_span_aware_matching_preserves_repeated_mentions(self):
        evaluator = load_script("evaluate_span_aware.py")
        job = {
            "segments": [
                {
                    "segment_id": "S1",
                    "text": "火灾后再次发生火灾",
                    "start": 100,
                    "end": 109,
                }
            ]
        }
        first = {
            "id": "E1",
            "text": "火灾",
            "type": "EVENT",
            "evidence": {"segment_id": "S1", "text": "火灾", "start": 100, "end": 102},
        }
        second = {
            "id": "E2",
            "text": "火灾",
            "type": "EVENT",
            "evidence": {"segment_id": "S1", "text": "火灾", "start": 107, "end": 109},
        }
        parent = {"entities": [first, second], "relations": []}
        gold = {
            "language": "zh",
            "document_id": "D",
            "entities": [
                {"id": "E1", "text": "火灾", "type": "EVENT"},
                {"id": "E2", "text": "火灾", "type": "EVENT"},
            ],
            "relations": [],
        }
        predicted = {"entities": [first], "relations": []}
        result = evaluator.analyze_job(gold, predicted, parent, job)
        self.assertEqual(result["entity"]["gold"], 2)
        self.assertEqual(result["entity"]["predicted"], 1)
        self.assertEqual(result["entity"]["correct"], 1)
        self.assertEqual(result["entity"]["recall"], 0.5)
        full_quote_entity_offsets = {
            **first,
            "evidence": {
                "segment_id": "S1",
                "text": "火灾后再次发生火灾",
                "start": 100,
                "end": 102,
            },
        }
        self.assertEqual(
            evaluator.resolve_entity_span(full_quote_entity_offsets, job),
            ((100, 102), "resolved"),
        )

    def test_span_aware_evaluator_refuses_non_validation_by_default(self):
        evaluator = load_script("evaluate_span_aware.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_gold = root / "source.jsonl"
            source_index = root / "source_index.jsonl"
            gold = root / "gold.jsonl"
            gold_index = root / "gold_index.jsonl"
            predictions = root / "predictions.jsonl"
            jobs = root / "jobs.jsonl"
            output = root / "metrics.json"
            source_gold.write_text('{"entities": [], "relations": []}\n', encoding="utf-8")
            source_index.write_text('{"record_index": 0, "job_id": "D_C1"}\n', encoding="utf-8")
            gold.write_text('{"entities": [], "relations": []}\n', encoding="utf-8")
            gold_index.write_text(
                '{"record_index": 0, "job_id": "D_C1_W001", "parent_job_id": "D_C1", "split": "test"}\n',
                encoding="utf-8",
            )
            predictions.write_text("", encoding="utf-8")
            jobs.write_text('{"job_id": "D_C1_W001", "segments": []}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                evaluator.run(
                    SimpleNamespace(
                        source_gold=source_gold,
                        source_gold_index=source_index,
                        gold=gold,
                        gold_index=gold_index,
                        predictions=predictions,
                        jobs=jobs,
                        output=output,
                        allow_non_validation=False,
                    )
                )

    def test_graph_canonical_name_strips_outer_markup_only(self):
        promoter = load_script("promote_reviewed_annotations.py")
        self.assertEqual(promoter.canonical_text("《安全生产法》"), "安全生产法")
        self.assertEqual(
            promoter.canonical_text("危险与可操作性分析（HAZOP）"),
            "危险与可操作性分析（hazop）",
        )

    def test_evidence_graph_metrics_detect_unsupported_relation(self):
        evaluator = load_script("evaluate_evidence_graph.py")
        gold = {
            "language": "zh",
            "entities": [
                {"id": "E1", "text": "火灾", "type": "EVENT"},
                {"id": "E2", "text": "灭火器", "type": "ASSET"},
            ],
            "relations": [{"source_id": "E1", "target_id": "E2", "type": "involves", "claim_status": "explicit"}],
        }
        predicted = {
            **gold,
            "entities": [
                {"id": "E1", "text": "火灾", "type": "EVENT", "evidence": {"segment_id": "S1", "text": "火灾"}},
                {"id": "E2", "text": "灭火器", "type": "ASSET", "evidence": {"segment_id": "S1", "text": "灭火器"}},
            ],
            "relations": [{"source_id": "E1", "target_id": "E2", "type": "involves", "claim_status": "explicit", "evidence": [{"segment_id": "S1", "text": "火灾"}]}],
        }
        result = evaluator.evaluate_job(gold, predicted, {"segments": [{"segment_id": "S1", "text": "火灾涉及灭火器"}]}, {"relation_types": {"involves": {}}, "claim_statuses": {"explicit": {}}, "allowed_relation_signatures": {"involves": {"source": ["EVENT"], "target": ["ASSET"]}}})
        self.assertEqual(result["entity_evidence_coverage"], 1.0)
        self.assertEqual(result["relation_evidence_correctness"], 0.0)
        self.assertEqual(result["unsupported_claim_rate"], 1.0)
        self.assertEqual(result["invalid_relation_rate"], 1.0)

    def test_evidence_graph_aggregate_uses_prediction_level_rates(self):
        evaluator = load_script("evaluate_evidence_graph.py")
        empty = evaluator.evaluate_job(
            {"language": "zh", "entities": [], "relations": []},
            {"entities": [], "relations": []},
            {"segments": []},
            {},
        )
        supported = evaluator.evaluate_job(
            {"language": "zh", "entities": [], "relations": []},
            {
                "entities": [
                    {
                        "id": "E1",
                        "text": "火灾",
                        "type": "EVENT",
                        "evidence": {"segment_id": "S1", "text": "火灾"},
                    }
                ],
                "relations": [],
            },
            {"segments": [{"segment_id": "S1", "text": "发生火灾"}]},
            {},
        )
        result = evaluator.aggregate([empty, supported])
        self.assertEqual(result["entity_evidence_coverage"], 1.0)
        self.assertEqual(result["entity_evidence_correctness"], 1.0)
        self.assertEqual(result["macro_by_job"]["entity_evidence_correctness"], 0.5)

    def test_window_merge_preserves_expanded_evidence(self):
        merger = load_script("merge_window_predictions.py")
        rows = merger.merge_rows(
            [
                {
                    "job_id": "D_C1_W001",
                    "parent_job_id": "D_C1",
                    "window_index": 1,
                    "annotation": {
                        "document_id": "D",
                        "language": "zh",
                        "entities": [{"id": "E1", "text": "火灾", "type": "EVENT", "evidence": {"text": "火灾", "segment_id": "S1"}}],
                        "relations": [],
                    },
                }
            ]
        )
        self.assertEqual(rows[0]["annotation"]["entities"][0]["evidence"]["segment_id"], "S1")

    def test_formal_distribution_reports_sparse_relation_coupling(self):
        analyzer = load_script("analyze_formal_distribution.py")
        rows = [
            (
                {
                    "document_id": "D1",
                    "language": "zh",
                    "entities": [
                        {"id": "E1", "text": "火灾", "type": "EVENT"},
                        {"id": "E2", "text": "灭火器", "type": "ASSET"},
                    ],
                    "relations": [{"source_id": "E1", "target_id": "E2", "type": "involves"}],
                },
                {"job_id": "D1_C1"},
            )
        ]
        result = analyzer.analyze_split(rows)
        self.assertEqual(result["relation_to_entity_ratio"], 0.5)
        self.assertEqual(result["isolated_entity_share"], 0.0)

    def test_relation_verifier_builds_diversified_weighted_negatives(self):
        verifier = load_script("train_relation_verifier.py")
        annotation = {
            "document_id": "D1",
            "entities": [
                {"id": "E1", "text": "火灾", "type": "EVENT", "evidence": {"segment_id": "S1", "text": "火灾影响灭火器"}},
                {"id": "E2", "text": "灭火器", "type": "ASSET", "evidence": {"segment_id": "S1", "text": "火灾影响灭火器"}},
            ],
            "relations": [{"source_id": "E1", "target_id": "E2", "type": "involves", "evidence": [{"segment_id": "S1", "text": "火灾影响灭火器"}]}],
        }
        features, labels, weights, kinds = verifier.build_training_examples(
            [annotation],
            {"D1": {"segments": [{"segment_id": "S1", "text": "火灾影响灭火器"}]}},
            {
                "allowed_relation_signatures": {
                    "involves": {"source": ["EVENT"], "target": ["ASSET"]},
                    "affects": {"source": ["EVENT"], "target": ["ASSET"]},
                }
            },
            seed=1,
            negatives_per_positive=2,
        )
        self.assertEqual(sum(labels), 1)
        self.assertEqual(len(features), len(weights))
        self.assertGreaterEqual(kinds["random_type_valid"], 1)

    def test_kg_concepts_prefer_graph_canonical_name(self):
        jobs = load_script("build_experiment_jobs.py")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mentions.jsonl"
            path.write_text(
                json.dumps(
                    {"split": "train", "type": "REGULATION", "text": "《安全生产法》", "canonical_name": "安全生产法", "document_id": "D1"},
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            concepts = jobs.train_concepts(path)
            self.assertEqual(concepts[0]["name"], "安全生产法")
            self.assertEqual(concepts[0]["language"], "unknown")

    def test_balanced_kg_retrieval_keeps_multiple_entity_types(self):
        jobs = load_script("build_experiment_jobs.py")
        concepts = [
            {"language": "zh", "type": "EVENT", "name": "火灾", "count": 10, "source_documents": {"D2"}},
            {"language": "zh", "type": "EVENT", "name": "爆炸", "count": 9, "source_documents": {"D2"}},
            {"language": "zh", "type": "ASSET", "name": "灭火器", "count": 1, "source_documents": {"D2"}},
        ]
        job = {"language": "zh", "document_id": "D1", "segments": [{"text": "火灾 爆炸 灭火器"}]}
        context = jobs.retrieve_concept_context(job, concepts, 2, balanced=True)
        self.assertIn("EVENT: 火灾", context)
        self.assertIn("ASSET: 灭火器", context)

    def test_semantic_kg_canonical_removes_outer_markup_only(self):
        probe = load_script("probe_semantic_kg_retrieval.py")
        self.assertEqual(probe.canonical("《安全生产法》"), "安全生产法")
        self.assertEqual(probe.canonical("危险与可操作性分析（HAZOP）"), "危险与可操作性分析（hazop）")

    def test_kg_v2_uses_evidence_gates_and_leave_one_document_out(self):
        builder = load_script("build_kg_v2_jobs.py")
        concepts = [
            {
                "concept_id": "C1", "canonical_name": "火灾", "language": "zh", "type": "EVENT",
                "mention_count": 2, "source_documents": ["D1", "D2"],
            },
            {
                "concept_id": "C2", "canonical_name": "灭火器", "language": "zh", "type": "ASSET",
                "mention_count": 1, "source_documents": ["D2"],
            },
            {
                "concept_id": "C3", "canonical_name": "检修", "language": "zh", "type": "OPERATION",
                "mention_count": 1, "source_documents": ["D1"],
            },
        ]
        job = {
            "job_id": "D1_W001", "document_id": "D1", "language": "zh",
            "segments": [{"segment_id": "S1", "text": "火灾涉及灭火器"}],
        }
        catalog = builder.build_anchor_catalog(concepts)
        anchors = builder.select_exact_anchors(job, catalog, 10, 0.8, 4, 2, 2)
        self.assertEqual({row["text"] for row in anchors}, {"火灾", "灭火器"})
        self.assertNotIn("检修", {row["text"] for row in anchors})

        relations = [
            {
                "source_concept_id": "C1", "target_concept_id": "C2", "type": "involves",
                "document_id": "D2", "evidence": [{"text": "火灾涉及灭火器"}],
            }
        ]
        edge_catalog = builder.build_edge_catalog(relations, builder.concept_map(concepts))
        edges = builder.select_edge_priors(job, edge_catalog, 5)
        self.assertEqual(edges[0]["relation"], "involves")
        no_cooccurrence = {**job, "segments": [{"text": "火灾"}, {"text": "灭火器"}]}
        self.assertEqual(builder.select_edge_priors(no_cooccurrence, edge_catalog, 5), [])
        context = builder.render_context(anchors, edges, [])
        self.assertIn("source text always overrides graph memory", context)
        self.assertIn("training examples, never facts", context)

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

    def test_relation_verifier_keeps_supported_relations_and_audits_rejections(self):
        verifier = load_script("verify_relations.py")
        ontology = {
            "relation_types": {"involves": {}, "causes": {}},
            "claim_statuses": {"explicit": {}, "uncertain": {}},
            "allowed_relation_signatures": {
                "involves": {"source": ["EVENT"], "target": ["ASSET"]},
                "causes": {"source": ["FAILURE"], "target": ["EVENT"]},
            },
        }
        annotation = {
            "document_id": "D1",
            "entities": [
                {"id": "E1", "text": "火灾", "type": "EVENT", "evidence": {"text": "火灾涉及灭火器", "segment_id": "S1"}},
                {"id": "E2", "text": "灭火器", "type": "ASSET", "evidence": {"text": "火灾涉及灭火器", "segment_id": "S1"}},
                {"id": "E3", "text": "设备故障", "type": "FAILURE", "evidence": {"text": "设备故障", "segment_id": "S2"}},
            ],
            "relations": [
                {"id": "R1", "source_id": "E1", "type": "involves", "target_id": "E2", "claim_status": "explicit", "evidence": [{"text": "火灾涉及灭火器", "segment_id": "S1"}]},
                {"id": "R2", "source_id": "E1", "type": "causes", "target_id": "E3", "claim_status": "explicit", "evidence": [{"text": "火灾", "segment_id": "S1"}]},
                {"id": "R3", "source_id": "E1", "type": "involves", "target_id": "E2", "claim_status": "missing", "evidence": [{"text": "火灾涉及灭火器", "segment_id": "S1"}]},
            ],
        }
        verified, audit = verifier.verify_annotation(annotation, ontology)
        self.assertEqual([relation["id"] for relation in verified["relations"]], ["R1"])
        self.assertEqual(audit[0]["reasons"], ["accepted"])
        self.assertIn("illegal_entity_type_signature", audit[1]["reasons"])
        self.assertIn("missing_or_unknown_claim_status", audit[2]["reasons"])

    def test_annotation_quality_audit_counts_source_and_relation_issues(self):
        auditor = load_script("audit_annotation_quality.py")
        ontology = {
            "entity_types": {"EVENT": {}, "ASSET": {}},
            "relation_types": {"involves": {}},
            "claim_statuses": {"explicit": {}},
            "allowed_relation_signatures": {"involves": {"source": ["EVENT"], "target": ["ASSET"]}},
        }
        annotation = {
            "language": "zh",
            "entities": [
                {"id": "E1", "text": "火灾", "type": "EVENT", "evidence": {"segment_id": "S1", "text": "火灾涉及灭火器"}},
                {"id": "E2", "text": "灭火器", "type": "ASSET", "evidence": {"segment_id": "S1", "text": "灭火器"}},
            ],
            "relations": [
                {"id": "R1", "source_id": "E1", "type": "involves", "target_id": "E2", "claim_status": "explicit", "evidence": [{"segment_id": "S1", "text": "火灾"}]},
            ],
        }
        issues, counts, diagnostics = auditor.audit_annotation(
            annotation,
            {"segments": [{"segment_id": "S1", "text": "火灾涉及灭火器"}]},
            ontology,
            "D1",
        )
        self.assertEqual(diagnostics["character_count"], 7)
        self.assertEqual(counts["relation_evidence_missing_entity"], 1)
        self.assertEqual(len(issues), 1)

    def test_review_queue_prioritizes_records_with_audit_findings(self):
        queue = load_script("build_review_queue.py")
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, rows in {
                "annotations.jsonl": [{"job_id": "D1", "annotation": {"entities": [], "relations": []}}],
                "errors.jsonl": [{"job_id": "D1", "candidate_errors": ["entity E1: not found"]}],
                "audit.jsonl": [{"job_id": "D1", "relation_id": "R1", "accepted": False, "reasons": ["bad evidence"]}],
            }.items():
                (root / name).write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            output = root / "queue.jsonl"
            args = type("Args", (), {"annotations": root / "annotations.jsonl", "normalize_errors": root / "errors.jsonl", "relation_audit": root / "audit.jsonl", "output": output})()
            self.assertEqual(queue.run(args), 0)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["review_meta"]["priority"], "high")
            self.assertEqual(result["review_meta"]["relation_rejected_count"], 1)


if __name__ == "__main__":
    unittest.main()
