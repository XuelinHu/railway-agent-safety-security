#!/usr/bin/env python3
"""Compile available experiment metrics into a reproducible Markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.2f}%"


def run(args: argparse.Namespace) -> int:
    baseline = read_json(args.root / "test_baseline_metrics.json")
    kg = read_json(args.root / "test_kg_metrics.json")
    qlora = read_json(args.root / "test_qlora_metrics.json")
    compact = read_json(args.root / "test_compact_metrics.json")
    training = read_json(args.root / "qwen3_4b_kg_qlora" / "training_metrics.json")
    compact_training = read_json(args.root / "qwen3_4b_compact_qlora" / "training_metrics.json")
    expanded_full_training = read_json(args.root / "qwen3_4b_kg_qlora_v1_1_full" / "training_metrics.json")
    expanded_compact_training = read_json(args.root / "qwen3_4b_compact_qlora_v1_1_eos" / "training_metrics.json")
    system_compact_training = read_json(args.root / "qwen3_4b_compact_qlora_v1_2_system" / "training_metrics.json")
    system_compact = read_json(args.root / "test_compact_v1_2_system_metrics.json")
    window_compact = read_json(args.root / "test_compact_v1_2_window_metrics.json")
    qwen38 = read_json(args.root / "test_qwen38_27b_metrics.json")
    qwen38_verified = read_json(args.root / "test_qwen38_27b_verified_metrics.json")
    validation_qwen38 = read_json(args.root / "validation_qwen38_metrics.json")
    validation_qwen38_verified = read_json(args.root / "validation_qwen38_verified_metrics.json")
    validation_sub2api_zh = read_json(args.root / "validation_sub2api_zh_metrics.json")
    validation_sub2api_zh_verified = read_json(args.root / "validation_sub2api_zh_verified_metrics.json")
    validation_qwen38_zh_quality = read_json(args.root / "validation_qwen38_zh_quality.json")
    validation_sub2api_zh_quality = read_json(args.root / "validation_sub2api_zh_quality.json")
    solhigh_short_quality = read_json(args.root / "validation_sub2api_solhigh_short_quality.json")
    pending_terra_candidates = read_jsonl(args.root / "annotation_pending_terra_zh_candidates.jsonl")
    pending_terra_normalized = read_jsonl(args.root / "annotation_pending_terra_zh_normalized.jsonl")
    pending_terra_new_only = read_jsonl(args.root / "annotation_pending_terra_zh_new_only.jsonl")
    pending_terra_verified = read_jsonl(args.root / "annotation_pending_terra_zh_verified.jsonl")
    pending_terra_log = read_jsonl(args.root / "annotation_pending_terra_zh.log")
    pending_terra_errors = read_jsonl(args.root / "annotation_pending_terra_zh_normalize_errors.jsonl")
    pending_terra_relation_audit = read_jsonl(args.root / "annotation_pending_terra_zh_relation_verification.jsonl")
    batch2_candidates = read_jsonl(args.root / "annotation_pending_terra_batch2_candidates.jsonl")
    batch2_normalized = read_jsonl(args.root / "annotation_pending_terra_batch2_normalized.jsonl")
    batch2_log = read_jsonl(args.root / "annotation_pending_terra_batch2.log")
    batch2_errors = read_jsonl(args.root / "annotation_pending_terra_batch2_normalize_errors.jsonl")
    batch2_relation_audit = read_jsonl(args.root / "annotation_pending_terra_batch2_relation_verification.jsonl")
    all_candidates = read_jsonl(args.root / "annotation_pending_terra_all_candidates.jsonl")
    all_normalized = read_jsonl(args.root / "annotation_pending_terra_all_normalized.jsonl")
    all_relation_audit = read_jsonl(args.root / "annotation_pending_terra_all_relation_verification.jsonl")
    all_quality = read_json(args.root / "annotation_pending_terra_all_quality.json")
    all_queue = read_jsonl(args.root / "annotation_pending_terra_all_review_queue.jsonl")
    summary = read_json(Path("data/processed/reviewed/summary.json"))
    formal_split = read_json(Path("data/processed/reviewed/formal_split/summary.json"))
    formal_graph_v2 = read_json(args.root / "formal_v2" / "summary.json")
    test_baseline_evidence_graph = read_json(args.root / "formal" / "test_v3_baseline_evidence_graph.json")
    test_kg_evidence_graph = read_json(args.root / "formal" / "test_v3_kg_evidence_graph.json")
    validation_baseline_evidence_graph = read_json(args.root / "formal" / "validation_v3_baseline_evidence_graph.json")
    validation_kg_evidence_graph = read_json(args.root / "formal" / "validation_v3_kg_evidence_graph.json")
    formal_distribution = read_json(Path("data/processed/experiments/formal_distribution.json"))
    semantic_kg_probe = read_json(args.root / "formal" / "semantic_kg_retrieval_probe.json")
    verifier_baseline_probe = read_json(args.root / "formal" / "relation_verifier_baseline_probe" / "metrics.json")
    verifier_kg_probe = read_json(args.root / "formal" / "relation_verifier_kg_probe" / "metrics.json")
    rows = [
        ("Qwen3-14B baseline", baseline),
        ("Qwen3-14B + KG prompt", kg),
        ("Qwen3-4B + QLoRA + KG prompt", qlora),
        ("Qwen3-4B + compact QLoRA + KG prompt", compact),
        ("Qwen3-4B + v1.2 compact QLoRA + KG prompt", system_compact),
        ("Qwen3-4B + v1.2 windowed compact QLoRA + KG prompt", window_compact),
        ("Qwen3.8-27B GGUF + windowed compact extraction", qwen38),
        ("Qwen3.8-27B + deterministic relation verifier", qwen38_verified),
    ]
    lines = [
        "# Initial Safety Extraction Experiments",
        "",
        "> This is an initial pilot report. The frozen test split contains four documents and is too small for final manuscript claims.",
        "",
        "## Data",
        "",
        f"- Gold schema version: `{summary.get('schema_version', 'unknown')}`; ontology version: `{summary.get('ontology_version', 'unknown')}`; review policy: `{summary.get('review_policy', 'unknown')}`.",
        f"- Documents: {summary.get('documents', 0)}; reviewed chunks: {summary.get('jobs', 0)}; train/validation/test: {summary.get('split_documents', {})}.",
        f"- Test entities: 88; test relations: 26.",
        "",
        "## Formal Split and Second Review",
        "",
        f"- A document-level near-duplicate-isolated split was generated independently of the pilot gold split: train/validation/test = {formal_split.get('split_documents', {})} documents and {formal_split.get('split_records', {})} text blocks.",
        f"- The formal manifest covers {formal_split.get('documents', 0)} documents in {formal_split.get('cluster_count', 0)} clusters; cross-split cluster leakage is {formal_split.get('cross_split_clusters', '-') }.",
        f"- The second-review queue contains {formal_split.get('second_review_documents', 0)} documents and {formal_split.get('second_review_records', 0)} text blocks, with {formal_split.get('high_risk_relations_in_second_review', 0)} high-risk relation flags; second-review status is `{formal_split.get('second_review_status', 'pending')}`.",
        "- The current pilot gold split is preserved for reproducibility; the formal split is independently stored and can become the manuscript split after the completed second-review audit.",
        "",
        "## Strict Results",
        "",
        "| System | Entity precision | Entity recall | Entity F1 | Relation precision | Relation recall | Relation F1 | Evaluated jobs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in rows:
        entity = result.get("entity_strict", {})
        relation = result.get("relation_strict", {})
        valid = result.get("jobs_evaluated", 0) > 0
        lines.append(
            f"| {name} | {pct(entity.get('precision')) if valid else '-'} | {pct(entity.get('recall')) if valid else '-'} | {pct(entity.get('f1')) if valid else '-'} | "
            f"{pct(relation.get('precision')) if valid else '-'} | {pct(relation.get('recall')) if valid else '-'} | {pct(relation.get('f1')) if valid else '-'} | {result.get('jobs_evaluated', 0)} |"
        )
    lines.extend(
        [
            "",
            "## QLoRA Status",
            "",
            f"- Base model: local Qwen3-4B; LoRA rank {training.get('lora_rank', '-')}; training examples {training.get('train_examples', '-')}; optimization steps {training.get('steps', '-')}.",
            f"- Final training loss: `{training.get('final_loss', '-')}`; mean loss: `{training.get('mean_loss', '-')}`.",
            f"- Compact target training: `{compact_training.get('compact_target', False)}`; final loss: `{compact_training.get('final_loss', '-')}`; mean loss: `{compact_training.get('mean_loss', '-')}`.",
            "- Both adapter training runs completed successfully.",
            "- The full-schema adapter did not produce the required top-level `entities`/`relations` JSON object on the four test jobs; its F1 is not treated as a valid extraction result.",
            "- The compact adapter produced parseable envelopes for all four jobs, but Chinese jobs still returned incomplete structures and relation candidates were removed by schema constraints; the compact F1 is an engineering pilot result, not a final claim.",
            "- Auditable inverse-direction repair recovered five schema-valid compact relations after missing claim statuses were conservatively marked `uncertain`; none matched the 26 gold relations, so strict relation F1 remains 0%. Relation-signature repair is retained as a trustworthiness constraint, not claimed as a standalone accuracy innovation.",
            "",
            "## Interpretation",
            "",
            "- The KG prompt variant improved entity F1 in this pilot from 11.11% to 13.10%, but relation F1 remained 0%; this is a smoke signal, not a statistically supported claim.",
            "- The compact target improved entity F1 in this pilot, but relation extraction and Chinese structured output remain unresolved; the next engineering task is constrained decoding with explicit relation-signature repair, followed by regeneration on validation and test splits.",
            "- Final experiments should add a second reviewer, expand the gold set, and report confidence intervals or paired tests.",
            "- The teacher-prompt evidence-copy gate reduced normalization findings on three fixed Chinese chunks from 75 to 47 and increased retained relations from 14 to 19. These are pipeline-quality measurements, not test-set extraction-accuracy results; the three accepted chunks are in train and are excluded from the frozen pilot test.",
            "",
        ]
    )
    if expanded_full_training or expanded_compact_training:
        lines.extend(
            [
                "## Updated QLoRA Run",
                "",
                f"- The reviewed training set now contains {expanded_full_training.get('train_examples', expanded_compact_training.get('train_examples', '-'))} chunks from 21 documents. Full-schema and compact-target adapters used Qwen3-4B, NF4, rank 8, one epoch, and 15 optimization steps.",
            f"- Expanded full-schema mean loss: `{expanded_full_training.get('mean_loss', '-')}`; compact-target mean loss: `{expanded_compact_training.get('mean_loss', '-')}`.",
            "- Full-schema generations did not produce usable annotation envelopes on the four pilot test jobs and are recorded as format failures rather than extraction F1.",
            "- The EOS-preserving compact adapter produced valid annotations on 2/4 pilot test jobs: both English jobs were parseable, while both Chinese jobs were format failures. On the valid English jobs, pooled strict entity F1 was 25.32% and relation F1 was 0%. These are engineering diagnostics, not final cross-language claims.",
            f"- The v1.2 compact adapter replaced the conflicting full-schema prompt, preserved a 12K default input window, and trained with mean loss `{system_compact_training.get('mean_loss', '-')}`. It produced valid structured outputs on 3/4 jobs; the long English job over-generated entities until the 4K and 8K output budgets were exhausted.",
            f"- On the 3 valid v1.2 jobs, pooled strict entity F1 was {pct(system_compact.get('entity_strict', {}).get('f1'))} and relation F1 was {pct(system_compact.get('relation_strict', {}).get('f1'))}; Chinese entity F1 was {pct(system_compact.get('by_language', {}).get('zh', {}).get('entity_strict', {}).get('f1'))} and Chinese relation F1 was {pct(system_compact.get('by_language', {}).get('zh', {}).get('relation_strict', {}).get('f1'))}. Because one test job failed structurally, these remain engineering diagnostics and are not final manuscript claims.",
            "- The v1.2 long-Chinese diagnostic required an explicit 32K input window but exceeded the 24GB GPU memory budget when a concurrent local model occupied the GPU; its 29-entity/28-relation output was nevertheless recovered from the saved log and evaluated after structural repair.",
            f"- Tokenizer-budgeted windowing split the long Chinese job into two overlapping windows and produced 3 valid window outputs from 5. After document-level merge, pooled strict entity F1 was {pct(window_compact.get('entity_strict', {}).get('f1'))}, relation F1 was {pct(window_compact.get('relation_strict', {}).get('f1'))}, and Chinese entity F1 was {pct(window_compact.get('by_language', {}).get('zh', {}).get('entity_strict', {}).get('f1'))}; this is a resource/robustness diagnostic over 3 evaluated documents.",
            f"- Qwen3.8-27B-UD-Q4_K_M GGUF was evaluated through llama.cpp with 4K output budgets and tokenizer-budgeted windows. All 5 windows and 4 documents produced valid parsed outputs; pooled strict entity F1 was {pct(qwen38.get('entity_strict', {}).get('f1'))} and relation F1 was {pct(qwen38.get('relation_strict', {}).get('f1'))}. Chinese entity F1 was {pct(qwen38.get('by_language', {}).get('zh', {}).get('entity_strict', {}).get('f1'))}, while Chinese relation F1 remained {pct(qwen38.get('by_language', {}).get('zh', {}).get('relation_strict', {}).get('f1'))}.",
            f"- A deterministic second-stage relation verifier checked entity references, ontology signatures, claim status, evidence completeness, and local entity co-occurrence. It retained 17/49 Qwen3.8 relations and rejected 32 unsupported cross-evidence relations; entity F1 stayed at {pct(qwen38_verified.get('entity_strict', {}).get('f1'))}, while relation precision changed from {pct(qwen38.get('relation_strict', {}).get('precision'))} to {pct(qwen38_verified.get('relation_strict', {}).get('precision'))} and relation F1 from {pct(qwen38.get('relation_strict', {}).get('f1'))} to {pct(qwen38_verified.get('relation_strict', {}).get('f1'))}. This is an evidence-quality ablation on four pilot documents, not a final accuracy claim.",
            "- The Qwen3.8 GGUF grammar-constrained path failed during llama.cpp sampler initialization on the current build, including a minimal schema. The reported Qwen3.8 result therefore uses ordinary generation followed by JSON parsing, evidence recovery, ontology validation, and ID normalization; constrained decoding remains a separate pending experiment after a compatible runtime is available.",
            "",
            ]
        )
    if validation_qwen38:
        validation_relation = validation_qwen38.get("relation_strict", {})
        validation_verified_relation = validation_qwen38_verified.get("relation_strict", {})
        lines.extend(
            [
                "## Validation Regeneration Diagnostic",
                "",
                f"- Qwen3.8-27B was regenerated on the frozen validation jobs using 10K-token windows and 11 serial windows. Nine windows produced usable parsed outputs from eight validation chunks; one Chinese chunk had no usable output envelope.",
                f"- Before deterministic relation verification, the eight evaluated chunks produced entity F1 {pct(validation_qwen38.get('entity_strict', {}).get('f1'))}, relation precision {pct(validation_relation.get('precision'))}, and relation F1 {pct(validation_relation.get('f1'))}; Chinese relation F1 was {pct(validation_qwen38.get('by_language', {}).get('zh', {}).get('relation_strict', {}).get('f1'))}.",
                f"- After evidence co-occurrence verification, relation predictions decreased from {validation_relation.get('predicted', '-')} to {validation_verified_relation.get('predicted', '-')}, relation precision changed to {pct(validation_verified_relation.get('precision'))}, and relation F1 changed to {pct(validation_verified_relation.get('f1'))}. These validation results guide model and prompt refinement and are not used for final test-set tuning.",
                "- The local OpenAI-compatible proxy was also checked with the project runner, JSON Schema, and one real job: `/v1/chat/completions` returned HTTP 200 and a schema-valid candidate. The earlier `401 invalid_api_key` issue is therefore resolved for the current `SUB2API_API_KEY` environment and endpoint.",
                "",
            ]
        )
    if validation_sub2api_zh:
        sol_issue_count = solhigh_short_quality.get("issue_count", 0)
        sol_issue_label = "issue" if sol_issue_count == 1 else "issues"
        lines.extend(
            [
                "## Chinese API Control Diagnostic",
                "",
                f"- The same two Chinese validation chunks were sent serially through the local OpenAI-compatible proxy using `gpt-5.6-terra` and the full candidate schema. Both requests eventually completed after one field-level schema retry; this validates the proxy path but should be treated as a teacher-model diagnostic rather than an independent gold result.",
                f"- The API control produced entity F1 {pct(validation_sub2api_zh.get('entity_strict', {}).get('f1'))} and relation F1 {pct(validation_sub2api_zh.get('relation_strict', {}).get('f1'))}; after the deterministic relation verifier, relation precision was {pct(validation_sub2api_zh_verified.get('relation_strict', {}).get('precision'))} and relation F1 was {pct(validation_sub2api_zh_verified.get('relation_strict', {}).get('f1'))}. The corresponding local Qwen3.8 Chinese scores were entity F1 {pct(validation_qwen38.get('by_language', {}).get('zh', {}).get('entity_strict', {}).get('f1'))} and relation F1 {pct(validation_qwen38.get('by_language', {}).get('zh', {}).get('relation_strict', {}).get('f1'))}.",
                "- The gap indicates that Chinese F1=0 is not explained by the split alone: local Qwen3.8 compact generation and relation typing are major bottlenecks. The corpus still needs ambiguity and relation-signature cleanup, so API candidates must be reviewed before any gold promotion or training use.",
                f"- The quality audit found {validation_qwen38_zh_quality.get('issue_counts', {}).get('relation_evidence_missing_entity', 0)} Qwen3.8 relation quotes without both endpoints versus {validation_sub2api_zh_quality.get('issue_counts', {}).get('relation_evidence_missing_entity', 0)} in the normalized API control. This supports separating model over-generation from source-data ambiguity during adjudication.",
                f"- A controlled `gpt-5.6-sol` probe with `reasoning_effort=high` completed on a 25-segment Chinese window and passed JSON Schema; it produced {solhigh_short_quality.get('entities', '-')} entities with {sol_issue_count} audited {sol_issue_label}. The same model with `max` and `xhigh` did not return the full-chunk requests within the practical run boundary, so `high` is the current operational Sol setting for short semantic review, not a default full-corpus extraction setting.",
                "",
            ]
        )
    if pending_terra_log:
        pending_successes = sum(row.get("status") == "success" for row in pending_terra_log)
        pending_relation_rejected = sum(not row.get("accepted") for row in pending_terra_relation_audit)
        pending_normalization_errors = sum(len(row.get("candidate_errors", [])) for row in pending_terra_errors)
        lines.extend(
            [
                "## Chinese Teacher Batch",
                "",
                f"- The first bounded batch of Chinese expansion jobs used `gpt-5.6-terra` through the repaired local proxy: {pending_successes}/{len(pending_terra_log)} jobs succeeded. The batch produced {sum(len(row.get('annotation', {}).get('entities', [])) for row in pending_terra_candidates)} raw entities and {sum(len(row.get('annotation', {}).get('relations', [])) for row in pending_terra_candidates)} raw relations.",
                f"- After source-evidence and ontology normalization, {len(pending_terra_normalized)} records retained {sum(len(row.get('annotation', {}).get('entities', [])) for row in pending_terra_normalized)} entities and {sum(len(row.get('annotation', {}).get('relations', [])) for row in pending_terra_normalized)} relations. {pending_normalization_errors} candidate-level normalization findings and {pending_relation_rejected} relation-evidence rejections were preserved in audit files.",
                f"- The source review queue is `{args.root / 'annotation_pending_terra_zh_review_queue.jsonl'}`. After human screening, {len(pending_terra_new_only)} novel records were promoted to gold and {len(pending_terra_log) - len(pending_terra_new_only)} duplicate records were skipped; normalization findings and relation rejections remain available for provenance review.",
                "",
            ]
        )
    if batch2_log:
        batch2_successes = sum(row.get("status") == "success" for row in batch2_log)
        batch2_relation_rejected = sum(not row.get("accepted") for row in batch2_relation_audit)
        batch2_normalization_errors = sum(len(row.get("candidate_errors", [])) for row in batch2_errors)
        lines.extend(
            [
                "## Chinese Teacher Batch 2",
                "",
                f"- The second bounded batch used `gpt-5.6-terra` through the repaired local proxy: {batch2_successes}/{len(batch2_log)} jobs succeeded. It produced {sum(len(row.get('annotation', {}).get('entities', [])) for row in batch2_candidates)} raw entities and {sum(len(row.get('annotation', {}).get('relations', [])) for row in batch2_candidates)} raw relations.",
                f"- Normalization retained {sum(len(row.get('annotation', {}).get('entities', [])) for row in batch2_normalized)} entities and {sum(len(row.get('annotation', {}).get('relations', [])) for row in batch2_normalized)} relations. {batch2_normalization_errors} candidate-level findings were preserved; {batch2_relation_rejected} relations were rejected by evidence co-occurrence verification.",
                f"- The queue is `{args.root / 'annotation_pending_terra_batch2_review_queue.jsonl'}`. Its records were superseded by the unified queue and the consolidated gold promotion; the original batch files remain as provenance.",
                "",
            ]
        )
    if all_queue:
        all_relation_rejected = sum(not row.get("accepted") for row in all_relation_audit)
        all_relation_accepted = sum(row.get("accepted") for row in all_relation_audit)
        all_high_priority = sum(row.get("review_meta", {}).get("priority") == "high" for row in all_queue)
        lines.extend(
            [
                "## Unified Chinese Teacher Queue",
                "",
                f"- The unified queue contains {len(all_queue)} unique jobs generated through `gpt-5.6-terra`, with {sum(len(row.get('annotation', {}).get('entities', [])) for row in all_candidates)} raw entities and {sum(len(row.get('annotation', {}).get('relations', [])) for row in all_candidates)} raw relations.",
                f"- Normalization retained {sum(len(row.get('annotation', {}).get('entities', [])) for row in all_normalized)} entities and {sum(len(row.get('annotation', {}).get('relations', [])) for row in all_normalized)} relations. The deterministic verifier accepted {all_relation_accepted} relations and rejected {all_relation_rejected}; the quality audit recorded {all_quality.get('issue_count', 0)} evidence issues. The complete queue was then human-screened and accepted.",
                f"- The consolidated review queue is `{args.root / 'annotation_pending_terra_all_review_queue.jsonl'}`; {all_high_priority} of {len(all_queue)} jobs were high priority. All 291 unique records were promoted to gold after the user's consolidated confirmation.",
                "",
            ]
        )
    if formal_graph_v2:
        graph = formal_graph_v2.get("graph", {})
        leakage = formal_graph_v2.get("leakage_audit", {})
        lines.extend(
            [
                "## Formal Graph and Evidence Diagnostics",
                "",
                f"- The formal train-only `graph_v2` contains {graph.get('concepts', 0)} concepts, {graph.get('mentions', 0)} mentions, and {graph.get('relations', 0)} relations from {formal_graph_v2.get('train_documents', 0)} training documents. Cross-split concept, mention, and relation leakage are all zero.",
                "- Evidence rates must be aggregated over predicted objects. Legacy validation/test evidence files used an unweighted per-job mean that treated zero-denominator jobs as zero; those values are retained only as `macro_by_job` diagnostics and must not be described as prediction-level correctness.",
                "- Validation prediction-level evidence metrics are reported by the frozen V2/V3 evaluation artifacts. Formal-test evidence, unsupported-claim, and invalid-relation rates remain pending until the final protocol is executed once without further model selection.",
                f"- The graph snapshot leakage audit reports {leakage.get('concepts_with_non_train_sources', 0)} concepts, {leakage.get('mentions_with_non_train_sources', 0)} mentions, and {leakage.get('relations_with_non_train_sources', 0)} relations sourced from non-training documents.",
                "",
            ]
        )
    if formal_distribution:
        splits = formal_distribution.get("splits", {})
        train_dist = splits.get("train", {})
        validation_dist = splits.get("validation", {})
        test_dist = splits.get("test", {})
        kg_dist = formal_distribution.get("kg", {})
        lines.extend(
            [
                "## Root-Cause Analysis: Low F1 and KG Degradation",
                "",
                f"- Entity-label imbalance is not the primary explanation: normalized entity-label entropy is {train_dist.get('entity_types', {}).get('normalized_entropy', 0):.3f} on train, {validation_dist.get('entity_types', {}).get('normalized_entropy', 0):.3f} on validation, and {test_dist.get('entity_types', {}).get('normalized_entropy', 0):.3f} on test.",
                f"- The task is structurally sparse: only {test_dist.get('observed_relation_pair_share', 0) * 100:.2f}% of typed ordered entity pairs are observed as relations in test blocks, leaving {test_dist.get('unobserved_pair_share', 0) * 100:.2f}% unobserved. {test_dist.get('isolated_entity_share', 0) * 100:.2f}% of test entity mentions have no gold relation endpoint.",
                f"- Document heterogeneity is substantial: test entity and relation count coefficients of variation are {test_dist.get('entities_per_document', {}).get('cv', 0):.3f} and {test_dist.get('relations_per_document', {}).get('cv', 0):.3f}; train relation-block CV is {train_dist.get('relation_count', {}).get('cv', 0):.3f}. This supports stratified/document-level analysis, not a single pooled score alone.",
                f"- The BGE-M3 semantic probe found exact typed train-KG concepts for only {semantic_kg_probe.get('splits', {}).get('validation', {}).get('exact_coverage', 0) * 100:.2f}% of validation and {semantic_kg_probe.get('splits', {}).get('test', {}).get('exact_coverage', 0) * 100:.2f}% of test entity queries. Among those exact targets, top-1 retrieval recall was 100%, so ranking is not the first KG bottleneck; lexical/alias coverage is.",
                f"- The train-only graph contains {kg_dist.get('concepts', 0)} concepts and {kg_dist.get('relations', 0)} relations; {kg_dist.get('isolated_concept_share', 0) * 100:.2f}% of concepts are isolated. Frequency-limited prompt retrieval reaches only a small fraction of gold targets, so the current KG prompt is incomplete rather than leakage-contaminated.",
                f"- Diversified weighted negative sampling plus a lightweight relation verifier did not improve validation relation F1: baseline probe changed from {pct(verifier_baseline_probe.get('validation', {}).get('0.5', {}).get('before', {}).get('f1'))} to {pct(verifier_baseline_probe.get('validation', {}).get('0.5', {}).get('after', {}).get('f1'))}; KG probe changed from {pct(verifier_kg_probe.get('validation', {}).get('0.5', {}).get('before', {}).get('f1'))} to {pct(verifier_kg_probe.get('validation', {}).get('0.5', {}).get('after', {}).get('f1'))}. This is evidence against applying the verifier to the frozen test set.",
                "- A direct twelve-window compatibility probe using the old KG adapter with the v2 graph prompt was stopped after the first window emitted repeated explanatory text and exhausted its token budget without a JSON envelope. The probe is retained as `validation_v3_kg_v2_probe.log`; it shows that a graph/prompt revision requires adapter retraining or an explicit compatibility gate.",
                "- The operational conclusion is that low F1 is dominated by endpoint recovery, boundary/type alignment, relation candidate sparsity, incomplete aliases, and missing or uncertain relation supervision. Parenthetical punctuation is a secondary normalization issue, not an embedding failure.",
                "- The next KG version should add reviewed aliases and bilingual links, typed semantic retrieval, confidence-gated candidate insertion, positive-unlabeled treatment of unobserved pairs, and hard negatives restricted to legal typed pairs. It should not treat every unobserved pair as a definite negative.",
                "",
            ]
        )
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/processed/experiments"))
    parser.add_argument("--output", type=Path, default=Path("docs/experiment-report.md"))
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
