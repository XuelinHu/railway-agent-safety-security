# Research and implementation work plan

> Status: historical implementation checklist. Several unchecked items were
> completed or superseded after this plan was written. The current executable
> plan is `docs/NEXT_SESSION_HANDOFF_2026-08-30.md`.

## Confirmed manuscript line

The paper studies knowledge-graph-constrained, evidence-grounded, low-resource language models for safety entity, relation, and causal-risk-structure extraction. Railway accident reports are the primary validation scenario; Chinese emergency and safety-management documents provide cross-language and cross-domain knowledge.

## Research questions

1. Does knowledge-graph augmentation improve safety entity and relation extraction under limited human annotation budgets?
2. Do deterministic evidence binding and graph-signature constraints reduce unsupported or structurally invalid outputs?
3. How well does the method transfer between English railway investigations and Chinese multi-domain safety documents?
4. What accuracy, latency, throughput, and memory trade-offs arise from QLoRA and model-size choices?

## Work packages

### WP1: Corpus governance and parsing

- [x] Inventory all local raw documents with content hashes and duplicate links.
- [x] Extract valid PDF, DOCX, and TXT content while preserving segments, pages, and global offsets.
- [x] Record corrupt, incomplete, unsupported, and tool-blocked files without modifying raw data.
- [x] Install project-local legacy Word extractors and parse the old `.doc` collection.
- [x] Build a near-duplicate-aware cluster inventory for the annotation expansion sample.
- [ ] Cluster near-duplicate Chinese templates before the final train/test split.

### WP2: Ontology and annotation protocol

- [x] Define 15 entity types.
- [x] Define 15 relation types and initial legal type signatures.
- [x] Define explicit, inferred, normative, and uncertain claim statuses.
- [x] Define candidate and reviewed-annotation JSON Schemas.
- [x] Define human review and adjudication rules.
- [x] Run a human pilot and freeze the ontology as version 1.0.

### WP3: Pilot dataset

- [x] Select 28 diverse English and Chinese pilot documents.
- [x] Generate chunked, provider-neutral pre-annotation jobs.
- [x] Validate local Qwen3 teacher output on English PDF and Chinese DOCX samples.
- [x] Screen the 28 pilot documents and bulk-accept the current candidates after review.
- [x] Freeze the current document-level train, validation, and test splits for the pilot.

### WP4: Teacher-assisted annotation

- [x] Implement Ollama and OpenAI pre-annotation runners.
- [x] Resolve evidence offsets deterministically rather than trusting model-generated numbers.
- [x] Reject paraphrased evidence and illegal graph signatures into an audit log.
- [x] Configure the OpenAI-compatible `sub2API` endpoint and run the 69-job serial teacher pass.
- [x] Normalize and validate all 69 teacher outputs with evidence and ontology constraints.
- [x] Promote the screened annotations to the current `gold v0.1.0` set.
- [x] Generate a 150-document expansion manifest with 122 pending documents.
- [x] Verify a ten-job serial sub2API batch through the Chat Completions fallback and preserve candidate-level errors.
- [x] Add auditable unique-span and inverse-relation repair without automatically promoting repaired candidates.
- [x] Reduce the 122-document expansion from 1,056 full-coverage jobs to 301 representative-chunk jobs.
- [x] Freeze teacher prompt `v1.1.0` after a three-job evidence-copy A/B gate.
- [x] Manually confirm and merge the three v1.1 gate chunks into reviewed gold and the provenance-bearing graph.
- [ ] Run two-model pre-annotation and human adjudication.
- [x] Double-annotate at least 20% of the gold set through the 30-document formal test review.

### WP5: Knowledge graph

- [x] Normalize accepted mentions to reproducible canonical concepts.
- [x] Build provenance-bearing entity mention and relation tables.
- [x] Version the initial graph snapshot as `gold v0.1.0`.
- [ ] Link aliases and bilingual terms with a manually reviewed lexicon.

### WP6: Models and experiments

- [x] Train a full-fine-tuned XLM-RoBERTa entity-stage baseline and add validation-only BIO-constrained decoding; the best one-seed validation entity F1 is 8.01%.
- [ ] Complete B1 with a learned CRF/span-boundary model and relation classifier; the present entity checkpoint exposes only 3.87% of gold relation endpoints and is not a complete B1 result.
- [x] Run the initial Qwen3-14B zero-shot and KG-prompt smoke comparison on the frozen pilot test split.
- [x] Train a local Qwen3-4B QLoRA adapter on the reviewed pilot jobs.
- [x] Train and evaluate a compact-output QLoRA variant; retain the structured-output and relation failures as pilot evidence.
- [x] Retrain full-schema and compact QLoRA variants on the expanded 59-chunk training split.
- [x] Diagnose QLoRA generation parsing, EOS truncation, and compact-target prompt mismatch; v1.2 removes the conflicting prompt and adds auditable structural JSON repair.
- [x] Add tokenizer-budgeted document windows and overlap-aware compact prediction merging for long-document inference.
- [ ] Evaluate Qwen3-4B and Qwen3-8B zero-shot baselines on the expanded benchmark.
- [x] Add Qwen3.8-27B GGUF baseline through llama.cpp and evaluate windowed compact extraction on the frozen pilot test set.
- [x] Add a deterministic second-stage relation verifier for entity references, ontology signatures, claim status, and local evidence co-occurrence; record an auditable before/after ablation.
- [x] Regenerate the frozen validation chunks with Qwen3.8-27B using tokenizer-budgeted windows and retain window-level failure logs.
- [x] Run a serial `sub2API` Chinese control on the same validation chunks and compare model, schema, evidence, and relation-signature failure categories.
- [x] Run the first bounded 10-job Chinese expansion batch with `gpt-5.6-terra`, normalize it, generate a priority-aware review queue, and promote the confirmed novel records to gold.
- [x] Run the second bounded 10-job Chinese expansion batch with `gpt-5.6-terra`, normalize it, and include it in the consolidated human acceptance.
- [x] Complete the remaining 281 representative teacher jobs with resumable Terra workers, build one 291-job unified audit queue, and promote the confirmed records to gold.
- [x] Probe `gpt-5.6-sol` at high reasoning on a controlled Chinese window; record `max`/`xhigh` long-request latency and use Sol only for bounded semantic review until proxy latency is resolved.
- [x] Build the independent document-level near-duplicate-isolated formal split with 100/20/30 documents and zero cross-split clusters.
- [x] Build a second-review queue for all 30 formal test documents, including 75 text blocks and 180 high-risk relation flags; reset item decisions to pending.
- [x] Complete the second review for all 30 formal test documents; accept 1,234 entities and 394 relations, including the 180 high-risk relation flags.
- [ ] Add vector RAG, KG retrieval, and graph-constrained decoding variants.
- [ ] Validate cross-architecture behavior with GLM-4-9B.
- [ ] Run low-resource budgets of 10, 25, 50, and 100 documents.
- [ ] Run ablations for ontology, evidence, graph constraints, weak labels, and active learning.

### WP7: Evaluation and paper

- [x] Generate an initial strict entity/relation evaluation report on the frozen four-job pilot test split.
- [x] Report strict entity F1 and relation F1 on the complete formal validation and frozen test comparisons; causal-edge F1 and graph similarity remain pending.
- [ ] Report corrected prediction-level evidence coverage, correctness, unsupported-claim, and invalid-relation rates on the formal test split; validation is corrected, while legacy test values were job-macro and are not current claims.
- [x] Produce a pilot relation-verifier audit with per-relation acceptance and rejection reasons; expand it to unsupported-claim and evidence-correctness rates on the larger gold set.
- [ ] Report peak VRAM, latency, throughput, training time, and annotation cost.
- [x] Perform strict relation endpoint, direction, type, and unsupported-pair error analysis; add evidence-quality and graph-overlap diagnostics.
- [x] Evaluate exact-text and boundary-tolerant entity/type matching as a diagnostic; retain strict matching as the primary metric.
- [x] Audit Chinese bracketed entity mentions; separate outer presentation marks from semantic parenthetical content and retain original evidence spans.
- [x] Complete document-level bootstrap confidence intervals and paired permutation tests for the formal comparison.
- [ ] Draft the method, experiment, limitation, and data-governance sections from frozen results.

## Immediate next gate

The pilot pipeline is executable end to end. Before treating the numbers as manuscript results:

1. The unified 291-job Terra queue has been human-screened and merged; keep its audit files as provenance for the current gold snapshot.
2. The independent second review of the 30-document formal test queue is complete; its 180 high-risk relation flags were included in the accepted review.
3. Keep the four pilot test documents frozen and excluded from pseudo-label training; the new formal split is an independent candidate benchmark.
4. Stabilize Chinese compact generation with language-balanced targets and constrained decoding; document windows now control long-input memory, but Chinese precision and relation F1 remain insufficient.
5. Freeze the generated 100/20/30 cluster-aware split and use it for the next cross-language and model-comparison experiments.
6. Use the verifier audit to select relation errors for semantic adjudication, then compare deterministic filtering with a Qwen3.8 or small cross-encoder relation verifier on validation data only.

## Current Experiment Checkpoint

- [x] Diagnose the interrupted unwindowed run: long inputs caused answer truncation and one high-memory sequence caused a CUDA out-of-memory failure.
- [x] Train the paired windowed Qwen3-4B baseline and KG-prompt adapters to completion with answer truncation rate 0%.
- [x] Complete the 51-block formal validation comparison with 100% generation success for both systems.
- [x] Freeze the current model selection: windowed baseline is the main model; windowed KG is an ablation.
- [x] Complete all 628 frozen test windows for both systems with 100% structured generation success.
- [x] Merge the window predictions into 75 formal test blocks and compute strict entity, relation, and claim-status-aware relation metrics.
- [x] Compute document-level paired bootstrap confidence intervals and permutation tests for validation and frozen test comparisons.
- [ ] Re-run causal edges, graph similarity, and corrected micro evidence metrics once under the final frozen-test protocol; do not reuse the legacy job-macro evidence interpretation.

## Root-cause decision

- [x] Quantify document/entity/relation dispersion, typed-pair sparsity, isolated entities, label entropy, and train-only KG coverage.
- [x] Diagnose Chinese outer presentation marks separately from semantic parenthetical content.
- [x] Probe BGE-M3 semantic retrieval and verify that exact KG targets are ranked correctly when they exist.
- [x] Test diversified weighted negative sampling with a validation-only lightweight relation verifier; no stable gain was observed.
- [x] Freeze the validation-selected V3 reject-option pipeline with immutable input hashes, a gold-free runner, and an explicit non-validation safety gate.
- The low strict F1 is not explained by a single extreme label imbalance. Entity-label normalized entropy is approximately 0.95 across splits, while relation/entity counts vary strongly by document and relation block.
- Relation extraction is sparse and endpoint-limited: only 1.59% of typed ordered entity pairs are observed as relations in the test split, 51.78% of test entities are isolated, and relation diagnostics identify endpoint recovery as the dominant failure category.
- The current KG is leakage-safe but incomplete. The train-only graph has 4,050 concepts, 1,510 relations, and 46.49% isolated concepts. Exact typed train-KG coverage is only 12.19% on validation and 8.51% on test. BGE-M3 top-1 recall is 100% conditional on an exact target existing, so retrieval ranking is not the first bottleneck.
- KG prompting currently adds incomplete, frequency-truncated context and increases unsupported/invalid relation outputs. It remains an ablation, not the proposed improvement.
- The next principled KG experiment is alias/bilingual linking plus typed semantic retrieval and confidence gating. Unobserved entity pairs should be treated as positive-unlabeled candidates; negative sampling must use legal typed pairs and down-weight uncertain negatives, with reversed and wrong-type pairs as hard negatives.
