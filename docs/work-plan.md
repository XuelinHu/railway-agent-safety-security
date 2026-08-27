# Research and implementation work plan

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
- [ ] Double-annotate at least 20% of the gold set.

### WP5: Knowledge graph

- [x] Normalize accepted mentions to reproducible canonical concepts.
- [x] Build provenance-bearing entity mention and relation tables.
- [x] Version the initial graph snapshot as `gold v0.1.0`.
- [ ] Link aliases and bilingual terms with a manually reviewed lexicon.

### WP6: Models and experiments

- [ ] Implement XLM-RoBERTa + CRF NER baseline.
- [x] Run the initial Qwen3-14B zero-shot and KG-prompt smoke comparison on the frozen pilot test split.
- [x] Train a local Qwen3-4B QLoRA adapter on the reviewed pilot jobs.
- [x] Train and evaluate a compact-output QLoRA variant; retain the structured-output and relation failures as pilot evidence.
- [x] Retrain full-schema and compact QLoRA variants on the expanded 59-chunk training split.
- [x] Diagnose QLoRA generation parsing, EOS truncation, and compact-target prompt mismatch; v1.2 removes the conflicting prompt and adds auditable structural JSON repair.
- [x] Add tokenizer-budgeted document windows and overlap-aware compact prediction merging for long-document inference.
- [ ] Evaluate Qwen3-4B and Qwen3-8B zero-shot baselines on the expanded benchmark.
- [x] Add Qwen3.8-27B GGUF baseline through llama.cpp and evaluate windowed compact extraction on the frozen pilot test set.
- [ ] Add vector RAG, KG retrieval, and graph-constrained decoding variants.
- [ ] Validate cross-architecture behavior with GLM-4-9B.
- [ ] Run low-resource budgets of 10, 25, 50, and 100 documents.
- [ ] Run ablations for ontology, evidence, graph constraints, weak labels, and active learning.

### WP7: Evaluation and paper

- [x] Generate an initial strict entity/relation evaluation report on the frozen four-job pilot test split.
- [ ] Report strict entity F1, relation F1, causal-edge F1, and graph similarity on the expanded benchmark.
- [ ] Report evidence coverage, evidence correctness, unsupported-claim rate, and invalid-relation rate.
- [ ] Report peak VRAM, latency, throughput, training time, and annotation cost.
- [ ] Perform error analysis and paired statistical tests.
- [ ] Draft the method, experiment, limitation, and data-governance sections from frozen results.

## Immediate next gate

The pilot pipeline is executable end to end. Before treating the numbers as manuscript results:

1. Run the remaining 298 representative teacher jobs in cost-controlled batches and review the resulting entities and relations.
2. Obtain an independent second review for at least 20% of the expanded gold set.
3. Keep the four pilot test documents frozen and excluded from pseudo-label training.
4. Stabilize Chinese compact generation with language-balanced targets and constrained decoding; document windows now control long-input memory, but Chinese precision and relation F1 remain insufficient.
5. Freeze a cluster-aware expanded split before making cross-language or model-comparison claims.
