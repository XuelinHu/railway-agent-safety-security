# JSSR special-issue experiment plan

## 1. Submission target

- Journal: *Journal of Safety Science and Resilience*
- Special issue: *Agent for Safety and Security: Complex Network Modeling, Substructure Analysis, Risk Discovery, and Resilience Assessment*
- Deadline: 28 February 2027
- Article type: Original research article; select `SI: Agent for Safety and Security`

The paper fits three stated themes: knowledge graphs for risk analysis, high-quality risk-identification data resources, and explainable/trustworthy AI. Complex-network resilience is not a current claim and should remain outside the title unless graph-level experiments are completed.

## 2. Current evidence base

| Item | Current amount | Publication status |
|---|---:|---|
| Raw documents | 1,515 | Sufficient as a corpus inventory |
| Successfully parsed documents | 1,458 | Sufficient for sampling and weak supervision |
| Chinese safety documents | 1,229 raw / 1,187 parsed | Broad but template de-duplication is incomplete |
| RAIB railway reports | 286 raw / 271 parsed | Strong primary-domain pool |
| Reviewed documents | 29 | Workflow pilot plus one manually accepted expansion document |
| Reviewed chunks | 72 | Too small for final model claims |
| Reviewed entities / relations | 1,611 / 477 | Useful for pilot training |
| Frozen pilot test documents | 4 | Not sufficient for final statistical comparison |

The current pilot results must be presented as engineering evidence, not as the final main experiment. In particular, relation F1 is currently zero and Chinese structured output is unstable.

## 3. Publication gates

All gates below are required before the main results are frozen.

1. Freeze ontology version 1.0 after reviewing relation direction and legal signatures. **Completed 27 August 2026.**
2. Cluster exact and near-duplicate Chinese templates before any new split.
3. Expand to at least 120 reviewed documents; target 150 documents.
4. Use a cluster-aware, document-level split. Recommended target: 100 train, 20 validation, and 30 test documents, with railway reports intentionally prominent in the test set.
5. Independently double-review at least 20% of the expanded benchmark and adjudicate disagreements.
6. Keep all validation/test text, mentions, concepts, and relations out of training-time KG retrieval and pseudo-labeling.
7. Require the proposed method to outperform the strongest comparable baseline with confidence intervals and a paired significance test; do not select a fixed favorable seed.
8. Report failed generations and invalid outputs in denominators rather than silently dropping them.

## 4. Required system variants

| ID | System | Purpose |
|---|---|---|
| B1 | Multilingual encoder NER + relation classifier | Non-generative baseline, preferably XLM-RoBERTa |
| B2 | Qwen3 zero-shot | Unadapted local LLM baseline |
| B2a | Qwen3.8-27B GGUF | Larger local GGUF baseline with post-generation validation |
| B3 | Qwen3 few-shot | Controls for demonstration-only gains |
| B4 | Qwen3 QLoRA | Parameter-efficient baseline without KG/evidence constraints |
| M1 | QLoRA + compact structured target | Tests output simplification |
| M2 | M1 + deterministic evidence grounding | Tests source-traceability contribution |
| M3 | M2 + training-only KG retrieval | Tests knowledge augmentation |
| M4 | M3 + schema and relation-signature constrained decoding | Tests structural validity |
| M5 | M4 + evidence and graph verifier agents | Complete proposed system |
| X1 | A second architecture such as GLM-4-9B | Checks architecture dependence |

Model comparisons must use identical document splits, ontology version, prompt budget, and evaluation code.

The current Qwen3.8-27B pilot uses the same four frozen test documents, tokenizer-budgeted windows, 4-bit GGUF weights, and post-generation evidence/ontology validation as the Qwen3-4B diagnostics. The current llama.cpp build cannot initialize its JSON grammar sampler for this GGUF, so grammar-constrained decoding is not included in the Qwen3.8 score.

## 5. Required experiments

### E1. Annotation quality and corpus reliability

- Report source, language, format, document length, duplicate clusters, and extraction failures.
- Report entity-span/type agreement, relation agreement, and adjudication rate.
- Publish ontology definitions, schemas, split manifests, and corpus redistribution rules.

### E2. Main extraction comparison

- Compare B1--B4 and M1--M5 on the same frozen test set.
- Report strict and relaxed entity micro/macro precision, recall, and F1.
- Report strict relation F1 both with and without claim-status matching.
- Report results by language and domain, not only as a pooled score.

### E3. Low-resource scaling

- Train with 10, 25, 50, and 100 reviewed documents.
- Keep validation and test documents fixed.
- Run at least three random seeds for trainable systems.
- Plot performance against reviewed-document count and annotation time.

### E4. Evidence-grounding and trustworthiness

- Evidence exact match and character-overlap F1.
- Evidence coverage and unsupported-claim rate.
- JSON/schema validity and entity-reference integrity.
- Invalid relation-signature rate before and after constraints.
- Human audit of a stratified sample of accepted and rejected outputs.

### E5. Cross-language and cross-domain transfer

- English railway training to English railway test.
- Chinese safety training to Chinese safety test.
- Joint bilingual training.
- English-to-Chinese and Chinese-to-English transfer where labels permit.
- Report domain and language effects separately to avoid treating generic Chinese templates as railway facts.

### E6. Risk-substructure recovery

- Define typed safety motifs such as hazard--cause--event and event--mitigation--barrier paths.
- Evaluate node, edge, typed-path, and motif recovery against reviewed test graphs.
- Report graph-edit or normalized structural distance only after defining its interpretation.
- Include representative recovered and missed substructures with source evidence.

This experiment connects the paper to the special issue without claiming full cascading-risk or resilience assessment.

### E7. Efficiency and resource constraints

- Peak GPU memory, training time, inference latency, throughput, and output tokens.
- Local versus API inference cost and teacher-annotation cost.
- Model size and quantization settings.
- Accuracy-efficiency trade-off for at least two model sizes.

### E8. Ablation and robustness

- Remove KG retrieval, evidence grounding, constrained decoding, verifier agents, and compact targets one at a time.
- Test prompt paraphrases and decoding seeds.
- Test long-document truncation and noisy PDF line breaks.
- Report common errors: wrong span, wrong type, illegal edge, hallucinated evidence, missing edge, malformed output, and cross-domain confusion.

### E9. Statistical analysis

- Bootstrap 95% confidence intervals over documents.
- Paired bootstrap or approximate-randomization tests for the main comparison.
- Mean and standard deviation over training seeds.
- Correct for multiple comparisons when interpreting many ablations.

## 6. Recommended tables and figures

1. Corpus composition and annotation statistics.
2. Ontology entity types, relation types, and legal signatures.
3. Main entity/relation/evidence results.
4. Trustworthiness and efficiency results.
5. Cross-language and cross-domain transfer matrix.
6. Full ablation table.
7. End-to-end agent and evidence-flow architecture.
8. Low-resource learning curves.
9. Risk-substructure example with provenance.
10. Error-category distribution.

## 7. Work schedule

| Target date | Deliverable |
|---|---|
| 15 Sep 2026 | Ontology 1.0, duplicate clusters, revised annotation guide |
| 31 Oct 2026 | Expanded and independently reviewed benchmark |
| 30 Nov 2026 | Constrained decoder, KG retriever, and verifier agents |
| 31 Dec 2026 | Baselines, low-resource experiments, and main ablations |
| 20 Jan 2027 | Graph-substructure, efficiency, statistics, and figures |
| 10 Feb 2027 | Full English manuscript and internal author review |
| 20 Feb 2027 | Submission package ready, leaving contingency before deadline |

## 8. Immediate next jobs

1. Run the remaining 298 representative teacher jobs in bounded batches; the 122-document selection is approved, but generated entities and relations are not yet human-accepted.
2. Review and adjudicate all generated entities and relations before gold promotion; the first three v1.1 chunks are already manually accepted.
3. Assign a second reviewer to at least 30 selected documents.
4. Implement schema-constrained generation and relation repair before another QLoRA run.
5. Implement the XLM-RoBERTa extraction baseline.
6. Replace the four-document smoke test with the expanded frozen test set.
7. Start writing Introduction, corpus governance, ontology, and method sections while annotation continues.

The expanded QLoRA run used 59 training chunks. The full-schema adapter produced no usable test annotation envelopes. The compact adapter produced valid annotations for 2/4 pilot test jobs: both English jobs were parseable, with pooled strict entity F1 of 25.32% and relation F1 of 0%; both Chinese jobs were format failures. These values are engineering diagnostics only, not final cross-language claims.

The representative teacher jobs are in `data/processed/experiments/annotation_pending_sampled_jobs.jsonl`. The serial proxy command is:

```bash
python3 scripts/run_preannotation.py \
  --provider openai --model gpt-5.6-terra \
  --base-url http://127.0.0.1:8999/v1 \
  --api-key-env SUB2API_API_KEY \
  --jobs data/processed/experiments/annotation_pending_sampled_jobs.jsonl \
  --output data/processed/experiments/annotation_pending_sampled_v2_candidates.jsonl \
  --log outputs/annotation_pending_sampled_v2_run.jsonl \
  --repair-schema --resume --max-jobs 10
```

The runner uses `/v1/chat/completions` with JSON Schema because the local proxy's `/v1/responses` endpoint currently returns an upstream 400 error. Never commit API keys or generated candidate files.

The prompt `teacher-preannotation-v1.1.0` was frozen after a three-job A/B gate. Compared with the prior prompt on the same chunks, audit findings fell from 75 to 47, deterministic repairs from 31 to 3, retained entities increased from 62 to 66, and retained relations from 14 to 19. All three normalized records passed schema, source-offset, and ontology checks. The gate used 78,209 tokens and 286.3 seconds in total; extrapolating directly to all 301 jobs would be costly, so production annotation must remain batched and monitored.
