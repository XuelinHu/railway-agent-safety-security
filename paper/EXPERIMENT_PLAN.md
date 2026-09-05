# JSSR special-issue experiment plan

> Status: broad reference plan. The frozen next execution phases, completed
> negative ablations, and current metric definitions are maintained in
> `docs/NEXT_SESSION_HANDOFF_2026-08-30.md`.

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
| Reviewed documents | 150 | Pilot plus the full representative expansion selection |
| Reviewed chunks | 370 | Main benchmark candidate; 75 formal test chunks have independent second review |
| Reviewed entities / relations | 6,570 / 2,247 | Expanded benchmark candidate |
| Frozen pilot test documents | 4 | Not sufficient for final statistical comparison |
| Formal cluster-isolated split | 100 / 20 / 30 documents | Frozen; zero cross-split clusters |
| Formal second-review queue | 30 documents / 75 chunks | Completed; 180 high-risk relation flags reviewed |

The current pilot results must be presented as engineering evidence, not as the final main experiment. In particular, relation F1 is currently zero and Chinese structured output is unstable.

## 3. Publication gates

All gates below are required before the main results are frozen.

1. Freeze ontology version 1.0 after reviewing relation direction and legal signatures. **Completed 27 August 2026.**
2. Cluster exact and near-duplicate Chinese templates before any new split.
3. Expand to at least 120 reviewed documents; target 150 documents.
4. Use a cluster-aware, document-level split. Recommended target: 100 train, 20 validation, and 30 test documents, with railway reports intentionally prominent in the test set.
5. Independently double-review at least 20% of the expanded benchmark and adjudicate disagreements. **Completed for 30 formal test documents on 28 August 2026.**
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

1. Keep the completed validation-selected V3 checkpoint frozen as V2 recall generation followed by the gold-independent V1-style entity gate and deterministic evidence/signature verifier; do not tune this rule on formal test.
2. Keep the BGE-M3 local-pair classifier and textual KG-feature variant as negative validation ablations; do not promote either to the main system or test it as a favorable checkpoint.
3. The repaired token head, learned CRF, and one frozen span-boundary candidate are complete validation ablations. The span model raised typed endpoint reachability to 25/491 (5.09%) but over-generated 8,115 spans and reached only 6.28% strict span F1, below hard BIO's 8.51%; its pooled difference was -2.23 points (95% CI -5.26 to +0.36; p=0.08395). Do not tune this candidate further on validation and do not add its relation classifier because the joint strict-F1/endpoint promotion gate failed.
4. Freeze the primary comparison, evaluation code, seeds, failure handling, and statistical test; then perform one formal test run for the selected method and baselines.
5. Add the 10/25/50/100-document scaling curve and at least three independent seeds for trainable systems that pass their first frozen candidate gate. Span-head latency/throughput/VRAM is now recorded; complete the same efficiency table for promoted systems.
6. Complete span-aware one-to-one evaluation and report annotation agreement/adjudication, since the current normalized-text evaluator collapses repeated mentions. Keep language results micro-aggregated within language and evidence rates micro-aggregated over predictions; retain job-macro values only as explicitly labeled diagnostics.
7. Replace manuscript placeholders and empty bibliography with the reproducible corpus, ontology, method, ablation, limitations, and verified literature sections.

The current diagnostic checkpoint is recorded in `docs/experiment-report.md` and `paper/RESULTS_DRAFT.md`. The low F1 is attributed to sparse relation candidate space, endpoint/boundary/type alignment, document heterogeneity, and incomplete KG aliases rather than a simple class-imbalance or parentheses-only problem. The sampled weighted-negative verifier did not produce a validation gain, so it must not be promoted to a full test claim.

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
