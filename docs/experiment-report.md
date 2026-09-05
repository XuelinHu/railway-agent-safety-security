# Initial Safety Extraction Experiments

> Status: historical chronological report. It mixes pilot splits, legacy
> normalized metrics, and later diagnostics. Preserve it for provenance; use
> `docs/NEXT_SESSION_HANDOFF_2026-08-30.md` for current span-aware results.

> This is an initial pilot report. The frozen test split contains four documents and is too small for final manuscript claims.

## Data

- Gold schema version: `0.1.0`; ontology version: `1.0.0`; review policy: `pilot_bulk_accepted_plus_formal_test_double_review`.
- Documents: 150; reviewed chunks: 370; train/validation/test: {'train': 142, 'test': 4, 'validation': 4}.
- Test entities: 88; test relations: 26.

## Formal Split and Second Review

- A document-level near-duplicate-isolated split was generated independently of the pilot gold split: train/validation/test = {'train': 100, 'test': 30, 'validation': 20} documents and {'train': 244, 'test': 75, 'validation': 51} text blocks.
- The formal manifest covers 150 documents in 150 clusters; cross-split cluster leakage is 0.
- The second-review queue contains 30 documents and 75 text blocks, with 180 high-risk relation flags; second-review status is `completed`.
- The current pilot gold split is preserved for reproducibility; the formal split is independently stored and can become the manuscript split after the completed second-review audit.

## Strict Results

| System | Entity precision | Entity recall | Entity F1 | Relation precision | Relation recall | Relation F1 | Evaluated jobs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-14B baseline | 18.42% | 7.95% | 11.11% | 0.00% | 0.00% | 0.00% | 4 |
| Qwen3-14B + KG prompt | 13.75% | 12.50% | 13.10% | 0.00% | 0.00% | 0.00% | 4 |
| Qwen3-4B + QLoRA + KG prompt | - | - | - | - | - | - | 0 |
| Qwen3-4B + compact QLoRA + KG prompt | 36.67% | 12.50% | 18.64% | 0.00% | 0.00% | 0.00% | 4 |
| Qwen3-4B + v1.2 compact QLoRA + KG prompt | 15.05% | 20.29% | 17.28% | 20.00% | 4.76% | 7.69% | 3 |
| Qwen3-4B + v1.2 windowed compact QLoRA + KG prompt | 15.73% | 20.29% | 17.72% | 20.00% | 4.76% | 7.69% | 3 |
| Qwen3.8-27B GGUF + windowed compact extraction | 24.62% | 36.36% | 29.36% | 6.12% | 11.54% | 8.00% | 4 |
| Qwen3.8-27B + deterministic relation verifier | 24.62% | 36.36% | 29.36% | 11.76% | 7.69% | 9.30% | 4 |

## QLoRA Status

- Base model: local Qwen3-4B; LoRA rank 8; training examples 56; optimization steps 14.
- Final training loss: `0.5822190642356873`; mean loss: `0.6156739784138543`.
- Compact target training: `True`; final loss: `0.6014580726623535`; mean loss: `0.4517741607768195`.
- Both adapter training runs completed successfully.
- The full-schema adapter did not produce the required top-level `entities`/`relations` JSON object on the four test jobs; its F1 is not treated as a valid extraction result.
- The compact adapter produced parseable envelopes for all four jobs, but Chinese jobs still returned incomplete structures and relation candidates were removed by schema constraints; the compact F1 is an engineering pilot result, not a final claim.
- Auditable inverse-direction repair recovered five schema-valid compact relations after missing claim statuses were conservatively marked `uncertain`; none matched the 26 gold relations, so strict relation F1 remains 0%. Relation-signature repair is retained as a trustworthiness constraint, not claimed as a standalone accuracy innovation.

## Interpretation

- The KG prompt variant improved entity F1 in this pilot from 11.11% to 13.10%, but relation F1 remained 0%; this is a smoke signal, not a statistically supported claim.
- The compact target improved entity F1 in this pilot, but relation extraction and Chinese structured output remain unresolved; the next engineering task is constrained decoding with explicit relation-signature repair, followed by regeneration on validation and test splits.
- Final experiments should add a second reviewer, expand the gold set, and report confidence intervals or paired tests.
- The teacher-prompt evidence-copy gate reduced normalization findings on three fixed Chinese chunks from 75 to 47 and increased retained relations from 14 to 19. These are pipeline-quality measurements, not test-set extraction-accuracy results; the three accepted chunks are in train and are excluded from the frozen pilot test.

## Updated QLoRA Run

- The reviewed training set now contains 59 chunks from 21 documents. Full-schema and compact-target adapters used Qwen3-4B, NF4, rank 8, one epoch, and 15 optimization steps.
- Expanded full-schema mean loss: `0.5559417386849721`; compact-target mean loss: `0.57162946164608`.
- Full-schema generations did not produce usable annotation envelopes on the four pilot test jobs and are recorded as format failures rather than extraction F1.
- The EOS-preserving compact adapter produced valid annotations on 2/4 pilot test jobs: both English jobs were parseable, while both Chinese jobs were format failures. On the valid English jobs, pooled strict entity F1 was 25.32% and relation F1 was 0%. These are engineering diagnostics, not final cross-language claims.
- The v1.2 compact adapter replaced the conflicting full-schema prompt, preserved a 12K default input window, and trained with mean loss `0.41431158085664116`. It produced valid structured outputs on 3/4 jobs; the long English job over-generated entities until the 4K and 8K output budgets were exhausted.
- On the 3 valid v1.2 jobs, pooled strict entity F1 was 17.28% and relation F1 was 7.69%; Chinese entity F1 was 11.62% and Chinese relation F1 was 0.00%. Because one test job failed structurally, these remain engineering diagnostics and are not final manuscript claims.
- The v1.2 long-Chinese diagnostic required an explicit 32K input window but exceeded the 24GB GPU memory budget when a concurrent local model occupied the GPU; its 29-entity/28-relation output was nevertheless recovered from the saved log and evaluated after structural repair.
- Tokenizer-budgeted windowing split the long Chinese job into two overlapping windows and produced 3 valid window outputs from 5. After document-level merge, pooled strict entity F1 was 17.72%, relation F1 was 7.69%, and Chinese entity F1 was 12.01%; this is a resource/robustness diagnostic over 3 evaluated documents.
- Qwen3.8-27B-UD-Q4_K_M GGUF was evaluated through llama.cpp with 4K output budgets and tokenizer-budgeted windows. All 5 windows and 4 documents produced valid parsed outputs; pooled strict entity F1 was 29.36% and relation F1 was 8.00%. Chinese entity F1 was 27.26%, while Chinese relation F1 remained 0.00%.
- A deterministic second-stage relation verifier checked entity references, ontology signatures, claim status, evidence completeness, and local entity co-occurrence. It retained 17/49 Qwen3.8 relations and rejected 32 unsupported cross-evidence relations; entity F1 stayed at 29.36%, while relation precision changed from 6.12% to 11.76% and relation F1 from 8.00% to 9.30%. This is an evidence-quality ablation on four pilot documents, not a final accuracy claim.
- The Qwen3.8 GGUF grammar-constrained path failed during llama.cpp sampler initialization on the current build, including a minimal schema. The reported Qwen3.8 result therefore uses ordinary generation followed by JSON parsing, evidence recovery, ontology validation, and ID normalization; constrained decoding remains a separate pending experiment after a compatible runtime is available.

## Validation Regeneration Diagnostic

- Qwen3.8-27B was regenerated on the frozen validation jobs using 10K-token windows and 11 serial windows. Nine windows produced usable parsed outputs from eight validation chunks; one Chinese chunk had no usable output envelope.
- Before deterministic relation verification, the eight evaluated chunks produced entity F1 17.90%, relation precision 0.88%, and relation F1 1.23%; Chinese relation F1 was 0.00%.
- After evidence co-occurrence verification, relation predictions decreased from 113 to 57, relation precision changed to 1.75%, and relation F1 changed to 1.87%. These validation results guide model and prompt refinement and are not used for final test-set tuning.
- The local OpenAI-compatible proxy was also checked with the project runner, JSON Schema, and one real job: `/v1/chat/completions` returned HTTP 200 and a schema-valid candidate. The earlier `401 invalid_api_key` issue is therefore resolved for the current `SUB2API_API_KEY` environment and endpoint.

## Chinese API Control Diagnostic

- The same two Chinese validation chunks were sent serially through the local OpenAI-compatible proxy using `gpt-5.6-terra` and the full candidate schema. Both requests eventually completed after one field-level schema retry; this validates the proxy path but should be treated as a teacher-model diagnostic rather than an independent gold result.
- The API control produced entity F1 36.84% and relation F1 7.69%; after the deterministic relation verifier, relation precision was 11.11% and relation F1 was 10.00%. The corresponding local Qwen3.8 Chinese scores were entity F1 13.63% and relation F1 0.00%.
- The gap indicates that Chinese F1=0 is not explained by the split alone: local Qwen3.8 compact generation and relation typing are major bottlenecks. The corpus still needs ambiguity and relation-signature cleanup, so API candidates must be reviewed before any gold promotion or training use.
- The quality audit found 30 Qwen3.8 relation quotes without both endpoints versus 6 in the normalized API control. This supports separating model over-generation from source-data ambiguity during adjudication.
- A controlled `gpt-5.6-sol` probe with `reasoning_effort=high` completed on a 25-segment Chinese window and passed JSON Schema; it produced 73 entities with 1 audited issue. The same model with `max` and `xhigh` did not return the full-chunk requests within the practical run boundary, so `high` is the current operational Sol setting for short semantic review, not a default full-corpus extraction setting.

## Chinese Teacher Batch

- The first bounded batch of Chinese expansion jobs used `gpt-5.6-terra` through the repaired local proxy: 10/10 jobs succeeded. The batch produced 215 raw entities and 117 raw relations.
- After source-evidence and ontology normalization, 10 records retained 185 entities and 55 relations. 92 candidate-level normalization findings and 14 relation-evidence rejections were preserved in audit files.
- The source review queue is `data/processed/experiments/annotation_pending_terra_zh_review_queue.jsonl`. After human screening, 7 novel records were promoted to gold and 3 duplicate records were skipped; normalization findings and relation rejections remain available for provenance review.

## Chinese Teacher Batch 2

- The second bounded batch used `gpt-5.6-terra` through the repaired local proxy: 10/10 jobs succeeded. It produced 198 raw entities and 96 raw relations.
- Normalization retained 151 entities and 57 relations. 86 candidate-level findings were preserved; 30 relations were rejected by evidence co-occurrence verification.
- The queue is `data/processed/experiments/annotation_pending_terra_batch2_review_queue.jsonl`. Its records were superseded by the unified queue and the consolidated gold promotion; the original batch files remain as provenance.

## Unified Chinese Teacher Queue

- The unified queue contains 291 unique jobs generated through `gpt-5.6-terra`, with 5837 raw entities and 3099 raw relations.
- Normalization retained 4841 entities and 1731 relations. The deterministic verifier accepted 827 relations and rejected 904; the quality audit recorded 903 evidence issues. The complete queue was then human-screened and accepted.
- The consolidated review queue is `data/processed/experiments/annotation_pending_terra_all_review_queue.jsonl`; 274 of 291 jobs were high priority. All 291 unique records were promoted to gold after the user's consolidated confirmation.

## Formal Graph and Evidence Diagnostics

- The formal train-only `graph_v2` contains 4050 concepts, 4335 mentions, and 1510 relations from 100 training documents. Cross-split concept, mention, and relation leakage are all zero.
- The evidence evaluator now reports prediction-level micro rates, with the prior unweighted per-job values retained under `macro_by_job`. Earlier 41.77/43.17% validation and 47.62/48.65% test figures were job-macro values that treated zero-denominator jobs as zero; they must not be described as evidence correctness.
- On the current validation artifacts, baseline/V1/V2/V3 entity evidence correctness is 100.00/99.87/99.96/99.95%. Relation evidence correctness is 95.16/86.96/58.32/100.00%. V3's 100% relation value is guaranteed by verifier acceptance rules, not a generator-level safety result.
- The existing formal-test evidence artifacts have not been rerun during validation model selection. Their legacy macro rates are excluded from current claims until the frozen evaluation protocol is executed once without further tuning.
- The graph snapshot leakage audit reports 0 concepts, 0 mentions, and 0 relations sourced from non-training documents.

## Root-Cause Analysis: Low F1 and KG Degradation

- Entity-label imbalance is not the primary explanation: normalized entity-label entropy is 0.954 on train, 0.948 on validation, and 0.947 on test.
- The task is structurally sparse: only 1.59% of typed ordered entity pairs are observed as relations in test blocks, leaving 98.41% unobserved. 51.78% of test entity mentions have no gold relation endpoint.
- Document heterogeneity is substantial: test entity and relation count coefficients of variation are 0.647 and 0.667; train relation-block CV is 0.756. This supports stratified/document-level analysis, not a single pooled score alone.
- The BGE-M3 semantic probe found exact typed train-KG concepts for only 12.19% of validation and 8.51% of test entity queries. Among those exact targets, top-1 retrieval recall was 100%, so ranking is not the first KG bottleneck; lexical/alias coverage is.
- The train-only graph contains 4050 concepts, 4335 mentions, and 1510 relations; 0.00% of concepts are isolated. Frequency-limited prompt retrieval reaches only a small fraction of gold targets, so the current KG prompt is incomplete rather than leakage-contaminated.
- Diversified weighted negative sampling plus a lightweight relation verifier did not improve validation relation F1: baseline probe changed from 1.01% to 0.00%; KG probe changed from 0.52% to 0.55%. This is evidence against applying the verifier to the frozen test set.
- A direct twelve-window compatibility probe using the old KG adapter with the v2 graph prompt was stopped after the first window emitted repeated explanatory text and exhausted its token budget without a JSON envelope. The probe is retained as `validation_v3_kg_v2_probe.log`; it shows that a graph/prompt revision requires adapter retraining or an explicit compatibility gate.
- The operational conclusion is that low F1 is dominated by endpoint recovery, boundary/type alignment, relation candidate sparsity, incomplete aliases, and missing or uncertain relation supervision. Parenthetical punctuation is a secondary normalization issue, not an embedding failure.
- The next KG version should add reviewed aliases and bilingual links, typed semantic retrieval, confidence-gated candidate insertion, positive-unlabeled treatment of unobserved pairs, and hard negatives restricted to legal typed pairs. It should not treat every unobserved pair as a definite negative.

## KG V2 Validation Experiment

- KG V2 uses a train-only graph, leave-one-document-out training prompts, source-grounded entity anchors, co-occurrence-gated edge priors, and provenance-bearing semantic relation patterns. It was trained on 983 effective Qwen3-4B QLoRA examples for 246 steps; final/mean loss was 0.14098/0.17199.
- Validation inference produced usable compact JSON for 419/434 windows (96.54%). The 15 remaining failures were repeated-entity or incomplete-envelope generations and are scored as empty predictions.
- On the same 434 validation windows, baseline entity precision/recall/F1 was 11.30/8.19/9.50%; KG V2 was 8.07/24.51/12.15%. Baseline relation precision/recall/F1 was 4.84/0.61/1.08%; KG V2 was 1.61/7.33/2.64%.
- Across 20 validation documents and 20000 bootstrap/permutation iterations, V2-minus-baseline pooled entity F1 was +2.65 points (95% bootstrap CI -0.37 to +5.85; paired permutation p=0.1723) and pooled relation F1 was +1.56 points (CI +0.15 to +3.09; p=0.0622). Document-macro differences were significant, but the primary pooled tests do not establish an overall improvement at alpha=0.05.
- KG V2 therefore improves validation recall and F1 but loses precision. It predicts 4929 entities and 2241 strict relation tuples for 1624 and 491 gold items, respectively.
- With prediction-level aggregation, KG V2 entity evidence coverage/correctness is 100.00/99.96%. Relation evidence coverage/correctness is 100.00/58.32%; unsupported-claim and invalid-relation rates are 41.68/57.62%. The former 96.31/96.29, 70.05/42.21, and 27.84/38.54 values are retained only as `macro_by_job` diagnostics and are not prediction-level rates.
- A deterministic evidence/signature verifier retained 970/2289 expanded relations and raised strict relation F1 from 2.64% to 4.38%, while reducing unsupported and invalid rates to zero by construction. This is recorded as an independent post-processing ablation, not the primary V2 result.
- No KG V2 test prediction was generated or inspected during validation design and model selection.

## KG V3 Validation Fusion and Reject-Option Ablation

- A direct comparison on the same 434 validation windows found V1 entity/relation F1 of 8.69/0.37% and V2 F1 of 12.15/2.64%. V1 is more conservative; V2 is substantially more recall-oriented. Their normalized entity intersection had 19.35% precision but only 4.80% recall, and their union reached 12.26% F1, so a set union/intersection was not adopted.
- The gold-independent V3 fusion gate (local artifact label `M2`, not the manuscript system ID) starts from V2 entities and retains candidates supported by exact V1/V2 `(text, type)` agreement, a same-type source-gated V2 anchor, or use as an endpoint of a deterministically verified relation. It retained 1857/5016 audited V2 candidates. Entity precision/recall/F1 changed from 8.07/24.51/12.15% to 12.44/14.22/13.27%.
- With raw endpoint-filtered relations, relation precision/recall/F1 was 2.95/6.31/4.02%. Relative to raw V2, pooled relation F1 increased by 1.38 points across 20 documents (95% bootstrap CI +0.60 to +2.41; paired permutation p=0.0005). The entity gain of 1.12 points was not significant (CI -1.28 to +3.22; p=0.4693).
- The fusion gate plus the deterministic relation verifier is the strongest combined validation checkpoint: entity F1 13.27% and relation F1 4.39%. Corrected language-micro Entity/Relation F1 is 15.08/6.06% for English and 11.86/3.05% for Chinese; the earlier 4.27/1.05 relation figures were job-macro values. The verified relation score is only 0.01 point above V2 plus the verifier, so this must be interpreted as the combination of a better entity checkpoint with an independently useful trustworthiness filter, not a new relation-classifier gain.
- The no-KG pair-classifier ablation (local artifact label `M3`) trained a BGE-M3 normalized-embedding weighted logistic classifier on 1619 positive and 13010 sampled negative local typed-pair candidates. Its best validation relation F1 was 0.92% at threshold 0.6. The KG-feature variant (artifact label `M4`) added textual KG edge-prior and semantic-pattern flags but reached only 0.85% at threshold 0.5; KG-feature-minus-no-KG was not significant (95% CI -0.79 to +0.25 points; p=0.7181). Both are negative ablations and are inferior to filtering V2 relations.
- A final deterministic verifier accepted every pair-classifier relation because legal signatures, same-segment co-occurrence, evidence, and claim status were already guaranteed at candidate construction. The no-op is expected and shows that structural validity alone cannot distinguish the sparse true relations from thousands of legal but unsupported semantic pairs.
- No formal test predictions were generated or inspected. The next test-eligible pipeline is V2 recall generation followed by the frozen M2 entity gate and deterministic relation verifier; the local-pair classifier is excluded from the main method unless a future validation-only redesign exceeds that checkpoint.

## XLM-R Entity-Stage Baseline Diagnostic

- The first weighted top-four-layer token head over-generated 8,816 entities and reached only 1.97% strict entity F1. Its confidence scores did not separate correct from incorrect spans, so post-hoc thresholding was rejected.
- A standard full-model run used all 11,556 train chunks, no token-class weighting, three epochs, one fixed seed, and no formal-test access. Training took 672 seconds and reduced greedy predictions to 2,579, but greedy strict entity F1 was still 2.71%.
- Hard BIO-transition Viterbi decoding removed illegal initial `I-*` sequences and produced 1,324 entities with strict precision/recall/F1 of 8.91/7.27/8.01%. Boundary-and-type relaxed F1 was 17.64%, boundary-only F1 was 21.60%, and source evidence coverage/correctness was 100/100%.
- Corrected language-micro entity F1 was 6.55% for English and 11.32% for Chinese. Only 19/491 gold relations had both strict typed endpoints recovered, giving a 3.87% endpoint-reachable recall ceiling. A relation head is therefore deferred until the learned boundary/entity stage improves; this is an entity-stage diagnostic, not completed B1.

## XLM-R BIO Repair and Learned CRF Follow-up

- Training labels were regenerated from the window-aligned gold artifact (`windowed_train_v2/gold.jsonl` plus its index), preserving the 11,556 chunks and 2,875 positive chunks. Chunk-local BIO repair converted 139 interrupted or chunk-initial `I-*` labels to `B-*` across 138 chunks; the count and entity-type breakdown are recorded in the CRF run summary.
- A learnable linear-chain CRF was trained jointly with the full XLM-R encoder for three epochs using seed 20260830, batch size 8, gradient accumulation 2, learning rate 1e-5, max length 192, stride 48, no class weighting, and no formal-test access. CRF parameters are stored separately in `crf.pt`; the Hugging Face token-classification checkpoint remains independently loadable.
- The same checkpoint was decoded with greedy, hard-BIO, and learned-CRF paths on the 434 validation windows. The confidence-window results are:

| Decoder | Predicted entities | Strict span F1 | Boundary + type F1 | Boundary-only F1 | Endpoint-reachable ceiling |
|---|---:|---:|---:|---:|---:|
| Greedy | 2,595 | 2.84% | 14.32% | 17.63% | 0.41% (2/491) |
| Hard BIO | 1,338 | **8.51%** | **18.03%** | **22.18%** | **2.24% (11/491)** |
| Learned CRF | 2,571 | 2.86% | 14.40% | 17.68% | 0.41% (2/491) |

- Longest-window selection changed learned-CRF strict span F1 to 3.07% and boundary-only F1 to 17.83%; it did not change the conclusion. Corrected language-micro strict entity F1 for learned CRF was 2.38% (English) and 4.00% (Chinese). Entity evidence coverage and correctness were both 100% because every emitted entity carries a recovered source segment; no relation objects were emitted.
- Learned CRF therefore does not materially improve boundary or endpoint recovery over the validation-selected hard-BIO ablation. No relation head was added; this result motivated the single frozen span-boundary follow-up reported below rather than further CRF tuning.

## XLM-R Span-Boundary Follow-up

- The frozen candidate uses per-type token start/end heads and a typed span classifier with `NONE`, followed by deterministic highest-score non-overlapping selection. The maximum span width is 64 tokens; all 6,619 exactly aligned training-span occurrences fit within 49 tokens. Sparse training uses four sampled negative spans per positive and at least eight negatives per chunk.
- Threshold selection used a deterministic document-level 79/20 split within the 99 training documents. The selected joint score threshold was 0.015 (inner-dev strict F1 8.84%). The final model was reinitialized and trained on all 11,556 chunks for three epochs with the same seed, batch, accumulation, learning-rate, sequence-length, stride, and full-encoder settings as the CRF run. Formal test artifacts were not read.
- On the 434-window validation set, the model emitted 8,115 spans and obtained strict precision/recall/F1 of 3.77/18.84/6.28%. Boundary+type F1 was 10.84%, boundary-only F1 was 17.24%, and language-micro strict F1 was 7.42% for English and 5.43% for Chinese. Entity evidence coverage and correctness were both 100%.
- Typed endpoint reachability increased to 25/491 (5.09%), exactly meeting the predeclared count target, but strict F1 remained below hard BIO's 8.51%. Relative to hard BIO, pooled strict F1 changed by -2.23 points under 20,000 document-level paired resamples (95% CI -5.26 to +0.36; paired permutation p=0.08395). The candidate therefore fails the joint promotion gate and remains a negative ablation; no relation head or additional span-head validation tuning is justified.
- The complete two-stage run took 1,358.5 seconds on an RTX 3090 and peaked at 10,713.6 MiB allocated CUDA memory. Reloaded validation-only inference took 44.51 seconds (102.56 ms/window job, 110.20 tokenizer chunks/s) and peaked at 2,417.1 MiB. The reloaded checkpoint reproduced the original prediction file byte-for-byte (SHA-256 `a9678147045030b7a721b71f53bf232744eb783b1ff65ad7ca0c35dc061cc2d1`).
