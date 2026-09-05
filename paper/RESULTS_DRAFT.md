# Current Results Draft

> Status: reference result archive. This file intentionally preserves older
> normalized-text results alongside later span-aware ablations. Use
> `docs/NEXT_SESSION_HANDOFF_2026-08-30.md` for the primary result table and
> metric precedence before transferring numbers into the manuscript.

This is an internal results draft based on the frozen formal 100/20/30 split. It records the completed KG V2 validation run and must not be presented as a final test-set claim until the remaining baselines, low-resource budgets, and independent seeds are complete.

## Dataset and split

The formal benchmark contains 100 training documents, 20 validation documents, and 30 frozen test documents, isolated by document-level near-duplicate clustering. The training graph is built only from the 100 training documents. It contains 4,050 canonical concepts, 4,335 entity mentions, and 1,510 relations; the graph leakage audit reports zero non-training concept, mention, or relation sources.

## Strict extraction comparison

The directly comparable window-level validation run uses 434 windows from 51 source blocks. The windowed baseline has entity precision/recall/F1 of 11.30/8.19/9.50% and relation precision/recall/F1 of 4.84/0.61/1.08%. KG V2 has entity precision/recall/F1 of 8.07/24.51/12.15% and relation precision/recall/F1 of 1.61/7.33/2.64%, with a 96.54% generation success rate before missing windows are scored as empty. Thus V2 increases recall and strict F1 on this validation run, but does so with substantial precision loss and relation over-generation. These numbers are validation evidence for the V2 design, not a claim of test-set generalization.

Document-level bootstrap and paired permutation analysis used 20 validation documents and 20,000 iterations. The V2-minus-baseline pooled entity-F1 difference was 2.65 percentage points (bootstrap 95% CI -0.37 to 5.85; paired permutation p=0.1723), and the pooled relation-F1 difference was 1.56 points (CI 0.15 to 3.09; p=0.0622). Document-macro differences were significant for entities (p=0.0126) and relations (p=0.0004), but the primary pooled tests do not support a robust overall significance claim at alpha=0.05.

For reference, the earlier KG prompt (not V2 and not directly comparable to the new window-level run) obtained validation entity/relation F1 of 10.55/0.52% and test F1 of 9.50/1.66% on the previous 51/75-block evaluation. Those results remain historical ablations.

### Direct V1--V2 window comparison

To isolate the version transition as fairly as the available artifacts allow, the previous windowed KG predictions were re-evaluated against the same 434-window gold and job list used by V2. V1 (the frequency-limited ``HINTS (train KG, verify in text)'' prompt) obtained entity precision/recall/F1 of 13.52/6.40/8.69% and relation precision/recall/F1 of 2.17/0.20/0.37%. V2 obtained 8.07/24.51/12.15% and 1.61/7.33/2.64%, respectively. V2 is therefore a recall-oriented upgrade: it recovers more gold entities and relations but emits many more unsupported candidates.

This is a system-version comparison, not a single-factor prompt ablation. V2 changes graph construction (document exclusion and evidence gates), context contents, compact training examples, and overlength handling in addition to the prompt wording. The V1--V2 relation-F1 difference is significant in the document-level paired permutation test (p=0.0074), while the entity-F1 difference is not (p=0.1176); the result should not be interpreted as proof that graph context alone caused the gain.

## Evidence and relation-verifier ablation

V2 compact outputs were expanded by deterministic source-span recovery. After correcting the evaluator to aggregate evidence rates over predicted objects rather than averaging zero-denominator jobs, entity evidence coverage/correctness are 100.00/99.96%. Before relation verification, relation evidence coverage/correctness are 100.00/58.32%, unsupported-claim rate is 41.68%, and invalid-relation rate is 57.62%. The previously recorded 96.31/96.29, 70.05/42.21, 27.84, and 38.54 figures are job-macro diagnostics, not prediction-level rates. A deterministic verifier accepting only valid entity references, ontology signatures, claim status, source evidence, and local co-occurrence retained 970 of 2,289 relation objects. It reduced unsupported and invalid rates to zero by construction and changed strict relation F1 from 2.64% to 4.38%. This is an independent trustworthiness ablation; the unverified V2 score remains the primary extraction result.

## Validation-only V1--V2 fusion and reject-option ablation

The V1--V2 comparison suggested a precision--recall complementarity, but neither a simple union nor a simple intersection was sufficient. On the same validation windows, the normalized V1/V2 entity intersection had 19.35% precision but only 4.80% recall, while the union reached only 12.26% F1. We therefore evaluated a gold-independent conservative gate over the higher-recall V2 entities. A V2 entity was retained if its normalized `(text, type)` agreed with V1, matched a source-gated V2 anchor with the same type, or was an endpoint of a relation accepted by the deterministic evidence/signature verifier. The gate retained 1,857 of 5,016 audited V2 entity candidates and increased entity precision/recall/F1 from 8.07/24.51/12.15% to 12.44/14.22/13.27%.

Applying the same endpoint gate to the unverified V2 relation set increased strict relation precision/recall/F1 from 1.61/7.33/2.64% to 2.95/6.31/4.02%. Across 20 validation documents, the pooled relation-F1 gain was 1.38 percentage points (document bootstrap 95% CI 0.60 to 2.41; paired permutation p=0.0005). The pooled entity-F1 gain was 1.12 points, but its CI crossed zero (-1.28 to 3.22; p=0.4693), so the entity improvement is not statistically established. Adding the deterministic verifier produced the strongest combined validation checkpoint: entity F1 13.27% and relation F1 4.39%. Corrected language-micro Entity/Relation F1 is 15.08/6.06% for English and 11.86/3.05% for Chinese; the prior 4.27/1.05 relation figures were job-macro values. Its relation score is essentially unchanged from applying the verifier to V2 alone (4.38%); the measurable raw-relation gain comes from conservative endpoint selection, whereas the final verifier supplies structural and evidence validity.

We also tested a local reject-option relation classifier over same-segment, ontology-legal ordered entity pairs. The classifier used normalized BGE-M3 embeddings and weighted logistic regression trained on 1,619 positive and 13,010 sampled negative candidates; validation contained 24,601 typed candidates for 8,096 ordered pairs. Without KG features, the best validation relation F1 was 0.92% at threshold 0.6. Encoding KG edge-prior and semantic-pattern flags in the classifier input did not improve the result: the best F1 was 0.85% at threshold 0.5, and the KG-feature-minus-no-KG difference was not significant (95% CI -0.79 to 0.25 points; p=0.7181). A final deterministic verifier was a no-op for these candidates because same-segment evidence and legal signatures had already been enforced during candidate construction. These negative ablations show that enlarging the legal pair space and then classifying it is inferior to filtering the grounded V2 output; they must not be presented as positive KG gains. The local artifact labels `M2`--`M5` used during this diagnostic are not the manuscript system IDs in `EXPERIMENT_PLAN.md`.

## Why F1 is weak

The corpus does not show severe entity-label imbalance: normalized entity-label entropy is approximately 0.95 in all three splits. Instead, the annotation graph is sparse and heterogeneous. Test documents have entity-count and relation-count coefficients of variation of 0.647 and 0.668, respectively. Only 1.59% of typed ordered entity pairs are observed as relations, 98.41% remain unobserved, and 51.78% of test entity mentions are isolated from gold relations.

The relation error decomposition supports this interpretation. Only 20 of 394 baseline gold relations and 12 of 394 KG gold relations had both endpoints recovered with the correct type. Relaxing entity boundaries raises test entity F1, but the resulting relation F1 remains low. Therefore the main problem is the combination of entity boundary/type alignment, endpoint recall, legal pair construction, relation long-tail coverage, and incomplete or uncertain relation supervision. Chinese brackets contribute a smaller normalization mismatch: stripping balanced outer presentation marks improves a diagnostic baseline score, while deleting internal parenthetical content does not.

## Why KG V2 still has failure modes

The graph is leakage-safe but sparse for evaluation-time concepts. Exact typed train-graph coverage is 12.19% of validation entity queries and 8.51% of test queries. A BGE-M3 probe retrieves the correct exact concept at top 1 whenever it exists, showing that the first limitation is lexical and bilingual alias coverage rather than vector ranking. The current prompt also frequency-truncates the candidate list, reducing practical coverage further.

V2 improves validation recall but produces 4,929 entity and 2,241 strict relation tuples for 1,624 and 491 gold items, respectively. The resulting precision loss, 41.68% unsupported-claim rate, and 57.62% invalid-relation rate show that added graph context increases candidate output faster than endpoint grounding and legal relation selection improve. Corrected language-micro Entity/Relation F1 is 8.92/1.90% for Chinese versus 18.33/3.83% for English. The current KG is therefore a promising evidence-gated recall aid, but not yet a reliable standalone accuracy component.

## Contributions that are currently supportable

1. **Evidence-gated KG augmentation.** Graph context is inserted only through source co-occurrence, type-purity, provenance, and explicit warnings that source text overrides graph memory; the graph is a candidate prior, not an additional gold source.
2. **Leakage-safe provenance construction.** Concepts, mentions, relations, source documents, and split provenance are retained, and the V2 training graph is built from the 100 training documents with leave-one-document-out prompt construction (zero cross-split graph leakage in the audit).
3. **Compact extraction with deterministic evidence recovery.** The adapted model emits compact entities and relations; a deterministic post-processor rebinds exact source spans, normalizes IDs, and validates relation references and evidence.
4. **Safety-specific typed ontology and audit metrics.** Fifteen entity types and fifteen typed relation signatures support structural checks beyond F1, including evidence correctness, unsupported claims, invalid relations, causal-edge recovery, and language-stratified results.
5. **Failure-aware experimental protocol.** The pipeline preserves failed generations and normalization audits, reports document-level uncertainty, and prohibits using the frozen test split for V2 prompt or model selection.
6. **Conservative V1--V2 fusion as validation evidence.** A gold-independent agreement/anchor/verified-endpoint gate combines V2 recall with V1-style abstention and improves raw validation relation F1 with a document-level paired test; this remains a validation-selected result until the rule and threshold are frozen and evaluated once on the untouched test set.

## Submission blockers

- `paper/elsarticle/manuscript.tex` is still a structured skeleton: the Introduction, Related Work, corpus, method, results, discussion, and limitations sections contain TODO/placeholders rather than the reproducible protocol and tables.
- `paper/elsarticle/references.bib` has no entries and the compiled manuscript reports an empty bibliography. Verified citations for the ontology, evidence grounding, KG retrieval, QLoRA, and safety-IE literature are required.
- The title currently claims an ``agent'' and the method plan claims constrained decoding, but the completed V2 evidence is a QLoRA extractor plus deterministic post-processing; the full multi-agent and grammar-constrained runtime must either be implemented and evaluated or removed from the claims.
- V2 has one validation seed and no 10/25/50/100-document scaling curve, independent seeds, latency/throughput/VRAM/cost table, or final frozen-test result. These are required for the planned RQ2--RQ4 claims after the protocol is locked.
- The primary evaluator collapses repeated mentions by normalized `(text, type)` and relations by endpoint text. A span-aware one-to-one metric and adjudicated inter-annotator agreement are required before reporting final scores.
- The current V2 relation output is not yet trustworthy without filtering: 41.68% of predicted relation objects lack source-valid endpoint evidence and 57.62% violate structural checks. The verifier result is an ablation, not evidence that the generator itself is safe.
- The XLM-R hard-BIO decoder remains the strongest encoder entity checkpoint after repaired-CRF and span-boundary follow-ups. B1 still lacks a viable relation classifier and independent successful seeds; the failed span candidate is reported below as a completed negative ablation.

### XLM-R BIO repair and learned CRF follow-up

The window-aligned training labels contain 139 interrupted or chunk-initial `I-*` transitions. The training encoder now repairs these labels locally to `B-*` (138 affected chunks) and records the count by entity type. A joint full-encoder learned linear-chain CRF was then trained for three epochs with the same fixed validation protocol (seed 20260830, batch size 8, accumulation 2, learning rate 1e-5, max length 192, stride 48, no class weighting). Its CRF parameters are saved separately from the Hugging Face checkpoint.

On the 434 validation windows, greedy, hard-BIO, and learned-CRF confidence-window decoding obtained strict span F1 of 2.84%, 8.51%, and 2.86%, respectively. Boundary+type F1 was 14.32%, 18.03%, and 14.40%; boundary-only F1 was 17.63%, 22.18%, and 17.68%. The learned CRF recovered both strict typed endpoints for only 2/491 gold relations (0.41%), versus 11/491 (2.24%) for hard BIO. Its language-micro strict entity F1 was 2.38% in English and 4.00% in Chinese. Entity evidence coverage and correctness remained 100%, with no relation objects emitted. The CRF decoder is therefore a completed negative ablation and motivated the single frozen span-boundary candidate below; no CRF relation head was justified.

### XLM-R span-boundary follow-up

The next frozen candidate replaced BIO sequence labeling with per-type token start/end heads and a typed span classifier including `NONE`. A 64-token maximum covered every aligned training span (observed maximum 49). A deterministic document-level 79/20 split within the training set selected the joint span/start/end threshold of 0.015; the final model was then reinitialized and trained on all 11,556 chunks for three epochs. The selection stage and final training used seed 20260830, batch size 8, accumulation 2, learning rate 1e-5, max length 192, stride 48, full-encoder training, and no formal-test access.

The span model emitted 8,115 validation entities and reached strict precision/recall/F1 of 3.77/18.84/6.28%. Boundary+type and boundary-only F1 were 10.84% and 17.24%; English and Chinese language-micro strict F1 were 7.42% and 5.43%. Typed endpoint reachability increased to 25/491 (5.09%), but strict F1 remained below hard BIO's 8.51%. Span-minus-hard-BIO pooled strict F1 was -2.23 points under 20,000 paired document resamples (95% CI -5.26 to +0.36; p=0.08395). The joint promotion gate was therefore not met, so this is a completed negative ablation and no relation head was trained.

The two-stage run took 1,358.5 seconds on an RTX 3090 and peaked at 10,713.6 MiB allocated CUDA memory. Reloaded inference over 434 windows took 44.51 seconds (102.56 ms/job; 110.20 chunks/s) and peaked at 2,417.1 MiB. Reloading reproduced the prediction artifact byte-for-byte. All emitted entity evidence was source-valid; the model emitted no relation objects.
- LaTeX compilation succeeds but reports an overfull repository URL and the empty bibliography; these are presentation issues in addition to the scientific blockers above.

A direct compatibility probe using the old KG adapter with the revised v2 graph prompt was also unsuccessful: the first validation window generated repeated explanatory text and exhausted its token budget without producing a JSON envelope. This indicates that changing the graph/prompt distribution requires adapter retraining or a compatibility gate; the old adapter cannot be treated as a drop-in evaluator for the revised KG.

## Sampled negative-sampling trial

A validation-only char-TF-IDF logistic relation verifier was trained on 1,510 positive relation examples and 4,544 weighted negatives, including legal random/type-valid pairs, reversed pairs, and wrong-relation-type hard negatives. At threshold 0.5, the baseline validation relation F1 changed from 1.01% to 0%, and the KG validation relation F1 changed from 0.52% to 0.55%. The change is not a meaningful improvement and was not applied to the frozen test predictions.

## Method revision required before full rerun

The next frozen checkpoint should use the V2 generator for recall, the V1-style entity agreement/source-anchor/verified-endpoint gate for abstention, and the deterministic evidence/signature verifier for the final relation graph. The validation-only local-pair classifier and its textual KG-feature variant should remain negative ablations, not components of the main system. Further KG work should target manually reviewed aliases and bilingual links, preserve `(language, entity_type)` in concept identity, retrieve concepts with typed semantic search, and calibrate explicit graph-prior features rather than embedding literal `yes/no` markers. For relation training, unobserved pairs should be treated as positive-unlabeled rather than definite negatives. No formal test result should be generated until this pipeline, its selection rule, and its primary comparison are frozen.
