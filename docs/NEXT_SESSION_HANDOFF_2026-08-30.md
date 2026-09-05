# Canonical next-session handoff and complete work plan

Last updated: 2026-08-30 (Asia/Shanghai)

Status: **authoritative entry point for the next session**

This document is self-contained. A new session should read it before opening
older reports. It records the frozen validation evidence, preserves every
important version and script, separates historical metrics from primary
metrics, and defines the remaining work before a one-time formal-test run.

## 1. Non-negotiable constraints

1. The repository is intentionally dirty. Do not commit, reset, clean, stash,
   revert, or delete existing files unless the user gives a new explicit
   instruction naming the exact operation.
2. Do not read, generate, score, summarize, or tune against the formal test
   split. Existing formal-test artifacts may be present; treat that namespace
   as sealed until the final protocol gate passes.
3. Use only training data, training-only inner splits, and the existing
   20-document validation artifacts during continued development.
4. BIO repair, learned CRF, and the frozen span-boundary model are completed
   negative ablations. Do not tune them further and do not train a relation
   head for the failed span candidate.
5. The BGE-M3 pair classifier, textual KG-feature classifier, and sampled
   lightweight relation verifier are also completed negative ablations. Do not
   promote them to the main system.
6. Do not overwrite or remove old predictions, metrics, audit logs, configs,
   model checkpoints, or scripts. New runs must use new versioned directories.
7. The primary metric is strict global-character-span one-to-one matching.
   Normalized-text and job-macro results are retained only as historical or
   diagnostic evidence.
8. Do not claim that deterministic filtering improves the generator itself.
   It guarantees properties only for the retained output.

## 2. Executive scientific decision

The paper should be positioned as a safety-oriented, provenance-preserving,
evidence-gated knowledge-graph extraction framework with conservative
abstention. It should not be positioned as a high-performance information
extraction model, a proven KG-prompting improvement, or a complete multi-agent
runtime.

Recommended working title:

> Provenance-Preserving Evidence-Gated Knowledge-Graph Augmentation for
> Low-Resource Safety Information Extraction

The word `Agents` and claims about grammar-constrained decoding should be
removed unless those components are actually implemented and evaluated.
Complex-network resilience should remain outside the claims unless a frozen
risk-substructure experiment is completed.

## 3. Dataset and protocol snapshot

- Corpus: 150 English and Chinese safety documents across railway and broader
  safety/emergency domains.
- Formal document split: 100 train / 20 validation / 30 test.
- Split policy: document-level, duplicate-cluster-aware, zero cross-split
  cluster leakage.
- Formal blocks: 244 train / 51 validation / 75 test.
- Primary validation set: 434 windows from 20 documents.
- Validation gold: 1,624 entities and 491 relations.
- Ontology: 15 entity types, 15 relation types, legal type signatures, claim
  status, source evidence, and provenance.
- XLM-R training: 11,556 chunks, full encoder, three epochs, seed 20260830.
- Formal test status for current model selection: untouched and prohibited.

Canonical data/config references:

- `configs/risk_ontology.yaml`
- `data/processed/reviewed/formal_split/split_manifest.jsonl`
- `data/processed/experiments/formal/windowed_train_v2/`
- `data/processed/experiments/formal/windowed_validation_v2/`
- `configs/kg_v3_frozen_validation.yaml`

Do not open test rows merely to verify this inventory.

## 4. Metric source of truth

The primary paper numbers are from JSON files whose metric is
`strict-global-character-span-one-to-one` and whose metadata states
`selection_split: validation` and `formal_test_read: false`.

Metric precedence:

1. `*_span_metrics.json` for strict entity and relation performance;
2. prediction-level `*_evidence_graph_metrics.json` for trustworthiness;
3. document-level 20,000-resample bootstrap JSON for uncertainty;
4. normalized-text, window-level, or job-macro values only as labelled
   diagnostics.

Important compatibility note: `configs/kg_v3_frozen_validation.yaml` retains
the earlier normalized selection record of entity/relation F1 13.27%/4.39%.
That record must stay unchanged for provenance. The final primary span-aware
V3 result is 12.64%/4.11% and is the number that belongs in the manuscript.
The current manuscript abstract still contains older 9.50%, 12.15%, 2.64%,
and 4.38% normalized figures and must be corrected before submission.

## 5. Frozen KG-version results

All values below are validation-only strict span-aware micro scores.

| System | Entity P/R/F1 | Relation P/R/F1 | Interpretation |
|---|---:|---:|---|
| Baseline | 10.71/7.82/9.04% | 1.61/0.20/0.36% | No-KG comparison |
| KG V1 | 12.71/6.03/8.18% | 2.17/0.20/0.37% | Conservative, low recall |
| KG V2 raw | 7.58/23.40/11.45% | 1.40/6.52/2.30% | Recall-oriented, substantial over-generation |
| KG V2 + verifier | 7.58/23.40/11.45% | 3.09/6.11/4.11% | Deterministic relation rejection |
| KG V3 raw gate | 11.85/13.55/12.64% | 2.74/6.11/3.78% | V1-style entity abstention before final verifier |
| KG V3 final | 11.85/13.55/12.64% | 3.10/6.11/4.11% | Frozen promoted validation pipeline |

V3 final language-micro results:

| Language | Entity F1 | Relation F1 |
|---|---:|---:|
| English | 14.16% | 5.73% |
| Chinese | 11.45% | 2.89% |

Correct interpretation:

- Absolute extraction accuracy is weak and is not production-ready.
- V2 recovers more entities and relations but loses substantial precision.
- V3 is valuable primarily as conservative relation selection and auditable
  output, not as evidence of high generator accuracy.
- The entity-gate gain is not statistically established.

Primary result artifacts:

| Version | Metrics | Predictions/audit |
|---|---|---|
| Baseline | `data/processed/experiments/formal/windowed_validation_v2/baseline_span_metrics.json` | `baseline_jobs.jsonl` and the frozen baseline prediction artifacts in the same directory |
| V1 | `data/processed/experiments/formal/windowed_validation_v2/kg_v1_span_metrics.json` | `data/processed/experiments/formal/validation_v3_kg_predictions_expanded.jsonl` |
| V2 raw | `data/processed/experiments/formal/windowed_validation_v2/kg_v2_span_metrics.json` | `kg_v2_predictions_expanded_complete.jsonl`, `kg_v2_audit.json`, inference logs |
| V2 verified | `data/processed/experiments/formal/windowed_validation_v2/kg_v2_verified_span_metrics.json` | `kg_v2_predictions_verified.jsonl`, `kg_v2_relation_verifier_audit.jsonl` |
| V3 raw | `data/processed/experiments/formal/windowed_validation_v2/kg_v3_fusion_raw_span_metrics.json` | `kg_v3_fusion_raw.jsonl`, `kg_v3_fusion_raw_audit.jsonl` |
| V3 final | `data/processed/experiments/formal/windowed_validation_v2/kg_v3_frozen/validation_span_metrics.json` | `kg_v3_frozen/final_predictions.jsonl`, both audit JSONL files, and `run_manifest.json` |

## 6. Statistical evidence

All comparisons use 20 validation documents, 20,000 document resamples, and
paired permutation tests.

| Comparison | Delta | 95% CI | p-value | Paper interpretation |
|---|---:|---:|---:|---|
| V2 - V1 entity F1 | +3.27 points | [-0.33, +6.70] | 0.11709 | Not significant |
| V2 - V1 relation F1 | +1.93 points | [+0.79, +3.22] | 0.02965 | Supported relation improvement |
| V3 raw - V2 raw entity F1 | +1.19 points | [-1.20, +3.31] | 0.42338 | Not significant |
| V3 raw - V2 raw relation F1 | +1.48 points | [+0.82, +2.45] | 0.00005 | Strong abstention/gating evidence |
| V3 final - V2 raw relation F1 | +1.81 points | [+1.03, +2.93] | 0.00005 | Strong combined-pipeline evidence |

Preserve:

- `data/processed/experiments/formal/windowed_validation_v2/kg_v1_v2_span_bootstrap.json`
- `data/processed/experiments/formal/windowed_validation_v2/kg_v2_v3_raw_span_bootstrap.json`
- `data/processed/experiments/formal/windowed_validation_v2/kg_v2_v3_span_bootstrap.json`
- `scripts/bootstrap_span_compare.py`

## 7. Evidence and safety results

Prediction-level micro aggregation is mandatory. Empty jobs must not be added
as zero-valued objects when calculating the primary evidence rate.

| System | Entity evidence correctness | Relation evidence correctness | Unsupported relation claims | Invalid relations |
|---|---:|---:|---:|---:|
| V2 raw | 99.96% | 58.32% | 41.68% | 57.62% |
| V3 final retained relations | -- | 100% | 0% | 0% |

The V3 result is guaranteed by deterministic rejection. The correct claim is
"all retained relations satisfy the implemented evidence and signature
checks," not "the generator is 100% accurate."

Preserve:

- `data/processed/experiments/formal/windowed_validation_v2/kg_v2_evidence_graph_metrics.json`
- `data/processed/experiments/formal/windowed_validation_v2/kg_v2_verified_evidence_graph_metrics.json`
- `data/processed/experiments/formal/windowed_validation_v2/kg_v3_frozen/validation_graph_metrics.json`
- `scripts/evaluate_evidence_graph.py`
- `scripts/verify_relations.py`

## 8. XLM-R, BIO, CRF, and span results

The original full unweighted XLM-R checkpoint reached 7.67% strict entity F1
under hard-BIO decoding. BIO audit found 139 illegal `I-*` transitions across
138 chunks: 92 chunk-initial and 47 after `O` or another entity type. Repairing
those labels was correct, but it did not make learned CRF competitive.

On the repaired checkpoint:

| Decoder/model | Strict entity F1 | Boundary + type F1 | Typed relation endpoints reachable |
|---|---:|---:|---:|
| Greedy | 2.84% | 14.32% | Not promoted |
| Hard BIO | 8.51% | 18.03% | 11/491 |
| Learned CRF | 2.86% | 14.40% | 2/491 |
| Span boundary | 6.28% | 10.84% | 25/491 |

The span model emitted 8,115 validation spans. Its strict precision/recall/F1
was 3.77/18.84/6.28%, boundary-only F1 was 17.24%, and English/Chinese strict
F1 was 7.42/5.43%. Span minus hard BIO was -2.23 points, 95% CI
[-5.26, +0.36], p=0.08395. The promotion gate failed.

Efficiency for the completed span ablation:

- Training: 1,358.5 seconds; 10,713.6 MiB peak allocated VRAM.
- Inference: 44.51 seconds for 434 jobs; 2,417.1 MiB peak allocated VRAM.
- Reloaded predictions match byte-for-byte.

Preserve:

- `data/processed/experiments/formal/windowed_validation_v2/xlmr_ner_full_unweighted_seed20260830/`
- `data/processed/experiments/formal/windowed_validation_v2/xlmr_ner_crf_repaired_seed20260830/`
- `data/processed/experiments/formal/windowed_validation_v2/xlmr_span_boundary_seed20260830/`
- `scripts/train_xlmr_ner_baseline.py`
- `scripts/run_xlmr_ner_inference.py`
- `scripts/xlmr_crf.py`
- `scripts/train_xlmr_span_ner.py`
- `scripts/run_xlmr_span_inference.py`

Decision: report these as a baseline and two negative ablations. Do not perform
more BIO/CRF/span validation tuning and do not add the span relation head.

## 9. Other completed negative or diagnostic experiments

Retain these results because they demonstrate experimental breadth and prevent
future sessions from repeating failed work:

1. BGE-M3 local pair classifier without KG: best validation relation F1 0.92%.
2. BGE-M3 textual KG-feature pair classifier: best relation F1 0.85%; KG-minus-
   no-KG difference not significant, p=0.7181.
3. Sampled char-TF-IDF/logistic verifier did not produce a meaningful gain.
4. The old KG adapter was incompatible with the revised V2 prompt and failed
   to return a JSON envelope in the compatibility probe.
5. The available llama.cpp build could not initialize the intended grammar
   sampler; ordinary generation plus deterministic validation was used.
6. Raw union retained too much V2 noise; intersection lost too much recall.

Preserve related scripts and artifacts:

- `scripts/train_v3_relation_classifier.py`
- `scripts/train_relation_verifier.py`
- `scripts/probe_semantic_kg_retrieval.py`
- `data/processed/experiments/formal/windowed_validation_v2/kg_v3_m3_no_kg/`
- `data/processed/experiments/formal/windowed_validation_v2/kg_v3_m4_with_kg/`
- `data/processed/experiments/formal/windowed_validation_v2/kg_v3_m3_vs_m4_bootstrap.json`

## 10. Frozen V3 method and essential scripts

Frozen pipeline:

```text
V2 high-recall generation
  -> deterministic evidence/signature relation verification
  -> V1-style entity reject-option gate
  -> final evidence-valid relation graph
```

An entity is retained when at least one auditable condition holds:

1. exact V1/V2 normalized `(text, type)` agreement;
2. source-gated V2 KG anchor with the same type; or
3. endpoint of a deterministically verified relation.

Core scripts/config that must remain independent and unchanged for the frozen
checkpoint:

- `configs/kg_v3_frozen_validation.yaml`
- `scripts/build_kg_v2_jobs.py`
- `scripts/fuse_kg_v1_v2_predictions.py`
- `scripts/verify_relations.py`
- `scripts/run_frozen_kg_v3.py`
- `scripts/evaluate_span_aware.py`
- `scripts/evaluate_relaxed_matching.py`
- `scripts/evaluate_evidence_graph.py`
- `scripts/bootstrap_span_compare.py`

Training, generation, windowing, and expansion scripts to preserve:

- `scripts/build_formal_experiment_assets.py`
- `scripts/build_paired_window_dataset.py`
- `scripts/train_qlora.py`
- `scripts/run_qlora_inference.py`
- `scripts/expand_compact_predictions.py`
- `scripts/merge_window_predictions.py`
- `scripts/rebind_window_instructions.py`

## 11. Defensible contributions

The supportable contribution bundle is:

1. Evidence-gated KG context used as a candidate prior while source text stays
   authoritative.
2. Leakage-safe graph construction with split, document, mention, relation,
   and source provenance plus leave-one-document-out training prompts.
3. Recall-abstention complementarity: V2 candidate recall plus V1-style
   agreement, anchors, and verified endpoints.
4. Deterministic evidence and signature verification for the retained graph.
5. Strict global-character-span one-to-one evaluation that preserves repeated
   mentions.
6. Prediction-level safety metrics exposing unsupported claims and illegal
   edges hidden by ordinary F1.
7. Failure-aware experimentation: failed generations, negative ablations,
   document bootstrap, immutable manifests, and no test tuning.
8. A 150-document bilingual/cross-domain safety corpus and typed ontology as a
   potential resource contribution, conditional on annotation reliability and
   data licensing.

Do not describe ordinary QLoRA, XLM-R, CRF, span heads, or generic KG prompting
as standalone innovations.

## 12. Current paper blockers

The project is not submission-ready. The main blockers are:

1. `paper/elsarticle/manuscript.tex` is largely a skeleton.
2. `paper/elsarticle/references.bib` is effectively empty.
3. The title claims LLM Agents, but the implemented main system is QLoRA plus
   deterministic post-processing.
4. The abstract uses obsolete normalized metrics rather than final span-aware
   results.
5. A reportable inter-annotator agreement and adjudication analysis is absent.
6. Main trainable systems have only one completed validation seed.
7. The 10/25/50/100-document scaling curve is absent.
8. Runtime, throughput, VRAM, token, and cost reporting is incomplete for the
   promoted baseline/V1/V2/V3 pipeline.
9. The final frozen formal-test evaluation has not been run.
10. Data licensing/release, limitations, ethics, error analysis, and verified
    bibliography sections are incomplete.
11. A viable end-to-end non-LLM relation baseline is absent. Do not respond by
    training a relation head over the failed span candidate; use hard BIO as an
    entity baseline and, if pre-registered, a clearly labelled oracle-gold-
    entity relation diagnostic for bottleneck analysis.

## 13. Complete next-step work plan

The next experiment is not another decoder. It is a frozen low-resource,
multi-seed study accompanied by annotation reliability and complete efficiency
reporting.

### Phase A: freeze the claim and protocol

Deliverable: `paper/PROTOCOL_FREEZE.md`.

Tasks:

1. Adopt the provenance/evidence/abstention paper positioning in Section 2.
2. Freeze the four research questions: extraction effectiveness,
   trustworthiness, bilingual/domain behavior, and low-resource efficiency.
3. Freeze the primary evaluator, prediction-level evidence aggregation,
   document bootstrap unit, failure-as-empty policy, language micro reporting,
   and claim-status-aware secondary metric.
4. Freeze all model hyperparameters from training or training-inner-dev data.
   Do not select new thresholds on formal validation.
5. Freeze three training seeds: 20260830, 20260831, and 20260901.
6. Freeze all file naming, run manifests, hashes, hardware telemetry, and stop
   rules before launching training.

Acceptance gate:

- Protocol contains no unresolved model choice or threshold.
- Every planned result row maps to a script, config, output directory, seed,
  and metric.
- Formal test remains sealed.

### Phase B: annotation reliability on training data only

Deliverables:

- a fixed, stratified training-only double-review manifest;
- two independent annotation files;
- disagreement queue and adjudicated result;
- `annotation_agreement_summary.json` and a manuscript-ready table.

Protocol:

1. Select at least 20 training documents, stratified by language and domain,
   without opening formal validation/test gold for selection.
2. Keep the two reviews independent until both are complete.
3. Report exact entity span agreement, entity type agreement conditional on
   matched spans, relation agreement conditional on matched endpoints,
   evidence-span/quote agreement, claim-status agreement, and adjudication
   rate. Report counts and confidence intervals; do not rely on one kappa value.
4. Preserve pre-adjudication annotations and every adjudication decision.
5. If agreement is weak, clarify annotation rules and adjudicate the training
   subset; do not silently rewrite formal validation or test labels.

Acceptance gate:

- The double-reviewed subset is complete and auditable.
- Agreement and adjudication statistics can be reproduced from saved inputs.
- No formal-test content has been read.

### Phase C: build frozen low-resource manifests

Deliverables:

- a deterministic script such as `scripts/build_low_resource_manifests.py`;
- nested 10/25/50/100-document train manifests;
- JSON summaries with language/domain/type coverage and SHA-256 hashes;
- focused tests for nesting, split isolation, determinism, and zero test access.

Protocol:

1. Construct nested subsets `10 subset 25 subset 50 subset 100`.
2. Preserve document and duplicate-cluster boundaries.
3. Balance language/domain coverage as far as the frozen training inventory
   permits; record rather than conceal unavoidable imbalance.
4. Never choose subsets according to validation F1.
5. Build all prompts/KG context from the selected training subset only. Keep
   leave-one-document-out KG construction for training prompts.

Acceptance gate:

- All four manifests are deterministic and hash-locked.
- There is no cross-split document or cluster leakage.
- Each run can reconstruct its exact training inputs from its manifest.

### Phase D: run the multi-seed scaling experiment

Trainable systems:

1. compact QLoRA baseline without KG;
2. KG V1 conservative generator;
3. KG V2 evidence-gated high-recall generator.

Derived systems require no extra model training:

4. V2 plus deterministic verifier;
5. V3 raw reject-option gate;
6. V3 final.

Full matrix: 4 budgets x 3 seeds x 3 trainable systems = 36 training runs,
followed by deterministic derivation and identical validation evaluation. Run
the 100-document three-seed rows first as an execution/telemetry gate, then run
10/25/50 if manifests and outputs pass structural checks. A negative outcome is
still reportable; do not modify the frozen method to rescue a weak seed.

For every run record:

- model and tokenizer hashes;
- training subset and seed;
- loss and generation success/failure;
- strict span-aware entity/relation P/R/F1;
- claim-status-aware relation F1;
- English and Chinese micro results;
- evidence correctness, unsupported claims, invalid signatures, JSON success;
- training/inference time, throughput, peak VRAM, output tokens, and estimated
  GPU/API cost;
- all predictions, logs, expansion errors, audits, and run manifests.

Acceptance gate:

- No missing or silently discarded run.
- Mean, standard deviation, per-seed values, and failure counts are reported.
- Main comparisons use identical subsets, seeds, prompts budgets, and metrics.

### Phase E: statistical and safety analysis

Tasks:

1. Plot validation learning curves for entity F1, relation F1, unsupported
   claims, and efficiency against 10/25/50/100 reviewed documents.
2. Report mean and standard deviation across seeds.
3. Use document-level paired bootstrap for the limited, pre-registered main
   comparisons; avoid interpreting a large collection of exploratory p-values.
4. Report verifier acceptance/rejection counts and error categories.
5. Separate English and Chinese results without converting job-macro values
   into language-micro claims.
6. Preserve low absolute F1 and negative ablations in the final paper.

Optional journal-fit diagnostic, only after the main matrix is stable:

- define safety motifs from the ontology and training data, such as
  hazard-cause-event and event-mitigation-barrier paths;
- freeze typed path definitions before validation evaluation;
- report node, edge, and path recovery without claiming network resilience.

### Phase F: freeze and run formal test once

Do not begin this phase until Phases A-E are complete.

Pre-test checklist:

- annotation reliability reported;
- primary and secondary metrics locked;
- all hyperparameters, seeds, system rows, and failure policies locked;
- low-resource and efficiency results complete;
- V3 config and ontology hashes recorded;
- table shells and analysis code prepared without test values;
- explicit user confirmation to unseal formal test.

Then execute one immutable formal-test batch containing every pre-registered
system/seed row. Save predictions and metrics once. Do not tune, choose a seed,
change a threshold, or regenerate after observing test results. Any execution
failure must remain a reported failure unless the protocol already defines a
mechanical retry rule.

### Phase G: finish the manuscript

1. Replace the title and abstract claims with the frozen framing and span-aware
   results.
2. Complete Introduction, Related Work, corpus/governance, method, experimental
   setup, results, ablations, discussion, limitations, ethics, and data/code
   availability.
3. Build a verified DOI-backed bibliography.
4. Include corpus/ontology statistics, main results, safety metrics, scaling,
   language analysis, efficiency, negative ablations, and representative
   evidence-valid/error examples.
5. State plainly that relation accuracy is low and that verifier guarantees
   apply only to retained outputs.
6. Resolve data redistribution and licensing before promising public release.

## 14. Document organization after this handoff

Use `docs/README.md` as the directory map.

Active source of truth:

- `docs/NEXT_SESSION_HANDOFF_2026-08-30.md`
- `docs/annotation-guidelines.md`
- `docs/paper-metadata.md`
- `paper/elsarticle/manuscript.tex`
- `paper/PROTOCOL_FREEZE.md`
- `paper/D100_EXECUTION_GATE.md`
- `paper/PROTOCOL_V2_AMENDMENT.md`
- `configs/low_resource_protocol_v2.yaml`
- `paper/D100_V2_EXECUTION_GATE.md`

Reference/provenance, not primary result sources:

- `docs/SESSION_HANDOFF_2026-08-30.md`
- `paper/EXPERIMENT_PLAN.md`
- `paper/RESULTS_DRAFT.md`
- `docs/preannotation-gate-report.md`
- `docs/research-requirements.md`

Historical mixed reports, excluded from the active workflow:

- `docs/experiment-report.md`
- `docs/work-plan.md`

No files were deleted during organization. This deliberately preserves old
version results and exact experimental provenance while giving the next
session one unambiguous entry point.

## 15. Last verified engineering state

The current checkpoint after the protocol v2 first-row execution gate is:

- 51 tests passing;
- both low-resource gate audit scripts pass `py_compile`;
- `compileall` passes for `scripts/` and `tests/`;
- `git diff --check` passes;
- the v1 d100 audit remains terminal and the v2 audit reports
  `passed_first_row`;
- the protocol v2 first-row gate passed and d100 rows are executing
  sequentially; use the machine gate audit for the live completed count, while
  lower budgets remain blocked until all nine pass;
- saved span predictions and the frozen low-resource assets remain preserved;
- formal test was not read and validation metrics were not calculated during
  the hardware gate;
- repository intentionally dirty;
- no commit created.

A new session should rerun relevant checks after implementing new scripts, but
must not clean or reset pre-existing changes.

## 16. Copy-ready first prompt for the new AI session

```text
请在当前项目目录 /ds1/workspace/ai/railway-agent-safety-security 中继续工作。

第一步请完整读取 docs/NEXT_SESSION_HANDOFF_2026-08-30.md 和 docs/README.md，再核对 git status、冻结配置以及交接文档列出的 validation artifacts。不要要求我重新解释旧实验。

严格约束：不要 commit、reset、clean、stash、revert 或删除现有文件；不要读取、生成、评分、汇总或调优 formal test；只可使用训练数据、训练内部划分和现有 20-document validation。仓库故意处于 dirty 状态，所有现有修改和未跟踪文件都要保留。不要继续调 BIO、learned CRF 或 span-boundary，也不要给失败的 span 模型训练 relation head；不要重复 BGE-M3 pair classifier、textual KG feature classifier 或 sampled lightweight verifier。

主论文定位已经冻结为 provenance-preserving、evidence-gated KG extraction with conservative abstention。不要把系统写成高性能 IE、完整 LLM Agents 或已实现 grammar-constrained decoding。主指标必须是 strict global-character-span one-to-one；旧 normalized/job-macro 指标只能作为历史诊断。V3 主验证结果是 Entity F1 12.64%、Relation F1 4.11%，不是冻结配置中保留的旧 13.27%/4.39%。

Phase A-C 已完成。Phase B 的两份真实人工复核仍待完成。Protocol v1 因 CUDA OOM 已终止，全部失败产物必须保持不变。Protocol v2 已冻结在 configs/low_resource_protocol_v2.yaml（SHA-256 `7ad5473c4dabbf7f3355ebf575209b041df9f47566e04c663ed6dc9a867405e5`），不得修改该配置及其中登记哈希的实现脚本；实质性变化必须另立 protocol v3。

Protocol v2 使用统一 `max_length=4096`、3,840-token 窗口构造上限、句子/段落感知切分、实体与关系证据保护、跨窗口关系 rescue windows，以及 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。权威窗口是 windowed_train_v6（1,266 个窗口，4,335/4,335 实体和 1,510/1,510 关系全覆盖）与 windowed_validation_v6（491 个窗口，1,001/1,001 实体和 343/343 关系全覆盖）。不要用 v3/v4/v5 中间窗口替代 v6。

首行 `lr_v2_d100_seed20260830_baseline` 已成功：1,266/1,266 样本、317 steps、零 prompt/answer truncation、零 overlength skip，峰值整卡显存 15,155 MiB。权威审计是 paper/D100_V2_EXECUTION_GATE.md 和 data/processed/experiments/formal/low_resource_v2/d100_execution_gate.json。

接下来不要只给建议，请继续 protocol v2 的剩余 d100 行：
1. 先检查 `systemctl --user status railway-low-resource-v2-d100.service`、GPU 和 machine gate audit；若该 unit 仍为 active，绝不能启动第二个控制器，只监控现有进程；
2. 若现有控制器已经结束，再运行完整单元测试、compileall、git diff --check，并对下一行执行 preflight；
3. 从 machine gate audit 中定位第一条 `not_started` 行，严格按 matrix 顺序串行运行剩余行；已经 `complete` 的行不得重跑，不得并行占用 GPU；
4. 每完成一行立即运行 `python scripts/audit_low_resource_v2_training_gate.py`，核对 complete、adapter、metrics、telemetry、零截断/零跳过和 formal_test_read=false；
5. 任一行失败立刻停止。只有满足冻结机械重试条件时才能使用许可的 retry，不得自行改参；
6. 九个 d100 行全部完成并通过前，不得启动 d10/d25/d50；不要启动 formal test，也不能伪造 IAA。

所有新产物必须写入 matrix 指定的独立目录，保留 predictions、metrics、audit、logs、configs 和 scripts。不要提交、清理或重置工作区。
```

## 17. Phase A-C implementation update

Implemented on 2026-08-30 without reading formal test rows and without starting
model training. Frozen validation source windows were used only to materialize
budget-specific prompts; validation scores were not used for subset selection:

- Phase A protocol is frozen in `paper/PROTOCOL_FREEZE.md` and
  `configs/low_resource_protocol_v1.yaml`. It fixes the four RQs, three seeds,
  36 trainable runs, 12 deterministic derivation groups, hyperparameters,
  metrics, bootstrap policy, failure handling, telemetry, output naming, and
  the explicit formal-test gate.
- Phase B infrastructure is ready at
  `data/processed/reviewed/annotation_agreement_v1/`. It contains a fixed,
  cluster-closed, training-only sample of 20 documents (11 English, 9 Chinese;
  12 category strata), 58 complete source annotation units (covering 220
  effective training windows), two independent label-free reviewer templates,
  instructions, hashes, and a document-bootstrap agreement
  evaluator. Two real human reviews and adjudication remain pending; no IAA
  value has been fabricated.
- Phase C manifest gate passed. The deterministic nested 10/25/50/100 manifests
  contain exactly 10/25/50/100 documents, have no partial duplicate cluster,
  and cover all 15 entity and 15 relation types at every budget. The complete
  36-run and 12-group derived matrices are in
  `data/processed/experiments/formal/low_resource_manifests_v1/`.
- Train-only subset graphs, aligned training window/gold assets, training and
  validation V1 LODO prompts, and hash manifests were materialized below
  `data/processed/experiments/formal/low_resource_v1/d*/assets/`. The historical
  window builder provides effective windows for 99 of the 100 formal training
  documents; `doc_5999b31d5117e686` has no retained window and remains explicitly
  present in the 100-document manifest rather than being replaced or hidden.
- New focused tests cover determinism, exact nesting, cluster closure,
  train-only path rejection, subset-KG provenance, agreement counting, blank
  reviewer templates, and the sealed-test protocol.
- Final verification after implementation: 48 tests passed, `compileall`
  passed, generated manifest/package hashes were identical across rebuilds,
  the matrices contain 36 unique trainable runs and 12 unique derived groups,
  and `git diff --check` passed.

Phase D was subsequently attempted and terminated at its hardware gate; see
Section 18. Phase B remains pending. Do not run formal test or modify protocol
v1 to rescue the failed run.

## 18. Phase D 100-document execution-gate update

Executed on 2026-08-30 without reading formal test and without calculating any
validation score. The immutable matrix executor and exact training telemetry
were added before the gate. All nine d100 rows passed input, configuration,
implementation-hash, and CUDA preflight checks.

- KG V2 d100 prompts were materialized for 986 training jobs and 434 validation
  jobs. Their SHA-256 values are
  `cf37781bd56ba8317c226eb5a646a91aee5364cddc324af447795a96872a8f1a` and
  `f52cfe69a22b8d68f345081be273e7df4c2e1734caedd485ac0a56b8a7ba6b1c`;
  both are byte-identical to the corresponding frozen historical assets.
  Training prompts retain train-subset-only graph provenance and
  leave-one-document-out context.
- `lr_v1_d100_seed20260830_baseline` first failed before model loading because
  `torch.cuda.reset_peak_memory_stats(0)` was incompatible with the installed
  PyTorch build. It produced no model or training output and qualified for the
  one frozen mechanical retry.
- `lr_v1_d100_seed20260830_baseline_retry01` used identical model, data, seed,
  and training parameters. It prepared all 986 examples, with maximum sequence
  length 5,027 and no truncation or overlength drop, then failed during the first
  backward pass with CUDA out-of-memory.
- The retry reached 23,952 MiB peak whole-device memory, 100% peak GPU
  utilization, 364.4 W peak draw, and an estimated 0.016121 kWh. No adapter
  checkpoint or `training_metrics.json` was emitted.
- Retained maximum lengths are 5,027 tokens for baseline, 5,086 for KG V1, and
  5,007 for KG V2. Because all occupy the same frozen memory-risk range, the
  remaining eight d100 rows were not launched. All d10/d25/d50 rows remain
  unstarted.
- The machine audit is
  `data/processed/experiments/formal/low_resource_v1/d100_execution_gate.json`;
  the readable report is `paper/D100_EXECUTION_GATE.md`. The audit records one
  `failed_terminal` row, eight `not_started` rows, two attempts, and zero
  remaining rows launched.
- Verification after the audit: 49 tests, audit-script `py_compile`, full
  `compileall`, and `git diff --check` passed. The audit's exit code 2 is
  expected because the gate failed.

Protocol v1 is now terminal under its frozen retry policy. Do not alter or
resume it, lower the sequence cap, change batching, discard the failed row, or
start lower budgets under v1. A further attempt requires a new versioned,
runtime-only amendment frozen before execution and applied identically to all
systems and seeds. Phase B remains pending on real human input, and formal test
remains sealed.

## 19. Protocol v2 semantic-window and first-row update

Protocol v2 was frozen and its first d100 row was executed on 2026-08-30 without
reading formal test or calculating validation metrics. Its machine-readable
configuration is `configs/low_resource_protocol_v2.yaml` (SHA-256
`7ad5473c4dabbf7f3355ebf575209b041df9f47566e04c663ed6dc9a867405e5`), and
the human-readable change record is `paper/PROTOCOL_V2_AMENDMENT.md`. Protocol
v1 and all its failure artifacts remain unchanged.

- Training uses a uniform hard `max_length` of 4,096 for every system and seed.
  Window construction is capped at 3,840 complete prompt-plus-gold tokens, with
  paragraph/sentence-aware splitting, 800-character segment caps,
  160-character overlap, protected entity/evidence spans, and evidence-focused
  rescue windows for cross-window relations. Truncation and overlength skipping
  are prohibited.
- `windowed_train_v6` is the canonical training asset: 1,266 windows (980
  sequential and 286 relation rescue), covering 4,335/4,335 entities and
  1,510/1,510 relations. `windowed_validation_v6` contains 491 windows (434
  sequential and 57 rescue), covering 1,001/1,001 entities and 343/343
  relations. Intermediate v3/v4/v5 assets are provenance only and must remain
  preserved.
- Exact d100 maximum complete sequence lengths are 3,165 for baseline, 3,968
  for KG V1, and 3,902 for KG V2. All systems retain the same 1,266 examples,
  with zero sequences over 4,096.
- `lr_v2_d100_seed20260830_baseline` completed all 1,266 examples and 317
  optimizer steps. Mean loss was 0.1728524578; training took 5,282.756 seconds.
  Peak CUDA allocated/reserved memory was 12,025.17/14,378 MiB and peak
  whole-device memory was 15,155 MiB. Energy was 0.534164 kWh. Prompt
  truncation, answer truncation, and overlength skipping were all zero.
- The adapter SHA-256 is
  `d98e499063a71e1a718e344612c7cb0f9659c6416284c8760541d0f8ef3a1f69`.
  The run is preserved at
  `data/processed/experiments/formal/low_resource_v2/d100/seed20260830/baseline/`.
- At the first-row gate snapshot, one d100 row was complete and eight were not
  started. Subsequent d100 rows are authorized sequentially; the machine audit
  is the source of truth for live progress. d10/d25/d50 remain blocked until
  all nine d100 rows pass artifact and telemetry checks.

The authoritative progress report is `paper/D100_V2_EXECUTION_GATE.md`; the
machine audit is
`data/processed/experiments/formal/low_resource_v2/d100_execution_gate.json`.
Any substantive change to the frozen v2 configuration or its registered
implementation hashes now requires protocol v3. Phase B still requires two
genuine independent reviews, and formal test remains sealed.

After the second d100 row passed, the remaining seven rows were placed under
the fail-fast sequential controller
`scripts/run_remaining_low_resource_v2_d100.sh` (SHA-256
`c501feea45782c9de8b7768de1bb60f416986d12053f0e1178e7fc545ebb62bd`).
It runs as the user service `railway-low-resource-v2-d100.service`, audits every
completed row, stops on any run/audit/artifact failure, and never starts lower
budgets. Its preserved log is
`data/processed/experiments/formal/low_resource_v2/remaining_d100_controller_attempt02.log`.
The earlier plain-`nohup` controller attempt was cleaned up by the execution
environment before preflight completed; it created no run directory, and its
one-line provenance log remains at `remaining_d100_controller.log`.

## 20. Deliberate background-service pause

At the user's request, the active controller was frozen on 2026-08-31 rather
than terminated. The user service
`railway-low-resource-v2-d100.service` has `FreezerState=frozen`; its current
row is `lr_v2_d100_seed20260831_kg_v2`, preserved at step 215/317. Five earlier
d100 rows are complete and audited. The frozen CUDA process retains about
18,130 MiB device memory but uses 0% GPU compute while paused.

In the next session, first run:

```bash
systemctl --user show railway-low-resource-v2-d100.service \
  --property=ActiveState,SubState,MainPID,FreezerState,Result
nvidia-smi
```

If the unit still reports `ActiveState=active` and `FreezerState=frozen`, resume
the exact preserved process in the background with:

```bash
systemctl --user thaw railway-low-resource-v2-d100.service
```

Then verify `FreezerState=running`, GPU utilization returns, and the current
training log advances beyond step 215. Do not launch another controller, do not
rerun the current row, and do not delete its partial output directory. Network
isolation does not require stopping this local user-systemd service. If the
unit has disappeared because the host or user manager restarted, do not assume
a retry is authorized: preserve the partial directory and audit the interruption
against the frozen retry policy before taking any further action.
