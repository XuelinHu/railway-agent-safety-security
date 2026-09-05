# Session handoff: KG V1/V2 fusion and XLM-R follow-up

> Status: reference snapshot. The canonical next-session entry point is
> `docs/NEXT_SESSION_HANDOFF_2026-08-30.md`. Preserve this file for provenance,
> but use the new handoff for current metrics and remaining work.

Last updated: 2026-08-30 (Asia/Shanghai)

## Scope and safety constraints

- The repository is intentionally dirty and contains the user's work from several sessions. Preserve all existing modifications and untracked files.
- Do not commit, reset, clean, stash, or revert anything unless the user explicitly changes this instruction.
- Do not read, generate, score, or tune against the formal test split. Continue with training data and the 20-document validation split only.
- The previous model-provider/session configuration may be unavailable. All claims below are recoverable from local scripts and artifacts; do not depend on the old chat transcript.
- BIO repair, learned CRF, and the first frozen span-boundary candidate are now implemented and evaluated. All three follow-ups are completed validation ablations; hard BIO remains the strongest XLM-R entity checkpoint.

## Frozen V1/V2 conclusion

V1 and V2 are complementary rather than interchangeable:

- V1 is conservative and precision-oriented, but misses many candidates.
- V2 uses evidence-gated KG context and is recall-oriented, but over-generates entities and especially relations.
- A raw union retains too much V2 noise; an intersection loses too much recall.

The validation-selected combined pipeline is therefore frozen as:

```text
V2 high-recall generation
  -> deterministic evidence/signature relation verifier
  -> V1-style entity reject-option gate
```

The entity gate keeps a V2 entity if at least one auditable signal holds:

1. exact V1/V2 `(text, type)` agreement;
2. a source-gated V2 KG anchor with the same type; or
3. the entity is an endpoint of a deterministically verified relation.

This is implemented by:

- `configs/kg_v3_frozen_validation.yaml`
- `scripts/fuse_kg_v1_v2_predictions.py`
- `scripts/run_frozen_kg_v3.py`
- `scripts/verify_relations.py`

Do not tune these rules further on formal test data.

## Primary span-aware validation results

The old normalized-text evaluator collapses repeated mentions. The primary evaluator is now strict global-character-span, one-to-one matching:

- `scripts/evaluate_span_aware.py`
- Gold spans are recovered with `parent_job_id + original entity id + provenance evidence`.
- All 1,624 validation gold entities resolve uniquely.
- Non-validation data are rejected by default.

| System | Entity F1 | Relation F1 |
|---|---:|---:|
| Baseline | 9.04% | 0.36% |
| V1 | 8.18% | 0.37% |
| V2 raw | 11.45% | 2.30% |
| V2 + verifier | 11.45% | 4.11% |
| V3 raw entity gate | 12.64% | 3.78% |
| V3 final | **12.64%** | **4.11%** |
| XLM-R hard BIO | 7.67% | 0% |

V3 final language-micro results:

- English Entity/Relation F1: 14.16% / 5.73%
- Chinese Entity/Relation F1: 11.45% / 2.89%

The main V3 artifact is:

- `data/processed/experiments/formal/windowed_validation_v2/kg_v3_frozen/validation_span_metrics.json`

## Paired span-aware statistics

Implemented in `scripts/bootstrap_span_compare.py`, using 20 validation documents and 20,000 iterations:

- V2 minus V1 entity F1: +3.27 points, 95% CI [-0.33, +6.70], p=0.1171.
- V2 minus V1 relation F1: +1.93 points, CI [+0.79, +3.22], p=0.0296.
- V3 raw gate minus V2 raw entity F1: +1.19 points, CI [-1.20, +3.31], p=0.42338.
- V3 raw gate minus V2 raw relation F1: +1.48 points, CI [+0.82, +2.45], p=0.00005.
- V3 final minus V2 raw relation F1: +1.81 points, CI [+1.03, +2.93], p=0.00005.

Interpretation: the significant relation improvement comes from V1-style endpoint abstention/gating. The verifier primarily guarantees evidence and structural validity; it is not evidence that the generator itself became safe.

## Corrected evidence metrics

Evidence rates must use prediction-level micro aggregation. Empty jobs must not be inserted as zeros into the primary rate.

- V2 entity/relation evidence correctness: 99.96% / 58.32%.
- V2 unsupported-claim/invalid-relation rates: 41.68% / 57.62%.
- V3 relation evidence correctness: 100%.
- V3 unsupported-claim/invalid-relation rates: 0% / 0%.

The V3 values are achieved by deterministic rejection and must not be described as intrinsic generator accuracy.

## Current XLM-R diagnostic

The full XLM-R run used all 11,556 training chunks, no class weighting, full-model fine-tuning, 3 epochs, and seed 20260830.

- Greedy normalized Entity F1: 2.71%.
- Hard-BIO normalized Entity F1: 8.01%.
- Hard-BIO span-aware Entity F1: 7.67%.
- Relaxed boundary + type F1: 17.64%.
- Boundary-only F1: 21.60%.
- Only 19/491 validation gold relations have both strict typed endpoints reachable (3.87%).

Artifact:

- `data/processed/experiments/formal/windowed_validation_v2/xlmr_ner_full_unweighted_seed20260830/`

Latest diagnosis:

- 1,601/1,624 validation gold spans align to both XLM-R token boundaries; 15 align only at the start and 8 only at the end. Tokenizer boundary mismatch is not the primary failure.
- Before repair, training labels contain 7,103 `B`, 34,401 `I`, and 139 illegal `I` transitions.
- Of those illegal transitions, 92 are chunk-initial `I` and 47 follow `O` or another entity type.

Relevant files:

- `scripts/train_xlmr_ner_baseline.py`
- `scripts/run_xlmr_ner_inference.py`
- `tests/test_pipeline.py`

## Completed XLM-R follow-ups

1. BIO repair converted 139 interrupted/chunk-initial `I-*` tags across 138 chunks and is covered by a focused test.
2. The learned CRF was trained jointly with the full encoder but reached only 2.86% strict span F1; hard BIO reached 8.51% on the same repaired checkpoint.
3. The frozen span-boundary candidate used start/end heads plus a typed span classifier, a document-level training-only inner-dev split, and full-data refitting. Its selected threshold was 0.015.
4. The span model reached strict P/R/F1 3.77/18.84/6.28%, boundary+type F1 10.84%, boundary-only F1 17.24%, and typed endpoint reachability 25/491 (5.09%). English/Chinese strict F1 was 7.42/5.43%.
5. Span-minus-hard-BIO pooled strict F1 was -2.23 points (95% CI [-5.26, +0.36], p=0.08395). The strict-F1 promotion gate failed even though the endpoint count reached its target. Do not train a relation head or tune this span candidate further on validation.
6. The saved span model reproduces its prediction file byte-for-byte. Training took 1,358.5 seconds with 10,713.6 MiB peak allocated VRAM; inference took 44.51 seconds for 434 jobs with 2,417.1 MiB peak allocated VRAM.

Artifact:

- `data/processed/experiments/formal/windowed_validation_v2/xlmr_span_boundary_seed20260830/`

The last verified repository checks were 41 passing tests, `py_compile` success, and `git diff --check` success. Do not create a commit.

## Supportable paper innovation

The defensible core contribution is not simply “KG prompting improves extraction.” It is a safety-oriented, provenance-preserving KG extraction framework in which:

1. graph context is an evidence-gated candidate prior rather than a source of truth;
2. training graph construction is split-safe and keeps document provenance;
3. V2 supplies recall while V1-style agreement/anchor/verified-endpoint signals implement conservative abstention;
4. a deterministic verifier guarantees source evidence and legal typed relation structure;
5. span-aware one-to-one evaluation and prediction-level safety metrics expose repeated mentions and unsupported graph claims that normalized set metrics hide.

The entity-gate gain is not statistically significant, so the paper should emphasize the significant relation improvement and trustworthiness/reject-option design, not claim a proven overall entity improvement.

## Remaining paper blockers

- The manuscript is still partly a skeleton and the bibliography is empty.
- The first learned span-boundary model is a completed negative ablation. B1 still has no viable relation head or independent successful seeds; do not spend additional validation budget on this failed candidate.
- Scaling curves (10/25/50/100 documents), at least three trainable seeds, and efficiency/VRAM/latency reporting remain incomplete.
- Annotation agreement/adjudication is still required.
- Claims involving a full multi-agent or grammar-constrained runtime must be implemented and evaluated or removed.

## Suggested first message for the new session

```text
请先完整读取 docs/SESSION_HANDOFF_2026-08-30.md，并核对 git status 和其中列出的本地 validation artifacts。不要提交代码，不要 reset/clean/stash，不要读取或评估 formal test。BIO repair、learned CRF 和首个 span-boundary 候选均已完成且是负消融，不要继续调它们，也不要训练 relation head。下一步先审计论文剩余 blocker 和可用的训练内层协议，再提出不重复消耗 validation 的后续实验；不要重复已经完成的 V1/V2 分析。
```
