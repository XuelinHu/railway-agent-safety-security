# Frozen protocol: low-resource evidence-gated safety extraction

Protocol ID: `low-resource-provenance-study-v1`  
Frozen: 2026-08-30 (Asia/Shanghai)  
Status: **frozen before multi-seed training**  
Machine-readable specification: `configs/low_resource_protocol_v1.yaml`

## 1. Claim and scope

Working title:

> Provenance-Preserving Evidence-Gated Knowledge-Graph Augmentation for
> Low-Resource Safety Information Extraction

The paper studies a safety-oriented, provenance-preserving, evidence-gated KG
extraction framework with conservative abstention. It does not claim a
high-performance extraction model, a complete LLM-agent runtime,
grammar-constrained decoding, production readiness, or complex-network
resilience. QLoRA, XLM-R, CRF, span heads, and generic KG prompting are not
standalone innovations.

The deterministic verifier claim is limited to retained output: every retained
relation satisfies the implemented local-evidence and ontology-signature checks.
It is not a claim that the generator or retained relation semantics are 100%
accurate.

## 2. Frozen research questions

1. **RQ1 — Extraction effectiveness.** Under strict global-character-span
   one-to-one matching, how do the no-KG baseline, KG V1, KG V2, V2 plus
   verification, V3 raw, and V3 final compare on entity and relation extraction?
2. **RQ2 — Trustworthiness.** How do KG context, deterministic verification,
   and abstention affect evidence correctness, unsupported relation claims,
   illegal signatures, JSON success, and rejection/acceptance counts?
3. **RQ3 — Bilingual and domain behavior.** How stable are extraction and
   safety results for English versus Chinese and across the frozen source/domain
   groups, without treating job-macro values as language-micro results?
4. **RQ4 — Low-resource efficiency.** How do effectiveness, trustworthiness,
   runtime, throughput, peak VRAM, tokens, and estimated cost change at nested
   budgets of 10, 25, 50, and 100 reviewed training documents?

No additional primary RQ will be introduced after observing validation or test
results. A typed safety-motif diagnostic is optional and secondary; if used, its
path definitions must be frozen before evaluation.

## 3. Data boundary and subset construction

- Formal split: 100 train / 20 validation / 30 sealed test documents.
- Development may read training data, training-only inner partitions, and the
  already-established 20-document validation artifacts.
- Formal test data, rows, predictions, metrics, and summaries remain sealed.
- The nested subsets are exactly `10 ⊂ 25 ⊂ 50 ⊂ 100` documents.
- Document and duplicate-cluster boundaries are indivisible.
- Selection balances language and source/category strata as far as exact nested
  cluster sizes permit. Entity/relation type coverage is a deterministic
  secondary tie-breaker. Validation performance is never a selection input.
- Selection salt: `low-resource-v1-20260830`.
- Every manifest and source file is SHA-256 locked by
  `scripts/build_low_resource_manifests.py`.
- Subset KGs contain only mentions/relations from selected training documents.
  Training prompts exclude graph evidence whose only provenance is the current
  document (leave-one-document-out).

Canonical outputs:

- `data/processed/experiments/formal/low_resource_manifests_v1/summary.json`
- `data/processed/experiments/formal/low_resource_manifests_v1/train_documents_010.jsonl`
- `data/processed/experiments/formal/low_resource_manifests_v1/train_documents_025.jsonl`
- `data/processed/experiments/formal/low_resource_manifests_v1/train_documents_050.jsonl`
- `data/processed/experiments/formal/low_resource_manifests_v1/train_documents_100.jsonl`
- `data/processed/experiments/formal/low_resource_manifests_v1/run_matrix.jsonl`
- `data/processed/experiments/formal/low_resource_manifests_v1/derived_matrix.jsonl`

The existing paired training-window inventory contains 99 effective documents
because empty windows were dropped. The 100-document manifest nevertheless
retains all 100 formal training documents and explicitly reports any document
without an effective window. Such a document is not silently replaced.

## 4. Annotation reliability protocol

`scripts/build_annotation_agreement_manifest.py` deterministically selects at
least 20 cluster-closed training documents, stratified by language and domain,
and creates a source-only review queue plus two label-free reviewer templates.
Formal validation/test gold is not used for selection.

The reviewers work independently until both files are frozen. The teacher model
and the existing gold annotation are not final judges. Pre-adjudication reviewer
files, the disagreement queue, and every adjudication decision are immutable
inputs to the reported summary.

`scripts/evaluate_annotation_agreement.py` reports counts and 95% confidence
intervals for:

1. exact global entity-span agreement;
2. entity-type agreement conditional on matched spans;
3. relation agreement conditional on matched endpoints;
4. entity evidence span/quote agreement;
5. relation evidence span/quote agreement;
6. claim-status agreement conditional on matched relations; and
7. adjudication completion rate.

Confidence intervals use 20,000 document-level bootstrap resamples with seed
20260830. No single kappa coefficient substitutes for these measures. Until two
real humans complete the templates, Phase B is reported as pending and no IAA
number is invented.

## 5. Frozen systems and matrix

Trainable systems:

| ID | Training input | KG behavior | Training script |
|---|---|---|---|
| `baseline` | Compact paired windows | No KG context | `scripts/train_qlora.py` |
| `kg_v1` | Same compact targets/windows | Exact-match concept hints, legal signatures, LODO provenance | `scripts/train_qlora.py` |
| `kg_v2` | Same compact targets/windows | Anchors, edge priors, semantic relation patterns, evidence gate, LODO provenance | `scripts/train_qlora.py` |

Derived systems require no additional training:

| ID | Inputs | Script(s) |
|---|---|---|
| `kg_v2_verified` | KG V2 predictions | `scripts/verify_relations.py` |
| `kg_v3_raw` | aligned KG V1, KG V2, and verified V2 predictions | `scripts/fuse_kg_v1_v2_predictions.py` |
| `kg_v3_final` | same aligned predictions and frozen acceptance rules | `scripts/run_frozen_kg_v3.py` |

The training matrix is exactly 4 budgets × 3 seeds × 3 trainable systems = 36
runs. Seeds are 20260830, 20260831, and 20260901. Each of the 12 budget/seed
groups produces the three derived rows. The generated `run_matrix.jsonl` maps
every trainable result row to its document manifest/hash, asset builder, trainer,
inference runner, evaluators, and unique output directory. The generated
`derived_matrix.jsonl` does the same for derived rows.

Execution begins with all nine 100-document trainable runs. The 10/25/50 rows
start only if their structural, provenance, and telemetry checks pass. A weak or
negative seed is retained and reported rather than rescued through retuning.

## 6. Frozen model, training, and prompt settings

Base model and tokenizer are the local Qwen3-4B snapshot at revision
`1cfa9a7208912126459214e8b04321603b3df60c`. Training uses local files only.
The frozen execution environment is the existing `rc-llm-comet` Python 3.11.15
environment with PyTorch 2.11.0+cu130, Transformers 4.57.6, PEFT 0.18.1,
Accelerate 1.13.0, and bitsandbytes 0.49.2. The executor refuses a run when a
frozen implementation hash, input hash, split boundary, or CUDA preflight fails.

All three trainable systems use:

- one epoch, batch size 1, gradient accumulation 4;
- AdamW learning rate `2e-4`;
- maximum sequence length 5,120;
- QLoRA rank 8, alpha 16, dropout 0.05;
- NF4 double quantization and bfloat16 compute;
- target modules `q_proj`, `k_proj`, `v_proj`, `o_proj`, `up_proj`,
  `down_proj`, and `gate_proj`;
- compact targets, length bucketing (bucket size 32), and skip-overlength;
- the matrix seed as the Torch/data-order seed.

The 5,120-token common cap is frozen for comparison parity using the already
successful KG V1/V2 training configuration; it was not selected by a new formal
validation sweep.

KG V1 uses 20 balanced exact concept hints and leave-one-document-out source
provenance. KG V2 uses the frozen BGE-M3 snapshot revision
`5617a9f61b028005a4858fdac845db406aefb181`, semantic threshold 0.72, semantic
limit 4, anchor limit 12, two anchors per type, edge limit 6, minimum type purity
0.8, and minimum content lengths 4 English / 2 Chinese characters. No threshold
will change after observing a seed.

Inference is greedy with maximum input 12,288 tokens, maximum output 4,096
tokens, 180 seconds per job, repeated-entity stop count 6, and complete-JSON
stopping enabled. Prompt and generation budgets are identical for like-for-like
system comparisons.

## 7. Metrics and aggregation

Primary entity and relation metrics use
`scripts/evaluate_span_aware.py` and strict global-character-span one-to-one
matching. Repeated mentions remain distinct. Primary values are
prediction-level micro precision, recall, and F1; job-macro and normalized-text
metrics are diagnostics only.

Secondary metrics are:

- claim-status-aware strict relation precision/recall/F1;
- English and Chinese micro results;
- evidence correctness, unsupported relation claims, illegal signatures, and
  verifier acceptance/rejection counts from
  `scripts/evaluate_evidence_graph.py` and audit logs;
- JSON success/failure and expansion failure counts.

Missing or failed generations are empty predictions in effectiveness scoring.
No run or job is silently dropped. For evidence-rate micro aggregation, an
empty job contributes no unsupported or supported claim object; it is not added
as a zero-valued per-job record.

## 8. Statistical analysis

- Report every seed, the mean, standard deviation, and failure count.
- Use documents—not windows or jobs—as resampling units.
- Use 20,000 paired document bootstrap resamples and paired permutation tests
  for the pre-registered comparisons only: KG V2 versus V1, V3 raw versus V2,
  and V3 final versus V2.
- Report 95% intervals and exact p-values without treating non-significance as
  equivalence.
- Plot validation learning curves for entity F1, relation F1, unsupported
  claims, and efficiency at 10/25/50/100 documents.
- English and Chinese results remain separate micro aggregates.

## 9. Telemetry and cost

Every run records wall-clock training/inference time, examples or jobs per
second, peak allocated and reserved VRAM, input/output tokens, generation
success/failure, model/tokenizer revision, manifest hash, seed, and estimated
GPU/API cost. Estimates state the price assumption and measurement timestamp.
No zero-cost claim is inferred from local execution.

For local GPU runs, the cost estimate is explicitly electricity-only: GPU power
is sampled every two seconds, integrated over wall-clock time, and multiplied by
the frozen accounting assumptions CNY 0.60/kWh and USD/CNY 0.14 (frozen
2026-08-30). Hardware amortization and labor are excluded and stated as such.

## 10. Output naming and immutability

Run root:

```text
data/processed/experiments/formal/low_resource_v1/
  d{010|025|050|100}/
    seed{20260830|20260831|20260901}/
      {baseline|kg_v1|kg_v2|derived}/
```

Each trainable run preserves `run_manifest.json`, training metrics/log,
inference log, compact and expanded predictions, expansion errors, strict span
metrics, evidence metrics, and telemetry. Derived directories preserve verifier
and entity-gate audits as well as all predictions. Outputs are never overwritten;
hash mismatches require a new versioned directory.

## 11. Mechanical retry and stopping rules

One mechanical retry is allowed only for an infrastructure failure that occurs
before model output is recorded. It must reuse the identical checkpoint, prompt,
seed, decoding parameters, and job. Generation failures, malformed output,
timeouts after generation begins, and repetitions remain reportable failures.

Do not:

- tune hyperparameters or thresholds on formal validation;
- choose the best seed as the reported system;
- discard negative or failed runs;
- further tune BIO repair, learned CRF, or the span-boundary model;
- train a relation head over the failed span candidate;
- repeat the negative BGE pair, textual KG-feature, or sampled verifier studies;
- read or run formal test before the gate below.

## 12. Formal-test gate

Formal test can be unsealed only after all of the following are true:

- two independent reviews, reproducible IAA, disagreement queue, and
  adjudication record are complete;
- all 36 trainable runs and 12 deterministic derivation groups are present;
- means, standard deviations, learning curves, safety metrics, and telemetry are
  complete;
- all hashes, settings, retry/failure policies, table shells, and analysis code
  are frozen;
- ontology and V3 rules match their recorded hashes;
- tests, compilation checks, and diff checks pass; and
- the user explicitly authorizes the one-time formal-test batch.

The test batch then executes every pre-registered row once. No seed selection,
threshold change, regeneration, or post-test method revision is permitted.
