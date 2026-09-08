# Full Public-Benchmark Run Plan

This plan is for the branch `experiment/public-small-benchmarks`. It evaluates
the existing method on the complete public CoNLL04, SciERC, and ADE splits
without mixing them with the railway benchmark.

## Fixed protocol

1. Use the complete published train/validation/test files.
2. Run one fixed seed (`42`) for three systems: `baseline`, `kg_prior`, and
   `pge` (KG prior plus evidence expansion and signature verifier).
3. Select systems using validation only. The main promotion condition is that
   PGE does not show a material relation-F1 regression against baseline while
   unsupported and illegal retained relations remain controlled.
4. Run the sealed test split only for the two promoted systems.
5. Repeat only `baseline` and `pge` with seeds `2026` and `3407`.
6. Run the registered external comparisons with the same complete public
   splits, document IDs, and evaluation code.

No threshold, prompt, checkpoint, or system selection may be changed after the
test run starts. Report every generation failure in the denominator.

### Pre-test promotion rule operationalization (2026-09-04)

Before any formal-test metric was computed, “no material regression” was made
operational as a validation point-estimate rule: for every dataset,
`PGE relation F1 - SOE relation F1 >= -0.01`. This is a pragmatic promotion
margin, not a preregistered statistical non-inferiority claim. The paired 95%
confidence intervals remain descriptive and are reported even when they cross
that margin; they are not used as a hidden release condition. Promotion also
requires complete closed-world audit coverage (54/54 system-dataset rows,
14/14 markers, and 3/3 comparisons), zero audit failures, and for each PGE
artifact: full evidence coverage/correctness plus zero unsupported or
signature-invalid retained relations. The formal marker must hash-bind all of
those inputs. Any incomplete, non-finite, hash-mismatched, or unrecognized
field fails closed.

## 当前自动编排（截至 2026-09-04）

The single-GPU lane is serialized as `PGE seed42` → fresh `SpERT` →
`GLiNER+GLiREL` with relation threshold `0` → train-calibrated
`GLiNER+GLiREL`. Low-priority CPU work runs alongside it: the formal GLiNER
entity-only validation artifact is complete; after PGE completes, each dataset
receives a 20,000-resample paired `SOE`-versus-`PGE` bootstrap; a consolidated
validation audit is then produced.

This automation is restricted to the complete public datasets and
validation-stage artifacts. Sealed test evaluation, seeds `2026` and `3407`,
and formal InstructUIE, OneKE, Mirror, and PL-Marker runs are **not**
automatically released; they remain behind the manual validation-promotion
decision.

Read-only status is available from:

```bash
bash scripts/public_experiment_status.sh
systemctl --user status public-experiment-watchdog.timer
```

The managed units are `public-pge-validation.service`,
`public-spert-fresh-validation.service`,
`public-gliner-glirel-t0-validation.service`,
`public-gliner-glirel-calibrated-validation.service`,
`public-gliner-entity-only-validation.service`,
`public-post-pge-bootstrap.service`, `public-validation-audit.service`, and
`public-experiment-watchdog.timer`.

The watchdog timer polls every 20 minutes. It attributes CUDA processes to the
owning user-systemd cgroup, never restarts a completed marker, and excludes
normal wait, CPU preflight, conversion, merge, evaluation, calibration, and
audit phases. A pending job is recovered only after its GPU phase and stopped
progress are rechecked immediately before restart; a still-present pre-CUDA
worker requires two consecutive unchanged observations. Automated GPU
restarts have a one-hour cooldown.

## RTX 3090 budget

The existing serial generation runner is the bottleneck. Approximate wall-clock
budget for full data is:

| Stage | Estimate |
|---|---:|
| Three systems, one seed: training | 5-10 h |
| Three systems, one seed: validation generation/evaluation | 5-10 h |
| Two promoted systems: test generation/evaluation | 4-8 h |
| Two systems, two additional seeds: training | 7-14 h |
| Two systems, two additional seeds: validation/test | 8-16 h |
| External baselines (GLiNER, SpERT, InstructUIE) | 8-18 h |
| **Total including horizontal comparisons** | **37-76 h** |

The lower bound assumes short outputs and no retries; the upper bound is the
safer planning value for a single 3090. CPU conversion, evidence expansion,
verification, and metric aggregation add less than one hour. A batched inference
runner can reduce the generation component substantially.

## Horizontal comparisons

The primary fair comparisons are:

| Method | Year | Scope | Public implementation |
|---|---:|---|---|
| SpERT | 2020 | Joint span entity and relation extraction | `lavis-nlp/spert` |
| InstructUIE | 2023 | Instruction-based unified generative IE | public project code/checkpoint |
| GLiNER | 2024 | Lightweight entity extraction; entity stage only | `urchade/GLiNER` |
| OneKE | 2025 | Optional agent/KG reference; report separately if schema alignment is clean | `zjunlp/OneKE` |

SpERT and InstructUIE are the main end-to-end comparisons. GLiNER is an
entity-only comparison and must not be presented as a relation competitor.
OneKE is supplementary because its agent orchestration introduces additional
system variables. All methods use the same public splits and strict evaluator;
raw and evidence-gated PGE scores are both reported.

## Publication sufficiency

This protocol is sufficient for a defensible methods paper if the following
evidence is reported:

- complete public split sizes and licenses;
- validation and test entity/relation/evidence metrics;
- mean and standard deviation for baseline versus PGE over three seeds;
- ablations for removing KG, evidence gate, and signature verifier;
- unsupported-claim, invalid-signature, evidence-coverage, and generation-failure
  rates;
- a clear statement that this phase evaluates public information-extraction
  transferability, not railway-domain effectiveness.

The public benchmarks in this phase are the complete experimental scope. They
demonstrate transferability, reproducibility, and the effect of evidence gates
without relying on a private or domain-specific corpus.

## Commands

Generate complete public artifacts (omit all `--*-limit` options):

```bash
python3 scripts/import_spert_benchmarks.py --dataset conll04
python3 scripts/import_spert_benchmarks.py --dataset scierc
python3 scripts/import_spert_benchmarks.py --dataset ade
```

Build one system's train/validation/test jobs with
`scripts/build_experiment_jobs.py`, then use the existing QLoRA training,
inference, compact-output expansion, relation verifier, and evaluation scripts.
Keep separate output directories for each dataset, system, and seed.
