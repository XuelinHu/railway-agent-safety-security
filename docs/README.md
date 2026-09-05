# Documentation index

Last organized: 2026-08-30 (Asia/Shanghai)

This directory contains both current research guidance and historical experiment
records. Use the status table below to avoid mixing obsolete normalized metrics
with the final span-aware validation results.

## Canonical entry point

- `NEXT_SESSION_HANDOFF_2026-08-30.md`: the authoritative handoff, result
  inventory, preserved-script index, next-work protocol, and copy-ready prompt
  for a new AI session.

## Active documents

| Document | Role |
|---|---|
| `NEXT_SESSION_HANDOFF_2026-08-30.md` | Current source of truth and next-session entry point |
| `annotation-guidelines.md` | Current annotation definitions and review rules |
| `paper-metadata.md` | Journal and manuscript metadata |
| `../paper/PROTOCOL_FREEZE.md` | Frozen RQs, systems, hyperparameters, metrics, failure policy, 36-run matrix, and formal-test gate |
| `../configs/low_resource_protocol_v1.yaml` | Machine-readable frozen low-resource protocol |
| `../paper/D100_EXECUTION_GATE.md` | Phase D hardware-gate audit; protocol v1 terminated on the first d100 baseline row |
| `../paper/PROTOCOL_V2_AMENDMENT.md` | Frozen semantic-windowing and 4,096-token protocol v2 amendment |
| `../configs/low_resource_protocol_v2.yaml` | Machine-readable frozen protocol v2; immutable after its first successful run |
| `../paper/D100_V2_EXECUTION_GATE.md` | Protocol v2 d100 gate; first baseline row passed and remaining d100 rows are authorized |
| `../paper/elsarticle/manuscript.tex` | Manuscript source; still incomplete and contains stale abstract metrics |
| `../paper/elsarticle/references.bib` | Bibliography source; currently effectively empty |

Active generated protocol assets:

- `../data/processed/experiments/formal/low_resource_manifests_v1/`: nested
  10/25/50/100-document manifests, hashes, 36-run matrix, and 12 derived groups.
- `../data/processed/experiments/formal/low_resource_v1/d*/assets/`: train-only
  subset graphs, aligned windows/gold, V1 prompts, and immutable asset manifests.
- `../data/processed/experiments/formal/windowed_train_v6/` and
  `windowed_validation_v6/`: canonical protocol v2 semantic windows, coverage
  audits, and hash summaries. Intermediate v3/v4/v5 windows remain provenance.
- `../data/processed/experiments/formal/low_resource_manifests_v2/`: protocol v2
  36-run matrix and 12 deterministic derived groups.
- `../data/processed/experiments/formal/low_resource_v2/d*/assets/`: protocol v2
  aligned train/validation assets. Do not overwrite completed run directories.
- `../data/processed/reviewed/annotation_agreement_v1/`: fixed 20-document
  source-only double-review package; its two human reviews are still pending.
- `../data/processed/experiments/formal/low_resource_v1/d100_execution_gate.json`:
  machine-readable Phase D gate result, including attempt hashes and telemetry.
- `../data/processed/experiments/formal/low_resource_v2/d100_execution_gate.json`:
  current machine-readable d100 progress and artifact/telemetry audit.

## Current execution status

Protocol v1 failed its 100-document hardware gate on 2026-08-30. The first
baseline attempt encountered a telemetry compatibility error before model
loading. Its one permitted mechanical retry retained all 986 baseline examples
and failed during the first backward pass with CUDA out-of-memory at 23,952 MiB
peak whole-device use. No checkpoint or training metrics were produced. v1 is
terminal and its failure artifacts remain immutable.

Protocol v2 applies the same 4,096-token training maximum and semantic-window
construction to every system and seed. Its first d100 baseline trained all
1,266 examples successfully, with zero truncation/skipping and 15,155 MiB peak
whole-device use. The d100 rows are running sequentially with an audit after
each row; consult `../paper/D100_V2_EXECUTION_GATE.md` or its machine audit for
the live completed count. The active controller is the user service
`railway-low-resource-v2-d100.service`; its log is
`../data/processed/experiments/formal/low_resource_v2/remaining_d100_controller_attempt02.log`.
The service was deliberately frozen at `lr_v2_d100_seed20260831_kg_v2` step
215/317 on 2026-08-31; check `FreezerState` before acting. Do not start a second
controller while that unit exists. Resume the preserved process with
`systemctl --user thaw railway-low-resource-v2-d100.service`. d10/d25/d50
remain blocked until all nine d100 rows pass. Formal test remains sealed, and
Phase B still awaits two independent human reviews.

## Reference documents

These files contain useful details, but their conclusions must be checked
against the canonical handoff before being cited.

| Document | Role |
|---|---|
| `SESSION_HANDOFF_2026-08-30.md` | Previous handoff covering V1/V2, BIO, CRF, and span follow-ups |
| `../paper/EXPERIMENT_PLAN.md` | Broad JSSR experiment plan and intended paper scope |
| `../paper/RESULTS_DRAFT.md` | Chronological result record containing both legacy normalized and newer span-aware results |
| `preannotation-gate-report.md` | Teacher pre-annotation quality-gate provenance |
| `research-requirements.md` | Early research and corpus concept; its agent/resilience claims are not current claims |

## Historical documents

| Document | Reason it is historical |
|---|---|
| `experiment-report.md` | Auto-generated chronological pilot report with obsolete split sizes and mixed metric definitions |
| `work-plan.md` | Earlier implementation checklist containing completed or superseded tasks |

Historical does not mean disposable. These files preserve experiment provenance
and should not be deleted, rewritten as current results, or used as the primary
source for paper numbers.

## Metric precedence

When two documents disagree, use this order:

1. strict global-character-span one-to-one JSON metrics listed in the canonical handoff;
2. prediction-level evidence and graph metrics;
3. normalized-text or job-macro metrics only as explicitly labelled historical diagnostics.

Do not inspect or evaluate formal-test artifacts. The v2 d100 matrix and all
lower-budget runs are incomplete, Phase B is pending, and the explicit
formal-test authorization gate has not been met.
