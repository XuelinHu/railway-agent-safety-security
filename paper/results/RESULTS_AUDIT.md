# ADE/CoNLL04 results audit

This audit describes the result snapshot used by the active manuscript. The
reporting scope is limited to the ADE and CoNLL04 benchmarks; SciERC, private
corpora, low-resource D10/D25/D50/D100 runs, API pilots, and exploratory
ablations are not primary paper evidence.

## Canonical reporting snapshot

- Location: `paper/results/ade_conll04/`
- Evaluator: strict source-character-span matching with set deduplication
- Main comparison: SOE and PGE on the held-out test split
- Development analysis: EAE, HRGE, EVGE, CFE, and PGE on validation
- Dataset descriptions: `dataset_statistics.csv`, `label_distribution.csv`,
  `lengths.json`, and the corresponding generated LaTeX tables
- Provenance: `protocol_snapshot.json`, `source_hashes.json`, and
  `evaluator_reconciliation.json`

The generated tables are the only result files included directly in the
manuscript. JSON and CSV artifacts provide auditable counts and intermediate
checks and should not be interpreted as additional experiments.

## Interpretation guardrails

The snapshot is a retrospective focused analysis, not a preregistered domain
survey. It does not establish semantic truth, clinical causality, or overall
state-of-the-art performance. The paired bootstrap file contains fixed-
prediction sentence resampling and must not be reported as training-seed
variation.

## Retained non-primary artifacts

Historical run summaries and exploratory directories remain under
`paper/results/` for traceability. They are explicitly separated in
[`README.md`](README.md) and must not be copied into the active tables without
an accompanying protocol and provenance update. The previous low-resource
audit is preserved at
`paper/results/archive/legacy-low-resource/RESULTS_AUDIT_low_resource.md`.
