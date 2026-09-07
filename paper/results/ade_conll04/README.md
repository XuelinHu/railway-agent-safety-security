# ADE and CoNLL04 paper evidence

This is the active two-benchmark reporting snapshot. Source experiment outputs
are read-only; no prediction or trained model was changed during preparation.

- `dataset_statistics.csv`: counts and reconstructed whitespace-token lengths
  for all three partitions of both datasets.
- `label_distribution.csv`: per-label counts and within-inventory proportions.
- `label_counts.tex`, `dataset_statistics.tex`: tables included directly in both manuscripts.
- `metrics/`: common strict-source-character-span re-evaluations. Repeated
  entity and relation keys receive one vote, consistently across model families.
- `results_snapshot.json`, `test_results.tex`, `validation_ablation.tex`: reported results.
- `evaluator_reconciliation.json`: original and common-evaluator SOE/PGE counts.
- `paired_test_bootstrap.json`: 20,000 paired sentence resamples on fixed predictions.
- `overlap_sensitivity.json`: fixed-prediction scores excluding exact train/test overlaps.
- `test_evidence.csv`: original object-level compliance counts, not deduplicated metric denominators.
- `protocol_snapshot.json`, `source_hashes.json`: method revisions, settings and input identities.

The main test comparison uses completed seed-42 SOE/PGE and completed SpERT,
Qwen3-4B zero-shot plus verifier, and train-calibrated GLiNER+GLiREL outputs.
Only SOE/PGE are reported as promoted internal test systems. The other four
internal configurations are development-set ablations.

Primary test relation F1: ADE SOE 73.67%, PGE 78.85%; CoNLL04 SOE 55.37%,
PGE 60.02%. These differ slightly from the old count-based artifacts because
duplicate keys are handled consistently here. They are not new inference results.

The reporting scope was selected retrospectively after available results were
inspected. Do not describe it as a preregistered representative domain sample,
pool test and validation rows, or report sentence-bootstrap intervals as
training-seed standard deviations. SpERT is stronger in overall F1 and remains
in the main table.

Regenerate from the repository root:

```bash
python3 scripts/build_ade_conll04_paper.py
```

Requires the original local prediction/data artifacts and Python with NumPy,
Matplotlib and PyYAML. The script reads released ADE/CoNLL04 test labels for
scoring and descriptive statistics, not for prompt, threshold or model selection.
