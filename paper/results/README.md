# Results organization

This directory is organized around the active two-benchmark manuscript. The
paper reports **ADE** and **CoNLL04** only.

## Canonical files used by the manuscript

All files consumed directly by `paper/cas-dc/manuscript.tex` are under
[`ade_conll04/`](ade_conll04/):

- `dataset_statistics.tex` and `label_counts.tex`: descriptive dataset tables;
- `test_results.tex`: the primary held-out comparison table;
- `validation_ablation.tex`: development-set component analysis;
- `repeated_run_stability.tex`: two-run test stability summary;
- `metrics/`: common strict source-character-span metric JSON files;
- `results_snapshot.json`: machine-readable metric snapshot;
- `paired_test_bootstrap.json`: paired sentence-level uncertainty analysis;
- `overlap_sensitivity.json`: exact train/test-overlap sensitivity check;
- `test_evidence.csv`: evidence and compliance counts;
- `protocol_snapshot.json` and `source_hashes.json`: protocol and provenance;
- `dataset_statistics.csv`, `label_distribution.csv`, and `lengths.json`:
  dataset-distribution inputs.
- `training_loss_availability.md`: audit explaining why no synthetic loss
  trajectory is reported for the additional runs.

The LaTeX source should reference only the generated `.tex` tables in this
directory. JSON/CSV files are audit sources and are not pasted into the paper.

## Supporting and historical material

- `RESULTS_AUDIT.md`: scope and provenance audit for the active snapshot.
- `archive/legacy-low-resource/`: historical run summaries, pause snapshots,
  and audits retained for traceability; they are not part of the two-benchmark
  test table.
- `api_validation_pilot_20260903/`, `external_baselines_d100/`, and
  `topk_d100_ablation/`: exploratory or out-of-scope experiments. The
  `document_level_metrics/` directory and root-level `paper_results_snapshot.json`
  are also historical low-resource summaries. They are retained, but must not
  be cited as primary ADE/CoNLL04 evidence.

## Reproducibility boundary

The active snapshot is a frozen reporting artifact. It records the predictions
and evaluator outputs used to build the manuscript; it is not regenerated from
the currently running matrix. Large raw model outputs, checkpoints, datasets,
and temporary inference shards remain outside this tracked reporting folder.
