# Validation-gated score-residual cross-domain NER

This directory contains the current 10-page CAS double-column manuscript, title page,
experiment scripts, compact result files, and publication figures for a
non-generative cross-domain NER study.

## Current experiment

Current title: *Validation-Gated Score-Residual Adaptation for Cross-Domain Named Entity Recognition*.

The manuscript now includes 22 DOI-bearing references, 10 numbered figures, and
10 tables (including the split-overlap audit and reproduction appendix).

The paper evaluates three controlled configurations on the five CrossNER
specialist domains:

1. frozen zero-shot GLiNER-small-v2.1;
2. the same recognizer after target-domain adaptation;
3. validation-gated score-residual fusion of the frozen and adapted scores.

The final three-run strict typed-span F1 scores are 69.19% (AI), 74.55%
(literature), 80.96% (music), 79.66% (politics), and 73.65% (science), with a
five-domain mean of 75.60%. The complete 15-model training and evaluation
matrix takes 562.1 seconds on one RTX 3090.

## Reproduction

Obtain the official BIO files from https://github.com/zliucr/CrossNER and run:

    python run_official_crossner_repeats.py --data-root /path/to/CrossNER/ner_data
    python analyze_official_results.py
    python figures/plot_high_performance_results.py

- run_official_crossner_repeats.py parses the official BIO train/development/
  test files, trains three independent target-domain branches, selects gates
  on development data, and performs exact typed-span test evaluation.
- analyze_official_results.py regenerates domain aggregates, repeated-run
  differences, the paired test, and the confidence interval.
- figures/plot_high_performance_results.py regenerates the result,
  precision--recall, and true training-loss figures.
- figures/generate_expanded_analysis.py regenerates the dataset profile,
  label-frequency, paired-run, and gate-selection figures.
- figures/make_cross_residual_figure_v2.py and
  figures/make_system_pipeline.py regenerate the method schematics.

Large optimizer states and model checkpoints are intentionally excluded from
Git. Small CSV/JSON result summaries, scripts, figures, manuscript.pdf, and
title-page.pdf are the archival artifacts.
