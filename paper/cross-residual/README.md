# Gated score-residual cross-domain NER

This directory contains the current CAS double-column manuscript, title page,
experiment scripts, compact result files, and publication figures for a
non-generative cross-domain NER study.

## Current experiment

The paper evaluates three controlled configurations on the five CrossNER
specialist domains:

1. frozen zero-shot GLiNER-small-v2.1;
2. the same recognizer after target-domain adaptation;
3. validation-gated score-residual fusion of the frozen and adapted scores.

The final strict typed-span F1 scores are 67.21% (AI), 72.31% (literature),
79.21% (music), 78.64% (politics), and 70.93% (science), with a five-domain
mean of 73.66%. The five adaptation runs take 168.5 seconds and a complete
residual evaluation takes 72.3 seconds on one RTX 3090.

## Reproduction

- finetune_gliner_crossner.py adapts the five target-domain branches.
- evaluate_gliner_crossner.py evaluates the frozen general model.
- evaluate_score_residual.py selects the residual gate on validation data and
  performs exact typed-span test evaluation.
- figures/plot_high_performance_results.py regenerates the result,
  precision--recall, and true training-loss figures.
- figures/make_cross_residual_figure_v2.py and
  figures/make_system_pipeline.py regenerate the method schematics.

Large optimizer states and model checkpoints are intentionally excluded from
Git. Small CSV/JSON result summaries, scripts, figures, manuscript.pdf, and
title-page.pdf are the archival artifacts.
