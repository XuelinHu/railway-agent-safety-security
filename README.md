# Auditable entity-relation extraction

This repository contains the code, frozen result evidence, and current paper for
a provenance-preserving, evidence-gated entity-relation extraction framework.
The paper evaluates ADE and CoNLL04 under one strict character-span protocol.

## Current paper

- English source: `paper/cas-dc/manuscript.tex`
- Chinese synchronized source: `paper/zh-cn/manuscript-zh.tex`
- Figures: `paper/figures/`
- Frozen paper evidence: `paper/results/ade_conll04/`
- Submission deliverables: `paper/submission/`

See `paper/README.md` for build and review commands. Large datasets, model
weights, checkpoints, caches, logs, and full experiment outputs are excluded
from version control.

## Project code

Active scripts cover corpus preparation, provenance-preserving graph context,
QLoRA training and inference, source-span evaluation, public baselines, and
paper-figure generation. Annotation contracts are in `schemas/`, and active
configuration is in `configs/`.

The repository does not retain superseded paper versions, obsolete
low-resource protocols, or historical generated-document bundles.
