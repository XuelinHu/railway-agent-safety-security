# ADE and CoNLL04 manuscript

The active paper evaluates **ADE and CoNLL04 only**. Its working title is:

> Provenance-Preserving Evidence-Gated Knowledge-Graph Augmentation for Auditable Entity–Relation Extraction

The target journal remains *Journal of Safety Science and Resilience*. The
motivation concerns auditable extraction and ADE's drug-safety relevance;
CoNLL04 is a general-domain structural comparison, not a safety-outcome dataset.

## Directory map

- `cas-dc/`: CAS double-column anonymous source, author page, bibliography, and
  journal class files.
- `figures/`: one authoritative editable method figure and its PDF/PNG/SVG
  exports. `methodology_detailed_draft.drawio` is the sole figure source.
- `results/ade_conll04/`: source-hashed statistics, common-evaluator scores,
  generated tables, repeated-run summaries, bootstrap intervals, and
  exact-overlap sensitivity results used by the manuscript.
- `results/archive/`: result audits retained for traceability but not cited by
  the active manuscript.
- `SUBMISSION_READINESS.md`: scope, known limitations, and author checks.
- `../output/pdf/ade-conll04/`: generated delivery PDFs and submission package;
  this directory is ignored by default except for already tracked final PDFs.

Historical low-resource protocols were moved to
[`../docs/legacy-experiments/`](../docs/legacy-experiments/README.md). Raw data,
checkpoints, runtime logs, and full prediction dumps do not belong in `paper/`.

## Rebuild

```bash
python3 scripts/build_ade_conll04_paper.py
# Draw.io needs a working DISPLAY/XAUTHORITY or Xvfb session.
python3 scripts/export_submission_artwork.py
cd paper/cas-dc
latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error title-page.tex
cd ../..
python3 scripts/package_paper_submission.py
python3 scripts/render_paper_review.py
```

The data builder reads completed, previously released predictions and labels.
It does not train, infer, change thresholds, regenerate predictions, or mutate
the original experiment directories. Bootstrap intervals are sentence-level
fixed-prediction analyses, not estimates of training-seed variation.

## Version history and submission

The previous mixed-scope manuscript is preserved locally under the ignored
`archive/` directory. Earlier root-level PDF names are historical, not the
active two-benchmark delivery. Existing experiment results were not deleted.

Submit the anonymous manuscript and author page separately, with editable
LaTeX, numbered references, and individual high-quality figures. Author
information and declarations still require author confirmation. The scope was
narrowed after results were available; the manuscript discloses that it is a
focused retrospective analysis, not an exhaustive or preregistered domain survey.

Official instructions:
https://www.keaipublishing.com/en/journals/journal-of-safety-science-and-resilience/guide-for-authors/

Submission system: https://www.editorialmanager.com/JNLSSR/default.aspx

Select the intended special issue in the submission system and recheck the
current deadline, scope, and APC before submission.
