# ADE and CoNLL04 manuscript

The active paper evaluates **ADE and CoNLL04 only**. Its working title is:

> Provenance-Preserving Evidence-Gated Knowledge-Graph Augmentation for Auditable Entity–Relation Extraction

The target journal remains *Journal of Safety Science and Resilience*. The
motivation concerns auditable extraction and ADE's drug-safety relevance;
CoNLL04 is a general-domain structural comparison, not a safety-outcome dataset.

## Active files

- `elsarticle/manuscript.tex`: English anonymous manuscript.
- `elsarticle/manuscript-zh.tex`: Chinese full-text checking version.
- `elsarticle/title-page.tex`: separate author page with the matching title.
- `elsarticle/references.bib`: cited literature, including benchmark and baseline sources.
- `results/ade_conll04/`: source-hashed statistics, common-evaluator scores,
  generated tables, bootstrap intervals, and exact-overlap sensitivity.
- `figures/ade_conll04/`: label distribution, sentence-length distribution,
  and the editable architecture adapted to the current benchmark protocol.
- `../output/pdf/ade-conll04/`: current PDFs, figure files, captions, and review source ZIP.
- [SUBMISSION_READINESS.md](SUBMISSION_READINESS.md): scope and final checks.

## Rebuild

```bash
python3 scripts/build_ade_conll04_paper.py
python3 scripts/build_ade_conll04_diagram.py
# Draw.io needs a working DISPLAY/XAUTHORITY or Xvfb session.
python3 scripts/export_submission_artwork.py
cd paper/elsarticle
latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript.tex
latexmk -xelatex -interaction=nonstopmode -halt-on-error manuscript-zh.tex
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

The previous mixed-scope manuscripts and metadata are preserved under
`archive/pre-ade-conll04-20260907/`; their PDFs and source package are in
`../output/pdf/archive/pre-ade-conll04-20260907/`. Earlier root-level PDF names
are historical, not the active two-benchmark delivery. Existing experiment
records and original Draw.io files were not deleted.

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
