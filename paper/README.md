# ADE and CoNLL04 manuscript

The current manuscript is titled *Provenance-Preserving Evidence-Gated
Knowledge-Graph Augmentation for Auditable Entity-Relation Extraction* and is
prepared for double-blind review at the *Journal of Safety Science and
Resilience*.

## Directory map

- `cas-dc/`: English CAS double-column source, separate title page, bibliography,
  and required class files.
- `zh-cn/`: synchronized Chinese double-column source.
- `figures/`: four final figures and the editable method-diagram source.
- `results/ade_conll04/`: frozen tables, statistics, scores, intervals, overlap
  analysis, loss observations, and provenance records used by the manuscript.
- `submission/`: the only current PDF and submission-package delivery directory.
- `SUBMISSION_READINESS.md`: scope, limitations, and final author checks.

## Rebuild

```bash
python3 scripts/build_paper_result_figures.py
python3 scripts/build_training_loss_figure.py
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error paper/cas-dc/manuscript.tex
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error paper/cas-dc/title-page.tex
latexmk -cd -xelatex -interaction=nonstopmode -halt-on-error paper/zh-cn/manuscript-zh.tex
python3 scripts/fill_submission_word_docs.py
python3 scripts/package_paper_submission.py
python3 scripts/render_paper_review.py
```

The seed-42 EAE/HRGE loss curves contain only attributable observations logged
every five optimizer steps. Later repetitions retain endpoint summaries only
and are not interpolated. Packaging reads the frozen paper evidence; it does not
train models, regenerate predictions, or alter thresholds.

Jingchao Wang is the corresponding author. The anonymous English manuscript
contains no author identity, AI-use statement, acknowledgement, or funding
section. Data and code availability is stated in one sentence.
