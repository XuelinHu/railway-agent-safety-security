# ADE-CoNLL04 submission readiness

## Current delivery

- English anonymous manuscript: `submission/manuscript.pdf`
- Chinese synchronized manuscript: `submission/manuscript-zh.pdf`
- Separate author title page: `submission/title-page.pdf`
- Editable review package: `submission/review-source.zip`
- Figure captions: `submission/figure-captions.tex`
- Word submission documents: `submission/submission-word/`
- Build checks: `submission/submission-build-checks.json`

The four final figures are maintained once in `figures/`; temporary renderings
and historical manuscript versions are not retained.

## Evidence boundary

The paper reports ADE and CoNLL04 only. Its main SOE/PGE relation F1 values are
73.67/78.85% on ADE and 55.37/60.02% on CoNLL04. SpERT remains stronger at
84.08% and 69.72%. Bootstrap intervals describe fixed-prediction sentence
sampling, and the two additional repetitions are only a limited stability check.
The 19 exact CoNLL04 train-test text overlaps are separately excluded in a
sensitivity analysis; this does not rule out near duplicates or pretraining
exposure.

Both imported benchmarks contain relation-bearing sentences. The results do
not establish false-positive behavior on unrestricted streams, clinical truth,
multilingual performance, or long-document performance.

## Author checks before upload

Jingchao Wang is the corresponding author. The authors must still confirm the
telephone number, complete postal data, author order, CRediT roles, competing
interests, journal scope, special-issue entry, licenses, and APC before formal
submission.
