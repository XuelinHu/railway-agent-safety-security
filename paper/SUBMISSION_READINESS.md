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

The primary application is drug--adverse-effect extraction from medical text,
evaluated on ADE. CoNLL04 is a supplementary general-domain benchmark for
applicability under different entity roles and relation structures. Each
dataset is trained separately; neither direct domain transfer nor universal
generalization is claimed. ADE is not described as a medical-accident dataset.

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

The file `submission/submission-word/author statement.docx` is explicitly a
draft: proposed roles for all seven authors require confirmation against their
actual work. The proposal emphasizes Xuelin Hu's research and implementation,
Jingchao Wang's supervision and project leadership, Xiaoqin Fu's data and
experimental work, and Youjing Fu's methodology and analysis. It is not a
verified contribution record and must not be uploaded as a completed statement
until every author has confirmed or corrected it. No workload percentages,
equal-contribution claims, or funding-acquisition roles have been asserted.

Before using the cover letter, confirm manuscript originality, prior
publication/preprint status, absence of simultaneous journal submission, and
approval by every author. Add the journal-required originality and author
approval declarations only after that confirmation. The revised cover letter
does not attest to these unverified facts. Every author must also confirm the
competing-interests statement already used in the manuscript.

## Author-approved title

The following shorter title is applied consistently to the manuscript, title
page, and Word documents following the author's approval:

*Provenance-Controlled Graph Augmentation for Auditable Entity-Relation Extraction*

Chinese: 面向可审计实体—关系抽取的来源受控图谱增强。

This preserves provenance control, auditability, and the extraction task while
leaving the evidence-gating mechanism to the abstract and method sections.
