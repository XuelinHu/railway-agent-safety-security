# ADE-CoNLL04 final review (2026-09-08)

## Delivery

- Anonymous manuscript: 12 PDF pages, comprising a separate Highlights sheet
  and 11 numbered double-column manuscript pages.
- Title page: one page, with a title matching the anonymous manuscript.
- Source package: anonymized LaTeX, cited bibliography entries, generated
  tables, result evidence, figure sources, and build scripts.

## Checks completed

- The manuscript and title page compile successfully with embedded fonts and
  no unresolved citations, references, or labels.
- All 4 figures, 5 tables, 14 numbered equation environments, and 14 cited
  references are present and referenced from the text.
- The only reported overfull box is the known empty CAS highlights-sheet
  container at `maketitle`; rendered pages show no visible overflow, clipping,
  overlap, missing glyphs, or black boxes.
- Figure 2 dataset-distribution panel titles and Figure 3 result-analysis panel
  titles were checked at manuscript size and no longer overlap their plots or
  neighboring panels.
- The anonymous manuscript contains no checked author emails, identifiers, or
  funding numbers. Author-identifying material remains in the separate title
  page.
- The dataset statistics, strict-span results, bootstrap intervals, overlap
  sensitivity, repeated-run summary, ablation table, and seed-42 loss traces
  remain linked to their tracked CSV/JSON provenance artifacts.
- The review-source archive contains no raw dataset text or model weights.

This revision reorganized the paper, regenerated figures, and repackaged
existing completed results. It did not launch new training, inference, or
test-set tuning.
