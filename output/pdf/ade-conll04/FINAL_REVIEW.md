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
- All 4 figures, 5 tables, 14 numbered equation environments, and 24 cited
  references are present and referenced from the text.
- The only reported overfull box is the known empty CAS highlights-sheet
  container at `maketitle`; rendered pages show no visible overflow, clipping,
  overlap, missing glyphs, or black boxes.
- Figure 2 uses a single-column CDF legend at the far right. Figure 3 places
  the SOE/PGE legend in panel (a)'s upper-right clear area. Both were checked at
  manuscript size and do not overlap data, labels, or neighboring panels.
- Equation 14 displays precision, recall, and F1 vertically on three lines
  under one equation number. Review line numbering is disabled.
- Ten strongly related 2022--2024 references were added to the related-work
  discussion; each includes a DOI that was checked through `doi.org`.
- The anonymous manuscript contains no checked author emails, identifiers, or
  funding numbers. It contains no acknowledgement/funding section or AI-use
  declaration. Author-identifying material remains in the separate title page.
- Data and code availability is stated in one sentence and directs reasonable
  requests to the corresponding author.
- The dataset statistics, strict-span results, bootstrap intervals, overlap
  sensitivity, repeated-run summary, ablation table, and seed-42 loss traces
  remain linked to their tracked CSV/JSON provenance artifacts.
- The review-source archive contains no raw dataset text or model weights.

This revision reorganized the paper, regenerated figures, and repackaged
existing completed results. It did not launch new training, inference, or
test-set tuning.
