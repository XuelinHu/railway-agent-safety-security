# Final paper review (2026-09-08)

## Delivery

- Anonymous English manuscript: 12 PDF pages, comprising one Highlights sheet
  and 11 numbered CAS double-column pages.
- Synchronized Chinese manuscript: 11 A4 double-column pages.
- Separate title page: one A4 page. Jingchao Wang is marked as corresponding
  author, and the two author-supplied funding grants are included.
- Review source: anonymized English LaTeX, cited bibliography entries, generated
  tables, four final figures, frozen result evidence, and rebuild scripts.

## Verified content and layout

- English and Chinese manuscripts contain the same six-section argument,
  4 figures, 5 tables, 14 numbered equation environments, and 24 cited works.
- The English abstract is one continuous paragraph and has four keywords.
- Figure 2 uses a one-column CDF legend at the far right. Figure 3 places the
  SOE/PGE legend in panel (a)'s upper-right clear area. Neither legend overlaps
  plotted data, labels, or adjacent panels.
- Equation 14 displays precision, recall, and F1 vertically on three lines.
- Review line numbering is disabled. Rendered pages contain no clipped text,
  overlapping floats, missing glyphs, unresolved citations, or black boxes.
- The only English overfull-log entry is the CAS class's invisible empty
  highlights-sheet container at `maketitle`; it does not affect rendered output.
- The anonymous manuscript contains no author identity, AI-use statement,
  acknowledgement, or funding section. Its data-and-code statement is one
  sentence directing reasonable requests to the corresponding author.
- The title page uses the exact two-grant funding statement requested by the
  authors. Word forms identify Jingchao Wang as corresponding author as well.

## Contribution-focused revision

- The argument now distinguishes provenance eligibility, evidence-gated
  acceptance, and empirical relation-extraction gains with component analysis.
- Experiments contain five subsections: setup, main results, loss diagnostics,
  additional-run stability, and ablation. Compliance and overlap controls are
  integrated into setup, with compact supporting results in the main analysis.
- Explicit training seed identifiers and epoch counts are removed from the
  narrative; the frozen experiment records remain unchanged.
- All numeric test-table values are unchanged. Twelve within-dataset metric
  maxima are bold and marked with an upward arrow. Citation numbering still
  follows first appearance, and Equation 14 retains its three-line layout.
- The five Word documents are rebuilt in `submission-word/`. Highlights are
  taken directly from the manuscript (three bullets, each under 85 characters),
  and the cover letter follows the same three-contribution argument.
- All five Word files were converted with LibreOffice in a temporary container
  and visually inspected as single-page A4 documents. No clipped text,
  overlapping elements, or unintended blank pages were observed. The container
  was removed automatically; PDF/PNG previews stay under ignored `tmp/pdfs/`.
- The CRediT author statement contains a proposed allocation, NOT a verified
  contribution record: all seven author roles require
  confirmation. The cover letter does not assert unverified originality,
  exclusive submission, or unanimous approval. See `../SUBMISSION_READINESS.md`.
- The author-approved shorter title, *Provenance-Controlled Graph Augmentation
  for Auditable Entity-Relation Extraction*, is applied across delivery formats.

## Evidence boundary

The dataset statistics, strict-span results, paired bootstrap intervals,
exact-overlap sensitivity, repeated-run summary, ablation table, rule-compliance
counts, and seed-42 loss observations remain linked to tracked CSV/JSON evidence.
No raw dataset text, model weights, checkpoints, caches, or historical manuscript
versions are included in the delivery or review-source archive.
