# Current paper delivery

This is the sole delivery directory for the current ADE-CoNLL04 paper.

- `manuscript.pdf`: anonymous English CAS double-column manuscript.
- `manuscript-zh.pdf`: synchronized Chinese double-column manuscript.
- `title-page.pdf`: separate author page; Jingchao Wang is corresponding author.
- `review-source.zip`: anonymized, buildable English source and evidence package.
- `figure-captions.tex`: standalone captions for all four figures.
- `submission-word/`: five journal-side editable Word documents. Highlights,
  title page, cover letter, and competing-interests wording are synchronized
  with the manuscript. `author statement.docx` is a clearly marked draft and
  requires confirmed CRediT roles before journal upload. See
  `../SUBMISSION_READINESS.md` for remaining author confirmations.
- `submission-build-checks.json`: automated build inventory.
- `FINAL_REVIEW.md`: final layout and reproducibility review.

Final figure masters are stored once in `paper/figures/`. Model weights, raw
datasets, caches, logs, and full training outputs are intentionally excluded.
