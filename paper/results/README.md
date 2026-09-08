# Frozen paper evidence

`ade_conll04/` is the sole result snapshot used by the current English and
Chinese manuscripts. It contains generated LaTeX tables, dataset distributions,
strict-span scores, repeated-run summaries, paired bootstrap intervals,
exact-overlap sensitivity, rule-compliance counts, audited loss observations,
and source hashes.

The LaTeX sources consume only the generated `.tex` tables. CSV and JSON files
are audit evidence and must not be edited to change reported values. Raw dataset
text, predictions, checkpoints, model weights, and historical experiment dumps
do not belong in this directory.
