# Cross-residual railway safety extraction paper

This branch contains a new manuscript built with the same CAS double-column template as the current submission. The primary corpus is the locally curated railway/safety collection (`data/catalog/*`); the public supplementary benchmark is CoNLL04. Raw documents and reviewed annotations remain governed by the repository data policy and are not committed.

## Three-hour experiment matrix

Run the compact multilingual encoder baseline, source-only adapter, graph-concatenation control, and cross-residual model on the two datasets. Use one small model (1--3B encoder or a 4B quantized generator), capped windows, and 1--3 epochs. Before launching the matrix, execute a dry run to estimate wall time. Record completed and interrupted jobs separately; do not fill missing scores.

## Recovered figures

`figures/cross_residual_architecture.drawio` is the historical Draw.io source with the largest number of text boxes (175 counted `value=` nodes). `cross_residual_architecture.pdf/png` is the new publication-ready schematic. Legacy distribution, result, and loss plots are retained only as candidate inputs until they are recomputed for this corpus.

No funding statement is included. Data are available from the corresponding author (Ai He, `deipss@gmail.com`) upon reasonable request and subject to source licences.
