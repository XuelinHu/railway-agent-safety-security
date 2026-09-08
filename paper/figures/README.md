# Paper figures

The manuscript uses four figures:

| Purpose | Reproducible source | PDF | PNG | SVG |
|---|---|---|---|---|
| Method architecture | `methodology.drawio` | `methodology.pdf` | `methodology.png` | `methodology.svg` |
| Dataset and sentence-length distributions | `../../scripts/build_paper_result_figures.py` | `dataset_distribution.pdf` | `dataset_distribution.png` | -- |
| Frozen test-result analysis | `../../scripts/build_paper_result_figures.py` | `result_analysis.pdf` | `result_analysis.png` | -- |
| Audited seed-42 training loss | `../../scripts/extract_paper_training_loss.py`, `../../scripts/build_training_loss_figure.py` | `training_loss.pdf` | `training_loss.png` | -- |

The editable Draw.io source is the authoritative figure source for the current
architecture. The distribution and result figures are generated directly from
the frozen CSV/JSON evidence and must not be edited by hand.

The loss figure contains only the attributable seed-42 EAE/HRGE observations
logged every five optimizer steps. It does not infer the four unlogged steps
between observations or mix endpoint-only repeated runs into the curves.
Future QLoRA runs also write `training_loss.csv` directly, avoiding dependence
on captured stdout.
