from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 400,
})

fig, ax = plt.subplots(figsize=(7.5, 3.55))
ax.set_xlim(0, 16.8)
ax.set_ylim(0, 7.0)
ax.axis("off")


def box(x, y, w, h, label, color, size=8.5, linewidth=1.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=.04,rounding_size=.08",
        facecolor=color, edgecolor="#333333", linewidth=linewidth,
    ))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=size)


def arrow(start, end, linestyle="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=11,
        color="#333333", linewidth=1.05, linestyle=linestyle,
        connectionstyle=f"arc3,rad={rad}",
    ))


box(0.15, 2.75, 1.8, 1.05, "Sentence $x$\nlabel set $\\mathcal{Y}$", "#e6f2f8")
box(2.55, 4.55, 2.65, 1.10, "Frozen general\nspan recognizer $f_{\\theta_0}$", "#cfe2f3")
box(2.55, 0.95, 2.65, 1.10, "Domain-adapted\nspan recognizer $f_{\\theta_1}$", "#fce5cd")
box(5.95, 4.55, 2.10, 1.10, "General scores\n$s_0(c)$", "#cfe2f3")
box(5.95, 0.95, 2.10, 1.10, "Adapted scores\n$s_1(c)$", "#fce5cd")
box(8.85, 1.70, 2.15, 1.10, "Score residual\n$\\Delta(c)=s_1-s_0$", "#fff2cc")
box(11.75, 3.00, 2.40, 1.20, "Validation-gated fusion\n$s_f=s_0+\\alpha\\Delta$", "#eadcf8")
box(14.85, 3.00, 1.75, 1.20, "Threshold $\\tau$\nTyped spans", "#d9d2e9")

arrow((1.95, 3.48), (2.55, 5.10))
arrow((1.95, 3.08), (2.55, 1.50))
arrow((5.20, 5.10), (5.95, 5.10))
arrow((5.20, 1.50), (5.95, 1.50))
arrow((8.05, 1.50), (8.85, 2.10))
arrow((8.05, 5.10), (9.60, 2.80), linestyle="--")
arrow((8.05, 5.10), (11.75, 3.88))
arrow((11.00, 2.25), (11.75, 3.30))
arrow((14.15, 3.60), (14.85, 3.60))

ax.text(3.88, 6.10, "Identity evidence", ha="center", color="#24557a", fontsize=8.5)
ax.text(3.88, 0.38, "Target-domain correction", ha="center", color="#8a4f16", fontsize=8.5)
ax.text(12.95, 0.70, "$\\alpha$ and $\\tau$ selected on validation data only", ha="center", fontsize=8)
ax.text(8.40, 6.65, "Parallel label-conditioned span scoring", ha="center", fontsize=9.5, weight="bold")

fig.tight_layout(pad=0.15)
fig.savefig(OUT / "cross_residual_architecture.pdf", bbox_inches="tight")
fig.savefig(OUT / "cross_residual_architecture.png", bbox_inches="tight")
