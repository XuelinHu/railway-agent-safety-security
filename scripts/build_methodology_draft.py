#!/usr/bin/env python3
"""Create a low-resolution methodology architecture draft for review."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/figures/ade_conll04"


def box(ax, xy, wh, text, face, edge="#29465b", fontsize=9, lw=1.2,
        rounded=0.04, weight="normal", ls="-"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0.012,rounding_size={rounded}",
        facecolor=face, edgecolor=edge, linewidth=lw, linestyle=ls,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color="#17324d", weight=weight,
            wrap=True, linespacing=1.25)
    return patch


def arrow(ax, start, end, color="#29465b", lw=1.2, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style,
                                 mutation_scale=11, linewidth=lw,
                                 linestyle=ls, color=color,
                                 connectionstyle="arc3,rad=0"))


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.text(8, 8.72, "Provenance-Preserving Evidence-Gated Extraction",
            ha="center", va="center", fontsize=15, weight="bold", color="#17324d")
    ax.text(8, 8.38, "Low-resolution methodology draft — examples are from ADE and CoNLL04",
            ha="center", va="center", fontsize=9, color="#526777")

    # 1. Real input and context.
    box(ax, (0.35, 5.85), (3.2, 1.55),
        "REAL CORPUS INPUT\n\nADE: “complete heart block ...\ndisopyramide phosphate”\nCoNLL04: “John Wilkes Booth ...\nPresident Lincoln”",
        "#fcecef", edge="#b84c6a", fontsize=8.5, weight="bold")
    box(ax, (0.35, 3.95), (3.2, 1.35),
        "SOURCE ANCHORING\nTokens → exact character spans\nSentence evidence retained\nNo annotation-guided window",
        "#fff7e6", edge="#c9892f", fontsize=9, weight="bold")
    arrow(ax, (1.95, 5.85), (1.95, 5.30))

    box(ax, (0.35, 1.65), (3.2, 1.65),
        "DATASET-SPECIFIC TRAINING KG\nTyped mentions + directed edges\nADE: Adverse-Effect → Drug\nCoNLL04: Kill / Live_In / Located_In\n/ OrgBased_In / Work_For",
        "#fcecef", edge="#b84c6a", fontsize=8.3, weight="bold")
    arrow(ax, (2.0, 3.95), (2.0, 3.30))
    arrow(ax, (3.55, 2.48), (5.05, 4.10), ls="--")
    ax.text(3.9, 3.26, "provenance-gated\ncontext", ha="center", va="center",
            fontsize=8, color="#526777", rotation=35)

    # 2. Actual neural architecture.
    box(ax, (5.05, 5.40), (6.0, 2.0),
        "FROZEN QWEN3-4B TRANSFORMER BACKBONE\n\nInput embeddings  →  self-attention  →  feed-forward\n        ↑ LoRA updates in Q/K/V/O and gate/up/down projections ↑\n\nBackbone weights frozen; three adapters share the configuration",
        "#eaf2fb", edge="#2f6f9f", fontsize=9.5, weight="bold", lw=1.5)
    arrow(ax, (3.55, 6.62), (5.05, 6.40))
    arrow(ax, (3.55, 2.30), (5.05, 5.78), ls="--")

    # Adapter branches inside/under backbone.
    branch_y = 3.55
    branch_w, branch_h = 1.8, 1.15
    xs = [5.1, 7.15, 9.2]
    labels = [
        "SOE\nsource only\nstructured JSON",
        "EAE\nexact concept hints\nstructured JSON",
        "HRGE\nanchors + edges + patterns\nstructured JSON",
    ]
    for x, label in zip(xs, labels):
        box(ax, (x, branch_y), (branch_w, branch_h), label,
            "#f1ecfa", edge="#7257a6", fontsize=7.5, weight="bold", ls="--")
        arrow(ax, (x + branch_w / 2, 5.40), (x + branch_w / 2, branch_y + branch_h),
              color="#7257a6", ls="--")
    ax.text(8.0, 3.15, "parallel QLoRA adapters (learned modules)", ha="center",
            fontsize=8.5, color="#7257a6")

    # 3. Deterministic gates/reconstruction.
    box(ax, (5.05, 1.55), (6.0, 1.25),
        "DETERMINISTIC EVIDENCE GATES (NOT NEURAL LAYERS)\nEntity: EAE agreement OR anchor OR verified endpoint\nRelation: valid references + type signature + explicit status + common evidence",
        "#eaf6ef", edge="#2f7d5b", fontsize=8.7, weight="bold")
    for x in [6.0, 8.05, 10.1]:
        arrow(ax, (x, branch_y), (x, 2.80), color="#7257a6")
    arrow(ax, (11.05, 6.40), (11.75, 2.18), color="#2f6f9f", ls="--")
    ax.text(11.55, 4.25, "candidate\nJSON", ha="center", va="center",
            fontsize=8, color="#2f6f9f", rotation=78)

    # 4. Real, typed graph outputs.
    box(ax, (11.75, 5.55), (3.9, 2.0),
        "OUTPUT KNOWLEDGE GRAPH\n\nADE (real labels)\n[complete heart block]₍Adverse-Effect₎\n      ── Adverse-Effect ──▶\n[disopyramide phosphate]₍Drug₎\n\nCoNLL04: [John Wilkes Booth]₍Peop₎\n      ── Kill ──▶ [President Lincoln]₍Peop₎",
        "#eaf6ef", edge="#2f7d5b", fontsize=8.5, weight="bold")
    arrow(ax, (11.05, 2.18), (11.75, 6.15), color="#2f7d5b")
    box(ax, (11.75, 1.65), (3.9, 1.65),
        "AUDIT RECORD\nsource evidence quote · character offsets\nrelation direction · provenance · accept/reject signal\nStructural admissibility ≠ clinical causality",
        "#fff7e6", edge="#c9892f", fontsize=8.7, weight="bold")
    arrow(ax, (13.7, 5.55), (13.7, 3.30), color="#c9892f", ls="--")

    # Legend.
    ax.add_patch(Circle((0.58, 0.72), 0.07, facecolor="#7257a6", edgecolor="none"))
    ax.text(0.76, 0.72, "learned QLoRA component", va="center", fontsize=8, color="#526777")
    ax.add_patch(Circle((3.05, 0.72), 0.07, facecolor="#2f7d5b", edgecolor="none"))
    ax.text(3.23, 0.72, "deterministic gate / graph", va="center", fontsize=8, color="#526777")
    ax.text(9.25, 0.52, "Entities and relations are copied from the source sentence\nand checked against the dataset ontology.",
            ha="center", va="center", fontsize=7.5, color="#526777")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "methodology_draft.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / "methodology_draft.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
