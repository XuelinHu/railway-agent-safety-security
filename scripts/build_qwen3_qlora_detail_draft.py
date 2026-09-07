#!/usr/bin/env python3
"""Draw a low-resolution, technically explicit Qwen3-4B + QLoRA block."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/figures/ade_conll04"


def box(ax, x, y, w, h, text, face, edge, fs=9, ls="-", lw=1.3, weight="normal"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.035",
                       facecolor=face, edgecolor=edge, linewidth=lw, linestyle=ls)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color="#17324d", weight=weight, linespacing=1.25,
            wrap=True)


def arr(ax, x1, y1, x2, y2, color="#29465b", ls="-", lw=1.2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=10, linewidth=lw,
                                 linestyle=ls, color=color))


def main():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(16, 8.8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.text(8, 8.65, "Qwen3-4B Transformer Block with QLoRA Updates",
            ha="center", fontsize=15, weight="bold", color="#17324d")
    ax.text(8, 8.30, "Technical draft — base weights are frozen; only low-rank A/B matrices are trained",
            ha="center", fontsize=9, color="#526777")

    # Input side.
    box(ax, 0.35, 5.85, 2.15, 1.35, "Token IDs\n[t₁, t₂, …, tₙ]", "#fcecef", "#b84c6a", fs=10, weight="bold")
    box(ax, 0.35, 3.75, 2.15, 1.35, "Embedding lookup\nX = E[token IDs]\nX in R^(n x d)", "#fff7e6", "#c9892f", fs=9, weight="bold")
    arr(ax, 1.42, 5.85, 1.42, 5.10)
    box(ax, 0.35, 1.65, 2.15, 1.35, "Prompt + KG hints\nsource text · anchors\nrelation patterns", "#fcecef", "#b84c6a", fs=8.7, weight="bold")
    arr(ax, 1.42, 3.75, 1.42, 3.02, ls="--", color="#c9892f")

    # Attention block.
    box(ax, 2.95, 5.15, 3.05, 2.05,
        "RMSNorm\n\nQ = XWq + X AᵩBᵩ\nK = XWk + X AₖBₖ\nV = XWv + X AᵥBᵥ\n\nW frozen; A/B trainable",
        "#eaf2fb", "#2f6f9f", fs=8.4, weight="bold")
    arr(ax, 2.5, 4.42, 2.95, 6.10)
    box(ax, 6.40, 5.15, 2.55, 2.05,
        "Scaled dot-product\n\nS = softmax(QKᵀ / √dₖ)\n\nH = S V\n(token-to-token attention)",
        "#eaf2fb", "#2f6f9f", fs=8.8, weight="bold")
    arr(ax, 6.0, 6.18, 6.4, 6.18)
    box(ax, 9.35, 5.15, 2.35, 2.05,
        "Output projection\n\nY = H Wₒ + H AₒBₒ\n\nresidual add + RMSNorm",
        "#eaf2fb", "#2f6f9f", fs=8.6, weight="bold")
    arr(ax, 8.95, 6.18, 9.35, 6.18)

    # FFN block.
    box(ax, 3.05, 2.05, 3.15, 1.95,
        "Gated FFN (SwiGLU)\n\nU = XWu + XAuBu\nG = XWg + XAgBg\nZ = SiLU(G) * U\n\nW frozen; A/B trainable",
        "#f1ecfa", "#7257a6", fs=8.2, weight="bold")
    arr(ax, 10.52, 5.15, 10.52, 4.22, ls="--", color="#2f6f9f")
    arr(ax, 3.05, 4.22, 3.05, 4.00, ls="--", color="#7257a6")
    box(ax, 6.55, 2.05, 2.95, 1.95,
        "Down projection\n\nO = ZWd + ZAdBd\n\nresidual add + RMSNorm",
        "#f1ecfa", "#7257a6", fs=8.7, weight="bold")
    arr(ax, 6.20, 3.02, 6.55, 3.02)
    box(ax, 9.85, 2.05, 2.35, 1.95,
        "Next Transformer block\n(repeated L times)\n\nFinal hidden states Hᴸ",
        "#eaf2fb", "#2f6f9f", fs=8.8, weight="bold")
    arr(ax, 9.50, 3.02, 9.85, 3.02)

    # Output head and explicit update.
    box(ax, 12.75, 5.15, 2.85, 2.05,
        "LM head / structured decoding\n\nlogits = Hᴸ W_vocab\n\nJSON: entities, types,\nrelations, evidence",
        "#eaf6ef", "#2f7d5b", fs=8.5, weight="bold")
    arr(ax, 11.70, 6.18, 12.75, 6.18)
    box(ax, 12.75, 2.05, 2.85, 1.95,
        "PGE output\n\naccepted spans + typed\ndirected relations\nwith provenance",
        "#eaf6ef", "#2f7d5b", fs=8.7, weight="bold")
    arr(ax, 12.20, 3.02, 12.75, 3.02, color="#2f7d5b")

    # Update explanation and legend.
    box(ax, 3.05, 0.45, 9.15, 0.95,
        "QLoRA update for each adapted matrix:  W' = W + (alpha/r) * BA   |   W remains frozen; only A in R^(d x r) and B in R^(r x k) receive gradients",
        "#fff7e6", "#c9892f", fs=8.4, weight="bold")
    ax.text(13.9, 0.92, "blue: frozen Transformer\npurple: trainable LoRA\ngreen: structured output",
            ha="center", va="center", fontsize=8, color="#526777")
    fig.savefig(OUT / "qwen3_qlora_detail_draft.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / "qwen3_qlora_detail_draft.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
