#!/usr/bin/env python3
"""Detailed low-resolution methodology figure for the ADE/CoNLL04 paper."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/figures/ade_conll04"

BLUE, PURPLE, GREEN, RED, GOLD, INK = "#2f6f9f", "#7257a6", "#2f7d5b", "#b84c6a", "#c9892f", "#17324d"

def box(ax, x, y, w, h, txt, fc, ec, fs=7.4, ls="-", weight="normal"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.012,rounding_size=.025",
                       facecolor=fc, edgecolor=ec, linewidth=1.2, linestyle=ls)
    ax.add_patch(p)
    ax.text(x+w/2, y+h/2, txt, ha="center", va="center", color=INK,
            fontsize=fs, weight=weight, linespacing=1.18, wrap=True)

def arrow(ax, a, b, color=INK, ls="-", lw=1.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9,
                                 linewidth=lw, linestyle=ls, color=color))

def node(ax, x, y, label, typ, color):
    ax.add_patch(Circle((x, y), .19, facecolor="white", edgecolor=color, linewidth=1.5))
    ax.text(x, y+.01, label, ha="center", va="center", fontsize=6.5, color=INK, weight="bold")
    ax.text(x, y-.31, typ, ha="center", va="center", fontsize=6.2, color=color)

def main():
    plt.rcParams.update({"font.family":"serif", "font.serif":["Times New Roman","Times","DejaVu Serif"],
                         "figure.facecolor":"white", "axes.facecolor":"white"})
    fig, ax = plt.subplots(figsize=(19, 11)); ax.set_xlim(0,19); ax.set_ylim(0,11); ax.axis("off")
    ax.text(9.5,10.72,"Detailed Methodology: Source-Grounded Qwen3-4B + QLoRA Extraction",
            ha="center", fontsize=14, weight="bold", color=INK)
    ax.text(9.5,10.42,"All examples below are taken from the released ADE/CoNLL04 benchmark records.",
            ha="center", fontsize=8, color="#526777")

    # Input and source anchoring.
    box(ax,.25,8.25,4.0,1.7,"REAL ADE TEST SENTENCE\nWe report a case of fulminant hepatic failure\nassociated with didanosine ...\n\nTokens: 0 We · 1 report · 2 a · 3 case · 4 of\n5 fulminant · 6 hepatic · 7 failure · 8 associated · 9 with · 10 didanosine",
        "#fcecef",RED,fs=7.3,weight="bold")
    box(ax,.25,5.95,4.0,1.8,"SOURCE ANCHORING\nWe report a case of\n[fulminant hepatic failure] [associated with] [didanosine]\n\nAdverse-Effect span = [20,45)\nDrug span = [62,72)\nToken text is copied; offsets are reconstructed deterministically.",
        "#fff7e6",GOLD,fs=7.2,weight="bold")
    arrow(ax,(2.25,8.25),(2.25,7.76));

    # Training KG graph.
    box(ax,.25,2.55,4.0,2.55,"", "#fcecef",RED,fs=7.5,weight="bold")
    ax.text(2.25,4.86,"DATASET-SPECIFIC TRAINING KG (ADE)",ha="center",fontsize=7.5,color=INK,weight="bold")
    ax.text(2.25,4.58,"Nodes and directed edges are derived only from training annotations.",ha="center",fontsize=6.4,color="#526777")
    node(ax,1.0,3.85,"heart\nblock","Adverse-Effect",RED); node(ax,2.45,3.85,"CHB","Adverse-Effect",RED)
    node(ax,1.0,3.0,"disopyramide\nphosphate","Drug",BLUE); node(ax,2.45,3.0,"Norpace","Drug",BLUE)
    for x in (1.0,2.45):
        arrow(ax,(x,3.62),(x,3.23),RED)
    ax.text(1.72,3.43,"Adverse-Effect",ha="center",fontsize=6.2,color=RED,rotation=90)
    ax.text(1.72,2.66,"source sentence: developed complete heart block (CHB)\nfollowing administration of disopyramide phosphate (Norpace)",ha="center",fontsize=6.2,color="#526777")
    arrow(ax,(2.25,5.95),(2.25,5.12),ls="--",color=GOLD)

    # Transformer body.
    box(ax,4.75,6.05,7.25,3.75,"FROZEN QWEN3-4B TRANSFORMER BACKBONE\n\nEmbedding: X = E[token IDs],  X ∈ R^(n×d)\n\nRMSNorm → Q/K/V projections\nQ = XWq + X AqBq     K = XWk + X AkBk     V = XWv + X AvBv\n\nSelf-attention:  S = softmax(QKᵀ / √dₖ),  H = SV\nOutput projection:  Y = HWo + X AoBo  + residual\n\nGated FFN (SwiGLU):\nU = XWu + XAuBu,   G = XWg + XAgBg,   Z = SiLU(G) ⊙ U\nDown projection: O = ZWd + ZAdBd + residual → next block (repeated L times)",
        "#eaf2fb",BLUE,fs=7.3,weight="bold")
    arrow(ax,(4.25,6.8),(4.75,7.0)); arrow(ax,(4.25,3.7),(4.75,6.15),ls="--",color=RED)
    box(ax,4.75,5.1,7.25,.65,"QLoRA: W′ = W + (α/r)BA    |    W frozen; only low-rank A ∈ R^(d×r), B ∈ R^(r×k) receive gradients",
        "#fff7e6",GOLD,fs=7.2,weight="bold")

    # Three branches with examples.
    branch_x=[4.85,7.15,9.45]
    branch_txt=[
        "SOE — Source-Only Extraction\nsource + ontology only\n{entities:[...], relations:[...]}",
        "EAE — Exact-Anchor Extraction\nsource + exact concept hints\n{entity:{text:'didanosine', type:'Drug'}}",
        "HRGE — High-Recall Graph Extraction\nsource + anchors + edge priors\n{relation:{source:'effect', target:'drug', evidence:'...'}}",
    ]
    for x,t in zip(branch_x,branch_txt):
        box(ax,x,3.95,2.05,.95,t,"#f1ecfa",PURPLE,fs=5.9,ls="--",weight="bold")
        arrow(ax,(x+1.02,6.05),(x+1.02,4.9),PURPLE,ls="--")
    ax.text(7.95,3.65,"three learned QLoRA adapters",ha="center",fontsize=7,color=PURPLE)

    # Gates and output.
    box(ax,4.75,1.75,7.25,1.45,"DETERMINISTIC EVIDENCE GATES (NOT NEURAL LAYERS)\nEntity: EAE agreement OR valid HRGE anchor OR verified endpoint\nRelation: valid references AND legal type signature AND explicit status AND common source evidence\nRejects structurally invalid relations; it is not a GNN decision.",
        "#eaf6ef",GREEN,fs=7.3,weight="bold")
    for x in (5.87,8.17,10.47): arrow(ax,(x,3.95),(x,3.2),PURPLE)

    box(ax,12.45,6.05,6.2,2.25,"OUTPUT KNOWLEDGE GRAPH (GRAPH DATA, NOT A GNN)\nADE real example:\n[fulminant hepatic failure]₍Adverse-Effect₎\n          ── Adverse-Effect ──▶\n[didanosine]₍Drug₎\n\nCoNLL04 real example:\n[John Wilkes Booth]₍Peop₎ ── Kill ──▶ [President Abraham Lincoln]₍Peop₎",
        "#eaf6ef",GREEN,fs=7.7,weight="bold")
    arrow(ax,(12.0,7.0),(12.45,7.0),GREEN)
    box(ax,12.45,3.0,6.2,1.75,"AUDIT RECORD (AUTOMATIC)\nEvidence quote · character offsets · entity types\nRelation direction · provenance · accept/reject signal\nOptional human spot-check; no manual review of every prediction required.",
        "#fff7e6",GOLD,fs=7.5,weight="bold")
    arrow(ax,(15.55,6.05),(15.55,4.75),GOLD,ls="--")
    box(ax,12.45,1.05,6.2,1.3,"PGE OUTPUT\naccepted spans + typed directed relations\nwith provenance linked back to source text",
        "#eaf6ef",GREEN,fs=7.8,weight="bold")
    arrow(ax,(12.0,2.45),(12.45,1.7),GREEN)

    ax.text(9.5,.35,"Blue = frozen Transformer computation    Purple = trainable LoRA adapters    Green = verified graph output    Gold = deterministic audit / offsets",
            ha="center",fontsize=7.5,color="#526777")
    OUT.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/"methodology_detailed_draft.png",dpi=150,bbox_inches="tight")
    fig.savefig(OUT/"methodology_detailed_draft.svg",bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__": main()
