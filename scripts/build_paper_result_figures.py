#!/usr/bin/env python3
"""Build a publication-ready analysis figure from the frozen paper results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "paper/results/ade_conll04"
OUT = ROOT / "paper/figures"
DATASETS = ("ade", "conll04")
DATASET_NAMES = {"ade": "ADE", "conll04": "CoNLL04"}
SYSTEMS = ("soe", "pge")
SYSTEM_NAMES = {"soe": "SOE", "pge": "PGE"}
COLORS = {"soe": "#0072B2", "pge": "#D55E00"}
HATCHES = {"soe": "///", "pge": "..."}


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 300,
    })


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight",
                facecolor="white", dpi=400)
    plt.close(fig)


def build_result_analysis() -> None:
    results = load("results_snapshot.json")
    bootstrap = load("paired_test_bootstrap.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.8),
                             gridspec_kw={"wspace": 0.34})

    ax = axes[0]
    groups = [(dataset, metric) for dataset in DATASETS
              for metric in ("entity_strict", "relation_strict")]
    x = np.arange(len(groups))
    width = 0.36
    for offset, system in zip((-width / 2, width / 2), SYSTEMS):
        values = [100 * results[dataset]["test"][system][metric]["f1"]
                  for dataset, metric in groups]
        bars = ax.bar(x + offset, values, width, label=SYSTEM_NAMES[system],
                      color=COLORS[system], edgecolor="black", linewidth=0.6,
                      hatch=HATCHES[system])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.8,
                    f"{value:.1f}", ha="center", va="bottom", fontsize=7.3,
                    rotation=90)
    ax.set_ylabel("Strict-span F1 (%)")
    ax.set_xticks(x, ("Entity", "Relation", "Entity", "Relation"))
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="lower left")
    ax.annotate("ADE", (0.25, -0.22), xycoords="axes fraction",
                ha="center", fontsize=9)
    ax.annotate("CoNLL04", (0.75, -0.22), xycoords="axes fraction",
                ha="center", fontsize=9)
    ax.set_title("(a) Matched test comparison", loc="left", fontweight="bold")

    ax = axes[1]
    items = [(dataset, metric) for dataset in DATASETS
             for metric in ("entity_strict", "relation_strict")]
    estimates, lower, upper = [], [], []
    for dataset, metric in items:
        row = bootstrap[dataset]["fields"][metric]
        estimate = 100 * row["difference"]
        lo, hi = (100 * value for value in row["ci95"])
        estimates.append(estimate)
        lower.append(estimate - lo)
        upper.append(hi - estimate)
    y = np.arange(len(items))
    point_colors = ["#009E73" if metric == "relation_strict" else "#666666"
                    for _, metric in items]
    ax.axvline(0, color="black", linewidth=0.9)
    for index, (estimate, lo, hi, color) in enumerate(
            zip(estimates, lower, upper, point_colors)):
        ax.errorbar(estimate, index, xerr=np.array([[lo], [hi]]), fmt="o",
                    color=color, ecolor=color, elinewidth=1.4, capsize=3,
                    markersize=5, markeredgecolor="black", markeredgewidth=0.4)
        ax.text(estimate + (0.35 if estimate >= 0 else -0.35), index - 0.17,
                f"{estimate:+.2f}", ha="left" if estimate >= 0 else "right",
                fontsize=8)
    ax.set_yticks(y, [f"{DATASET_NAMES[d]} {'Entity' if m == 'entity_strict' else 'Relation'}"
                      for d, m in items])
    ax.invert_yaxis()
    ax.set_xlabel("PGE minus SOE F1 (percentage points)")
    ax.grid(axis="x", color="#D0D0D0", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.set_title("(b) Paired difference (95% CI)", loc="left", fontweight="bold")

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.87, bottom=0.27)
    save(fig, "result_analysis")


def build_dataset_distribution() -> None:
    with (RESULTS / "label_distribution.csv").open(newline="", encoding="utf-8") as handle:
        labels = [row for row in csv.DictReader(handle) if row["split"] == "train"]
    lengths = load("lengths.json")
    display = {
        "Adverse-Effect": "Adverse effect", "Drug": "Drug", "Loc": "Location",
        "Org": "Organization", "Other": "Other", "Peop": "Person",
        "Kill": "Kill", "Live_In": "Live in", "Located_In": "Located in",
        "OrgBased_In": "Org. based in", "Work_For": "Work for",
    }
    kind_mark = {"entity": "E", "relation": "R"}
    dataset_color = {"ade": "#0072B2", "conll04": "#D55E00"}
    dataset_hatch = {"entity": "", "relation": "///"}

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.55),
                             gridspec_kw={"width_ratios": (1.08, 1), "wspace": 0.3})
    ax = axes[0]
    values = [100 * float(row["proportion"]) for row in labels]
    names = [f"{DATASET_NAMES[row['dataset']]} · {display[row['label']]} "
             f"({kind_mark[row['kind']]})" for row in labels]
    y = np.arange(len(labels))
    for index, (row, value) in enumerate(zip(labels, values)):
        ax.barh(index, value, color=dataset_color[row["dataset"]],
                hatch=dataset_hatch[row["kind"]], edgecolor="black", linewidth=0.55)
        ax.text(min(value + 1.2, 96), index, f"{value:.1f}%", va="center",
                ha="left" if value < 94 else "right", fontsize=7.4)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Share within entity/relation inventory (%)")
    ax.grid(axis="x", color="#D0D0D0", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.set_title("(a) Training-label composition", loc="left", fontweight="bold")

    ax = axes[1]
    split_style = {"train": "-", "validation": "--", "test": ":"}
    split_marker = {"train": None, "validation": "o", "test": "s"}
    for dataset in DATASETS:
        for split in ("train", "validation", "test"):
            ordered = np.sort(np.asarray(lengths[dataset][split], dtype=float))
            cumulative = np.arange(1, len(ordered) + 1) / len(ordered)
            markevery = max(1, len(ordered) // 12)
            ax.plot(ordered, cumulative, color=dataset_color[dataset],
                    linestyle=split_style[split], linewidth=1.6,
                    marker=split_marker[split], markersize=3.2,
                    markevery=markevery,
                    label=f"{DATASET_NAMES[dataset]} {split}")
    ax.set_xlabel("Sentence length (whitespace tokens)")
    ax.set_ylabel("Empirical cumulative probability")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    ax.grid(color="#D0D0D0", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="lower right", fontsize=7.6,
              columnspacing=0.8, handlelength=2.2)
    ax.set_title("(b) Sentence-length distributions", loc="left", fontweight="bold")

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.22, right=0.99, top=0.91, bottom=0.17)
    save(fig, "dataset_distribution")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    configure_style()
    build_dataset_distribution()
    build_result_analysis()


if __name__ == "__main__":
    main()
