#!/usr/bin/env python3
"""Generate paper figures from the official three-seed CrossNER experiment."""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RESULT = ROOT / "paper/cross-residual/results/official_crossner"
DOMAINS = ["ai", "literature", "music", "politics", "science"]
METHODS = ["zero-shot", "domain-adapted", "score-residual"]
NAMES = {"zero-shot": "Zero-shot GLiNER", "domain-adapted": "Domain-adapted",
         "score-residual": "Score-residual"}
COLORS = {"zero-shot": "#56B4E9", "domain-adapted": "#E69F00",
          "score-residual": "#009E73"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 400,
})

runs = list(csv.DictReader((RESULT / "runs.csv").open()))
grouped = defaultdict(list)
for row in runs:
    grouped[(row["method"], row["domain"])].append(row)

# Main F1 comparison with variability from three independently trained models.
x = np.arange(len(DOMAINS))
width = 0.25
fig, ax = plt.subplots(figsize=(7.1, 3.5))
for index, method in enumerate(METHODS):
    values = [[float(row["f1"]) * 100 for row in grouped[(method, domain)]]
              for domain in DOMAINS]
    means = [np.mean(items) for items in values]
    stds = [np.std(items, ddof=1) if len(set(items)) > 1 else 0 for items in values]
    ax.bar(x + (index - 1) * width, means, width, yerr=stds, capsize=2.5,
           label=NAMES[method], color=COLORS[method], edgecolor="black", linewidth=0.45)
ax.set_xticks(x, [domain.title() for domain in DOMAINS])
ax.set_ylabel("Strict typed-span F1 (%)")
ax.set_ylim(0, 90)
ax.set_title("Cross-domain NER performance")
ax.grid(axis="y", color="#dddddd", linewidth=0.5)
ax.legend(frameon=False, ncol=3)
fig.tight_layout()
fig.savefig(OUT / "high_f1_results.pdf", bbox_inches="tight")
fig.savefig(OUT / "high_f1_results.png", bbox_inches="tight")
plt.close(fig)

# Precision and recall of the final method.
fig, ax = plt.subplots(figsize=(6.2, 3.5))
for metric, color, marker, linestyle in [
    ("precision", "#0072B2", "o", "-"),
    ("recall", "#D55E00", "s", "--"),
]:
    values = [[float(row[metric]) * 100 for row in grouped[("score-residual", domain)]]
              for domain in DOMAINS]
    means = [np.mean(items) for items in values]
    stds = [np.std(items, ddof=1) for items in values]
    ax.errorbar(DOMAINS, means, yerr=stds, color=color, marker=marker,
                linestyle=linestyle, linewidth=1.8, markersize=4.5,
                capsize=2.5, label=metric.title())
ax.set_ylabel("Score (%)")
ax.set_ylim(55, 90)
ax.set_title("Score-residual precision and recall")
ax.grid(color="#dddddd", linewidth=0.5)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "precision_recall.pdf", bbox_inches="tight")
fig.savefig(OUT / "precision_recall.png", bbox_inches="tight")
plt.close(fig)

# Mean training curves with seed variability.
loss_rows = list(csv.DictReader((RESULT / "loss_history.csv").open()))
fig, ax = plt.subplots(figsize=(6.2, 3.5))
domain_colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]
markers = ["o", "s", "^", "D", "v"]
for domain, color, marker in zip(DOMAINS, domain_colors, markers):
    steps = sorted({int(row["step"]) for row in loss_rows if row["domain"] == domain})
    values = [[float(row["loss"]) for row in loss_rows
               if row["domain"] == domain and int(row["step"]) == step] for step in steps]
    means = np.array([np.mean(items) for items in values])
    stds = np.array([np.std(items, ddof=1) for items in values])
    ax.plot(steps, means, marker=marker, markersize=4, linewidth=1.6,
            label=domain.title(), color=color)
    ax.fill_between(steps, means - stds, means + stds, color=color, alpha=0.10)
ax.set_xlabel("Optimization step")
ax.set_ylabel("Training loss")
ax.set_title("Domain-adaptation convergence")
ax.grid(color="#dddddd", linewidth=0.5)
ax.legend(frameon=False, ncol=3)
fig.tight_layout()
fig.savefig(OUT / "training_loss_diagnostic.pdf", bbox_inches="tight")
fig.savefig(OUT / "training_loss_diagnostic.png", bbox_inches="tight")
plt.close(fig)

# Official train/development/test sentence distribution.
stats = list(csv.DictReader((RESULT / "dataset_statistics.csv").open()))
fig, ax = plt.subplots(figsize=(6.2, 3.5))
split_specs = [("train_sentences", "Train", "#56B4E9"),
               ("dev_sentences", "Development", "#E69F00"),
               ("test_sentences", "Test", "#009E73")]
for index, (field, label, color) in enumerate(split_specs):
    ax.bar(x + (index - 1) * width, [int(row[field]) for row in stats], width,
           label=label, color=color, edgecolor="black", linewidth=0.4)
ax.set_xticks(x, [domain.title() for domain in DOMAINS])
ax.set_ylabel("Sentences")
ax.set_title("Official CrossNER split distribution")
ax.grid(axis="y", color="#dddddd", linewidth=0.5)
ax.legend(frameon=False, ncol=3)
fig.tight_layout()
fig.savefig(OUT / "crossner_distribution.pdf", bbox_inches="tight")
fig.savefig(OUT / "crossner_distribution.png", bbox_inches="tight")
