#!/usr/bin/env python3
"""Plot the audited seed-42 optimizer-step loss observations."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("paper/results/ade_conll04/training_loss_seed42.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("paper/figures/training_loss.pdf"))
    parser.add_argument("--smooth-window", type=int, default=10,
                        help="Trailing mean over logged observations")
    return parser.parse_args()


def trailing_mean(values: list[float], window: int) -> list[float]:
    result = []
    for index in range(len(values)):
        selected = values[max(0, index - window + 1):index + 1]
        result.append(sum(selected) / len(selected))
    return result


def main() -> None:
    args = parse_args()
    if args.smooth_window < 1:
        raise SystemExit("--smooth-window must be at least 1")
    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"dataset", "system", "seed", "step", "loss"}
    if not rows or not required <= set(rows[0]):
        raise SystemExit(f"{args.input}: missing audited loss columns")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["system"])].append(
            (int(row["step"]), float(row["loss"])))
    expected = {(dataset, system) for dataset in ("ade", "conll04")
                for system in ("eae", "hrge")}
    if set(grouped) != expected:
        raise SystemExit(f"Expected {sorted(expected)}, found {sorted(grouped)}")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 300,
    })
    colors = {"eae": "#0072B2", "hrge": "#D55E00"}
    styles = {"eae": "-", "hrge": "--"}
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.55), sharey=True)
    for ax, dataset, title in zip(axes, ("ade", "conll04"), ("ADE", "CoNLL04")):
        for system in ("eae", "hrge"):
            observations = sorted(grouped[(dataset, system)])
            steps = [step for step, _ in observations]
            losses = [loss for _, loss in observations]
            if steps != sorted(set(steps)):
                raise SystemExit(f"{dataset}/{system}: steps are not unique and increasing")
            ax.plot(steps, losses, color=colors[system], linewidth=0.55, alpha=0.18)
            ax.plot(steps, trailing_mean(losses, args.smooth_window),
                    label=system.upper(), color=colors[system],
                    linestyle=styles[system], linewidth=1.8)
        ax.set_yscale("log")
        ax.set_xlabel("Optimizer step")
        ax.set_title(title)
        ax.grid(color="#D0D0D0", linewidth=0.5, which="both")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Training loss (log scale)")
    fig.subplots_adjust(left=0.09, right=0.99, top=0.9, bottom=0.2, wspace=0.12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", facecolor="white")
    fig.savefig(args.output.with_suffix(".png"), bbox_inches="tight",
                facecolor="white", dpi=400)
    plt.close(fig)


if __name__ == "__main__":
    main()
