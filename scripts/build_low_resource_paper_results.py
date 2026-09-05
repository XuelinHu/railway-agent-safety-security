#!/usr/bin/env python3
"""Build auditable paper tables and figures from completed low-resource runs.

The script deliberately treats ``pipeline_complete.json`` as the inclusion
gate for validation results.  Training summaries use only runs with a complete
``training_metrics.json`` and telemetry file, and loss curves use the JSON
records written verbatim to each ``training.log``.  It never writes into the
experiment tree.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


SYSTEMS = {
    "baseline": ("SOE", "Source-Only Extraction"),
    "kg_v1": ("EAE", "Exact-Anchor Extraction"),
    "kg_v2": ("HRGE", "High-Recall Graph Extraction"),
    "kg_v2_verified": ("EVGE", "Evidence-Verified Graph Extraction"),
    "kg_v3_raw": ("CFE", "Conservative-Fusion Extraction"),
    "kg_v3_final": ("PGE", "Provenance-Gated Extraction"),
}

GENERATOR_SYSTEMS = ("baseline", "kg_v1", "kg_v2")
DISPLAY_ORDER = tuple(SYSTEMS)
COLORS = {
    "SOE": "#0072B2",
    "EAE": "#E69F00",
    "HRGE": "#D55E00",
    "EVGE": "#CC79A7",
    "CFE": "#56B4E9",
    "PGE": "#009E73",
}
LINESTYLES = {"SOE": "-", "EAE": "--", "HRGE": "-.", "PGE": ":"}
MARKERS = {"SOE": "o", "EAE": "s", "HRGE": "^", "PGE": "D"}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def budget_from_path(path: Path) -> int:
    for part in path.parts:
        match = re.fullmatch(r"d(\d{3})", part)
        if match:
            return int(match.group(1))
    raise ValueError(f"No budget component in {path}")


def seed_from_path(path: Path) -> int:
    for part in path.parts:
        match = re.fullmatch(r"seed(\d+)", part)
        if match:
            return int(match.group(1))
    raise ValueError(f"No seed component in {path}")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sample_sd(values: Iterable[float]) -> float | None:
    seq = list(values)
    return stdev(seq) if len(seq) > 1 else None


def fmt(value: float | None, digits: int = 4) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Tinos", "DejaVu Serif"],
            "font.size": 22,
            "axes.titlesize": 22,
            "axes.labelsize": 22,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "legend.fontsize": 20,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 1.0,
        }
    )


def collect_validation(
    root: Path, document_metric_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    language_rows: list[dict[str, Any]] = []
    for completion in sorted(root.glob("d*/seed*/derived_validation/pipeline_complete.json")):
        derived = completion.parent
        gate = read_json(completion)
        if gate.get("status") != "complete" or gate.get("formal_test_read") is not False:
            continue
        budget = budget_from_path(completion)
        seed = seed_from_path(completion)
        for internal_id in DISPLAY_ORDER:
            span_path = (
                document_metric_root
                / f"d{budget:03d}"
                / f"seed{seed}"
                / f"{internal_id}_document_span_metrics.json"
            )
            window_span_path = derived / f"{internal_id}_span_metrics.json"
            graph_path = derived / f"{internal_id}_graph_metrics.json"
            if not span_path.exists() or not window_span_path.exists() or not graph_path.exists():
                raise FileNotFoundError(f"Completed gate lacks metrics: {span_path}")
            span = read_json(span_path)
            window_span = read_json(window_span_path)
            graph = read_json(graph_path)
            if (
                span.get("metric")
                != "strict-global-character-span-document-deduplicated"
                or span.get("selection_split") != "validation"
                or span.get("formal_test_read") is not False
            ):
                raise ValueError(f"Invalid validation provenance in {span_path}")
            abbreviation, name = SYSTEMS[internal_id]
            overall = span["overall"]
            graph_overall = graph["overall"]
            metric_rows.append(
                {
                    "budget_documents": budget,
                    "seed": seed,
                    "system": abbreviation,
                    "system_name": name,
                    "entity_precision": overall["entity"]["precision"],
                    "entity_recall": overall["entity"]["recall"],
                    "entity_f1": overall["entity"]["f1"],
                    "relation_precision": overall["relation"]["precision"],
                    "relation_recall": overall["relation"]["recall"],
                    "relation_f1": overall["relation"]["f1"],
                    "claim_relation_f1": overall["relation_with_claim_status"]["f1"],
                    "generation_success_rate": window_span["generation_success_rate"],
                    "entity_evidence_correctness": graph_overall["entity_evidence_correctness"],
                    "relation_evidence_correctness": graph_overall["relation_evidence_correctness"],
                    "unsupported_claim_rate": graph_overall["unsupported_claim_rate"],
                    "invalid_relation_rate": graph_overall["invalid_relation_rate"],
                }
            )
            for language, values in span.get("by_language", {}).items():
                language_rows.append(
                    {
                        "budget_documents": budget,
                        "seed": seed,
                        "system": abbreviation,
                        "language": language,
                        "entity_f1": values["entity"]["f1"],
                        "relation_f1": values["relation"]["f1"],
                        "claim_relation_f1": values["relation_with_claim_status"]["f1"],
                    }
                )
    if not metric_rows:
        raise RuntimeError("No completed validation pipelines found")
    return metric_rows, language_rows


def summarize_validation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["budget_documents"], row["system"])].append(row)
    output: list[dict[str, Any]] = []
    for (budget, system), group in sorted(grouped.items(), key=lambda item: (item[0][0], DISPLAY_ORDER.index(next(k for k, v in SYSTEMS.items() if v[0] == item[0][1])))):
        result: dict[str, Any] = {
            "budget_documents": budget,
            "system": system,
            "n_completed_seeds": len(group),
            "seeds": ";".join(str(row["seed"]) for row in sorted(group, key=lambda row: row["seed"])),
        }
        for field in (
            "entity_precision",
            "entity_recall",
            "entity_f1",
            "relation_precision",
            "relation_recall",
            "relation_f1",
            "claim_relation_f1",
            "generation_success_rate",
            "entity_evidence_correctness",
            "relation_evidence_correctness",
            "unsupported_claim_rate",
            "invalid_relation_rate",
        ):
            values = [float(row[field]) for row in group]
            result[f"{field}_mean"] = mean(values)
            result[f"{field}_sd"] = sample_sd(values)
        output.append(result)
    return output


def summarize_language(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["budget_documents"], row["system"], row["language"])].append(row)
    output: list[dict[str, Any]] = []
    for (budget, system, language), group in sorted(grouped.items()):
        result: dict[str, Any] = {
            "budget_documents": budget,
            "system": system,
            "language": language,
            "n_completed_seeds": len(group),
        }
        for field in ("entity_f1", "relation_f1", "claim_relation_f1"):
            values = [float(row[field]) for row in group]
            result[f"{field}_mean"] = mean(values)
            result[f"{field}_sd"] = sample_sd(values)
        output.append(result)
    return output


def parse_loss_log(path: Path) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "step" in record and "loss" in record:
                points.append((int(record["step"]), float(record["loss"])))
    if not points:
        raise ValueError(f"No real loss observations found in {path}")
    return points


def collect_training(root: Path) -> tuple[list[dict[str, Any]], dict[tuple[int, int, str], list[tuple[int, float]]]]:
    rows: list[dict[str, Any]] = []
    loss_points: dict[tuple[int, int, str], list[tuple[int, float]]] = {}
    for metrics_path in sorted(root.glob("d*/seed*/*/training_metrics.json")):
        internal_id = metrics_path.parent.name
        if internal_id not in GENERATOR_SYSTEMS:
            continue
        telemetry_path = metrics_path.parent / "telemetry.json"
        log_path = metrics_path.parent / "training.log"
        if not telemetry_path.exists() or not log_path.exists():
            raise FileNotFoundError(f"Incomplete training artifacts beside {metrics_path}")
        metrics = read_json(metrics_path)
        telemetry = read_json(telemetry_path)
        budget = budget_from_path(metrics_path)
        seed = seed_from_path(metrics_path)
        abbreviation, name = SYSTEMS[internal_id]
        points = parse_loss_log(log_path)
        if points[-1][0] > int(metrics["steps"]):
            raise ValueError(f"Loss step exceeds completed step count in {log_path}")
        loss_points[(budget, seed, abbreviation)] = points
        rows.append(
            {
                "budget_documents": budget,
                "seed": seed,
                "system": abbreviation,
                "system_name": name,
                "train_examples": metrics["train_examples"],
                "steps": metrics["steps"],
                "loss_observations": len(points),
                "final_loss": metrics["final_loss"],
                "mean_loss": metrics["mean_loss"],
                "wall_clock_seconds": telemetry["wall_clock_seconds"],
                "examples_per_second": telemetry["examples_per_training_second"],
                "tokens_per_second": telemetry["tokens_per_training_second"],
                "prompt_tokens": telemetry["prompt_tokens"],
                "target_tokens": telemetry["target_tokens"],
                "peak_allocated_mib": telemetry["peak_cuda_memory_allocated_mib"],
                "peak_reserved_mib": telemetry["peak_cuda_memory_reserved_mib"],
                "peak_device_mib": telemetry["peak_device_memory_used_mib"],
                "energy_kwh": telemetry["estimated_energy_kwh"],
                "electricity_cost_cny": telemetry["estimated_electricity_cost_cny"],
                "truncated_prompts": metrics["truncated_prompts"],
                "truncated_answers": metrics["truncated_answers_with_eos"],
                "skipped_overlength": metrics["skipped_overlength"],
            }
        )
    if len(rows) != 36:
        raise RuntimeError(f"Expected 36 completed training runs, found {len(rows)}")
    return rows, loss_points


def summarize_training(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["budget_documents"], row["system"])].append(row)
    output: list[dict[str, Any]] = []
    for (budget, system), group in sorted(grouped.items()):
        result: dict[str, Any] = {
            "budget_documents": budget,
            "system": system,
            "n_completed_seeds": len(group),
            "train_examples_mean": mean(float(row["train_examples"]) for row in group),
            "steps_mean": mean(float(row["steps"]) for row in group),
            "total_truncated_prompts": sum(int(row["truncated_prompts"]) for row in group),
            "total_truncated_answers": sum(int(row["truncated_answers"]) for row in group),
            "total_skipped_overlength": sum(int(row["skipped_overlength"]) for row in group),
        }
        for field in (
            "final_loss",
            "mean_loss",
            "wall_clock_seconds",
            "examples_per_second",
            "tokens_per_second",
            "peak_allocated_mib",
            "peak_reserved_mib",
            "peak_device_mib",
            "energy_kwh",
            "electricity_cost_cny",
        ):
            values = [float(row[field]) for row in group]
            result[f"{field}_mean"] = mean(values)
            result[f"{field}_sd"] = sample_sd(values)
        output.append(result)
    return output


def summary_lookup(rows: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    return {(int(row["budget_documents"]), str(row["system"])): row for row in rows}


def plot_low_resource(summary: list[dict[str, Any]], figure_dir: Path) -> None:
    lookup = summary_lookup(summary)
    budgets = [10, 25, 50, 100]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.5), sharex=True)
    for axis, field, title in zip(
        axes,
        ("entity_f1", "relation_f1"),
        ("(a) Entity extraction", "(b) Relation extraction"),
    ):
        for system in ("SOE", "HRGE", "PGE"):
            x = np.array(budgets, dtype=float)
            y = np.array(
                [
                    np.nan
                    if (budget, system) not in lookup
                    else 100.0 * float(lookup[(budget, system)][f"{field}_mean"])
                    for budget in budgets
                ]
            )
            yerr = np.array(
                [
                    np.nan
                    if (budget, system) not in lookup
                    else (
                        0.0
                        if lookup[(budget, system)][f"{field}_sd"] is None
                        else 100.0 * float(lookup[(budget, system)][f"{field}_sd"])
                    )
                    for budget in budgets
                ]
            )
            axis.errorbar(
                x,
                y,
                yerr=yerr,
                label=system,
                color=COLORS[system],
                linestyle=LINESTYLES[system],
                marker=MARKERS[system],
                linewidth=2.1,
                markersize=7,
                capsize=5,
            )
        axis.set_title(title)
        axis.set_xlabel("Reviewed training documents")
        axis.set_ylabel("Strict micro F1 (%)")
        axis.set_xticks(budgets)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=2.2)
    for suffix in ("pdf", "png"):
        fig.savefig(figure_dir / f"low_resource_scaling.{suffix}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def moving_average(values: np.ndarray, width: int = 5) -> np.ndarray:
    if len(values) < width:
        return values.copy()
    left = width // 2
    right = width - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def plot_training_loss(
    loss_points: dict[tuple[int, int, str], list[tuple[int, float]]], figure_dir: Path
) -> None:
    budgets = (10, 25, 50, 100)
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 12.5))
    for axis, budget, panel in zip(axes.flat, budgets, ("a", "b", "c", "d")):
        for system in ("SOE", "EAE", "HRGE"):
            runs = [points for (b, _seed, s), points in loss_points.items() if b == budget and s == system]
            if len(runs) != 3:
                raise RuntimeError(f"Expected three loss logs for D{budget}/{system}, got {len(runs)}")
            common_steps = sorted(set.intersection(*(set(step for step, _ in points) for points in runs)))
            matrix = np.array(
                [[dict(points)[step] for step in common_steps] for points in runs], dtype=float
            )
            raw_mean = matrix.mean(axis=0)
            raw_sd = matrix.std(axis=0, ddof=1)
            smooth_mean = moving_average(raw_mean, width=5)
            axis.fill_between(
                common_steps,
                np.maximum(0.0, raw_mean - raw_sd),
                raw_mean + raw_sd,
                color=COLORS[system],
                alpha=0.10,
                linewidth=0,
            )
            axis.scatter(
                common_steps,
                raw_mean,
                color=COLORS[system],
                marker=MARKERS[system],
                s=12,
                alpha=0.25,
                edgecolors="none",
            )
            axis.plot(
                common_steps,
                smooth_mean,
                label=system,
                color=COLORS[system],
                linestyle=LINESTYLES[system],
                linewidth=2.1,
            )
        axis.set_title(f"({panel}) D{budget}")
        axis.set_xlabel("Optimization step")
        axis.set_ylabel("Training loss")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.005))
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.0, w_pad=1.6)
    for suffix in ("pdf", "png"):
        fig.savefig(figure_dir / f"training_loss.{suffix}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_constraint_compliance(summary: list[dict[str, Any]], figure_dir: Path) -> None:
    lookup = summary_lookup(summary)
    systems = ("HRGE", "PGE")
    metrics = ("Unsupported relation", "Rule-invalid relation")
    fields = ("unsupported_claim_rate_mean", "invalid_relation_rate_mean")
    values = np.array([[100.0 * float(lookup[(100, system)][field]) for field in fields] for system in systems])
    errors = np.array(
        [
            [
                0.0
                if lookup[(100, system)][field.replace("_mean", "_sd")] is None
                else 100.0 * float(lookup[(100, system)][field.replace("_mean", "_sd")])
                for field in fields
            ]
            for system in systems
        ]
    )
    x = np.arange(len(metrics))
    width = 0.34
    fig, axis = plt.subplots(figsize=(12.5, 7.5))
    for index, system in enumerate(systems):
        offset = (index - 0.5) * width
        bars = axis.bar(
            x + offset,
            values[index],
            width,
            yerr=errors[index],
            capsize=5,
            label=system,
            color=COLORS[system],
            edgecolor="black",
            linewidth=0.8,
            hatch="" if system == "HRGE" else "//",
        )
        for bar, value in zip(bars, values[index]):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.0,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=20,
            )
    axis.set_ylabel("Rate (%)")
    axis.set_xticks(x, metrics)
    axis.set_ylim(0, max(65, math.ceil(values.max() / 10) * 10 + 10))
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(figure_dir / f"constraint_compliance_effects.{suffix}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def write_audit(
    path: Path,
    validation_rows: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
    complete_paths: list[Path],
) -> None:
    budgets = defaultdict(list)
    for row in validation_rows:
        if row["system"] == "SOE":
            budgets[row["budget_documents"]].append(row["seed"])
    lines = [
        "# Low-resource result audit",
        "",
        "This snapshot includes validation metrics only when the run directory contains a valid `pipeline_complete.json` gate. The sealed formal-test namespace was not read.",
        "Strict span metrics are recomputed after merging and de-duplicating all windows of each document. Original experiment directories are read-only.",
        "",
        "## Included validation groups",
        "",
    ]
    for budget in sorted(budgets):
        seeds = ", ".join(str(seed) for seed in sorted(set(budgets[budget])))
        lines.append(f"- D{budget}: {len(set(budgets[budget]))} completed seed(s): {seeds}")
    lines.extend(
        [
            "",
            "D50 has no completed validation pipeline and is excluded from validation tables and scaling plots. D10 and D25 are single-seed observations; D100 is summarized across three seeds.",
            "",
            "## Training evidence",
            "",
            f"- Completed training runs: {len(training_rows)}/36",
            f"- Parsed loss logs: {len(training_rows)}/36",
            f"- Total recorded loss observations: {sum(int(row['loss_observations']) for row in training_rows)}",
            f"- Prompt truncations: {sum(int(row['truncated_prompts']) for row in training_rows)}",
            f"- Answer truncations: {sum(int(row['truncated_answers']) for row in training_rows)}",
            f"- Skipped overlength examples: {sum(int(row['skipped_overlength']) for row in training_rows)}",
            "",
            "## Completion gates read",
            "",
        ]
    )
    lines.extend(f"- `{item.resolve()}`" for item in complete_paths)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("data/processed/experiments/formal/low_resource_v2"),
    )
    parser.add_argument("--results-dir", type=Path, default=Path("paper/results"))
    parser.add_argument(
        "--document-metric-dir",
        type=Path,
        default=Path("paper/results/document_level_metrics"),
    )
    parser.add_argument("--figure-dir", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    validation_rows, language_rows = collect_validation(
        args.experiment_root, args.document_metric_dir
    )
    validation_summary = summarize_validation(validation_rows)
    language_summary = summarize_language(language_rows)
    training_rows, loss_points = collect_training(args.experiment_root)
    training_summary = summarize_training(training_rows)

    write_csv(
        args.results_dir / "validation_available_runs.csv",
        validation_rows,
        list(validation_rows[0]),
    )
    write_csv(
        args.results_dir / "validation_available_summary.csv",
        validation_summary,
        list(validation_summary[0]),
    )
    write_csv(
        args.results_dir / "validation_language_summary.csv",
        language_summary,
        list(language_summary[0]),
    )
    write_csv(args.results_dir / "training_runs.csv", training_rows, list(training_rows[0]))
    write_csv(
        args.results_dir / "training_summary.csv",
        training_summary,
        list(training_summary[0]),
    )
    snapshot = {
        "selection_split": "validation",
        "formal_test_read": False,
        "effectiveness_evaluation_unit": "document-deduplicated global spans",
        "completed_validation_groups": sorted(
            {
                (row["budget_documents"], row["seed"])
                for row in validation_rows
                if row["system"] == "SOE"
            }
        ),
        "completed_training_runs": len(training_rows),
        "validation_rows": validation_rows,
        "validation_summary": validation_summary,
        "language_summary": language_summary,
        "training_summary": training_summary,
    }
    (args.results_dir / "paper_results_snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    configure_plots()
    plot_low_resource(validation_summary, args.figure_dir)
    plot_training_loss(loss_points, args.figure_dir)
    plot_constraint_compliance(validation_summary, args.figure_dir)
    complete_paths = sorted(args.experiment_root.glob("d*/seed*/derived_validation/pipeline_complete.json"))
    write_audit(args.results_dir / "RESULTS_AUDIT.md", validation_rows, training_rows, complete_paths)


if __name__ == "__main__":
    main()
