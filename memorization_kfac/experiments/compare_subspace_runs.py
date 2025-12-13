import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

CATEGORY_STYLES = {
    "memory": {"color": "#f6d55c", "hatch": ".."},
    "math": {"color": "#4c72b0", "hatch": ""},
    "closed-book qa": {"color": "#55a868", "hatch": "xx"},
    "open-book qa": {"color": "#e26f46", "hatch": "//"},
    "logic": {"color": "#7f3c8d", "hatch": "\\\\"},
}
DEFAULT_STYLE = {"color": "#999999", "hatch": ""}


def _perplexity_to_loss(perplexity: float) -> float:
    """Convert perplexity to average negative log-likelihood (natural units)."""
    if perplexity <= 0:
        raise ValueError(f"Perplexity must be positive, got {perplexity}.")
    return math.log(perplexity)


def _average_canary_loss(canary_metrics: List[Dict[str, float]]) -> float:
    """Return the mean negative log-probability across canaries."""
    if not canary_metrics:
        return float("nan")
    total = 0.0
    for item in canary_metrics:
        avg_logprob = item.get("avg_logprob")
        if avg_logprob is None:
            raise ValueError("Canary metric missing 'avg_logprob'.")
        total += -avg_logprob
    return total / len(canary_metrics)


def _parameter_counts(compression_stats: List[Dict[str, float]]) -> Tuple[int, int]:
    """Aggregate original and compressed parameter counts across layers."""
    original = 0
    compressed = 0
    for entry in compression_stats:
        original += int(entry.get("original_params", 0))
        compressed += int(entry.get("compressed_params", 0))
    if original == 0:
        raise ValueError("No parameter statistics found in compression report.")
    return original, compressed


def _summarise(path: Path) -> Dict[str, float]:
    """Load a curvature subspace artifact and extract comparison metrics."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    baseline = data["baseline"]
    compressed = data["compressed"]
    compression_stats = data.get("compression", [])

    orig_params, comp_params = _parameter_counts(compression_stats)

    summary = {
        "val_loss_baseline": _perplexity_to_loss(baseline["val_perplexity"]),
        "val_loss_compressed": _perplexity_to_loss(compressed["val_perplexity"]),
        "train_loss_baseline": _perplexity_to_loss(baseline["train_perplexity"]),
        "train_loss_compressed": _perplexity_to_loss(compressed["train_perplexity"]),
        "canary_loss_baseline": _average_canary_loss(baseline.get("canary_metrics", [])),
        "canary_loss_compressed": _average_canary_loss(compressed.get("canary_metrics", [])),
        "params_original": orig_params,
        "params_compressed": comp_params,
    }
    summary["param_ratio"] = summary["params_compressed"] / summary["params_original"]
    return summary


def _format_loss(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def _format_params(original: int, compressed: int, ratio: float) -> str:
    return f"{compressed:,} / {original:,} ({ratio:.3f}×)"


def _load_task_results(path: Path) -> List[Dict[str, float]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    tasks = data.get("tasks", data)
    parsed: List[Dict[str, float]] = []
    for entry in tasks:
        baseline = float(entry.get("baseline_accuracy", entry.get("baseline", 0.0)))
        edited = float(entry.get("edited_accuracy", entry.get("edited", 0.0)))
        ratio = float("nan") if baseline == 0 else edited / baseline
        parsed.append(
            {
                "task": entry.get("task", entry.get("name", "unknown")),
                "category": entry.get("category", entry.get("group", "uncategorized")).lower(),
                "baseline_accuracy": baseline,
                "edited_accuracy": edited,
                "ratio": ratio,
            }
        )
    return parsed


def plot_task_results(task_results: Dict[str, List[Dict[str, float]]], output_dir: Path) -> None:
    if not task_results:
        return

    run_items = [(run, tasks) for run, tasks in task_results.items() if tasks]
    if not run_items:
        return

    fig, axes = plt.subplots(len(run_items), 1, figsize=(14, 4.2 * len(run_items)), constrained_layout=True)
    if len(run_items) == 1:
        axes = [axes]

    for ax, (run, tasks) in zip(axes, run_items):
        filtered = [t for t in tasks if not math.isnan(t["ratio"])]
        if not filtered:
            ax.set_title(f"{run} (no valid task ratios)")
            ax.axis("off")
            continue

        sorted_tasks = sorted(filtered, key=lambda item: item["ratio"])
        task_names = [item["task"] for item in sorted_tasks]
        ratios = [item["ratio"] for item in sorted_tasks]
        colors = []
        hatches = []
        for item in sorted_tasks:
            style = CATEGORY_STYLES.get(item["category"], DEFAULT_STYLE)
            colors.append(style["color"])
            hatches.append(style["hatch"])

        bars = ax.bar(task_names, ratios, color=colors)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

        ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
        ax.set_ylabel("Edited / Baseline accuracy")
        ax.set_title(f"Relative task performance – {run}")
        ax.set_xticklabels(task_names, rotation=25, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ymax = max(ratios)
        ax.set_ylim(0, max(1.1, ymax * 1.1))

        categories_in_plot = {item["category"] for item in sorted_tasks}
        handles = []
        labels = []
        for category in sorted(categories_in_plot):
            style = CATEGORY_STYLES.get(category, DEFAULT_STYLE)
            patch = plt.Rectangle((0, 0), 1, 1, color=style["color"])
            patch.set_hatch(style["hatch"])
            handles.append(patch)
            labels.append(category.title())
        if handles:
            ax.legend(handles, labels, fontsize="small", loc="upper left")

    output_path = output_dir / "comparison_tasks_plot.png"
    fig.savefig(output_path, dpi=180)
    print(f"Saved task comparison plot to {output_path}")


def compare_runs(paths: List[Path], task_results: Dict[str, List[Dict[str, float]]]) -> None:
    summaries = []
    for path in paths:
        summaries.append((path, _summarise(path)))

    header = (
        "Run",
        "Val loss (base → comp)",
        "Train loss (base → comp)",
        "Canary loss (base → comp)",
        "Params (comp / orig)",
    )
    rows = [header]
    for path, summary in summaries:
        rows.append(
            (
                path.name,
                f"{_format_loss(summary['val_loss_baseline'])} → {_format_loss(summary['val_loss_compressed'])}",
                f"{_format_loss(summary['train_loss_baseline'])} → {_format_loss(summary['train_loss_compressed'])}",
                f"{_format_loss(summary['canary_loss_baseline'])} → {_format_loss(summary['canary_loss_compressed'])}",
                _format_params(
                    summary["params_original"], summary["params_compressed"], summary["param_ratio"]
                ),
            )
        )

    # Compute pairwise deltas if exactly two runs are provided.
    if len(summaries) == 2:
        (_, a), (_, b) = summaries
        rows.append(
            (
                "Δ (run2 - run1)",
                f"{_format_loss(b['val_loss_compressed'] - a['val_loss_compressed'])}",
                f"{_format_loss(b['train_loss_compressed'] - a['train_loss_compressed'])}",
                f"{_format_loss(b['canary_loss_compressed'] - a['canary_loss_compressed'])}",
                _format_params(
                    b["params_compressed"] - a["params_compressed"],
                    b["params_original"] - a["params_original"],
                    b["param_ratio"] / a["param_ratio"] if a["param_ratio"] != 0 else float("nan"),
                ),
            )
        )

    col_widths = [max(len(row[idx]) for row in rows) for idx in range(len(header))]
    for row in rows:
        formatted = " | ".join(cell.ljust(col_widths[idx]) for idx, cell in enumerate(row))
        print(formatted)

    # Plotting section -------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    runs = [path.stem for path, _ in summaries]

    # Loss plot
    width = 0.2
    positions = range(len(runs))
    losses = {
        "Train (baseline)": [summary["train_loss_baseline"] for _, summary in summaries],
        "Train (compressed)": [summary["train_loss_compressed"] for _, summary in summaries],
        "Val (baseline)": [summary["val_loss_baseline"] for _, summary in summaries],
        "Val (compressed)": [summary["val_loss_compressed"] for _, summary in summaries],
        "Canary (baseline)": [summary["canary_loss_baseline"] for _, summary in summaries],
        "Canary (compressed)": [summary["canary_loss_compressed"] for _, summary in summaries],
    }
    offsets = [-1.5, -0.9, -0.3, 0.3, 0.9, 1.5]
    colors = ["#4c72b0", "#4c72b0", "#55a868", "#55a868", "#c44e52", "#c44e52"]
    alphas = [0.6, 1.0, 0.6, 1.0, 0.6, 1.0]

    for idx, (label, values) in enumerate(losses.items()):
        axs[0].bar(
            [p + offsets[idx] * width for p in positions],
            values,
            width=width,
            label=label,
            color=colors[idx],
            alpha=alphas[idx],
        )
    axs[0].set_xticks(list(positions))
    axs[0].set_xticklabels(runs, rotation=20)
    axs[0].set_ylabel("Loss (nats)")
    axs[0].set_title("Cross-entropy losses")
    axs[0].legend(loc="upper left", fontsize="small", ncol=2)
    axs[0].grid(True, axis="y", linestyle="--", alpha=0.4)

    # Parameter plot
    orig_params = [summary["params_original"] for _, summary in summaries]
    comp_params = [summary["params_compressed"] for _, summary in summaries]
    for idx, (values, label, color) in enumerate(
        [
            (orig_params, "Original params", "tab:gray"),
            (comp_params, "Compressed params", "tab:green"),
        ]
    ):
        axs[1].bar(
            [p + (idx - 0.5) * width * 2 for p in positions],
            values,
            width=width * 1.8,
            label=label,
            color=color,
            alpha=0.85,
        )
    axs[1].set_xticks(list(positions))
    axs[1].set_xticklabels(runs, rotation=20)
    axs[1].set_ylabel("Parameters")
    axs[1].set_title("Parameter counts")
    axs[1].legend()
    axs[1].grid(True, axis="y", linestyle="--", alpha=0.4)

    # Compression ratio plot
    param_ratios = [summary["param_ratio"] for _, summary in summaries]
    axs[2].bar(runs, param_ratios, color="tab:purple", alpha=0.85)
    axs[2].axhline(1.0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    axs[2].set_ylabel("Compressed / original")
    axs[2].set_title("Compression ratios")
    axs[2].grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt_path = paths[0].parent / "comparison_plot.png"
    fig.savefig(plt_path, dpi=150)
    print(f"\nSaved comparison plot to {plt_path}")

    run_task_subset = {}
    for path, _ in summaries:
        run_name = path.stem
        if run_name in task_results:
            run_task_subset[run_name] = task_results[run_name]
    if run_task_subset:
        plot_task_results(run_task_subset, paths[0].parent)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare curvature subspace compression runs (losses and parameter counts)."
    )
    parser.add_argument(
        "artifacts",
        nargs="+",
        type=Path,
        help="One or more JSON artifacts produced by run_curvature_subspace_tiny_transformer.py",
    )
    parser.add_argument(
        "--task-results",
        action="append",
        default=[],
        help="Optional per-run task evaluation JSON (format: run_name=path or just path).",
    )
    args = parser.parse_args()

    task_results: Dict[str, List[Dict[str, float]]] = {}
    for entry in args.task_results:
        if "=" in entry:
            run_name, path_str = entry.split("=", 1)
        else:
            path_str = entry
            run_name = Path(path_str).stem
        path = Path(path_str)
        task_results[run_name] = _load_task_results(path)

    compare_runs(args.artifacts, task_results)


if __name__ == "__main__":
    main()

