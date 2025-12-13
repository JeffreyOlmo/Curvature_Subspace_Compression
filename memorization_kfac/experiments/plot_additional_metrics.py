import argparse
import math
import sys
from pathlib import Path
from collections import OrderedDict
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from curvature_subspace_compression.memorization_kfac.experiments.compare_subspace_runs import _summarise

FRACTIONS = ["baseline", "comp050", "comp020", "comp010", "comp005", "comp002"]
FRACTION_LABELS = {
    "baseline": "Baseline",
    "comp050": "Keep 50%",
    "comp020": "Keep 20%",
    "comp010": "Keep 10%",
    "comp005": "Keep 5%",
    "comp002": "Keep 2%",
    "comp001": "Keep 1%",
}
DEFAULT_SERIES_DIRS = OrderedDict(
    {
        "d192 (3L)": "d192",
        "d256 (4L)": "d256",
        "d320 (5L)": "d320",
        "d384 (6L)": "d384",
    }
)
EXCLUDED_RELATIVE = {
    "tiny_gpt2_subspace",
    "tiny_fromscratch",
    "plots",
    "tiny_dim064",
    "tiny_dim096",
}
DISPLAY_OVERRIDES = {
    "tiny_dim128": "d128 (Tiny)",
    "tiny_dim192": "d192 (Tiny)",
    "tiny_dim256": "d256 (Tiny)",
    "tiny_dim320": "d320 (Tiny)",
    "tiny_dim384": "d384 (Tiny)",
    "tiny_dim640": "d640 (Tiny)",
}
SERIES_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]


def _discover_series(base_dir: Path) -> OrderedDict:
    # Prefer explicit subdirectories if present
    if any((base_dir / subdir).is_dir() for subdir in DEFAULT_SERIES_DIRS.values()):
        return DEFAULT_SERIES_DIRS.copy()

    discovered_dirs = sorted(
        [path for path in base_dir.iterdir() if path.is_dir()],
        key=lambda item: item.name,
    )
    if discovered_dirs:
        series: OrderedDict[str, str] = OrderedDict()
        for subdir in discovered_dirs:
            key = subdir.name
            if key in EXCLUDED_RELATIVE:
                continue
            if not any((subdir / f"{fraction}.json").exists() for fraction in FRACTIONS):
                continue
            display_name = DISPLAY_OVERRIDES.get(key, key.replace("_", " "))
            series[display_name] = key
        if series:
            return series

    # Otherwise, infer from flat files matching *_baseline.json
    series = OrderedDict()
    for baseline_path in sorted(base_dir.glob("*_baseline.json")):
        prefix = baseline_path.stem[: -len("_baseline")]
        if prefix in EXCLUDED_RELATIVE:
            continue
        if prefix in series:
            continue
        display_name = DISPLAY_OVERRIDES.get(prefix, prefix.replace("_", " "))
        series[display_name] = prefix
    return series


def _collect_series(base_dir: Path) -> Dict[str, List[Dict[str, float]]]:
    series: Dict[str, List[Dict[str, float]]] = {}
    series_dirs = _discover_series(base_dir)
    for display_name, relative in series_dirs.items():
        entries: List[Dict[str, float]] = []
        for fraction in FRACTIONS:
            if (base_dir / relative).is_dir():
                path = base_dir / relative / f"{fraction}.json"
            else:
                path = base_dir / f"{relative}_{fraction}.json"
            if not path.exists():
                continue
            summary = _summarise(path)
            val_loss = summary["val_loss_compressed"]
            train_loss = summary["train_loss_compressed"]
            canary_loss = summary["canary_loss_compressed"]
            entry = {
                "label": FRACTION_LABELS[fraction],
                "param_ratio": summary["param_ratio"],
                "val_perplexity": math.exp(val_loss) if not math.isnan(val_loss) else float("nan"),
                "train_perplexity": math.exp(train_loss) if not math.isnan(train_loss) else float("nan"),
                "canary_perplexity": math.exp(canary_loss) if not math.isnan(canary_loss) else float("nan"),
            }
            entries.append(entry)
        if entries:
            series[display_name] = entries
    return series


def _plot_perplexities(ax: plt.Axes, series: Dict[str, List[Dict[str, float]]]) -> tuple[List[tuple], List]:
    metric_defs = [
        ("val_perplexity", "-", "Validation"),
        ("canary_perplexity", "--", "Canary"),
    ]
    label_positions: Dict[str, List[float]] = {}
    series_handles = []
    for idx, (series_label, points) in enumerate(series.items()):
        color = SERIES_COLORS[idx % len(SERIES_COLORS)]
        for metric_name, linestyle, offset in metric_defs:
            filtered = [
                (p["param_ratio"], p[metric_name], p["label"])
                for p in points
                if not math.isnan(p[metric_name])
            ]
            if not filtered:
                continue
            filtered.sort(key=lambda item: item[0])
            xs = [item[0] for item in filtered]
            ys = [item[1] for item in filtered]
            labels = [item[2] for item in filtered]
            if metric_name == "val_perplexity":
                line, = ax.plot(
                    xs,
                    ys,
                    marker="o",
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.8,
                    markersize=5.5,
                    label=series_label,
                )
                series_handles.append(line)
            else:
                ax.plot(
                    xs,
                    ys,
                    marker="o",
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.8,
                    markersize=5.5,
                    label="_nolegend_",
                )
            for x, y, label in zip(xs, ys, labels):
                label_positions.setdefault(label, []).append(x)
                if label.lower() == "baseline":
                    ax.scatter(
                        [x],
                        [y],
                        marker="o",
                        color=color,
                        edgecolor="black",
                        linewidths=1.0,
                        s=42,
                        zorder=5,
                    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Parameter ratio (compressed / original)")
    ax.set_ylabel("Perplexity")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)

    tick_pairs: List[tuple] = []
    if label_positions:
        for label, xs in label_positions.items():
            if not xs:
                continue
            geom_mean = math.exp(sum(math.log(val) for val in xs) / len(xs))
            tick_pairs.append((geom_mean, label))
        tick_pairs.sort(key=lambda item: item[0])
        ticks = [item[0] for item in tick_pairs]
        tick_labels = [item[1] for item in tick_pairs]
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, rotation=20, ha="right")

    # Remove legends from here.
    return tick_pairs, series_handles, metric_defs


def _plot_ratio(
    ax: plt.Axes, series: Dict[str, List[Dict[str, float]]], tick_pairs: List[tuple]
) -> None:
    for idx, (series_label, points) in enumerate(series.items()):
        color = SERIES_COLORS[idx % len(SERIES_COLORS)]
        val_lookup = {
            p["param_ratio"]: p["val_perplexity"]
            for p in points
            if not math.isnan(p["val_perplexity"]) and p["val_perplexity"] > 0
        }
        canary_lookup = {
            p["param_ratio"]: p["canary_perplexity"]
            for p in points
            if not math.isnan(p["canary_perplexity"]) and p["canary_perplexity"] > 0
        }
        common_keys = sorted(val_lookup.keys() & canary_lookup.keys())
        if not common_keys:
            continue
        ratio_xs = common_keys
        ratio_ys = [val_lookup[x] / canary_lookup[x] for x in common_keys]
        ax.plot(
            ratio_xs,
            ratio_ys,
            marker="d",
            color=color,
            linestyle=":",
            linewidth=1.8,
            markersize=5.0,
            alpha=0.95,
            label="_nolegend_",
        )

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Parameter ratio (compressed / original)")
    ax.set_ylabel("Validation / Canary perplexity")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)

    if tick_pairs:
        ticks = [item[0] for item in tick_pairs]
        tick_labels = [item[1] for item in tick_pairs]
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, rotation=20, ha="right")


def plot_additional_metrics(series: Dict[str, List[Dict[str, float]]], output: Path) -> None:
    if not series:
        print("No series found to plot.")
        return
    fig, (ax_val, ax_ratio) = plt.subplots(1, 2, figsize=(8, 3), sharex=True)

    # Collect series and metric details for legend
    tick_pairs, series_handles, metric_defs = _plot_perplexities(ax_val, series)
    _plot_ratio(ax_ratio, series, tick_pairs)
    ax_val.set_title("Validation vs canary perplexity")
    ax_ratio.set_title("Validation / canary perplexity ratio")
    fig.tight_layout(rect=[0, 0, 0.85, 1])

    # Compose model and metric handles for legend
    # Model handles: series_handles, Model; Metric handles: line styles, Metric
    # Place both model and metric legends together to the right

    # Create handles for the metrics
    metric_handles = [
        Line2D([0], [0], color="black", linestyle=linestyle, linewidth=1.8, label=display)
        for (_, linestyle, display) in metric_defs
    ]

    # Combine handles and labels for a single legend
    all_handles = series_handles + metric_handles
    model_labels = [h.get_label() for h in series_handles]
    metric_labels = [h.get_label() for h in metric_handles]
    all_labels = model_labels + metric_labels

    # Create a single legend for both model and metric, titled by grouping
    # To visually indicate which are models and which are metrics, use a title with two lines
    # But matplotlib only supports a single title string, so we can use a linebreak
    legend_title = "Model\nMetric"

    # Place legend to the right of both plots
    fig.legend(
        handles=all_handles,
        labels=all_labels,
        loc="center left",
        bbox_to_anchor=(0.87, 0.5),  # moved legend closer to the plots
        borderaxespad=0.0,
        fontsize="small",
        title=None,
        ncol=1,
        frameon=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved metric plot to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot additional metrics from curvature subspace runs.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("curvature_subspace_compression/artifacts/v2_scaling"),
        help="Directory containing subdirectories with run artifacts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for the plot image.",
    )
    args = parser.parse_args()

    series = _collect_series(args.base_dir)
    output = args.output or args.base_dir / "plots" / "additional_metrics.png"
    plot_additional_metrics(series, output)


if __name__ == "__main__":
    main()

