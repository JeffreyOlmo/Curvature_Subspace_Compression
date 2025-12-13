import argparse
import math
import sys
from collections import OrderedDict
from pathlib import Path
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
}
DEFAULT_SERIES_DIRS = OrderedDict(
    {
        "d192 (3L)": "d192",
        "d256 (4L)": "d256",
        "d320 (5L)": "d320",
        "d384 (6L)": "d384",
    }
)
EXCLUDED_RELATIVE = {"tiny_gpt2_subspace", "tiny_fromscratch", "plots"}
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


def _gather_points(base_dir: Path) -> Dict[str, Dict[str, List[Dict[str, float]]]]:
    series_dirs = _discover_series(base_dir)
    baselines: Dict[str, Dict[str, float]] = {}
    compressed: Dict[str, List[Dict[str, float]]] = {}

    for display_name, relative in series_dirs.items():
        compressed.setdefault(display_name, [])
        for fraction in FRACTIONS:
            if (base_dir / relative).is_dir():
                path = base_dir / relative / f"{fraction}.json"
            else:
                path = base_dir / f"{relative}_{fraction}.json"
            if not path.exists():
                continue
            summary = _summarise(path)
            baseline_perplexity = math.exp(summary["val_loss_baseline"]) if not math.isnan(
                summary["val_loss_baseline"]
            ) else float("nan")
            compressed_perplexity = math.exp(summary["val_loss_compressed"]) if not math.isnan(
                summary["val_loss_compressed"]
            ) else float("nan")
            baseline_params = summary["params_original"]
            compressed_params = summary["params_compressed"]

            if fraction == "baseline":
                baselines[display_name] = {
                    "params": baseline_params,
                    "perplexity": baseline_perplexity,
                }
                continue

            compressed[display_name].append(
                {
                    "label": FRACTION_LABELS[fraction],
                    "params": compressed_params,
                    "perplexity": compressed_perplexity,
                    "baseline_params": baseline_params,
                    "baseline_perplexity": baseline_perplexity,
                }
            )

    return {"baselines": baselines, "compressed": compressed}


def plot(points: Dict[str, Dict[str, List[Dict[str, float]]]], output: Path) -> None:
    baselines = points["baselines"]
    compressed = points["compressed"]
    if not baselines and all(not values for values in compressed.values()):
        print("No data to plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    handles: List[Line2D] = []
    baseline_points = []

    for series_label, baseline_point in baselines.items():
        if baseline_point and not math.isnan(baseline_point["perplexity"]):
            baseline_points.append(
                (baseline_point["params"], baseline_point["perplexity"], series_label)
            )

    for idx, (series_label, entries) in enumerate(compressed.items()):
        points = [
            {
                "label": entry["label"],
                "params": entry["params"],
                "perplexity": entry["perplexity"],
            }
            for entry in entries
            if not math.isnan(entry["perplexity"])
        ]
        if not points:
            continue
        color = SERIES_COLORS[idx % len(SERIES_COLORS)]
        points.sort(key=lambda item: item["params"])
        xs = [item["params"] for item in points]
        ys = [item["perplexity"] for item in points]
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=1.6,
            alpha=0.9,
        )
        for item in points:
            ax.scatter(
                item["params"],
                item["perplexity"],
                marker="o",
                color=color,
                s=55,
                alpha=0.95,
            )
            ax.annotate(
                item["label"],
                (item["params"], item["perplexity"]),
                textcoords="offset points",
                xytext=(-6, 6),
                fontsize=8,
                color=color,
            )
        handles.append(
            Line2D([], [], color=color, linewidth=1.6, label=series_label)
        )

    baseline_points = sorted(baseline_points, key=lambda item: item[0])
    if baseline_points:
        bx = [item[0] for item in baseline_points]
        by = [item[1] for item in baseline_points]
        baseline_line = Line2D(
            bx,
            by,
            color="#444444",
            linestyle="--",
            linewidth=1.6,
            marker="s",
            markersize=6,
            markerfacecolor="white",
            markeredgecolor="#444444",
            label="Baselines (uncompressed)",
        )
        ax.add_line(baseline_line)
        handles.append(baseline_line)
        for x, y, label in baseline_points:
            ax.scatter(
                [x],
                [y],
                marker="s",
                color="white",
                edgecolor="#444444",
                linewidths=1.0,
                s=65,
                zorder=6,
            )
            ax.annotate(
                label,
                (x, y),
                textcoords="offset points",
                xytext=(4, -10),
                fontsize=8,
                color="#303030",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compressed parameter count")
    ax.set_ylabel("Validation perplexity")
    ax.set_title("Validation perplexity vs parameter count")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(handles=handles, title="Model", loc="upper right")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)
    print(f"Saved validation vs params plot to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot validation perplexity against parameter count.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("curvature_subspace_compression/artifacts/v2_scaling"),
        help="Directory containing model artifacts (baseline and compression runs).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output filepath for the generated plot.",
    )
    args = parser.parse_args()

    data = _gather_points(args.base_dir)
    output_path = args.output or args.base_dir / "plots" / "val_perplexity_vs_params.png"
    plot(data, output_path)


if __name__ == "__main__":
    main()

