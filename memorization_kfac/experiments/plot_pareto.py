import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from curvature_subspace_compression.memorization_kfac.experiments.compare_subspace_runs import _summarise


FRACTIONS = ["baseline", "comp050", "comp020", "comp010", "comp005", "comp002", "comp001"]
FRACTION_LABELS = {
    "baseline": "Baseline",
    "comp050": "Keep 50%",
    "comp020": "Keep 20%",
    "comp010": "Keep 10%",
    "comp005": "Keep 5%",
    "comp002": "Keep 2%",
    "comp001": "Keep 1%",
}
DEFAULT_SERIES_DIRS = {
    "d192 (3L)": "d192",
    "d256 (4L)": "d256",
    "d320 (5L)": "d320",
    "d384 (6L)": "d384",
}
DISPLAY_OVERRIDES = {
    "tiny_dim128": "d128 (Tiny)",
    "tiny_dim192": "d192 (Tiny)",
    "tiny_dim256": "d256 (Tiny)",
    "tiny_dim320": "d320 (Tiny)",
    "tiny_dim384": "d384 (Tiny)",
    "tiny_dim640": "d640 (Tiny)",
}
EXCLUDED_RELATIVE = {"tiny_gpt2_subspace", "tiny_fromscratch", "plots"}


def _discover_series(base_dir: Path) -> Dict[str, str]:
    if any((base_dir / subdir).is_dir() for subdir in DEFAULT_SERIES_DIRS.values()):
        return DEFAULT_SERIES_DIRS.copy()

    discovered_dirs = sorted(
        [path for path in base_dir.iterdir() if path.is_dir()],
        key=lambda item: item.name,
    )
    if discovered_dirs:
        series: Dict[str, str] = {}
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

    series = {}
    for baseline_path in sorted(base_dir.glob("*_baseline.json")):
        prefix = baseline_path.stem[: -len("_baseline")]
        if prefix in EXCLUDED_RELATIVE:
            continue
        if prefix in series.values():
            continue
        display_name = DISPLAY_OVERRIDES.get(prefix, prefix.replace("_", " "))
        series[display_name] = prefix
    return series


def gather_series(entries: List[Tuple[str, Path]]) -> List[Dict[str, float]]:
    points = []
    for label, path in entries:
        summary = _summarise(path)
        if label.lower().startswith("baseline"):
            canary_loss = summary["canary_loss_baseline"]
            val_loss = summary["val_loss_baseline"]
            param_ratio = 1.0
        else:
            canary_loss = summary["canary_loss_compressed"]
            val_loss = summary["val_loss_compressed"]
            param_ratio = summary["param_ratio"]
        ratio = math.exp(canary_loss) / math.exp(val_loss) if val_loss is not None else float("inf")
        points.append(
            {
                "label": label,
                "param_ratio": param_ratio,
                "memorization_capability_ratio": ratio,
            }
        )
    return sorted(points, key=lambda item: item["param_ratio"])


def plot(series: Dict[str, List[Dict[str, float]]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    colors = plt.cm.viridis_r([i / max(1, len(series) - 1) for i in range(len(series))])

    for (series_label, points), color in zip(series.items(), colors):
        xs = [p["param_ratio"] for p in points]
        ys = [p["memorization_capability_ratio"] for p in points]
        # plot line without baseline first
        non_baseline = [(x, y, label) for x, y, label in zip(xs, ys, [p["label"] for p in points]) if label.lower() != "baseline"]
        baseline = [(x, y, label) for x, y, label in zip(xs, ys, [p["label"] for p in points]) if label.lower() == "baseline"]
        if non_baseline:
            xs_nb, ys_nb, labels_nb = zip(*non_baseline)
            ax.plot(xs_nb, ys_nb, marker="o", label=series_label, color=color)
            for x, y, label in non_baseline:
                ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8, color=color)
        if baseline:
            for x, y, label in baseline:
                ax.scatter([x], [y], marker="o", color=color, edgecolor="black", linewidths=0.7, zorder=5)
                ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8, color=color)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Parameter ratio (compressed / original)")
    ax.set_ylabel("Canary perplexity / Validation perplexity")
    ax.set_title("Memorization vs Capability Pareto Frontier")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot memorization vs capability Pareto frontier.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("curvature_subspace_compression/artifacts/v2_scaling"),
        help="Directory containing subfolders for each model size.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for the plot PNG.",
    )
    args = parser.parse_args()

    series_dirs = _discover_series(args.base_dir)
    collected = {}
    for label, subdir in series_dirs.items():
        entries = []
        subdir_path = args.base_dir / subdir
        for fraction in FRACTIONS:
            if subdir_path.is_dir():
                path = subdir_path / f"{fraction}.json"
            else:
                path = args.base_dir / f"{subdir}_{fraction}.json"
            if path.exists():
                entries.append((FRACTION_LABELS[fraction], path))
        if not entries:
            continue
        points = gather_series(entries)
        collected[label] = points
        for p in points:
            print(
                f"{label} - {p['label']}: param_ratio={p['param_ratio']:.3f}, "
                f"memorization/capability={p['memorization_capability_ratio']:.2f}"
            )

    output_path = (
        args.output
        if args.output is not None
        else args.base_dir / "plots" / "multi_pareto.png"
    )
    plot(collected, output_path)


if __name__ == "__main__":
    main()

