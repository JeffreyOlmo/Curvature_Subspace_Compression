import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_cifar_results(path: Path) -> Tuple[List[str], List[float], List[float]]:
    with open(path, "r") as f:
        data = json.load(f)

    labels: List[str] = ["baseline"]
    train_noisy: List[float] = [data["baseline"]["train_noisy_acc"]]
    test_acc: List[float] = [data["baseline"]["test_acc"]]

    for run in data.get("crt_runs", []):
        lam_fc2 = run.get("lambda_crt_fc2")
        lam_fc1 = run.get("lambda_crt_fc1")
        label = f"λ_fc2={lam_fc2:g}, λ_fc1={lam_fc1:g}"
        metrics = run["metrics"]
        labels.append(label)
        train_noisy.append(metrics["train_noisy_acc"])
        test_acc.append(metrics["test_acc"])

    return labels, train_noisy, test_acc


def load_transformer_results(path: Path) -> Tuple[List[str], List[float], List[float]]:
    with open(path, "r") as f:
        data = json.load(f)

    def avg_canary_perplexity(canary_metrics: List[Dict[str, float]]) -> float:
        if not canary_metrics:
            return float("nan")
        return float(np.mean([float(m["perplexity"]) for m in canary_metrics]))

    labels: List[str] = ["baseline"]
    canary_ppl: List[float] = [avg_canary_perplexity(data["baseline"]["canary_metrics"])]
    val_ppl: List[float] = [data["baseline"]["val_perplexity"]]

    for run in data.get("runs", []):
        lam_fc2 = run.get("lambda_crt_fc2")
        lam_mlp = run.get("lambda_crt_mlp")
        label = f"λ_fc2={lam_fc2:g}, λ_mlp={lam_mlp:g}"
        labels.append(label)
        canary_ppl.append(avg_canary_perplexity(run["canary_metrics"]))
        val_ppl.append(run["val_perplexity"])

    return labels, canary_ppl, val_ppl


def plot_bars(
    ax: plt.Axes,
    labels: List[str],
    series_a: List[float],
    series_b: List[float],
    ylabel: str,
    legend_a: str,
    legend_b: str,
) -> None:
    x = np.arange(len(labels))
    width = 0.38

    ax.bar(x - width / 2, series_a, width, label=legend_a, color="#1f77b4")
    ax.bar(x + width / 2, series_b, width, label=legend_b, color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for xpos, val in zip(x - width / 2, series_a):
        ax.text(xpos, val, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    for xpos, val in zip(x + width / 2, series_b):
        ax.text(xpos, val, f"{val:.2f}", ha="center", va="bottom", fontsize=8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CRT memorization vs capability metrics.")
    parser.add_argument("--cifar-results", type=Path, required=True, help="Path to CIFAR noisy-label JSON results.")
    parser.add_argument(
        "--transformer-results",
        type=Path,
        default=None,
        help="Optional path to transformer canary JSON results.",
    )
    parser.add_argument("--output", type=Path, default=Path("crt_results.png"), help="Output image path.")
    args = parser.parse_args()

    cifar_labels, train_noisy, test_acc = load_cifar_results(args.cifar_results)
    include_transformer = args.transformer_results is not None
    num_panels = 2 if include_transformer else 1

    fig_width = 6 * num_panels
    fig, axes = plt.subplots(1, num_panels, figsize=(fig_width, 4.5), constrained_layout=True)
    if num_panels == 1:
        axes = [axes]

    plot_bars(
        axes[0],
        cifar_labels,
        train_noisy,
        test_acc,
        ylabel="Accuracy",
        legend_a="Train noisy acc (lower = less memorization)",
        legend_b="Test acc (higher = capability)",
    )
    axes[0].set_title("CIFAR noisy-label CRT sweep")

    if include_transformer:
        transformer_labels, canary_ppl, val_ppl = load_transformer_results(args.transformer_results)
        plot_bars(
            axes[1],
            transformer_labels,
            canary_ppl,
            val_ppl,
            ylabel="Perplexity",
            legend_a="Canary perplexity (higher = less memorization)",
            legend_b="Validation perplexity (lower = better)",
        )
        axes[1].set_title("Tiny transformer CRT sweep")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()

