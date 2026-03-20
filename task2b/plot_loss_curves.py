"""Plot loss curves from distributed training results.

Supports two modes:
  1. Single task: plot each rank's loss curve for one task
  2. Cross-task comparison: overlay 2a vs 2b per rank to verify they match

Usage:
    # Single task
    python3 plot_loss_curves.py --results_dir results/

    # Compare task 2a vs 2b (to verify loss curves match)
    python3 plot_loss_curves.py --results_dir task2a/results/ --compare_dir task2b/results/
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def load_losses(results_dir):
    """Load all losses_rank*.json files from a directory."""
    loss_files = sorted(glob.glob(os.path.join(results_dir, "losses_rank*.json")))
    data = {}
    for loss_file in loss_files:
        with open(loss_file, "r") as f:
            d = json.load(f)
        data[d["rank"]] = d
    return data


def plot_single_task(results_dir, output):
    """Plot each rank's loss curve for a single task."""
    data = load_losses(results_dir)
    if not data:
        print(f"No losses_rank*.json files found in {results_dir}")
        return

    method = list(data.values())[0].get("method", "unknown")
    plt.figure(figsize=(10, 6))

    for rank in sorted(data.keys()):
        losses = data[rank]["losses"]
        steps = [e["step"] for e in losses]
        values = [e["loss"] for e in losses]
        plt.plot(steps, values, label=f"Rank {rank}", alpha=0.8, linewidth=1.5)

    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(f"Loss Curves ({method})", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Loss curve saved to {output}")

    for rank in sorted(data.keys()):
        losses = [e["loss"] for e in data[rank]["losses"]]
        print(f"  Rank {rank}: {len(losses)} steps, "
              f"initial={losses[0]:.4f}, final={losses[-1]:.4f}")


def plot_comparison(dir_a, dir_b, output):
    """Compare loss curves between two tasks (e.g. 2a vs 2b) per rank."""
    data_a = load_losses(dir_a)
    data_b = load_losses(dir_b)

    if not data_a or not data_b:
        print("Missing data in one of the directories")
        return

    method_a = list(data_a.values())[0].get("method", "task_a")
    method_b = list(data_b.values())[0].get("method", "task_b")

    ranks = sorted(set(data_a.keys()) & set(data_b.keys()))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, rank in enumerate(ranks[:4]):
        ax = axes[i]

        losses_a = data_a[rank]["losses"]
        losses_b = data_b[rank]["losses"]

        steps_a = [e["step"] for e in losses_a]
        vals_a = [e["loss"] for e in losses_a]
        steps_b = [e["step"] for e in losses_b]
        vals_b = [e["loss"] for e in losses_b]

        ax.plot(steps_a, vals_a, label=method_a, alpha=0.8, linewidth=1.5)
        ax.plot(steps_b, vals_b, label=method_b, alpha=0.8, linewidth=1.5, linestyle='--')

        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title(f"Rank {rank}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Check if they match
        if len(vals_a) == len(vals_b):
            max_diff = max(abs(a - b) for a, b in zip(vals_a, vals_b))
            ax.text(0.02, 0.98, f"Max diff: {max_diff:.6f}",
                    transform=ax.transAxes, va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle(f"Loss Comparison: {method_a} vs {method_b}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Comparison plot saved to {output}")


def main():
    parser = argparse.ArgumentParser(description="Plot loss curves from distributed training")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory containing losses_rank*.json files")
    parser.add_argument("--compare_dir", type=str, default=None,
                        help="Second results directory for cross-task comparison")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image filename")
    args = parser.parse_args()

    if args.compare_dir:
        output = args.output or "loss_comparison_2a_vs_2b.png"
        plot_comparison(args.results_dir, args.compare_dir, output)
    else:
        output = args.output or "loss_curves.png"
        plot_single_task(args.results_dir, output)


if __name__ == "__main__":
    main()
