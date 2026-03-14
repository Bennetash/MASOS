"""
MASOS - Generate comparison plots from training logs.

Usage:
    python scripts/plot_results.py --log_dir results/logs --task 1
"""
import sys
import os
import argparse
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="MASOS Plot Results")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Directory containing CSV logs")
    parser.add_argument("--task", type=int, default=None,
                        help="Task number to plot (plots all tasks if not specified)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save plots")
    parser.add_argument("--smooth", type=int, default=50,
                        help="Smoothing window size")
    return parser.parse_args()


def load_csv(path):
    """Load a CSV log file into a dictionary of arrays."""
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                if key not in data:
                    data[key] = []
                try:
                    data[key].append(float(val))
                except (ValueError, TypeError):
                    data[key].append(0.0)
    return {k: np.array(v) for k, v in data.items()}


def smooth(values, window):
    """Apply moving average smoothing."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_metric(all_data, metric, title, ylabel, output_path, smooth_window=50):
    """Plot a single metric comparing algorithms."""
    plt.figure(figsize=(10, 6))

    colors = {"ac": "#1f77b4", "maac": "#ff7f0e", "maddpg": "#2ca02c"}
    labels = {"ac": "AC (Baseline)", "maac": "MAAC (Ours)", "maddpg": "MADDPG"}

    for algo, data in all_data.items():
        if metric in data and "step" in data:
            steps = data["step"]
            values = data[metric]
            smoothed = smooth(values, smooth_window)
            x = steps[:len(smoothed)]
            color = colors.get(algo, "gray")
            label = labels.get(algo, algo.upper())
            plt.plot(x, smoothed, label=label, color=color, linewidth=2)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    args = parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = args.log_dir or os.path.join(project_root, "results", "logs")
    output_dir = args.output_dir or os.path.join(project_root, "results", "plots")
    os.makedirs(output_dir, exist_ok=True)

    algorithms = ["ac", "maac", "maddpg"]
    tasks = [args.task] if args.task else list(range(1, 7))

    for task in tasks:
        print(f"\n[Plot] Task {task}:")
        all_data = {}

        for algo in algorithms:
            # Search for matching CSV files
            pattern = f"{algo}_task{task}_seed"
            for fname in os.listdir(log_dir) if os.path.isdir(log_dir) else []:
                if fname.startswith(pattern) and fname.endswith(".csv"):
                    path = os.path.join(log_dir, fname)
                    data = load_csv(path)
                    all_data[algo] = data
                    print(f"  Found: {fname}")
                    break

        if not all_data:
            print(f"  No data found for task {task}")
            continue

        # Generate plots
        metrics = [
            ("avg_reward", "Average Reward", "Reward"),
            ("avg_found", "Targets Found", "Targets Found"),
            ("avg_coverage", "Coverage Ratio", "Coverage"),
        ]

        for metric, title, ylabel in metrics:
            output_path = os.path.join(output_dir, f"task{task}_{metric}.png")
            plot_metric(
                all_data, metric,
                f"Task {task}: {title}",
                ylabel, output_path,
                smooth_window=args.smooth,
            )

    print("\n[Plot] Done!")


if __name__ == "__main__":
    main()
