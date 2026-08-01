"""
Plot Joint Allocation Results for QATNet
=========================================
Reads `QAT/joint_allocation_results.json` and generates a 3-panel publication-quality
figure saved to `QAT/joint_allocation_plots.png`.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def plot_joint_allocation(json_path: str, out_img_path: str):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON results file not found at '{json_path}'. Run run_joint_allocation_experiments.py first.")

    with open(json_path, "r") as f:
        data = json.load(f)

    if "primary_formulation_results" in data:
        results_by_mode = data["primary_formulation_results"]
    elif "results_by_mode" in data:
        results_by_mode = data["results_by_mode"]
    else:
        raise KeyError("Could not find formulation results in JSON file.")

    budget_points = data["budget_points"]
    labels = list(budget_points.keys())

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

    # Color Palette
    color_joint = "#2b5c8f"      # Deep Blue
    color_bw = "#d95f02"         # Vibrant Orange
    color_prune = "#7570b3"      # Purple
    color_fc1 = "#1b9e77"        # Green
    color_fc2 = "#e7298a"        # Pink/Magenta
    color_fc3 = "#e6ab02"        # Gold/Yellow

    # --- Panel 1: Pareto Frontier (Distortion vs FHE Cost Budget) ---
    ax1 = axes[0]
    for mode, color, marker, label_name in [
        ("joint", color_joint, "o", "Joint (Bit-width + Sparsity)"),
        ("bitwidth_only", color_bw, "s", "Bit-width-Only (Baseline #2)"),
        ("pruning_only", color_prune, "^", "Pruning-Only (Baseline #3)"),
    ]:
        costs = [res["total_cost"] for res in results_by_mode[mode]]
        dists = [res["total_distortion"] for res in results_by_mode[mode]]
        ax1.plot(costs, dists, marker=marker, color=color, linewidth=2.5, markersize=8, label=label_name)

    ax1.set_xlabel("Total FHE Cost Budget (Operational Units)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Estimated Distortion Penalty (Sensitivity Loss)", fontsize=11, fontweight="bold")
    ax1.set_title("Pareto Frontiers\n(Joint vs Single-Dimension Ablations)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True)

    # --- Panel 2: Per-Layer Bit-Width Allocations (Joint Mode) ---
    ax2 = axes[1]
    joint_res = results_by_mode["joint"]
    x = np.arange(len(labels))
    w = 0.25

    b_fc1 = [res["config"]["fc1"][0] for res in joint_res]
    b_fc2 = [res["config"]["fc2"][0] for res in joint_res]
    b_fc3 = [res["config"]["fc3"][0] for res in joint_res]

    ax2.bar(x - w, b_fc1, w, label="FC1 Bit-width", color=color_fc1, alpha=0.85)
    ax2.bar(x, b_fc2, w, label="FC2 Bit-width", color=color_fc2, alpha=0.85)
    ax2.bar(x + w, b_fc3, w, label="FC3 Bit-width", color=color_fc3, alpha=0.85)

    ax2.set_xlabel("FHE Budget Target", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Allocated Bit-width (bits)", fontsize=11, fontweight="bold")
    ax2.set_title("Layer Bit-Width Allocation\n(Joint Optimization Mode)", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([l.replace("B", "B").replace("_", "\n") for l in labels], fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 9)
    ax2.legend(loc="upper left", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # --- Panel 3: Per-Layer Structured Sparsity Allocations (Joint Mode) ---
    ax3 = axes[2]
    s_fc1 = [res["config"]["fc1"][1] * 100 for res in joint_res]
    s_fc2 = [res["config"]["fc2"][1] * 100 for res in joint_res]
    s_fc3 = [res["config"]["fc3"][1] * 100 for res in joint_res]

    ax3.bar(x - w, s_fc1, w, label="FC1 Sparsity %", color=color_fc1, alpha=0.85)
    ax3.bar(x, s_fc2, w, label="FC2 Sparsity %", color=color_fc2, alpha=0.85)
    ax3.bar(x + w, s_fc3, w, label="FC3 Sparsity %", color=color_fc3, alpha=0.85)

    ax3.set_xlabel("FHE Budget Target", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Structured Neuron Sparsity (%)", fontsize=11, fontweight="bold")
    ax3.set_title("Layer Structured Sparsity Allocation\n(Joint Optimization Mode)", fontsize=12, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels([l.replace("B", "B").replace("_", "\n") for l in labels], fontsize=10, fontweight="bold")
    ax3.set_ylim(0, 85)
    ax3.legend(loc="upper right", frameon=True)
    ax3.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_img_path)), exist_ok=True)
    plt.savefig(out_img_path, dpi=300, bbox_inches="tight")
    print(f"[+] Plot saved successfully to '{out_img_path}'")


def main():
    parser = argparse.ArgumentParser(description="Plot Joint Allocation Results")
    parser.add_argument("--json_path", type=str, default="./joint_allocation_results.json", help="Path to JSON results")
    parser.add_argument("--out_img", type=str, default="./joint_allocation_plots.png", help="Output PNG path")
    args = parser.parse_args()

    plot_joint_allocation(args.json_path, args.out_img)


if __name__ == "__main__":
    main()
