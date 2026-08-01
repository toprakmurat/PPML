"""
Plot Hessian Sensitivity Analysis Results for QATNet
=====================================================
Reads `QAT/hessian_sensitivity_results.json` and generates a 3-panel publication-quality
plot saved to `QAT/hessian_sensitivity_plots.png`.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def plot_hessian_sensitivity(json_path: str, out_img_path: str):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON results file not found at '{json_path}'. Run run_hessian_experiments.py first.")

    with open(json_path, "r") as f:
        data = json.load(f)

    bit_sens = data["bit_width_sensitivity"]
    neuron_sens = data["neuron_pruning_sensitivity"]["neuron_importance"]
    prune_verif = data["fhe_pruning_verification"]["pruning_comparison"]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

    # Palette
    color_total = "#2b5c8f"
    color_norm = "#d95f02"
    color_fc1 = "#1b9e77"
    color_fc2 = "#7570b3"
    color_struct = "#e7298a"
    color_unstruct = "#666666"

    # --- Panel 1: Layer-wise Hessian Trace (Bit-Width Sensitivity) ---
    ax1 = axes[0]
    layers = list(bit_sens["total_trace"].keys())
    total_traces = [bit_sens["total_trace"][l] for l in layers]
    norm_traces = [bit_sens["normalized_trace"][l] for l in layers]

    x = np.arange(len(layers))
    width = 0.35

    ax1_twin = ax1.twinx()
    b1 = ax1.bar(x - width/2, total_traces, width, label="Total Hessian Trace Tr(H_l)", color=color_total, alpha=0.85)
    b2 = ax1_twin.bar(x + width/2, norm_traces, width, label="Norm Trace (Tr/N_param)", color=color_norm, alpha=0.85)

    ax1.set_xlabel("Model Layer", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Total Hessian Trace", fontsize=11, fontweight="bold", color=color_total)
    ax1_twin.set_ylabel("Normalized Trace per Parameter", fontsize=11, fontweight="bold", color=color_norm)
    ax1.set_xticks(x)
    ax1.set_xticklabels([l.upper() for l in layers], fontsize=10, fontweight="bold")
    ax1.set_title("Layer Bit-Width Sensitivity\n(Hutchinson Trace Estimator)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True)

    # --- Panel 2: Per-Neuron OBD Importance Distribution ---
    ax2 = axes[1]
    fc1_imp = np.sort(np.array(neuron_sens["fc1"]))
    fc2_imp = np.sort(np.array(neuron_sens["fc2"]))

    ax2.plot(np.linspace(0, 100, len(fc1_imp)), fc1_imp, label="FC1 Neurons (784→92)", color=color_fc1, linewidth=2.5)
    ax2.plot(np.linspace(0, 100, len(fc2_imp)), fc2_imp, label="FC2 Neurons (92→92)", color=color_fc2, linewidth=2.5)

    ax2.fill_between(np.linspace(0, 100, len(fc1_imp)), fc1_imp, color=color_fc1, alpha=0.15)
    ax2.fill_between(np.linspace(0, 100, len(fc2_imp)), fc2_imp, color=color_fc2, alpha=0.15)

    ax2.set_xlabel("Neuron Percentile (Sorted Low → High Importance)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("OBD Neuron Importance Score (1/2 H_ww w²)", fontsize=11, fontweight="bold")
    ax2.set_title("Structured Neuron Pruning Sensitivity\n(Optimal Brain Damage OBD)", fontsize=12, fontweight="bold")
    ax2.set_yscale("log")
    ax2.legend(loc="upper left", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # --- Panel 3: Structured vs Unstructured Pruning Impact on FHE b_acc ---
    ax3 = axes[2]
    ratios = [f"{row['prune_ratio']*100:.0f}%" for row in prune_verif]
    struct_bacc = [row["structured_b_acc"] for row in prune_verif]
    unstruct_bacc = [row["unstructured_b_acc"] for row in prune_verif]

    x3 = np.arange(len(ratios))
    w3 = 0.35

    rects1 = ax3.bar(x3 - w3/2, struct_bacc, w3, label="Structured (Neuron Pruning)", color=color_struct, alpha=0.85)
    rects2 = ax3.bar(x3 + w3/2, unstruct_bacc, w3, label="Unstructured (Weight Masking)", color=color_unstruct, alpha=0.85)

    ax3.set_xlabel("Pruning Ratio", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Max FHE Accumulator Bit-width b_acc (bits)", fontsize=11, fontweight="bold")
    ax3.set_title("FHE Accumulator Impact\n(Structured vs Unstructured Pruning)", fontsize=12, fontweight="bold")
    ax3.set_xticks(x3)
    ax3.set_xticklabels(ratios, fontsize=10, fontweight="bold")
    ax3.set_ylim(0, max(max(struct_bacc), max(unstruct_bacc)) + 3)
    ax3.legend(loc="upper right", frameon=True)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # Value labels on bars
    for rect in rects1:
        h = rect.get_height()
        ax3.annotate(f"{h}b", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold', color=color_struct)
    for rect in rects2:
        h = rect.get_height()
        ax3.annotate(f"{h}b", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold', color=color_unstruct)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_img_path)), exist_ok=True)
    plt.savefig(out_img_path, dpi=300, bbox_inches="tight")
    print(f"[+] Plot saved successfully to '{out_img_path}'")


def main():
    parser = argparse.ArgumentParser(description="Plot Hessian sensitivity analysis results")
    parser.add_argument("--json_path", type=str, default="./hessian_sensitivity_results.json", help="Path to JSON results")
    parser.add_argument("--out_img", type=str, default="./hessian_sensitivity_plots.png", help="Output PNG path")
    args = parser.parse_args()

    plot_hessian_sensitivity(args.json_path, args.out_img)


if __name__ == "__main__":
    main()
