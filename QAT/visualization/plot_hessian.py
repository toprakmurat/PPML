"""
Plot Hessian Sensitivity Analysis Results for QATNet
=====================================================
"""

import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

_BASE_DIR = Path(__file__).resolve().parents[1]
_DEFAULT_RESULTS_JSON = _BASE_DIR / "experiments" / "results" / "hessian_sensitivity_results.json"
_DEFAULT_OUTPUT_IMG = _BASE_DIR / "experiments" / "plots" / "hessian_sensitivity_plots.png"


def plot_hessian_sensitivity(json_path: str = None, out_img_path: str = None):
    in_json = json_path if json_path else str(_DEFAULT_RESULTS_JSON)
    out_img = out_img_path if out_img_path else str(_DEFAULT_OUTPUT_IMG)

    if not os.path.exists(in_json):
        raise FileNotFoundError(f"JSON results file not found at '{in_json}'. Run scripts/run_hessian_analysis.py first.")

    with open(in_json, "r") as f:
        data = json.load(f)

    bit_sens = data["bit_width_sensitivity"]
    neuron_sens = data["neuron_pruning_sensitivity"]["neuron_importance"]
    prune_verif = data["fhe_pruning_verification"]["pruning_comparison"]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

    color_total = "#2b5c8f"
    color_norm = "#d95f02"
    color_fc1 = "#1b9e77"
    color_fc2 = "#7570b3"
    color_struct = "#e7298a"
    color_unstruct = "#666666"

    # Panel 1: Layer-wise Hessian Trace
    ax1 = axes[0]
    layers = list(bit_sens["total_trace"].keys())
    total_traces = [bit_sens["total_trace"][l] for l in layers]
    norm_traces = [bit_sens["normalized_trace"][l] for l in layers]

    x = np.arange(len(layers))
    width = 0.35

    rects1 = ax1.bar(x - width/2, total_traces, width, label="Total Trace Tr(H_l)", color=color_total)
    ax1_twin = ax1.twinx()
    rects2 = ax1_twin.bar(x + width/2, norm_traces, width, label="Norm Trace Tr(H_l)/#params", color=color_norm, alpha=0.85)

    ax1.set_xticks(x)
    ax1.set_xticklabels([l.upper() for l in layers], fontweight="bold")
    ax1.set_ylabel("Total Hessian Trace Tr(H_l)", color=color_total, fontweight="bold")
    ax1_twin.set_ylabel("Normalized Trace (Per-Parameter)", color=color_norm, fontweight="bold")
    ax1.set_title("1. Layer Quantization Sensitivity (Hutchinson Trace)", fontsize=11, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    # Panel 2: OBD Neuron Importance Distribution
    ax2 = axes[1]
    sorted_fc1 = np.sort(neuron_sens.get("fc1", []))
    sorted_fc2 = np.sort(neuron_sens.get("fc2", []))

    if len(sorted_fc1) > 0:
        ax2.plot(np.linspace(0, 100, len(sorted_fc1)), sorted_fc1, label="FC1 Neurons (92)", color=color_fc1, linewidth=2)
    if len(sorted_fc2) > 0:
        ax2.plot(np.linspace(0, 100, len(sorted_fc2)), sorted_fc2, label="FC2 Neurons (92)", color=color_fc2, linewidth=2)

    ax2.set_xlabel("Neuron Percentile (Sorted by OBD Importance)", fontweight="bold")
    ax2.set_ylabel("OBD Neuron Importance Score (1/2 H_ii w_i^2)", fontweight="bold")
    ax2.set_title("2. Structured OBD Neuron Importance Spectrum", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper left")

    # Panel 3: Structured vs Unstructured Pruning Comparison
    ax3 = axes[2]
    ratios = [p["prune_ratio"] * 100 for p in prune_verif]
    b_acc_struct = [p["structured_b_acc"] for p in prune_verif]
    b_acc_unstruct = [p["unstructured_b_acc"] for p in prune_verif]

    ax3.plot(ratios, b_acc_struct, "o-", label="Structured Pruning (Neuron Removal)", color=color_struct, linewidth=2.5, markersize=7)
    ax3.plot(ratios, b_acc_unstruct, "s--", label="Unstructured Zeroing (Sparse Matrix)", color=color_unstruct, linewidth=2, markersize=6)

    ax3.set_xlabel("Pruning / Zeroing Ratio (%)", fontweight="bold")
    ax3.set_ylabel("Concrete ML Accumulator Bit-Width b_acc (bits)", fontweight="bold")
    ax3.set_title("3. FHE Compilation Impact: Structured vs Unstructured", fontsize=11, fontweight="bold")
    ax3.set_ylim([0, max(b_acc_unstruct) + 2])
    ax3.legend(loc="lower left")

    plt.tight_layout()
    Path(out_img).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_img, dpi=300)
    plt.close()
    return out_img
