"""
Plot Joint Allocation Results for QATNet
=========================================
"""

import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

_BASE_DIR = Path(__file__).resolve().parents[1]
_DEFAULT_RESULTS_JSON = _BASE_DIR / "experiments" / "results" / "joint_allocation_results.json"
_DEFAULT_OUTPUT_IMG = _BASE_DIR / "experiments" / "plots" / "joint_allocation_plots.png"


def plot_joint_allocation(json_path: str = None, out_img_path: str = None):
    in_json = json_path if json_path else str(_DEFAULT_RESULTS_JSON)
    out_img = out_img_path if out_img_path else str(_DEFAULT_OUTPUT_IMG)

    if not os.path.exists(in_json):
        raise FileNotFoundError(f"JSON results file not found at '{in_json}'. Run scripts/run_joint_allocation.py first.")

    with open(in_json, "r") as f:
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

    color_joint = "#2b5c8f"
    color_bw = "#d95f02"
    color_prune = "#7570b3"
    color_fc1 = "#1b9e77"
    color_fc2 = "#e7298a"

    # Panel 1: Distortion vs Cost (Pareto Frontiers)
    ax1 = axes[0]
    for mode, color, mark, name in [
        ("joint", color_joint, "o-", "Joint Allocation (b_l + s_l)"),
        ("bitwidth_only", color_bw, "s--", "Bit-width-Only (Baseline #2)"),
        ("pruning_only", color_prune, "^:", "Pruning-Only (Baseline #3)"),
    ]:
        if mode in results_by_mode:
            res_list = results_by_mode[mode]
            costs = [r["total_cost"] for r in res_list]
            dists = [r["total_distortion"] for r in res_list]
            ax1.plot(costs, dists, mark, color=color, label=name, linewidth=2, markersize=6)

    ax1.set_xlabel("FHE Operational Cost Budget (Log Scale)", fontweight="bold")
    ax1.set_ylabel("Hessian Sensitivity-Weighted Distortion", fontweight="bold")
    ax1.set_xscale("log")
    ax1.set_title("1. Distortion vs. Cost (Pareto Frontier)", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right")

    # Panel 2: Optimal Bit-Width Allocation across Layers
    ax2 = axes[1]
    if "joint" in results_by_mode:
        joint_res = results_by_mode["joint"]
        fc1_b = [r["layer_breakdown"]["fc1"]["bit_width"] for r in joint_res]
        fc2_b = [r["layer_breakdown"]["fc2"]["bit_width"] for r in joint_res]
        fc3_b = [r["layer_breakdown"]["fc3"]["bit_width"] for r in joint_res]

        x = np.arange(len(labels))
        ax2.plot(x, fc1_b, "o-", color=color_fc1, label="FC1 Bit-Width", linewidth=2)
        ax2.plot(x, fc2_b, "s--", color=color_fc2, label="FC2 Bit-Width", linewidth=2)
        ax2.plot(x, fc3_b, "^:", color="#666666", label="FC3 Bit-Width", linewidth=2)

        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, fontweight="bold")
        ax2.set_xlabel("Budget Level", fontweight="bold")
        ax2.set_ylabel("Allocated Bit-Width (bits)", fontweight="bold")
        ax2.set_title("2. Optimal Bit-Width per Layer", fontsize=11, fontweight="bold")
        ax2.legend(loc="upper left")

    # Panel 3: Optimal Sparsity Allocation across Layers
    ax3 = axes[2]
    if "joint" in results_by_mode:
        joint_res = results_by_mode["joint"]
        fc1_s = [r["layer_breakdown"]["fc1"]["sparsity"] * 100 for r in joint_res]
        fc2_s = [r["layer_breakdown"]["fc2"]["sparsity"] * 100 for r in joint_res]

        x = np.arange(len(labels))
        ax3.plot(x, fc1_s, "o-", color=color_fc1, label="FC1 Sparsity (%)", linewidth=2)
        ax3.plot(x, fc2_s, "s--", color=color_fc2, label="FC2 Sparsity (%)", linewidth=2)

        ax3.set_xticks(x)
        ax3.set_xticklabels(labels, fontweight="bold")
        ax3.set_xlabel("Budget Level", fontweight="bold")
        ax3.set_ylabel("Neuron Sparsity / Pruning (%)", fontweight="bold")
        ax3.set_title("3. Optimal Structured Pruning per Layer", fontsize=11, fontweight="bold")
        ax3.legend(loc="upper left")

    plt.tight_layout()
    Path(out_img).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_img, dpi=300)
    plt.close()
    return out_img
