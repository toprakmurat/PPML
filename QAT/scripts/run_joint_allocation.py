"""
Run Joint Knapsack Allocation Experiments
=========================================
Runs joint optimization across:
  1. Primary Formulation: Minimize Distortion s.t. FHE Cost <= Budget Target
  2. Dual Formulation:    Minimize FHE Cost s.t. Distortion <= SLA Accuracy Bound

Evaluates three modes:
  - Joint Allocation (Bit-width + Structured Pruning)
  - Bit-width-Only Ablation (Baseline #2)
  - Pruning-Only Ablation (Baseline #3)

Verifies recommended configurations with Concrete ML and exports results.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import brevitas.nn as qnn
from concrete.ml.torch.compile import compile_brevitas_qat_model

# Add QAT root to Python path
QAT_ROOT = Path(__file__).resolve().parents[1]
if str(QAT_ROOT) not in sys.path:
    sys.path.insert(0, str(QAT_ROOT))

from qat.allocation.allocator import JointKnapsackAllocator
from visualization.plot_joint_allocation import plot_joint_allocation


def sanitize_for_json(obj):
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(v) for v in obj]
    else:
        return str(obj)


def compile_custom_config_in_concrete_ml(config: dict, seed: int = 42):
    (b1, s1) = config["fc1"]
    (b2, s2) = config["fc2"]
    (b3, s3) = config["fc3"]

    h1_dim = max(1, int(round(92 * (1.0 - s1))))
    h2_dim = max(1, int(round(92 * (1.0 - s2))))

    class HeterogeneousQATNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.quant_input = qnn.QuantIdentity(bit_width=b1, return_quant_tensor=True)
            self.fc1 = qnn.QuantLinear(784, h1_dim, bias=True, weight_bit_width=b1, bias_quant=None)
            self.relu1 = qnn.QuantReLU(bit_width=b1, return_quant_tensor=True)
            self.fc2 = qnn.QuantLinear(h1_dim, h2_dim, bias=True, weight_bit_width=b2, bias_quant=None)
            self.relu2 = qnn.QuantReLU(bit_width=b2, return_quant_tensor=True)
            self.fc3 = qnn.QuantLinear(h2_dim, 10, bias=True, weight_bit_width=b3, bias_quant=None)

        def forward(self, x):
            x = self.quant_input(x)
            x = self.relu1(self.fc1(x))
            x = self.relu2(self.fc2(x))
            x = self.fc3(x)
            return x

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = HeterogeneousQATNet().eval()
    calib_data = rng.uniform(0.0, 1.0, size=(50, 784)).astype(np.float32)

    t0 = time.perf_counter()
    try:
        q_mod = compile_brevitas_qat_model(
            torch_model=model,
            torch_inputset=calib_data,
            rounding_threshold_bits={"n_bits": 6, "method": "approximate"},
        )
        compile_time = time.perf_counter() - t0
        b_acc = q_mod.fhe_circuit.graph.maximum_integer_bit_width()
        return {"status": "success", "b_acc": b_acc, "compile_time": round(compile_time, 3)}
    except Exception as e:
        return {"status": "error", "error": str(e), "compile_time": round(time.perf_counter() - t0, 3)}


def main():
    default_hessian_json = QAT_ROOT / "experiments" / "results" / "hessian_sensitivity_results.json"
    default_out_json = QAT_ROOT / "experiments" / "results" / "joint_allocation_results.json"
    default_out_img = QAT_ROOT / "experiments" / "plots" / "joint_allocation_plots.png"

    parser = argparse.ArgumentParser(description="Run Joint Allocation Knapsack Experiments")
    parser.add_argument("--hessian_json", type=str, default=str(default_hessian_json), help="Path to Hessian results")
    parser.add_argument("--out_json", type=str, default=str(default_out_json), help="Output JSON path")
    parser.add_argument("--out_img", type=str, default=str(default_out_img), help="Output PNG path")
    args = parser.parse_args()

    print("==================================================================")
    print(" Joint Knapsack Allocation Optimization for QATNet")
    print("==================================================================")

    allocator = JointKnapsackAllocator(hessian_results_path=args.hessian_json)

    # Define 4 Budget Constraints
    budget_points = {
        "Tight Budget (C <= 5,000)": 5000.0,
        "Moderate Budget (C <= 15,000)": 15000.0,
        "High Budget (C <= 50,000)": 50000.0,
        "Unconstrained (C <= 200,000)": 200000.0,
    }

    print("\n--- Part 1: Primary Formulation (Minimize Distortion s.t. FHE Cost <= Budget) ---")
    results_by_mode = {}
    for mode in ["joint", "bitwidth_only", "pruning_only"]:
        print(f"\nMode: {mode.upper()}")
        mode_results = []
        for label, budget in budget_points.items():
            res = allocator.solve_knapsack(max_budget=budget, mode=mode)
            mode_results.append(res)
            print(f"  {label:<30s} | Distortion={res['total_distortion']:8.4f} | "
                  f"Cost={res['total_cost']:10,.0f} | FC1=(b={res['config']['fc1'][0]}, s={res['config']['fc1'][1]:.2f}) "
                  f"FC2=(b={res['config']['fc2'][0]}, s={res['config']['fc2'][1]:.2f}) "
                  f"FC3=(b={res['config']['fc3'][0]}, s={res['config']['fc3'][1]:.2f})")
        results_by_mode[mode] = mode_results

    print("\n--- Part 2: Dual Formulation (Minimize FHE Cost s.t. Distortion <= SLA Bound) ---")
    sla_bounds = {
        "Strict SLA (Distortion <= 0.05)": 0.05,
        "Moderate SLA (Distortion <= 0.20)": 0.20,
        "Relaxed SLA (Distortion <= 0.50)": 0.50,
    }
    dual_results = []
    for label, max_dist in sla_bounds.items():
        res_dual = allocator.solve_min_cost_for_distortion(max_allowed_distortion=max_dist, mode="joint")
        dual_results.append({"sla_label": label, "max_allowed_distortion": max_dist, "result": res_dual})
        print(f"  {label:<35s} | Achieved Cost={res_dual['total_cost']:10,.0f} | "
              f"Actual Distortion={res_dual['total_distortion']:8.4f}")

    print("\n--- Part 3: Concrete ML Compilation Verification of Recommended Configs ---")
    recommended_configs = {
        "Tight Budget (C <= 5k)": results_by_mode["joint"][0]["config"],
        "Moderate Budget (C <= 15k)": results_by_mode["joint"][1]["config"],
        "High Budget (C <= 50k)": results_by_mode["joint"][2]["config"],
    }
    verification_results = {}
    for name, cfg in recommended_configs.items():
        v_res = compile_custom_config_in_concrete_ml(cfg)
        verification_results[name] = {"config": cfg, "concrete_ml_verification": v_res}
        print(f"  {name:<28s} | Config: {cfg} | Concrete ML b_acc={v_res.get('b_acc')} | Status={v_res.get('status')}")

    output_data = sanitize_for_json({
        "budget_points": budget_points,
        "primary_formulation_results": results_by_mode,
        "dual_formulation_results": dual_results,
        "concrete_ml_verifications": verification_results,
    })

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to '{args.out_json}'")

    print("\n--- Part 4: Generating Visualization Figure ---")
    plot_saved = plot_joint_allocation(args.out_json, args.out_img)
    print(f"Plot saved to '{plot_saved}'")


if __name__ == "__main__":
    main()
