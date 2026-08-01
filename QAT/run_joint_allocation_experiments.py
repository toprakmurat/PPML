"""
Run Phase 3 Joint Allocation Experiments
=========================================
Runs joint knapsack optimization across 4 total budget points for:
  1. Joint Allocation (Bit-width + Structured Pruning)
  2. Bit-width-Only Ablation (Baseline #2)
  3. Pruning-Only Ablation (Baseline #3)

Verifies recommended configurations with Concrete ML and exports results to
`QAT/joint_allocation_results.json`.
"""

import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import brevitas.nn as qnn
from concrete.ml.torch.compile import compile_brevitas_qat_model
from joint_allocator import JointKnapsackAllocator


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
    """
    Constructs a Brevitas QATNet matching a per-layer configuration dict:
      config = {
        "fc1": (b1, s1),
        "fc2": (b2, s2),
        "fc3": (b3, s3),
      }
    and compiles it via Concrete ML to extract exact circuit graph metrics.
    """
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

    rng = np.random.default_rng(seed)
    calib_data = rng.uniform(0.0, 1.0, size=(100, 784)).astype(np.float32)

    torch.manual_seed(seed)
    model = HeterogeneousQATNet().eval()

    t0 = time.perf_counter()
    q_module = compile_brevitas_qat_model(
        torch_model=model,
        torch_inputset=calib_data,
        rounding_threshold_bits={"n_bits": 6, "method": "approximate"},
    )
    compile_time = time.perf_counter() - t0
    b_acc_compiled = q_module.fhe_circuit.graph.maximum_integer_bit_width()

    return {
        "compiled_b_acc": b_acc_compiled,
        "compile_time_sec": round(compile_time, 3),
        "h1_dim": h1_dim,
        "h2_dim": h2_dim,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Phase 3 Joint Allocation Experiments")
    parser.add_argument("--hessian_json", type=str, default="./hessian_sensitivity_results.json", help="Path to Hessian results")
    parser.add_argument("--out_json", type=str, default="./joint_allocation_results.json", help="Output JSON path")
    args = parser.parse_args()

    hessian_path = os.path.abspath(args.hessian_json)
    if not os.path.exists(hessian_path):
        raise FileNotFoundError(f"Hessian sensitivity file not found at '{hessian_path}'. Run Phase 2 first.")

    print(f"\n=======================================================")
    print(f" Phase 3 — Joint Allocation Algorithm Runner          ")
    print(f"=======================================================\n")

    allocator = JointKnapsackAllocator(hessian_results_path=hessian_path)

    # 1. Determine FHE Budget Range
    # Evaluate max unconstrained baseline cost (b=8, s=0%)
    max_cfg = {"fc1": (8, 0.0), "fc2": (8, 0.0), "fc3": (8, 0.0)}
    max_eval = allocator.evaluate_configuration(max_cfg)
    max_cost = max_eval["total_cost"]

    # Evaluate min baseline cost (b=3, s=0.75)
    min_cfg = {"fc1": (3, 0.75), "fc2": (3, 0.75), "fc3": (3, 0.75)}
    min_eval = allocator.evaluate_configuration(min_cfg)
    min_cost = min_eval["total_cost"]

    print(f"  [+] Calculated FHE Cost Range: Min Cost = {min_cost:,.0f} | Max Cost = {max_cost:,.0f}")

    # Define 4 Budget Target Points
    budget_points = [
        float(round(min_cost + 0.15 * (max_cost - min_cost))),  # Budget 1: Low / Tight
        float(round(min_cost + 0.40 * (max_cost - min_cost))),  # Budget 2: Medium
        float(round(min_cost + 0.70 * (max_cost - min_cost))),  # Budget 3: High
        float(round(max_cost)),                                 # Budget 4: Unconstrained / Max
    ]
    budget_labels = ["B1_Tight", "B2_Medium", "B3_High", "B4_Max"]

    # 2. Run Allocation Sweeps across Modes
    modes = ["joint", "bitwidth_only", "pruning_only"]
    results_by_mode = {}

    for mode in modes:
        print(f"\n--- Running Optimization Mode: '{mode.upper()}' ---")
        mode_results = []
        for idx, target in enumerate(budget_points):
            label = budget_labels[idx]
            res = allocator.solve_knapsack(max_budget=target, mode=mode)
            
            # Concrete ML Compilation Validation
            c_info = compile_custom_config_in_concrete_ml(res["config"])
            res["concrete_ml_validation"] = c_info
            res["budget_label"] = label
            
            mode_results.append(res)

            cfg_str = " | ".join([f"{l}: {b}b, {int(s*100)}%s" for l, (b, s) in res["config"].items()])
            print(f"  [{label:<9}] Budget <= {target:>9,.0f} | Cost: {res['total_cost']:>8,.0f} | Dist: {res['total_distortion']:>8.4f} | Config: {cfg_str}")

        results_by_mode[mode] = mode_results

    # 3. Format and Export Structured JSON Results
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "budget_points": {label: target for label, target in zip(budget_labels, budget_points)},
        "fhe_cost_range": {"min_cost": min_cost, "max_cost": max_cost},
        "results_by_mode": results_by_mode,
    }

    sanitized_payload = sanitize_for_json(payload)
    out_json_path = os.path.abspath(args.out_json)
    with open(out_json_path, "w") as f:
        json.dump(sanitized_payload, f, indent=2)

    print(f"\n[+] Results successfully exported to '{out_json_path}'")

    # 4. Print Summary Comparison Table
    print("\n╔════════════════════════════════════════════════════════════════════════════════════════╗")
    print("║ Phase 3 Joint Allocation Summary (Distortion @ Budget Targets)                          ║")
    print("╠════════════════════════════════════════════════════════════════════════════════════════╣")
    print(f"║ {'Budget Target':<12} | {'Joint Distortion':<18} | {'Bit-width-Only':<18} | {'Pruning-Only':<18} ║")
    print("╠════════════════════════════════════════════════════════════════════════════════════════╣")
    for i in range(len(budget_points)):
        label = budget_labels[i]
        d_joint = results_by_mode["joint"][i]["total_distortion"]
        d_bw = results_by_mode["bitwidth_only"][i]["total_distortion"]
        d_prune = results_by_mode["pruning_only"][i]["total_distortion"]
        print(f"║ {label:<12} | {d_joint:<18.4f} | {d_bw:<18.4f} | {d_prune:<18.4f} ║")
    print("╚════════════════════════════════════════════════════════════════════════════════════════╝\n")


if __name__ == "__main__":
    main()
