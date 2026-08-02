"""
FHE Cost Model Benchmark Harness
=================================
Compiles each layer of QATNet in isolation using Brevitas QAT primitives and Zama Concrete ML.
"""

import os
import json
import time
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import brevitas.nn as qnn
from concrete.ml.torch.compile import compile_brevitas_qat_model

_BASE_DIR = Path(__file__).resolve().parents[2]
_RESULTS_JSON = _BASE_DIR / "experiments" / "results" / "cost_model_results.json"


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


class SingleLayerQAT(nn.Module):
    """
    Isolated single-layer wrapper for Brevitas QAT compilation.
    """
    def __init__(self, in_features: int, out_features: int, n_bits: int, has_relu: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_relu = has_relu

        self.quant_input = qnn.QuantIdentity(
            bit_width=n_bits,
            return_quant_tensor=True,
        )
        self.fc = qnn.QuantLinear(
            in_features, out_features, bias=True,
            weight_bit_width=n_bits,
            bias_quant=None,
            return_quant_tensor=False,
        )
        if has_relu:
            self.relu = qnn.QuantReLU(
                bit_width=n_bits,
                return_quant_tensor=False,
            )

    def forward(self, x):
        x = self.quant_input(x)
        x = self.fc(x)
        if self.has_relu:
            x = self.relu(x)
        return x


def apply_fan_in_pruning(model: SingleLayerQAT, fan_in_ratio: float, seed: int = 42):
    if fan_in_ratio >= 1.0:
        return
    rng = np.random.default_rng(seed)
    weight_data = model.fc.weight.data
    out_dim, in_dim = weight_data.shape
    k = max(1, int(round(in_dim * fan_in_ratio)))
    mask = torch.zeros_like(weight_data)
    for i in range(out_dim):
        active_indices = rng.choice(in_dim, size=k, replace=False)
        mask[i, active_indices] = 1.0
    model.fc.weight.data.mul_(mask)


def generate_calibration_data(in_features: int, num_samples: int = 100, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = rng.uniform(low=0.0, high=1.0, size=(num_samples, in_features)).astype(np.float32)
    return data


def compile_and_profile_layer(
    layer_name: str,
    in_features: int,
    out_features: int,
    n_bits: int,
    has_relu: bool,
    fan_in_ratio: float = 1.0,
    num_samples: int = 100,
):
    torch.manual_seed(42)
    model = SingleLayerQAT(in_features, out_features, n_bits=n_bits, has_relu=has_relu)
    model.eval()

    if fan_in_ratio < 1.0:
        apply_fan_in_pruning(model, fan_in_ratio)

    calib_data = generate_calibration_data(in_features, num_samples=num_samples)

    t0 = time.perf_counter()
    try:
        quantized_module = compile_brevitas_qat_model(
            torch_model=model,
            torch_inputset=calib_data,
            rounding_threshold_bits={"n_bits": 6, "method": "approximate"},
        )
        compile_time = time.perf_counter() - t0
        circuit = quantized_module.fhe_circuit
        b_acc = circuit.graph.maximum_integer_bit_width()
        
        stats = {}
        if hasattr(circuit, "statistics"):
            circuit_stats = circuit.statistics
            if isinstance(circuit_stats, dict):
                stats = sanitize_for_json(circuit_stats)
            elif hasattr(circuit_stats, "__dict__"):
                stats = sanitize_for_json(circuit_stats.__dict__)

        unit_pbs_cost = 2 ** b_acc
        pbs_count = stats.get("programmable_bootstrap_count", out_features if has_relu else 0)
        if pbs_count == 0 and has_relu:
            pbs_count = out_features

        total_cost = out_features * unit_pbs_cost

        return {
            "layer_name": layer_name,
            "in_features": in_features,
            "out_features": out_features,
            "num_neurons": out_features,
            "n_bits": n_bits,
            "fan_in_ratio": fan_in_ratio,
            "actual_fan_in": int(round(in_features * fan_in_ratio)),
            "b_acc": b_acc,
            "pbs_count": pbs_count,
            "unit_pbs_cost": unit_pbs_cost,
            "total_cost": total_cost,
            "normalized_cost": total_cost / out_features,
            "compile_time": round(compile_time, 3),
            "status": "success",
            "stats": stats,
        }

    except Exception as e:
        return {
            "layer_name": layer_name,
            "in_features": in_features,
            "out_features": out_features,
            "num_neurons": out_features,
            "n_bits": n_bits,
            "fan_in_ratio": fan_in_ratio,
            "actual_fan_in": int(round(in_features * fan_in_ratio)),
            "b_acc": None,
            "pbs_count": None,
            "unit_pbs_cost": None,
            "total_cost": None,
            "normalized_cost": None,
            "compile_time": round(time.perf_counter() - t0, 3),
            "status": "error",
            "error_msg": str(e),
        }


def run_experiments(min_bit: int = 2, max_bit: int = 8, hidden_dim: int = 92, out_json_path: str = None):
    layers_config = [
        ("fc1", 784, hidden_dim, True),
        ("fc2", hidden_dim, hidden_dim, True),
        ("fc3", hidden_dim, 10, False),
    ]

    bitwidth_results = []
    fanin_results = []

    print(f"\n=======================================================")
    print(f" Experiment 1: Sweeping Bit-Widths (b = {min_bit} to {max_bit})")
    print(f"=======================================================")
    
    for layer_name, in_dim, out_dim, has_relu in layers_config:
        print(f"\n--- Profiling Layer: {layer_name} ({in_dim} -> {out_dim}) ---")
        for b in range(min_bit, max_bit + 1):
            res = compile_and_profile_layer(
                layer_name=layer_name,
                in_features=in_dim,
                out_features=out_dim,
                n_bits=b,
                has_relu=has_relu,
                fan_in_ratio=1.0,
            )
            if res["status"] == "success":
                print(f"  b={b:2d} | b_acc={res['b_acc']:2d} bits | PBS count={res['pbs_count']:4d} | "
                      f"Unit PBS Cost=2^{res['b_acc']} ({res['unit_pbs_cost']:6d}) | "
                      f"Total Cost={res['total_cost']:10,d} | Cost/Neuron={res['normalized_cost']:.1f}")
            else:
                print(f"  b={b:2d} | Error: {res.get('error_msg')}")
            bitwidth_results.append(res)

    print(f"\n=======================================================")
    print(f" Experiment 2: Sweeping Fan-In / Sparsity (Fixed b = 4)")
    print(f"=======================================================")

    fixed_b = 4
    fan_in_ratios = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]

    for layer_name, in_dim, out_dim, has_relu in layers_config:
        print(f"\n--- Pruning Sweep Layer: {layer_name} ({in_dim} -> {out_dim}, b={fixed_b}) ---")
        for k_ratio in fan_in_ratios:
            res = compile_and_profile_layer(
                layer_name=layer_name,
                in_features=in_dim,
                out_features=out_dim,
                n_bits=fixed_b,
                has_relu=has_relu,
                fan_in_ratio=k_ratio,
            )
            if res["status"] == "success":
                print(f"  fan_in={res['actual_fan_in']:4d}/{in_dim} ({k_ratio*100:3.0f}%) | "
                      f"b_acc={res['b_acc']:2d} bits | Unit Cost={res['unit_pbs_cost']:6d} | "
                      f"Total Cost={res['total_cost']:10,d}")
            else:
                print(f"  fan_in={res['actual_fan_in']:4d}/{in_dim} ({k_ratio*100:3.0f}%) | Error: {res.get('error_msg')}")
            fanin_results.append(res)

    output_data = sanitize_for_json({
        "bitwidth_results": bitwidth_results,
        "fanin_results": fanin_results,
    })

    target_json = Path(out_json_path) if out_json_path else _RESULTS_JSON
    target_json.parent.mkdir(parents=True, exist_ok=True)
    with open(target_json, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults successfully saved to {target_json}")
    return output_data
