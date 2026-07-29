"""
FHE Cost Model Module
=====================
Deliverable for Phase 1: Validated FHE Cost Estimator Function.

Provides `estimated_cost(layer, bitwidth, sparsity)` to calculate Concrete ML
FHE operational cost (PBS operations * unit PBS complexity 2^b_acc) based on
layer dimensions, bit-width precision, and sparsity (fan-in pruning fraction).
"""

import os
import json
import math
import numpy as np
from typing import Union, Tuple, Dict, Any

DEFAULT_LAYER_DIMS = {
    "fc1": (784, 92, True),
    "fc2": (92, 92, True),
    "fc3": (92, 10, False),
}


def _load_calibration_data() -> Dict[str, Any]:
    """
    Loads empirical benchmark results if available.
    """
    json_path = os.path.join(os.path.dirname(__file__), "cost_model_results.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def estimate_accumulator_bitwidth(
    in_features: int,
    bitwidth: int,
    sparsity: float = 0.0,
    calib_data: Dict[str, Any] = None,
    layer_name: str = None,
) -> int:
    """
    Estimates maximum integer accumulator bit-width (b_acc) for a given layer.

    Args:
        in_features: Number of input features/connections.
        bitwidth: Quantization precision bit-width (2 to 16).
        sparsity: Fraction of zeroed weights (0.0 = dense, 0.8 = 80% sparse).
        calib_data: Optional benchmark calibration results.
        layer_name: Optional layer name ('fc1', 'fc2', 'fc3').

    Returns:
        Estimated accumulator bit-width b_acc (integer, typically in 2..16).
    """
    fan_in_ratio = max(0.01, 1.0 - sparsity)
    active_fan_in = max(1, int(round(in_features * fan_in_ratio)))

    # Check empirical lookup table if calibration data is provided and contains successful runs
    if calib_data and "bitwidth_results" in calib_data:
        bw_matches = [
            r for r in calib_data["bitwidth_results"]
            if r.get("layer_name") == layer_name and r.get("n_bits") == bitwidth
            and r.get("status") == "success" and r.get("b_acc") is not None
        ]
        if bw_matches and sparsity == 0.0:
            return bw_matches[0]["b_acc"]

        fan_matches = [
            r for r in calib_data.get("fanin_results", [])
            if r.get("layer_name") == layer_name and abs(r.get("fan_in_ratio", 1.0) - fan_in_ratio) < 0.05
            and r.get("status") == "success" and r.get("b_acc") is not None
        ]
        if fan_matches:
            base_b = fan_matches[0].get("n_bits", 4)
            delta_b = bitwidth - base_b
            return max(2, min(16, fan_matches[0]["b_acc"] + delta_b))

    # Analytical / calibrated fallback formula:
    # b_acc = b_weight/act + ceil(log2(active_fan_in * average_magnitude_factor))
    magnitude_scale = 0.35
    accum_bits = bitwidth + int(math.ceil(math.log2(max(1, active_fan_in * magnitude_scale))))
    
    # Concrete ML bounds accumulator bit-widths to [2, 16]
    return max(2, min(16, accum_bits))


def estimated_cost(
    layer: Union[str, Tuple[int, int], Any],
    bitwidth: int,
    sparsity: float = 0.0,
) -> float:
    """
    Computes the calibrated FHE execution cost estimate for a layer.

    Args:
        layer: Layer specification ('fc1', 'fc2', 'fc3', or (in_features, out_features)).
        bitwidth: Integer weight/activation quantization bit-width (2 to 16).
        sparsity: Float fraction of zeroed/pruned connections [0.0, 1.0).

    Returns:
        Estimated FHE operational cost (units of PBS operation complexity, O(N_neurons * 2^b_acc)).
    """
    # 1. Parse layer geometry
    if isinstance(layer, str):
        layer_key = layer.lower()
        if layer_key in DEFAULT_LAYER_DIMS:
            in_features, out_features, has_relu = DEFAULT_LAYER_DIMS[layer_key]
        else:
            raise ValueError(f"Unknown layer name '{layer}'. Expected one of {list(DEFAULT_LAYER_DIMS.keys())} or (in_features, out_features) tuple.")
    elif isinstance(layer, (tuple, list)) and len(layer) >= 2:
        in_features, out_features = layer[0], layer[1]
        has_relu = True
        layer_key = f"custom_{in_features}x{out_features}"
    elif hasattr(layer, "in_features") and hasattr(layer, "out_features"):
        in_features, out_features = layer.in_features, layer.out_features
        has_relu = True
        layer_key = getattr(layer, "layer_name", f"layer_{in_features}x{out_features}")
    else:
        raise TypeError(f"Unsupported layer type: {type(layer)}")

    # 2. Validate bounds
    if not (2 <= bitwidth <= 16):
        raise ValueError(f"bitwidth must be between 2 and 16, got {bitwidth}")
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"sparsity must be in range [0.0, 1.0), got {sparsity}")

    calib_data = _load_calibration_data()

    # 3. Estimate accumulator bit-width
    b_acc = estimate_accumulator_bitwidth(
        in_features=in_features,
        bitwidth=bitwidth,
        sparsity=sparsity,
        calib_data=calib_data,
        layer_name=layer_key if isinstance(layer, str) else None,
    )

    # 4. Compute cost decomposition:
    # layer_cost(b, s) = num_neurons * unit_pbs_cost(b_acc)
    # unit_pbs_cost = 2^b_acc
    num_neurons = out_features
    unit_pbs_cost = 2 ** b_acc
    cost = num_neurons * unit_pbs_cost

    return float(cost)


if __name__ == "__main__":
    print("Testing FHE Cost Estimator Function:")
    print("------------------------------------")
    for l_name in ["fc1", "fc2", "fc3"]:
        cost_b3 = estimated_cost(l_name, bitwidth=3, sparsity=0.0)
        cost_b4 = estimated_cost(l_name, bitwidth=4, sparsity=0.0)
        cost_b4_sparse50 = estimated_cost(l_name, bitwidth=4, sparsity=0.5)
        print(f"Layer {l_name:3s} | b=3, s=0.0: cost={cost_b3:10,.0f} | b=4, s=0.0: cost={cost_b4:10,.0f} | b=4, s=0.5: cost={cost_b4_sparse50:10,.0f}")
