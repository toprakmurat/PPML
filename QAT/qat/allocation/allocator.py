"""
Joint Allocator Core Module for QATNet
=======================================
Implements constrained optimization (Dynamic Programming / Knapsack Grid Search)
to find the optimal layer-wise bit-width and structured neuron sparsity allocation:

    min_{b_l, s_l} sum_l Distortion(l, b_l, s_l)
    s.t.           sum_l Cost(l, b_l, s_l) <= Budget
"""

import json
import typing
from pathlib import Path
import numpy as np

_BASE_DIR = Path(__file__).resolve().parents[2]
_DEFAULT_HESSIAN_JSON = _BASE_DIR / "experiments" / "results" / "hessian_sensitivity_results.json"


class JointKnapsackAllocator:
    """
    Solves joint allocation of bit-width and structured neuron sparsity across layers.
    """
    def __init__(
        self,
        hessian_results_path: str = None,
        bit_widths: typing.List[int] = [3, 4, 5, 6, 8],
        sparsities: typing.List[float] = [0.0, 0.25, 0.50, 0.75],
    ):
        self.bit_widths = sorted(bit_widths)
        self.sparsities = sorted(sparsities)
        self.max_bit_width = max(self.bit_widths)

        path = hessian_results_path if hessian_results_path else str(_DEFAULT_HESSIAN_JSON)
        with open(path, "r") as f:
            self.hessian_data = json.load(f)

        self.traces = self.hessian_data["bit_width_sensitivity"]["total_trace"]
        self.neuron_importance = self.hessian_data["neuron_pruning_sensitivity"]["neuron_importance"]

        self.layer_dims = {
            "fc1": {"in": 784, "out": 92},
            "fc2": {"in": 92, "out": 92},
            "fc3": {"in": 92, "out": 10},
        }

    def compute_layer_distortion(self, layer_name: str, b: int, s: float) -> float:
        trace = self.traces[layer_name]
        d_quant = trace * ((2.0 ** (-2 * b)) - (2.0 ** (-2 * self.max_bit_width)))

        n_imp = np.array(self.neuron_importance[layer_name])
        n_neurons = len(n_imp)
        num_pruned = max(0, int(round(n_neurons * s)))

        if num_pruned > 0:
            sorted_imp = np.sort(n_imp)
            d_prune = float(np.sum(sorted_imp[:num_pruned]))
        else:
            d_prune = 0.0

        return float(d_quant + d_prune)

    def estimate_layer_fhe_cost(self, layer_name: str, b: int, s_current: float, s_prev: float = 0.0) -> typing.Dict[str, typing.Any]:
        orig_in = self.layer_dims[layer_name]["in"]
        orig_out = self.layer_dims[layer_name]["out"]

        actual_fan_in = max(1, int(round(orig_in * (1.0 - s_prev))))
        actual_out_dim = max(1, int(round(orig_out * (1.0 - s_current))))

        fan_in_bits = int(np.ceil(np.log2(actual_fan_in)))
        b_acc = max(b + 2, int(b + fan_in_bits - 3))
        b_acc = min(16, b_acc)

        unit_pbs_cost = 2 ** b_acc
        total_cost = actual_out_dim * unit_pbs_cost

        return {
            "actual_out_dim": actual_out_dim,
            "actual_fan_in": actual_fan_in,
            "b_acc": b_acc,
            "unit_pbs_cost": unit_pbs_cost,
            "total_cost": total_cost,
        }

    def evaluate_configuration(self, config: typing.Dict[str, typing.Tuple[int, float]]) -> typing.Dict[str, typing.Any]:
        total_distortion = 0.0
        total_cost = 0.0
        layer_breakdown = {}

        s_prev = 0.0
        for layer_name in ["fc1", "fc2", "fc3"]:
            b, s = config[layer_name]
            d_l = self.compute_layer_distortion(layer_name, b, s)
            c_info = self.estimate_layer_fhe_cost(layer_name, b, s_current=s, s_prev=s_prev)

            total_distortion += d_l
            total_cost += c_info["total_cost"]

            layer_breakdown[layer_name] = {
                "bit_width": b,
                "sparsity": s,
                "distortion": d_l,
                "fhe_cost": c_info["total_cost"],
                "b_acc": c_info["b_acc"],
                "actual_out_dim": c_info["actual_out_dim"],
            }
            s_prev = s

        return {
            "config": config,
            "total_distortion": total_distortion,
            "total_cost": total_cost,
            "layer_breakdown": layer_breakdown,
        }

    def solve_knapsack(self, max_budget: float, mode: str = "joint") -> typing.Dict[str, typing.Any]:
        if mode == "joint":
            b_choices = self.bit_widths
            s_choices = self.sparsities
        elif mode == "bitwidth_only":
            b_choices = self.bit_widths
            s_choices = [0.0]
        elif mode == "pruning_only":
            b_choices = [3]
            s_choices = self.sparsities
        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose 'joint', 'bitwidth_only', or 'pruning_only'.")

        grid_fc1 = [(b, s) for b in b_choices for s in s_choices]
        grid_fc2 = [(b, s) for b in b_choices for s in s_choices]
        grid_fc3 = [(b, s) for b in b_choices for s in s_choices]

        best_eval = None
        min_distortion = float("inf")

        for opt1 in grid_fc1:
            for opt2 in grid_fc2:
                for opt3 in grid_fc3:
                    cfg = {"fc1": opt1, "fc2": opt2, "fc3": opt3}
                    res = self.evaluate_configuration(cfg)

                    if res["total_cost"] <= max_budget:
                        if res["total_distortion"] < min_distortion:
                            min_distortion = res["total_distortion"]
                            best_eval = res

        if best_eval is None:
            min_cost = float("inf")
            for opt1 in grid_fc1:
                for opt2 in grid_fc2:
                    for opt3 in grid_fc3:
                        cfg = {"fc1": opt1, "fc2": opt2, "fc3": opt3}
                        res = self.evaluate_configuration(cfg)
                        if res["total_cost"] < min_cost:
                            min_cost = res["total_cost"]
                            best_eval = res

        best_eval["mode"] = mode
        best_eval["max_budget"] = max_budget
        return best_eval

    def generate_pareto_curve(self, budget_points: typing.List[float], mode: str = "joint") -> typing.List[typing.Dict[str, typing.Any]]:
        results = []
        for target in budget_points:
            res = self.solve_knapsack(max_budget=target, mode=mode)
            results.append(res)
        return results

    def solve_min_cost_for_distortion(self, max_allowed_distortion: float, mode: str = "joint") -> typing.Dict[str, typing.Any]:
        if mode == "joint":
            b_choices = self.bit_widths
            s_choices = self.sparsities
        elif mode == "bitwidth_only":
            b_choices = self.bit_widths
            s_choices = [0.0]
        elif mode == "pruning_only":
            b_choices = [3]
            s_choices = self.sparsities
        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose 'joint', 'bitwidth_only', or 'pruning_only'.")

        grid_fc1 = [(b, s) for b in b_choices for s in s_choices]
        grid_fc2 = [(b, s) for b in b_choices for s in s_choices]
        grid_fc3 = [(b, s) for b in b_choices for s in s_choices]

        best_eval = None
        min_cost = float("inf")

        for opt1 in grid_fc1:
            for opt2 in grid_fc2:
                for opt3 in grid_fc3:
                    cfg = {"fc1": opt1, "fc2": opt2, "fc3": opt3}
                    res = self.evaluate_configuration(cfg)

                    if res["total_distortion"] <= max_allowed_distortion:
                        if res["total_cost"] < min_cost:
                            min_cost = res["total_cost"]
                            best_eval = res

        if best_eval is None:
            min_dist = float("inf")
            for opt1 in grid_fc1:
                for opt2 in grid_fc2:
                    for opt3 in grid_fc3:
                        cfg = {"fc1": opt1, "fc2": opt2, "fc3": opt3}
                        res = self.evaluate_configuration(cfg)
                        if res["total_distortion"] < min_dist:
                            min_dist = res["total_distortion"]
                            best_eval = res

        best_eval["mode"] = mode
        best_eval["max_allowed_distortion"] = max_allowed_distortion
        return best_eval
