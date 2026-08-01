"""
Joint Allocator Core Module for QATNet
=======================================
Implements constrained optimization (Dynamic Programming / Knapsack Grid Search)
to find the optimal layer-wise bit-width and structured neuron sparsity allocation:

    min_{b_l, s_l} sum_l Distortion(l, b_l, s_l)
    s.t.           sum_l Cost(l, b_l, s_l) <= Budget

Modes:
  1. Joint Optimization (varying both b_l and s_l)
  2. Bit-Width-Only Ablation (s_l = 0 fixed, varying b_l)
  3. Pruning-Only Ablation (b_l = 3 fixed, varying s_l)
"""

import json
import typing
import numpy as np


class JointKnapsackAllocator:
    """
    Solves joint allocation of bit-width and structured neuron sparsity across layers.
    """
    def __init__(
        self,
        hessian_results_path: str,
        bit_widths: typing.List[int] = [3, 4, 5, 6, 8],
        sparsities: typing.List[float] = [0.0, 0.25, 0.50, 0.75],
    ):
        self.bit_widths = sorted(bit_widths)
        self.sparsities = sorted(sparsities)
        self.max_bit_width = max(self.bit_widths)

        # Load Phase 2 Hessian sensitivity metrics
        with open(hessian_results_path, "r") as f:
            self.hessian_data = json.load(f)

        self.traces = self.hessian_data["bit_width_sensitivity"]["total_trace"]
        self.neuron_importance = self.hessian_data["neuron_pruning_sensitivity"]["neuron_importance"]

        # Layer dimensions for QATNet: fc1 (784->92), fc2 (92->92), fc3 (92->10)
        self.layer_dims = {
            "fc1": {"in": 784, "out": 92},
            "fc2": {"in": 92, "out": 92},
            "fc3": {"in": 92, "out": 10},
        }

    def compute_layer_distortion(self, layer_name: str, b: int, s: float) -> float:
        """
        Computes layer distortion penalty combining quantization noise (Hessian trace)
        and OBD neuron pruning loss.
        """
        # 1. Quantization distortion penalty: Tr(H_l) * (2^{-2b} - 2^{-2b_max})
        trace = self.traces[layer_name]
        d_quant = trace * ((2.0 ** (-2 * b)) - (2.0 ** (-2 * self.max_bit_width)))

        # 2. Structured neuron pruning distortion penalty: sum of OBD scores for pruned neurons
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
        """
        Estimates layer FHE operational cost:
            out_dim = ceil(orig_out_dim * (1 - s_current))
            actual_fan_in = ceil(orig_in_dim * (1 - s_prev))
            b_acc ≈ max(b + 2, b + ceil(log2(actual_fan_in)) - 3)
            cost = out_dim * 2^{b_acc}
        """
        orig_in = self.layer_dims[layer_name]["in"]
        orig_out = self.layer_dims[layer_name]["out"]

        actual_fan_in = max(1, int(round(orig_in * (1.0 - s_prev))))
        actual_out_dim = max(1, int(round(orig_out * (1.0 - s_current))))

        # Estimating FHE accumulator bit-width b_acc
        fan_in_bits = int(np.ceil(np.log2(actual_fan_in)))
        # Calibrated empirical accumulator bit-width model for Brevitas + Concrete ML
        b_acc = max(b + 2, int(b + fan_in_bits - 3))
        b_acc = min(16, b_acc)  # Cap at reasonable FHE bound

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
        """
        Evaluates total distortion and total FHE cost for a 3-layer configuration dict:
            config = {
                "fc1": (b1, s1),
                "fc2": (b2, s2),
                "fc3": (b3, s3),
            }
        """
        total_distortion = 0.0
        total_cost = 0.0
        layer_breakdown = {}

        s_prev = 0.0  # Input dimension to fc1 is fixed at 784
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
            # Output of current layer is input dimension to next layer
            s_prev = s

        return {
            "config": config,
            "total_distortion": total_distortion,
            "total_cost": total_cost,
            "layer_breakdown": layer_breakdown,
        }

    def solve_knapsack(self, max_budget: float, mode: str = "joint") -> typing.Dict[str, typing.Any]:
        """
        Solves discrete knapsack / grid optimization under specified mode:
          - 'joint': vary both b in bit_widths and s in sparsities
          - 'bitwidth_only': s = 0.0 fixed, vary b in bit_widths
          - 'pruning_only': b = 3 fixed, vary s in sparsities
        """
        if mode == "joint":
            b_choices = self.bit_widths
            s_choices = self.sparsities
        elif mode == "bitwidth_only":
            b_choices = self.bit_widths
            s_choices = [0.0]
        elif mode == "pruning_only":
            b_choices = [3]  # Baseline uniform bit-width
            s_choices = self.sparsities
        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose 'joint', 'bitwidth_only', or 'pruning_only'.")

        # Candidate choices per layer
        grid_fc1 = [(b, s) for b in b_choices for s in s_choices]
        grid_fc2 = [(b, s) for b in b_choices for s in s_choices]
        grid_fc3 = [(b, s) for b in b_choices for s in s_choices]

        best_eval = None
        min_distortion = float("inf")

        # Exact grid evaluation over search space
        for opt1 in grid_fc1:
            for opt2 in grid_fc2:
                for opt3 in grid_fc3:
                    cfg = {"fc1": opt1, "fc2": opt2, "fc3": opt3}
                    res = self.evaluate_configuration(cfg)

                    if res["total_cost"] <= max_budget:
                        if res["total_distortion"] < min_distortion:
                            min_distortion = res["total_distortion"]
                            best_eval = res

        # If no configuration fits strictly within budget, fallback to minimum cost config
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
        """
        Runs solver across multiple budget targets to construct Pareto frontier.
        """
        results = []
        for target in budget_points:
            res = self.solve_knapsack(max_budget=target, mode=mode)
            results.append(res)
        return results
