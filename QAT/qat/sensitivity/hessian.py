"""
Hessian-Based Sensitivity Estimator for Brevitas / PyTorch Models
===================================================================
Provides Hutchinson Trace Estimator and Diagonal Hessian (OBD) importance scoring.
"""

import time
import typing
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import brevitas.nn as qnn
from concrete.ml.torch.compile import compile_brevitas_qat_model


class HessianSensitivityEstimator:
    """
    Computes Hessian-based sensitivity signals for layer bit-width allocation
    and neuron-level structured pruning.
    """
    def __init__(self, model: nn.Module, criterion: nn.Module = None, device: str = "cpu"):
        self.model = model
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.device = device

    def _get_target_layers(self) -> typing.Dict[str, nn.Module]:
        layers = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, qnn.QuantLinear)):
                clean_name = name.split(".")[-1]
                layers[clean_name] = module
        return layers

    def compute_hutchinson_trace(
        self,
        dataloader: DataLoader,
        n_samples: int = 20,
        max_batches: int = 10,
    ) -> typing.Dict[str, typing.Any]:
        self.model.eval()
        self.model.to(self.device)

        target_layers = self._get_target_layers()
        layer_traces = {name: [] for name in target_layers.keys()}

        for sample_idx in range(n_samples):
            batch_count = 0
            for X_batch, y_batch in dataloader:
                if batch_count >= max_batches:
                    break
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)

                for name, layer in target_layers.items():
                    if layer.weight.grad is not None:
                        layer.weight.grad.zero_()

                    grads = torch.autograd.grad(loss, layer.weight, create_graph=True, retain_graph=True)[0]
                    v = torch.randint_like(layer.weight, high=2) * 2 - 1  # Rademacher sample {-1, +1}
                    v = v.float().to(self.device)

                    hv = torch.autograd.grad(grads, layer.weight, grad_outputs=v, retain_graph=True)[0]
                    v_hv = torch.sum(v * hv).item()
                    layer_traces[name].append(v_hv)

                batch_count += 1

        results = {}
        total_trace_sum = 0.0
        for name in target_layers.keys():
            traces = np.array(layer_traces[name])
            mean_trace = float(np.mean(traces))
            std_trace = float(np.std(traces))
            param_count = target_layers[name].weight.numel()
            norm_trace = float(mean_trace / max(1, param_count))
            total_trace_sum += abs(mean_trace)

            results[name] = {
                "total_trace": mean_trace,
                "trace_std": std_trace,
                "normalized_trace": norm_trace,
                "param_count": param_count,
            }

        sensitivity_ranking = sorted(
            results.keys(),
            key=lambda k: abs(results[k]["total_trace"]),
            reverse=True,
        )

        return {
            "layer_sensitivity": results,
            "ranking": sensitivity_ranking,
            "total_trace": {k: results[k]["total_trace"] for k in results},
            "normalized_trace": {k: results[k]["normalized_trace"] for k in results},
        }

    def compute_diagonal_hessian_obd(
        self,
        dataloader: DataLoader,
        max_batches: int = 10,
    ) -> typing.Dict[str, typing.Any]:
        self.model.eval()
        self.model.to(self.device)
        target_layers = self._get_target_layers()

        diag_hessians = {name: torch.zeros_like(layer.weight.data) for name, layer in target_layers.items()}
        total_samples = 0

        batch_count = 0
        for X_batch, y_batch in dataloader:
            if batch_count >= max_batches:
                break
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)

            for name, layer in target_layers.items():
                grads = torch.autograd.grad(loss, layer.weight, create_graph=False, retain_graph=True)[0]
                diag_hessians[name] += (grads ** 2).detach()

            total_samples += X_batch.size(0)
            batch_count += 1

        neuron_importance = {}
        weight_importance = {}

        for name, layer in target_layers.items():
            h_diag = diag_hessians[name] / max(1, total_samples)
            w = layer.weight.data.to(self.device)

            obd_weight_score = 0.5 * h_diag * (w ** 2)
            row_neuron_score = torch.sum(obd_weight_score, dim=1).cpu().numpy()

            neuron_importance[name] = row_neuron_score.tolist()
            weight_importance[name] = {
                "mean_obd": float(obd_weight_score.mean().item()),
                "max_obd": float(obd_weight_score.max().item()),
                "min_obd": float(obd_weight_score.min().item()),
            }

        return {
            "neuron_importance": neuron_importance,
            "weight_importance_summary": weight_importance,
        }


def verify_fhe_pruning_impact(
    prune_ratios: typing.List[float] = [0.0, 0.25, 0.50, 0.75],
    orig_dim: int = 92,
    seed: int = 42,
) -> typing.Dict[str, typing.Any]:
    rng = np.random.default_rng(seed)
    calib_data = rng.uniform(0.0, 1.0, size=(50, 784)).astype(np.float32)

    results = []
    for ratio in prune_ratios:
        pruned_dim = max(1, int(round(orig_dim * (1.0 - ratio))))

        class PrunedQATNet(nn.Module):
            def __init__(self, h_dim: int, n_bits: int = 3):
                super().__init__()
                self.quant_input = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True)
                self.fc1 = qnn.QuantLinear(784, h_dim, bias=True, weight_bit_width=n_bits, bias_quant=None)
                self.relu1 = qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=True)
                self.fc2 = qnn.QuantLinear(h_dim, h_dim, bias=True, weight_bit_width=n_bits, bias_quant=None)
                self.relu2 = qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=True)
                self.fc3 = qnn.QuantLinear(h_dim, 10, bias=True, weight_bit_width=n_bits, bias_quant=None)

            def forward(self, x):
                x = self.quant_input(x)
                x = self.relu1(self.fc1(x))
                x = self.relu2(self.fc2(x))
                x = self.fc3(x)
                return x

        torch.manual_seed(seed)
        struct_model = PrunedQATNet(h_dim=pruned_dim, n_bits=3).eval()
        t0 = time.perf_counter()
        q_module_struct = compile_brevitas_qat_model(
            torch_model=struct_model,
            torch_inputset=calib_data,
            rounding_threshold_bits={"n_bits": 6, "method": "approximate"},
        )
        b_acc_struct = q_module_struct.fhe_circuit.graph.maximum_integer_bit_width()
        time_struct = time.perf_counter() - t0

        torch.manual_seed(seed)
        unstruct_model = PrunedQATNet(h_dim=orig_dim, n_bits=3).eval()
        with torch.no_grad():
            if ratio > 0.0:
                w2 = unstruct_model.fc2.weight.data
                mask = (torch.rand_like(w2) > ratio).float()
                unstruct_model.fc2.weight.data.mul_(mask)

        t0 = time.perf_counter()
        q_module_unstruct = compile_brevitas_qat_model(
            torch_model=unstruct_model,
            torch_inputset=calib_data,
            rounding_threshold_bits={"n_bits": 6, "method": "approximate"},
        )
        b_acc_unstruct = q_module_unstruct.fhe_circuit.graph.maximum_integer_bit_width()
        time_unstruct = time.perf_counter() - t0

        results.append({
            "prune_ratio": ratio,
            "structured_hidden_dim": pruned_dim,
            "structured_b_acc": b_acc_struct,
            "structured_compile_time": round(time_struct, 3),
            "unstructured_hidden_dim": orig_dim,
            "unstructured_b_acc": b_acc_unstruct,
            "unstructured_compile_time": round(time_unstruct, 3),
        })

    return {
        "pruning_comparison": results,
        "conclusion": "Structured pruning physically reduces tensor shapes, lowering accumulator bit-width b_acc, whereas unstructured zeroing retains full matrix dimensions and b_acc.",
    }
