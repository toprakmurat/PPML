"""
QAT Model Training & Concrete ML FHE Compilation Runner
=======================================================
Runs QAT training on MNIST, compiles to Concrete ML FHE circuit,
and evaluates baseline PyTorch, FHE simulation, and FHE encrypted execution.
"""

import time
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from concrete.ml.torch.compile import compile_brevitas_qat_model

# Add QAT root to Python path
QAT_ROOT = Path(__file__).resolve().parents[1]
if str(QAT_ROOT) not in sys.path:
    sys.path.insert(0, str(QAT_ROOT))

from qat.models.qat_net import (
    QATNet,
    DEFAULT_CONFIG,
    load_mnist,
    train_qat,
    evaluate_pytorch,
    evaluate_fhe_simulation,
    evaluate_fhe_execute,
)


def main():
    parser = argparse.ArgumentParser(description="Concrete ML + Brevitas QAT Training Runner")
    parser.add_argument("--n_bits", type=int, default=DEFAULT_CONFIG["n_bits"], help="Bit-width (default: 3)")
    parser.add_argument("--hidden_dim", type=int, default=DEFAULT_CONFIG["hidden_dim"], help="Hidden dim (default: 92)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"], help="Training epochs (default: 35)")
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_CONFIG["weight_decay"], help="Weight decay (default: 1e-4)")
    parser.add_argument("--val_ratio", type=float, default=DEFAULT_CONFIG["val_ratio"], help="Val ratio (default: 0.10)")
    parser.add_argument("--train_subset", type=int, default=DEFAULT_CONFIG["train_subset"], help="Train subset (default: 10000)")
    parser.add_argument("--test_subset", type=int, default=DEFAULT_CONFIG["test_subset"], help="Test subset (default: 500)")
    parser.add_argument("--fhe_samples", type=int, default=DEFAULT_CONFIG["fhe_samples"], help="FHE samples (default: 5)")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_CONFIG["data_dir"], help="Data directory")
    parser.add_argument("--checkpoint_path", type=str, default=DEFAULT_CONFIG["checkpoint_path"], help="Checkpoint path")
    parser.add_argument("--skip_fhe_execute", action="store_true", help="Skip encrypted execution")
    args = parser.parse_args()

    config = {**DEFAULT_CONFIG, **vars(args)}
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    print("Step 1 - Loading MNIST dataset")
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist(config)

    print("\nStep 2 - Building Brevitas QAT Model")
    model = QATNet(n_bits=config["n_bits"], hidden_dim=config["hidden_dim"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture: 784 → {config['hidden_dim']} → {config['hidden_dim']} → 10")
    print(f"  Bit-width:    {config['n_bits']}b weights / {config['n_bits']}b activations")
    print(f"  Parameters:   {n_params:,}")

    print("\nStep 3 - Quantization-Aware Training")
    model, best_val_acc = train_qat(model, X_train, y_train, X_val, y_val, config)

    print("\nStep 4 - Compiling to FHE circuit")
    calibration_data = X_train[:100].numpy()
    t0 = time.perf_counter()
    quantized_module = compile_brevitas_qat_model(
        torch_model=model,
        torch_inputset=calibration_data,
        rounding_threshold_bits={"n_bits": 6, "method": "approximate"},
    )
    compile_time = time.perf_counter() - t0
    print(f"Compilation complete in {compile_time:.1f}s")

    bitwidth = quantized_module.fhe_circuit.graph.maximum_integer_bit_width()
    print(f"  Accumulator bit-width: {bitwidth} bits")

    print("\nStep 5a - PyTorch Baseline (float)")
    _, acc_pt = evaluate_pytorch(model, X_test, y_test)
    print(f"  PyTorch accuracy:  {acc_pt:.4f}  ({int(acc_pt * len(y_test))}/{len(y_test)})")

    print("\nStep 5b - FHE Simulation (cleartext integers)")
    t0 = time.perf_counter()
    _, acc_sim = evaluate_fhe_simulation(quantized_module, X_test, y_test)
    sim_time = time.perf_counter() - t0
    print(f"  Simulated accuracy: {acc_sim:.4f}  ({int(acc_sim * len(y_test))}/{len(y_test)})")
    print(f"  Simulation time:    {sim_time:.2f}s for {len(y_test)} samples")

    acc_fhe = None
    if not config["skip_fhe_execute"]:
        print("\nStep 5c - FHE Execution (encrypted inference)")
        _, acc_fhe = evaluate_fhe_execute(
            quantized_module, X_test, y_test,
            n_samples=config["fhe_samples"],
        )
        print(f"\n  FHE accuracy: {acc_fhe:.4f}  "
              f"({int(acc_fhe * config['fhe_samples'])}/{config['fhe_samples']})")
    else:
        print("\n  (Skipping FHE execution — use without --skip_fhe_execute to enable)")

    print("\n╔══════════════════════════════════════════════════╗")
    print("║  Summary                                         ║")
    print("╚══════════════════════════════════════════════════╝")
    print(f"  Model:           QATNet ({config['n_bits']}b, H={config['hidden_dim']})")
    print(f"  Parameters:      {n_params:,}")
    print(f"  Accumulator:     {bitwidth} bits")
    print(f"  Compile time:    {compile_time:.1f}s")
    print(f"  PyTorch acc:     {acc_pt:.4f}")
    print(f"  Best Val acc:    {best_val_acc:.4f}")
    print(f"  FHE sim acc:     {acc_sim:.4f}")
    if acc_fhe is not None:
        print(f"  FHE exec acc:    {acc_fhe:.4f}")
    print()


if __name__ == "__main__":
    main()
