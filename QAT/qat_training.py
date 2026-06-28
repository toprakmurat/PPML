"""
Concrete ML + Brevitas QAT Demo — MNIST Handwritten Digit Classification
=========================================================================
This script demonstrates the full Quantization-Aware Training (QAT) → FHE
compilation → encrypted inference pipeline using:
  • Brevitas  — PyTorch-based QAT library (QuantLinear, QuantReLU, QuantIdentity)
  • Concrete ML — Zama's privacy-preserving ML toolkit that compiles quantized
                  models into Fully Homomorphic Encryption (FHE) circuits
Dataset: MNIST (28×28 grayscale images → 10 digit classes)
The model is intentionally kept small (MLP, not CNN) so that FHE execution
completes in a reasonable time on commodity hardware.

Steps
--------
  1. Load & preprocess input dataset (MNIST) (flatten, normalize, subsample for FHE speed)
  2. Define a small MLP using Brevitas quantized layers
  3. Train with QAT (Brevitas fake-quantizes during forward/backward)
  4. Compile to an FHE circuit via Concrete ML
  5. Evaluate in three modes:
       a) PyTorch (float) — baseline accuracy
       b) FHE simulation  — quantized integer ops on cleartext
       c) FHE execution    — actual encrypted inference (slow but private)
"""

import time
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, classification_report
import brevitas.nn as qnn
from concrete.ml.torch.compile import compile_brevitas_qat_model


DEFAULT_CONFIG = {
    "n_bits": 3,              # weight & activation bit-width (2-4 typical for FHE)
    "hidden_dim": 92,         # hidden-layer width
    "epochs": 15,             # QAT training epochs
    "batch_size": 64,         # training batch size
    "lr": 1e-3,               # learning rate
    "train_subset": 10_000,   # use a subset to speed up the demo
    "test_subset": 500,       # smaller test set for FHE execution timing
    "fhe_samples": 5,         # number of samples for actual FHE inference
    "seed": 42,
    "data_dir": "./data",
}

# Data Loading & Preprocessing
def load_mnist(config: dict):
    transform = transforms.Compose([
        transforms.ToTensor(),                    # [0, 255] → [0.0, 1.0]
        transforms.Lambda(lambda x: x.view(-1)),  # 28×28 → 784
    ])
    train_ds = datasets.MNIST(config["data_dir"], train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST(config["data_dir"], train=False, download=True, transform=transform)
    # Subsample for speed
    rng = np.random.default_rng(config["seed"])
    train_idx = rng.choice(len(train_ds), size=config["train_subset"], replace=False)
    test_idx  = rng.choice(len(test_ds),  size=config["test_subset"],  replace=False)
    X_train = torch.stack([train_ds[i][0] for i in train_idx])
    y_train = torch.tensor([train_ds[i][1] for i in train_idx])
    X_test  = torch.stack([test_ds[i][0]  for i in test_idx])
    y_test  = torch.tensor([test_ds[i][1]  for i in test_idx])
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test

# MLP Model Definition using Brevitas
class QATNet(nn.Module):
    """
    A 3-layer quantized MLP for MNIST classification.
    Architecture:
        QuantIdentity (quantize input)
        → QuantLinear(784, H) → QuantReLU
        → QuantLinear(H, H)   → QuantReLU
        → QuantLinear(H, 10)
    Key design choices for FHE compatibility:
        • QuantIdentity at the entry point quantizes the raw float input
          into the integer domain. Without this layer, the first QuantLinear
          would receive unquantized floats and break the FHE circuit.
        • bias_quant=None lets Concrete ML handle bias quantization
          automatically during compilation.
        • return_quant_tensor=True keeps the output in Brevitas's
          QuantTensor format so downstream layers receive quantization
          metadata (scale, zero-point, bit-width).
    """
    def __init__(self, n_bits: int = 3, hidden_dim: int = 92):
        super().__init__()
        self.quant_input = qnn.QuantIdentity(
            bit_width=n_bits,
            return_quant_tensor=True,
        )
        self.fc1 = qnn.QuantLinear(
            784, hidden_dim, bias=True,
            weight_bit_width=n_bits,
            bias_quant=None,
        )
        self.relu1 = qnn.QuantReLU(
            bit_width=n_bits,
            return_quant_tensor=True,
        )
        self.fc2 = qnn.QuantLinear(
            hidden_dim, hidden_dim, bias=True,
            weight_bit_width=n_bits,
            bias_quant=None,
        )
        self.relu2 = qnn.QuantReLU(
            bit_width=n_bits,
            return_quant_tensor=True,
        )
        self.fc3 = qnn.QuantLinear(
            hidden_dim, 10, bias=True,
            weight_bit_width=n_bits,
            bias_quant=None,
        )
    def forward(self, x):
        x = self.quant_input(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

# Main Quantization-Aware Training Loop
def train_qat(model: nn.Module, X_train, y_train, config: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    dataset = TensorDataset(X_train, y_train)
    loader  = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    model.train()
    for epoch in range(1, config["epochs"] + 1):
        epoch_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += xb.size(0)
        avg_loss = epoch_loss / total
        acc = correct / total
        print(f"    Epoch {epoch:>2}/{config['epochs']}  "
              f"loss={avg_loss:.4f}  acc={acc:.4f}")
    model = model.cpu().eval()
    return model

# Evaluation Helpers
def evaluate_pytorch(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        preds = logits.argmax(dim=1).numpy()
    acc = accuracy_score(y_test.numpy(), preds)
    return preds, acc

def evaluate_fhe_simulation(quantized_module, X_test, y_test):
    X_np = X_test.numpy()
    y_np = y_test.numpy()
    # forward() returns class logits; argmax to get predictions
    logits_sim = quantized_module.forward(X_np, fhe="simulate")
    preds_sim = np.argmax(logits_sim, axis=1)
    acc = accuracy_score(y_np, preds_sim)
    return preds_sim, acc

def evaluate_fhe_execute(quantized_module, X_test, y_test, n_samples: int = 5):
    X_np = X_test.numpy()[:n_samples]
    y_np = y_test.numpy()[:n_samples]
    print(f"\n  Running FHE inference on {n_samples} encrypted samples...")
    preds = []
    for i in range(n_samples):
        t0 = time.perf_counter()
        logits = quantized_module.forward(X_np[i : i + 1], fhe="execute")
        dt = time.perf_counter() - t0
        pred = int(np.argmax(logits, axis=1)[0])
        preds.append(pred)
        label = int(y_np[i])
        status = "✓" if pred == label else "✗"
        print(f"    Sample {i+1}: true={label}  pred={pred}  {status}  ({dt:.1f}s)")
    acc = accuracy_score(y_np, preds)
    return np.array(preds), acc

# Driver Code
def main():
    parser = argparse.ArgumentParser(
        description="Concrete ML + Brevitas QAT demo on MNIST"
    )
    parser.add_argument("--n_bits", type=int, default=DEFAULT_CONFIG["n_bits"],
                        help="Bit-width for weights & activations (default: 3)")
    parser.add_argument("--hidden_dim", type=int, default=DEFAULT_CONFIG["hidden_dim"],
                        help="Hidden layer width (default: 92)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"],
                        help="Training epochs (default: 15)")
    parser.add_argument("--train_subset", type=int, default=DEFAULT_CONFIG["train_subset"],
                        help="Number of training samples (default: 10000)")
    parser.add_argument("--test_subset", type=int, default=DEFAULT_CONFIG["test_subset"],
                        help="Number of test samples (default: 500)")
    parser.add_argument("--fhe_samples", type=int, default=DEFAULT_CONFIG["fhe_samples"],
                        help="Samples for actual FHE execution (default: 5)")
    parser.add_argument("--skip_fhe_execute", action="store_true",
                        help="Skip actual FHE execution (slow)")
    args = parser.parse_args()
    config = {**DEFAULT_CONFIG, **vars(args)}
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    print("\nStep 1 - Loading MNIST dataset")
    X_train, y_train, X_test, y_test = load_mnist(config)
    
    print("\nStep 2 - Building Brevitas QAT Model║")
    model = QATNet(n_bits=config["n_bits"], hidden_dim=config["hidden_dim"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture: 784 → {config['hidden_dim']} → {config['hidden_dim']} → 10")
    print(f"  Bit-width:    {config['n_bits']}b weights / {config['n_bits']}b activations")
    print(f"  Parameters:   {n_params:,}")
    # print(f"\n  Model structure:\n{model}")
    
    print("\nStep 3 - Quantization-Aware Training")
    model = train_qat(model, X_train, y_train, config)
    
    print("\nStep 4 - Compiling to FHE circuit") 
    # Calibration set is a small subset of training data
    # Concrete ML uses this to determine quantization ranges for the circuit
    calibration_data = X_train[:100].numpy()
    print("  Compiling with compile_brevitas_qat_model()...")
    t0 = time.perf_counter()
    quantized_module = compile_brevitas_qat_model(
        torch_model=model,
        torch_inputset=calibration_data,
        rounding_threshold_bits={"n_bits": 6, "method": "approximate"},
    )
    compile_time = time.perf_counter() - t0
    print(f"Compilation complete in {compile_time:.1f}s")
    
    # Accumulator bit-width is the key FHE cost metric
    # A higher value means more expensive (slower) bootstrapping operations
    bitwidth = quantized_module.fhe_circuit.graph.maximum_integer_bit_width()
    print(f"  Accumulator bit-width: {bitwidth} bits")
    
    print("Step 5a - PyTorch Baseline (float)")
    _, acc_pt = evaluate_pytorch(model, X_test, y_test)
    print(f"  PyTorch accuracy:  {acc_pt:.4f}  ({int(acc_pt * len(y_test))}/{len(y_test)})")
    
    print("Step 5b - FHE Simulation (cleartext integers)")
    t0 = time.perf_counter()
    _, acc_sim = evaluate_fhe_simulation(quantized_module, X_test, y_test)
    sim_time = time.perf_counter() - t0
    print(f"  Simulated accuracy: {acc_sim:.4f}  ({int(acc_sim * len(y_test))}/{len(y_test)})")
    print(f"  Simulation time:    {sim_time:.2f}s for {len(y_test)} samples")
    
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
    # ── Summary ───────────────────────────────────────────────────────────
    print("\n╔══════════════════════════════════════════════════╗")
    print("║  Summary                                         ║")
    print("╚══════════════════════════════════════════════════╝")
    print(f"  Model:           QATNet ({config['n_bits']}b, H={config['hidden_dim']})")
    print(f"  Parameters:      {n_params:,}")
    print(f"  Accumulator:     {bitwidth} bits")
    print(f"  Compile time:    {compile_time:.1f}s")
    print(f"  PyTorch acc:     {acc_pt:.4f}")
    print(f"  FHE sim acc:     {acc_sim:.4f}")
    if not config["skip_fhe_execute"]:
        print(f"  FHE exec acc:    {acc_fhe:.4f}")
    print()
if __name__ == "__main__":
    main()

