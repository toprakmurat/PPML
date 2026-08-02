"""
Concrete ML + Brevitas QAT Model Core
=====================================
Defines QATNet architecture, MNIST data loader, training loop, and evaluation tools.
"""

import time
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
import brevitas.nn as qnn

# Calculate default paths relative to this file
_BASE_DIR = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = _BASE_DIR / "experiments" / "data"
_DEFAULT_CHECKPOINT = _BASE_DIR / "experiments" / "checkpoints" / "best_qat_model.pth"

DEFAULT_CONFIG = {
    "n_bits": 3,              # weight & activation bit-width
    "hidden_dim": 92,         # hidden-layer width
    "epochs": 35,             # QAT training epochs
    "batch_size": 64,         # training batch size
    "lr": 1e-3,               # learning rate
    "weight_decay": 1e-4,     # L2 penalty
    "val_ratio": 0.10,        # 90/10 train/validation split
    "train_subset": 10_000,   # training subset
    "test_subset": 500,       # test subset
    "fhe_samples": 5,         # FHE inference sample count
    "seed": 42,
    "data_dir": str(_DEFAULT_DATA_DIR),
    "checkpoint_path": str(_DEFAULT_CHECKPOINT),
}


def load_mnist(config: dict):
    transform = transforms.Compose([
        transforms.ToTensor(),                    # [0, 255] → [0.0, 1.0]
        transforms.Lambda(lambda x: x.view(-1)),  # 28×28 → 784
    ])
    data_dir = config.get("data_dir", str(_DEFAULT_DATA_DIR))
    os.makedirs(data_dir, exist_ok=True)
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    rng = np.random.default_rng(config.get("seed", 42))
    train_idx = rng.choice(len(train_ds), size=config.get("train_subset", 10000), replace=False)
    test_idx = rng.choice(len(test_ds), size=config.get("test_subset", 500), replace=False)
    X_train_full = torch.stack([train_ds[i][0] for i in train_idx])
    y_train_full = torch.tensor([train_ds[i][1] for i in train_idx])
    X_test = torch.stack([test_ds[i][0] for i in test_idx])
    y_test = torch.tensor([test_ds[i][1] for i in test_idx])

    val_size = int(len(X_train_full) * config.get("val_ratio", 0.10))
    train_size = len(X_train_full) - val_size
    X_train, X_val = X_train_full[:train_size], X_train_full[train_size:]
    y_train, y_val = y_train_full[:train_size], y_train_full[train_size:]

    print(f"  Train: {X_train.shape}  |  Val: {X_val.shape}  |  Test: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test


class QATNet(nn.Module):
    """
    A 3-layer quantized MLP for MNIST classification using Brevitas.
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


def train_qat(model: nn.Module, X_train, y_train, X_val, y_val, config: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=config["batch_size"], shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=1e-5
    )

    best_val_acc = -1.0
    checkpoint_path = Path(config.get("checkpoint_path", str(_DEFAULT_CHECKPOINT)))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += xb.size(0)

        train_loss = epoch_loss / total
        train_acc = correct / total

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss_sum += loss.item() * xb.size(0)
                val_correct += (logits.argmax(dim=1) == yb).sum().item()
                val_total += xb.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            saved_str = " -> Best Saved"
        else:
            saved_str = ""

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"    Epoch {epoch:>2}/{config['epochs']}  "
              f"loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"lr={current_lr:.1e}{saved_str}")

    if checkpoint_path.exists():
        print(f"\n  Loading best model checkpoint from {checkpoint_path} (val_acc={best_val_acc:.4f})...")
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

    model = model.cpu().eval()
    return model, best_val_acc


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
