import time
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from tqdm.auto import tqdm
from torch.nn.utils import prune

import brevitas.nn as qnn
from brevitas.core.bit_width import BitWidthImplType
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import FloatToIntImplType, RestrictValueType
from brevitas.core.scaling import ScalingImplType
from brevitas.core.zero_point import ZeroZeroPoint
from brevitas.inject import ExtendedInjector
from brevitas.quant.solver import ActQuantSolver, WeightQuantSolver


from concrete.ml.torch.compile import compile_brevitas_qat_model


# =========================
# CONFIG
# =========================

IN_FEAT = 2
OUT_FEAT = 2

N_SIDE = 100
N_EXAMPLE_TOTAL = N_SIDE * N_SIDE
N_TEST = 500
CLUSTERS = 3


# =========================
# DATA GENERATION
# =========================

def generate_dataset():
    xx, yy = np.meshgrid(
        np.linspace(0, 1, N_SIDE),
        np.linspace(0, 1, N_SIDE)
    )

    X = np.c_[np.ravel(xx), np.ravel(yy)]

    y = (
        (np.rint(xx * CLUSTERS).astype(np.int64) % 2)
        ^ (np.rint(yy * CLUSTERS).astype(np.int64) % 2)
    ).ravel()

    X += np.random.randn(*X.shape) * 0.01

    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title("Original dataset")
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=N_TEST / N_EXAMPLE_TOTAL, random_state=42
    )

    return X_train, X_test, y_train, y_test


# =========================
# TRAINING LOOP
# =========================

def train(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    criterion,
    optimizer,
    epochs=10,
    batch_size=1,
    device="cpu",
):
    X_train = torch.tensor(X_train).float()
    X_test = torch.tensor(X_test).float()
    y_train = torch.tensor(y_train)

    loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )

    model.train()

    for epoch in range(epochs):
        losses = []
        preds, labels = [], []

        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            out = model(X_batch)

            preds.append(out.argmax(1).detach().cpu().numpy())
            labels.append(y_batch.cpu().numpy())

            loss = criterion(out, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        preds = np.concatenate(preds)
        labels = np.concatenate(labels)

        acc = np.mean(preds == labels)

        print(
            f"Epoch {epoch:02} | "
            f"Loss {np.mean(losses):.4f} | "
            f"Acc {acc * 100:.2f}%"
        )

    # test
    model.eval()
    test_out = model(torch.tensor(X_test).float().to(device))
    test_pred = test_out.argmax(1).cpu().numpy()

    test_acc = np.mean(test_pred == y_test)

    print(f"\nTest Accuracy (FP32): {test_acc * 100:.2f}%")
    return test_acc


# =========================
# FHE TESTING
# =========================

def test_in_fhe(quantized_model, X_test, y_test, simulate=True):
    mode = "simulate" if simulate else "execute"

    if not simulate:
        print("Generating FHE key...")
        start = time.time()
        quantized_model.fhe_circuit.keygen()
        print(f"Keygen: {time.time() - start:.2f}s")

    start = time.time()
    preds = quantized_model.forward(X_test, fhe=mode).argmax(1)
    end = time.time()

    print(f"Inference time: {end - start:.2f}s")

    acc = np.mean(preds == y_test) * 100
    print(f"FHE accuracy ({mode}): {acc:.2f}%")

    return preds


# =========================
# BREVITAS QUANT CONFIG
# =========================

class CommonQuant(ExtendedInjector):
    bit_width_impl_type = BitWidthImplType.CONST
    scaling_impl_type = ScalingImplType.CONST
    restrict_scaling_type = RestrictValueType.FP
    zero_point_impl = ZeroZeroPoint
    float_to_int_impl_type = FloatToIntImplType.ROUND
    scaling_per_output_channel = False
    narrow_range = True
    signed = True

    @staticmethod
    def quant_type(bit_width):
        if bit_width is None:
            return QuantType.FP
        if bit_width == 1:
            return QuantType.BINARY
        return QuantType.INT


class CommonWeightQuant(CommonQuant, WeightQuantSolver):
    scaling_const = 1.0


class CommonActQuant(CommonQuant, ActQuantSolver):
    min_val = -1.0
    max_val = 1.0


# =========================
# MODEL
# =========================

class QATPrunedSimpleNet(nn.Module):
    def __init__(self, n_hidden, qlinear_args, qidentity_args):
        super().__init__()

        self.pruned_layers = set()

        self.quant_inp = qnn.QuantIdentity(**qidentity_args)

        self.fc1 = qnn.QuantLinear(IN_FEAT, n_hidden, **qlinear_args)
        self.relu1 = qnn.QuantReLU(bit_width=qidentity_args["bit_width"])

        self.fc2 = qnn.QuantLinear(n_hidden, n_hidden, **qlinear_args)
        self.relu2 = qnn.QuantReLU(bit_width=qidentity_args["bit_width"])

        self.fc3 = qnn.QuantLinear(n_hidden, OUT_FEAT, **qlinear_args)

        for m in self.modules():
            if isinstance(m, qnn.QuantLinear):
                torch.nn.init.uniform_(m.weight.data, -1, 1)

    def forward(self, x):
        x = self.quant_inp(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

    def prune(self, max_non_zero):
        for name, layer in self.named_modules():
            if isinstance(layer, qnn.QuantLinear):
                total = layer.weight.shape[0] * layer.weight.shape[1]
                keep = layer.weight.shape[0] * max_non_zero
                amount = total - keep

                if amount > 0:
                    print(f"Pruning {name}: {amount} weights")
                    prune.l1_unstructured(layer, "weight", amount=amount)
                    self.pruned_layers.add(name)

    def unprune(self):
        for name, layer in self.named_modules():
            if name in self.pruned_layers:
                prune.remove(layer, "weight")
                self.pruned_layers.remove(name)


# =========================
# MAIN
# =========================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_train, X_test, y_train, y_test = generate_dataset()

    model = QATPrunedSimpleNet(
        n_hidden=100,
        qlinear_args={
            "weight_bit_width": 3,
            "weight_quant": CommonWeightQuant,
            "bias": True,
            "bias_quant": None,
            "narrow_range": True,
        },
        qidentity_args={
            "bit_width": 3,
            "act_quant": CommonActQuant,
        },
    )

    model.prune(20)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train(
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        criterion,
        optimizer,
        epochs=7,
        batch_size=1,
        device=device,
    )

    model.unprune()


if __name__ == "__main__":
    main()