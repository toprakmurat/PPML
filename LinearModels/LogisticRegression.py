import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from concrete.ml.sklearn import LogisticRegression as ConcreteLogisticRegression
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=2,
    n_clusters_per_class=1,
)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)

# Bounding box for the decision boundary grid
b_min = np.min(X, axis=0)
b_max = np.max(X, axis=0)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Dense 2-D grid for plotting decision boundaries — not used for accuracy scoring
x_test_grid, y_test_grid = np.meshgrid(
    np.linspace(b_min[0], b_max[0], 30),
    np.linspace(b_min[1], b_max[1], 30),
)
x_grid = np.vstack([x_test_grid.ravel(), y_test_grid.ravel()]).T

# ---------------------------------------------------------------------------
# Scikit-learn model
# ---------------------------------------------------------------------------
sklearn_logr = SklearnLogisticRegression()
sklearn_logr.fit(x_train, y_train)

y_pred_test = sklearn_logr.predict(x_test)
sklearn_accuracy = accuracy_score(y_test, y_pred_test)

# Probabilities over the grid (class 1) for contour plot
y_proba_sklearn_grid = sklearn_logr.predict_proba(x_grid)[:, 1]

# ---------------------------------------------------------------------------
# Concrete ML model — trained on clear data, quantized to 8 bits
# ---------------------------------------------------------------------------
concrete_logr = ConcreteLogisticRegression(n_bits=8)
concrete_logr.fit(x_train, y_train)

# Quantized clear predictions on the test set
y_proba_q = concrete_logr.predict_proba(x_test)[:, 1]
y_pred_q = concrete_logr.predict(x_test)
quantized_accuracy = accuracy_score(y_test, y_pred_q)

# Grid predictions for decision boundary contour
y_proba_q_grid = concrete_logr.predict_proba(x_grid)[:, 1]
y_pred_q_grid = concrete_logr.predict(x_grid)

# ---------------------------------------------------------------------------
# Plot 1: Decision boundaries — sklearn (float) vs Concrete ML (quantized clear)
# Points misclassified by the quantized model are circled in red
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, proba_grid, title in zip(
    axes,
    [y_proba_sklearn_grid, y_proba_q_grid],
    [
        f"Scikit-Learn (clear float)\nAccuracy={sklearn_accuracy:.4f}",
        f"Concrete ML (quantized clear)\nAccuracy={quantized_accuracy:.4f}",
    ],
):
    ax.contourf(
        x_test_grid, y_test_grid,
        proba_grid.reshape(x_test_grid.shape),
        alpha=0.4, levels=20, cmap="RdBu",
    )
    ax.scatter(
        x_test[y_test == 0, 0], x_test[y_test == 0, 1],
        c="blue", marker="o", s=20, label="Class 0",
    )
    ax.scatter(
        x_test[y_test == 1, 0], x_test[y_test == 1, 1],
        c="red", marker="x", s=20, label="Class 1",
    )
    ax.set_title(title)
    ax.legend(loc="upper right")

# Circle points where quantized prediction differs from sklearn
misclassified = y_pred_q != y_pred_test
axes[1].scatter(
    x_test[misclassified, 0], x_test[misclassified, 1],
    facecolors="none", edgecolors="red", linewidths=1.5, s=80,
    label="Differs from sklearn",
)
axes[1].legend(loc="upper right")

plt.suptitle("Logistic Regression: decision boundaries")
plt.tight_layout()
plt.savefig("plot_logr_clear.png", dpi=150)
plt.show()

print(f"Scikit-Learn accuracy      : {sklearn_accuracy:.4f}")
print(f"Quantized clear accuracy   : {quantized_accuracy:.4f}")

# ---------------------------------------------------------------------------
# FHE compilation
# compile() traces the quantized model and determines cryptographic parameters.
# x_train is the calibration set used to fix intermediate bit-widths.
# ---------------------------------------------------------------------------
print("\nCompiling model for FHE…")
fhe_circuit = concrete_logr.compile(x_train)
print(f"Generating keys for a {fhe_circuit.graph.maximum_integer_bit_width()}-bit circuit…")
fhe_circuit.keygen(force=True)

# ---------------------------------------------------------------------------
# FHE simulation — fast, bit-exact dry-run of the compiled circuit (no encryption)
# Useful to confirm quantization impact before paying the cost of real FHE
# ---------------------------------------------------------------------------
y_pred_simulated = concrete_logr.predict(x_test, fhe="simulate")
simulated_accuracy = accuracy_score(y_test, y_pred_simulated)
print(f"Simulated (circuit, clear) accuracy: {simulated_accuracy:.4f}")

# ---------------------------------------------------------------------------
# FHE inference — real encrypted execution on a small subset (slow)
# ---------------------------------------------------------------------------
N_TEST_FHE = 10  # increase for fuller evaluation; each sample takes ~1ms for linear models
x_test_fhe = x_test[:N_TEST_FHE]
y_test_fhe = y_test[:N_TEST_FHE]

print(f"\nRunning FHE inference on {N_TEST_FHE} samples…")
start = time.time()
y_pred_fhe = concrete_logr.predict(x_test_fhe, fhe="execute")
elapsed = time.time() - start
print(f"FHE inference time: {elapsed:.2f}s ({elapsed / N_TEST_FHE:.2f}s per sample)")

fhe_accuracy = accuracy_score(y_test_fhe, y_pred_fhe)

# ---------------------------------------------------------------------------
# Plot 2: Prediction comparison on the FHE subset
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4))
x_idx = np.arange(N_TEST_FHE)
ax.scatter(x_idx, y_test_fhe, c="black", marker="D", s=40, zorder=3, label="Ground truth")
ax.scatter(x_idx, y_pred_q[:N_TEST_FHE], c="blue", marker="o", s=40, zorder=2,
           label=f"Quantized clear (acc={accuracy_score(y_test_fhe, y_pred_q[:N_TEST_FHE]):.2f})")
ax.scatter(x_idx, y_pred_simulated[:N_TEST_FHE], c="green", marker="^", s=40, zorder=2,
           label=f"Simulated (acc={accuracy_score(y_test_fhe, y_pred_simulated[:N_TEST_FHE]):.2f})")
ax.scatter(x_idx, y_pred_fhe, c="orange", marker="x", s=60, zorder=4,
           label=f"FHE execute (acc={fhe_accuracy:.2f})")
ax.set_xlabel("Sample index")
ax.set_ylabel("Predicted class")
ax.set_yticks([0, 1])
ax.set_title(f"Prediction comparison on {N_TEST_FHE} samples")
ax.legend()
plt.tight_layout()
plt.savefig("plot_logr_fhe.png", dpi=150)
plt.show()

# ---------------------------------------------------------------------------
# Score comparison summary
# ---------------------------------------------------------------------------
concrete_score_difference = abs(quantized_accuracy - fhe_accuracy) / quantized_accuracy * 100
sklearn_fhe_difference = abs(sklearn_accuracy - fhe_accuracy) / sklearn_accuracy * 100

print("\n=== Score comparison ===")
print(f"Scikit-Learn (clear float)        : {sklearn_accuracy:.4f}")
print(f"Concrete ML (quantized clear)     : {quantized_accuracy:.4f}")
print(f"Concrete ML (simulated)           : {simulated_accuracy:.4f}")
print(f"Concrete ML (FHE, N={N_TEST_FHE})        : {fhe_accuracy:.4f}")
print(f"\nRelative diff quantized clear vs FHE : {concrete_score_difference:.2f}%")
print(f"Relative diff sklearn vs FHE         : {sklearn_fhe_difference:.2f}%")
