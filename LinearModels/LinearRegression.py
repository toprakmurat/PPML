import time
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from concrete.ml.sklearn import LinearRegression as ConcreteLinearRegression
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Plot style helpers
# ---------------------------------------------------------------------------
train_plot_config = {"c": "black", "marker": "D", "s": 15, "label": "Train data"}
test_plot_config = {"c": "red", "marker": "x", "s": 15, "label": "Test data"}

def get_sklearn_plot_config(r2=None):
    label = "Scikit-Learn"
    if r2 is not None:
        label += f", R²={r2:.4f}"
    return {"c": "blue", "linewidth": 2.5, "label": label}

def get_concrete_plot_config(r2=None):
    label = "Concrete ML"
    if r2 is not None:
        label += f", R²={r2:.4f}"
    return {"c": "orange", "linewidth": 2.5, "label": label}

# ---------------------------------------------------------------------------
# Data generation & splitting
# ---------------------------------------------------------------------------
X, y = make_regression(
    n_samples=200, n_features=1, n_targets=1, bias=5.0, noise=30.0, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Sort test set by X so regression line plots cleanly
sorted_indexes = np.argsort(np.squeeze(X_test))
X_test = X_test[sorted_indexes, :]
y_test = y_test[sorted_indexes]

# ---------------------------------------------------------------------------
# Scikit-learn model
# ---------------------------------------------------------------------------
sklearn_lr = SklearnLinearRegression()
sklearn_lr.fit(X_train, y_train)
y_pred = sklearn_lr.predict(X_test)
sklearn_r2_score = r2_score(y_test, y_pred)

# ---------------------------------------------------------------------------
# Concrete ML model — trained on clear data, quantized to 8 bits
# ---------------------------------------------------------------------------
concrete_lr = ConcreteLinearRegression(n_bits=8)
concrete_lr.fit(X_train, y_train)

# Dense x-space for a smooth regression curve
x_space = np.linspace(X_test.min(), X_test.max(), num=300)
x_space = x_space[:, np.newaxis]
y_pred_q_space = concrete_lr.predict(x_space)

# Quantized clear predictions on the test set
y_pred_q = concrete_lr.predict(X_test)
quantized_r2_score = r2_score(y_test, y_pred_q)

# ---------------------------------------------------------------------------
# Plot 1: Data + both regression lines (clear / quantized clear)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X_train, y_train, **train_plot_config)
ax.scatter(X_test, y_test, **test_plot_config)
ax.plot(X_test, y_pred, **get_sklearn_plot_config(sklearn_r2_score))
ax.plot(x_space, y_pred_q_space, **get_concrete_plot_config(quantized_r2_score))
ax.set_xlabel("X")
ax.set_ylabel("y")
ax.set_title("Linear Regression: Scikit-Learn vs Concrete ML (quantized, clear)")
ax.legend()
plt.tight_layout()
plt.savefig("plot_clear.png", dpi=150)
plt.show()

print(f"Scikit-Learn R² score  : {sklearn_r2_score:.4f}")
print(f"Quantized clear R² score: {quantized_r2_score:.4f}")

# ---------------------------------------------------------------------------
# FHE compilation
# The compile() call traces the quantized model to build an FHE circuit.
# X_train is used as a representative calibration set.
# ---------------------------------------------------------------------------
print("\nCompiling model for FHE…")
fhe_circuit = concrete_lr.compile(X_train)
print(f"Generating keys for a {fhe_circuit.graph.maximum_integer_bit_width()}-bit circuit…")
fhe_circuit.keygen(force=True)

# ---------------------------------------------------------------------------
# FHE inference — run a small subset because FHE is much slower than clear
# ---------------------------------------------------------------------------
N_TEST_FHE = 10  # keep small for demo; increase for full evaluation
X_test_fhe = X_test[:N_TEST_FHE]
y_test_fhe = y_test[:N_TEST_FHE]

print(f"\nRunning FHE inference on {N_TEST_FHE} samples…")
start = time.time()
y_pred_fhe = concrete_lr.predict(X_test_fhe, fhe="execute")
elapsed = time.time() - start
print(f"FHE inference time: {elapsed:.2f}s ({elapsed / N_TEST_FHE:.2f}s per sample)")

fhe_r2_score = r2_score(y_test_fhe, y_pred_fhe)

# Also get the quantized-clear predictions on the same small subset for fair comparison
y_pred_q_fhe_subset = concrete_lr.predict(X_test_fhe)

# ---------------------------------------------------------------------------
# Plot 2: Actual vs predicted for the FHE subset
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(
    range(N_TEST_FHE), y_test_fhe,
    c="black", marker="D", s=30, label="Ground truth"
)
ax.plot(
    range(N_TEST_FHE), y_pred_q_fhe_subset,
    c="blue", linewidth=2, marker="o", markersize=5,
    label=f"Quantized clear (R²={r2_score(y_test_fhe, y_pred_q_fhe_subset):.4f})"
)
ax.plot(
    range(N_TEST_FHE), y_pred_fhe,
    c="orange", linewidth=2, marker="x", markersize=8,
    label=f"FHE (R²={fhe_r2_score:.4f})"
)
ax.set_xlabel("Sample index")
ax.set_ylabel("Predicted value")
ax.set_title(f"FHE vs quantized-clear predictions (N={N_TEST_FHE})")
ax.legend()
plt.tight_layout()
plt.savefig("plot_fhe.png", dpi=150)
plt.show()

# ---------------------------------------------------------------------------
# Score comparison summary
# ---------------------------------------------------------------------------
score_diff_quantized = abs(sklearn_r2_score - quantized_r2_score) / abs(sklearn_r2_score) * 100
score_diff_fhe = abs(quantized_r2_score - fhe_r2_score) / abs(quantized_r2_score) * 100 if quantized_r2_score != 0 else float("nan")

print("\n=== Score comparison ===")
print(f"Scikit-Learn (clear float)   : {sklearn_r2_score:.6f}")
print(f"Concrete ML (quantized clear): {quantized_r2_score:.6f}")
print(f"Concrete ML (FHE, N={N_TEST_FHE})   : {fhe_r2_score:.6f}")
print(f"\nRelative diff sklearn vs quantized clear : {score_diff_quantized:.2f}%")
print(f"Relative diff quantized clear vs FHE     : {score_diff_fhe:.2f}%")
