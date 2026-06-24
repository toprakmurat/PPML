import time
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.svm import LinearSVR as SklearnLinearSVR
from concrete.ml.sklearn.svm import LinearSVR as ConcreteLinearSVR
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Data loading & preparation
# ---------------------------------------------------------------------------
X, y = load_diabetes(return_X_y=True)

# Use only one feature so we can plot a 1-D regression curve (same as notebook)
X = X[:, np.newaxis, 2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=23)

# Sort test set so the regression line plots cleanly
sorted_indexes = np.argsort(np.squeeze(X_test))
X_test = X_test[sorted_indexes, :]
y_test = y_test[sorted_indexes]

# ---------------------------------------------------------------------------
# Grid search — scorer uses negative MSE so GridSearchCV maximises it
# ---------------------------------------------------------------------------
grid_scorer = make_scorer(mean_squared_error, greater_is_better=False)
kfold_cv = KFold(n_splits=5, shuffle=True, random_state=13)

param_grid_sklearn = {
    "epsilon": [0.0, 1.0, 10.0, 20.0],
    "C": [0.1, 100.0, 10000.0, 100000.0],
}

print("Grid search — Scikit-Learn LinearSVR…")
sklearn_rgs = SklearnLinearSVR()
gs_sklearn = GridSearchCV(
    sklearn_rgs,
    param_grid_sklearn,
    cv=kfold_cv,
    scoring=grid_scorer,
    verbose=1,
).fit(X_train, y_train)

print(f"\nBest sklearn params : {gs_sklearn.best_params_}")
print(f"Best sklearn CV MSE : {-gs_sklearn.best_score_:.4f}")

param_grid_concrete = {
    "n_bits": [6, 8, 12],
    "epsilon": [0.0, 1.0, 10.0, 20.0],
    "C": [0.1, 100.0, 10000.0, 100000.0],
}

print("\nGrid search — Concrete ML LinearSVR…")
concrete_rgs = ConcreteLinearSVR()
gs_concrete = GridSearchCV(
    concrete_rgs,
    param_grid_concrete,
    cv=kfold_cv,
    scoring=grid_scorer,
    verbose=1,
).fit(X_train, y_train)

print(f"\nBest Concrete ML params : {gs_concrete.best_params_}")
print(f"Best Concrete ML CV MSE : {-gs_concrete.best_score_:.4f}")

# ---------------------------------------------------------------------------
# Best estimators & test-set predictions
# ---------------------------------------------------------------------------
best_sklearn = gs_sklearn.best_estimator_
best_concrete = gs_concrete.best_estimator_

y_pred_sklearn = best_sklearn.predict(X_test)
sklearn_mse = mean_squared_error(y_test, y_pred_sklearn)
sklearn_r2 = r2_score(y_test, y_pred_sklearn)

# Quantized clear predictions
y_pred_q = best_concrete.predict(X_test)
quantized_mse = mean_squared_error(y_test, y_pred_q)
quantized_r2 = r2_score(y_test, y_pred_q)

# Dense x-space for a smooth regression curve
x_space = np.linspace(X_test.min(), X_test.max(), num=300)[:, np.newaxis]
y_pred_sklearn_space = best_sklearn.predict(x_space)
y_pred_q_space = best_concrete.predict(x_space)

# ---------------------------------------------------------------------------
# Grid-search results table — show top-5 configs for each model
# ---------------------------------------------------------------------------
sklearn_results = pd.DataFrame(gs_sklearn.cv_results_)
concrete_results = pd.DataFrame(gs_concrete.cv_results_)

print("\n--- Top-5 sklearn configs (by mean CV MSE) ---")
print(
    sklearn_results[["param_C", "param_epsilon", "mean_test_score"]]
    .sort_values("mean_test_score", ascending=False)
    .head(5)
    .to_string(index=False)
)

print("\n--- Top-5 Concrete ML configs (by mean CV MSE) ---")
print(
    concrete_results[["param_n_bits", "param_C", "param_epsilon", "mean_test_score"]]
    .sort_values("mean_test_score", ascending=False)
    .head(5)
    .to_string(index=False)
)

# ---------------------------------------------------------------------------
# Plot 1: Regression curves on the test set
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X_train, y_train, c="black", marker="D", s=15, label="Train data")
ax.scatter(X_test, y_test, c="red", marker="x", s=15, label="Test data")
ax.plot(
    x_space, y_pred_sklearn_space,
    c="blue", linewidth=2.5,
    label=f"Scikit-Learn (MSE={sklearn_mse:.1f}, R²={sklearn_r2:.4f})",
)
ax.plot(
    x_space, y_pred_q_space,
    c="orange", linewidth=2.5, linestyle="--",
    label=f"Concrete ML quantized clear (MSE={quantized_mse:.1f}, R²={quantized_r2:.4f})",
)
ax.set_xlabel("Feature")
ax.set_ylabel("Target")
ax.set_title("LinearSVR: Scikit-Learn vs Concrete ML (quantized clear)")
ax.legend()
plt.tight_layout()
plt.savefig("plot_svr_clear.png", dpi=150)
plt.show()

print(f"\nScikit-Learn  — MSE: {sklearn_mse:.4f} | R²: {sklearn_r2:.4f}")
print(f"Quantized clear — MSE: {quantized_mse:.4f} | R²: {quantized_r2:.4f}")

# ---------------------------------------------------------------------------
# FHE compilation — must be called on the best estimator, not the base model
# ---------------------------------------------------------------------------
print("\nCompiling best Concrete ML model for FHE…")
fhe_circuit = best_concrete.compile(X_train)
print(f"Generating keys for a {fhe_circuit.graph.maximum_integer_bit_width()}-bit circuit…")
fhe_circuit.keygen(force=True)

# ---------------------------------------------------------------------------
# FHE simulation — fast, bit-exact, runs on the full test set
# ---------------------------------------------------------------------------
y_pred_simulated = best_concrete.predict(X_test, fhe="simulate")
simulated_mse = mean_squared_error(y_test, y_pred_simulated)
simulated_r2 = r2_score(y_test, y_pred_simulated)
print(f"Simulated — MSE: {simulated_mse:.4f} | R²: {simulated_r2:.4f}")

# ---------------------------------------------------------------------------
# FHE inference — real encrypted execution on a small subset (slow)
# ---------------------------------------------------------------------------
N_TEST_FHE = 10  # increase as needed; linear SVR is ~1ms per sample
X_test_fhe = X_test[:N_TEST_FHE]
y_test_fhe = y_test[:N_TEST_FHE]

print(f"\nRunning FHE inference on {N_TEST_FHE} samples…")
start = time.time()
y_pred_fhe = best_concrete.predict(X_test_fhe, fhe="execute")
elapsed = time.time() - start
print(f"FHE inference time: {elapsed:.2f}s ({elapsed / N_TEST_FHE:.2f}s per sample)")

fhe_mse = mean_squared_error(y_test_fhe, y_pred_fhe)
fhe_r2 = r2_score(y_test_fhe, y_pred_fhe)

# ---------------------------------------------------------------------------
# Plot 2: Actual vs predicted on the FHE subset
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
x_idx = np.arange(N_TEST_FHE)
ax.scatter(x_idx, y_test_fhe, c="black", marker="D", s=40, zorder=3, label="Ground truth")
ax.plot(
    x_idx, y_pred_sklearn[:N_TEST_FHE],
    c="blue", linewidth=2, marker="o", markersize=5,
    label=f"Sklearn (MSE={mean_squared_error(y_test_fhe, y_pred_sklearn[:N_TEST_FHE]):.1f})",
)
ax.plot(
    x_idx, y_pred_q[:N_TEST_FHE],
    c="orange", linewidth=2, marker="s", markersize=5, linestyle="--",
    label=f"Quantized clear (MSE={mean_squared_error(y_test_fhe, y_pred_q[:N_TEST_FHE]):.1f})",
)
ax.plot(
    x_idx, y_pred_simulated[:N_TEST_FHE],
    c="green", linewidth=1.5, marker="^", markersize=5, linestyle=":",
    label=f"Simulated (MSE={mean_squared_error(y_test_fhe, y_pred_simulated[:N_TEST_FHE]):.1f})",
)
ax.plot(
    x_idx, y_pred_fhe,
    c="red", linewidth=2, marker="x", markersize=8,
    label=f"FHE execute (MSE={fhe_mse:.1f})",
)
ax.set_xlabel("Sample index")
ax.set_ylabel("Predicted value")
ax.set_title(f"Prediction comparison on {N_TEST_FHE} samples")
ax.legend()
plt.tight_layout()
plt.savefig("plot_svr_fhe.png", dpi=150)
plt.show()

# ---------------------------------------------------------------------------
# Score comparison summary
# ---------------------------------------------------------------------------
concrete_mse_diff = abs(quantized_mse - fhe_mse) / quantized_mse * 100
sklearn_fhe_diff = abs(sklearn_mse - fhe_mse) / sklearn_mse * 100

print("\n=== Score comparison ===")
print(f"Scikit-Learn (clear float)        : MSE={sklearn_mse:.4f} | R²={sklearn_r2:.4f}")
print(f"Concrete ML (quantized clear)     : MSE={quantized_mse:.4f} | R²={quantized_r2:.4f}")
print(f"Concrete ML (simulated)           : MSE={simulated_mse:.4f} | R²={simulated_r2:.4f}")
print(f"Concrete ML (FHE, N={N_TEST_FHE})        : MSE={fhe_mse:.4f} | R²={fhe_r2:.4f}")
print(f"\nRelative MSE diff quantized clear vs FHE : {concrete_mse_diff:.2f}%")
print(f"Relative MSE diff sklearn vs FHE         : {sklearn_fhe_diff:.2f}%")
