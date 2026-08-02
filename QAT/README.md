# Quantization-Aware Training (QAT) & FHE Optimization Framework

A production-ready framework for PyTorch and Brevitas models compiled with Zama Concrete ML for Fully Homomorphic Encryption (FHE) inference.

---

## Repository Architecture

The repository is organized by functional domain and end-result artifacts:

```
QAT/
├── qat/                          # Core Python library package
│   ├── models/                   # QATNet PyTorch architecture & training
│   │   ├── __init__.py
│   │   └── qat_net.py
│   ├── cost_model/               # FHE operational cost estimation & harness
│   │   ├── __init__.py
│   │   ├── estimator.py
│   │   └── harness.py
│   ├── sensitivity/              # Hessian curvature & OBD neuron importance
│   │   ├── __init__.py
│   │   └── hessian.py
│   └── allocation/               # Joint knapsack allocator & optimization
│       ├── __init__.py
│       └── allocator.py
│
├── scripts/                      # CLI execution entrypoints
│   ├── run_cost_model.py         # Benchmark isolated layer FHE compilation
│   ├── run_hessian_analysis.py   # Run Hutchinson trace & OBD neuron scoring
│   ├── run_joint_allocation.py   # Solve joint knapsack optimization
│   └── run_qat_training.py       # Full QAT training & encrypted inference
│
├── visualization/                # Publication-quality plot generators
│   ├── plot_cost_model.py
│   ├── plot_hessian.py
│   └── plot_joint_allocation.py
│
└── experiments/                  # Generated experiment outputs & data
    ├── data/                     # Cached MNIST datasets
    ├── checkpoints/              # Model weights (`best_qat_model.pth`)
    ├── results/                  # Execution JSON metrics
    ├── reports/                  # Baseline evaluation summaries
    └── plots/                    # Rendered diagnostic charts (.png)
```

---

## Quick Start & Usage

### 1. Python Package Import
```python
from qat.models.qat_net import QATNet, train_qat
from qat.cost_model.estimator import estimated_cost
from qat.sensitivity.hessian import HessianSensitivityEstimator
from qat.allocation.allocator import JointKnapsackAllocator
```

### 2. Run CLI Scripts
```bash
# Run FHE cost model benchmarks and plot exponential cost scaling
python scripts/run_cost_model.py

# Run Hessian sensitivity analysis & OBD neuron importance scoring
python scripts/run_hessian_analysis.py

# Run Joint Knapsack Allocation optimization across budget targets
python scripts/run_joint_allocation.py

# Run QAT model training and Concrete ML encrypted execution
python scripts/run_qat_training.py
```
