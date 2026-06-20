# Logistic Regression with ±1 Labels — Gradient Descent Debugging

A from-scratch implementation of logistic regression using gradient descent with ±1 label formulation. Built for Stanford CS229 Problem Set 2 — all NumPy, no ML libraries.

## What I Learned

### The ±1 Label Formulation

Standard logistic regression uses labels `y ∈ {0, 1}`. The CS229 formulation uses `y ∈ {−1, +1}`, which simplifies the math:

```
Loss:     L(θ) = (1/m) Σ log(1 + exp(-y_i · θ·x_i))
Gradient: ∇L  = -(1/m) · Xᵀ · (σ(-Y·Xθ) · Y)
```

Both formulations are equivalent, but ±1 gives cleaner gradients and a more symmetric loss function.

### The `.copy()` Bug

The hardest bug: `prev_theta = theta` doesn't create a copy — both variables point to the same NumPy array. When you update `theta`, `prev_theta` changes too. The convergence check `np.linalg.norm(prev_theta - theta) < tol` evaluates to `0 < tol` every time, so the loop exits after ONE iteration.

**Fix**: `prev_theta = theta.copy()`

This is a classic Python/NumPy gotcha. Mutable objects are passed by reference. Assignment doesn't copy.

### Learning Rate Matters

- `lr = 10`: Oscillated wildly, took 30,000+ iterations on separable data
- `lr = 1`: Smooth convergence in ~6,500 iterations on the same data

With gradient descent (not Newton-Raphson), the learning rate determines everything. Too high = oscillation. Too low = slow. There's no Hessian to automatically scale the step.

### Separable vs Non-Separable Data

- **Dataset A**: Linearly separable → converges to perfect separation in ~6,500 iterations
- **Dataset B**: Not linearly separable → converges to best linear boundary in ~17,000 iterations, but never reaches zero loss

The loss plateaus at a non-zero value on B. The gradient gets small but never zero because no linear boundary can perfectly separate the classes. This is correct behavior.

### Convergence Tolerance

Using `1e-15` for gradient descent is unrealistic with floating-point arithmetic. A tolerance of `1e-3` to `1e-6` is more practical and catches actual convergence rather than numerical noise.

## Project Structure

```
├── fixedfromdefault.py          # Logistic regression training + plotting
├── util.py          # CS229 helper functions (load_csv, plot)
├── ds1_a.csv        # Dataset A — linearly separable
├── ds1_b.csv        # Dataset B — not linearly separable
├── plot.py          # Plotting Dataset A and B
└── README.md
```

## Usage

```bash
python main.py
```

## Key Takeaways

1. Always use `.copy()` when you need a snapshot of a NumPy array
2. Gradient descent needs learning rate tuning — there's no free lunch
3. Non-separable data will never reach zero loss — plateau is expected
4. Convergence tolerance should match your optimizer (tight for Newton, loose for GD)
5. The ±1 label formulation is mathematically elegant and worth knowing

## Dependencies

```bash
pip install numpy matplotlib
```
