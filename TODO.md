# OPTINUM - Optimization & Numerics Library

> **HEADER-ONLY C++20 SIMD-ACCELERATED LIBRARY**
> Just `#include <optinum/optinum.hpp>` - No compilation required

---

## Module Status Overview

| Module | Status | Lines | Description |
|--------|--------|-------|-------------|
| `simd/` | **COMPLETE** | ~20,000 | SIMD pack<T,W>, views, 40+ math functions (SSE/AVX/AVX-512/NEON) |
| `lina/` | **COMPLETE** | ~2,800 | Linear algebra: 5 decompositions, solvers, DARE, Jacobian, Hessian |
| `lie/` | **COMPLETE** | ~4,400 | Lie groups: SO2/SE2/SO3/SE3, Sim2/Sim3, batched SIMD |
| `opti/` | **COMPLETE** | ~3,500 | 12 optimizers, 7 decay policies, 4 test problems, line search |
| `meta/` | **TODO** | ~0 | Metaheuristic optimizers: MPPI, PSO, CEM, SA |

**Test Status:** 87/87 test suites passing (400+ test cases)

---

## Completed Modules

### SIMD Module (`simd/`) - 89 files, ~20,000 lines

**Pack Types (`simd/pack/`):**
- `pack<T,W>` for SSE, AVX, AVX-512, NEON
- `pack<complex<T>,W>` for complex SIMD
- `pack<quaternion<T>,W>` for quaternion SIMD (Hamilton product, exp, log, slerp)

**Views (`simd/view/`):**
- `Vector<T,N>`, `Matrix<T,R,C>`, `Tensor<T,dims...>` - non-owning SIMD views
- `quaternion_view` - transparent SIMD over quaternion arrays
- `complex_view`, `diagonal_view`, `filter_view`, `slice_view`

**Math Functions (`simd/math/`) - 40+ functions:**
- Exponential: exp, exp2, expm1, log, log2, log10, log1p
- Power: pow, sqrt, cbrt, rsqrt, hypot
- Trigonometric: sin, cos, tan, sincos, asin, acos, atan, atan2
- Hyperbolic: sinh, cosh, tanh, asinh, acosh, atanh
- Special: erf, erfc, tgamma, lgamma
- Rounding: ceil, floor, round, trunc, frac
- Utility: abs, copysign, fmod, clamp, isnan, isinf

**Algorithms (`simd/algo/`):**
- Elementwise: add, sub, mul, div, axpy, scale, fill, copy
- Reductions: sum, min, max, dot, norm

**Backend (`simd/backend/`):**
- Optimized 2×2, 3×3, 4×4 kernels (32-243x speedup)
- matmul, transpose, determinant, inverse for small matrices

---

### Linear Algebra Module (`lina/`) - 28 files, ~2,800 lines

**Decompositions (`lina/decompose/`):**
| Function | Algorithm | Lines |
|----------|-----------|-------|
| `lu()` | LU with partial pivoting | 215 |
| `qr()` | Householder reflections | 158 |
| `svd()` | One-sided Jacobi | 234 |
| `cholesky()` | SPD matrices | 84 |
| `eigen()` | Symmetric eigendecomposition | 144 |

**Solvers (`lina/solve/`):**
| Function | Purpose | Lines |
|----------|---------|-------|
| `solve()` | Ax = b via LU | 81 |
| `lstsq()` | Least squares via QR | 65 |
| `dare()` | Discrete Algebraic Riccati Equation | 270 |
| `triangular_solve()` | Forward/back substitution | 104 |

**Basic Operations (`lina/basic/`):**
- `matmul`, `transpose`, `inverse`, `determinant`
- `adjoint`, `cofactor`, `trace`, `norm`
- `jacobian`, `gradient`, `hessian` - numerical differentiation
- `hessian_vector_product`, `laplacian` - efficient Hessian operations
- `pinv`, `null`, `orth`, `rank`, `cond`
- `expmat`, `log_det`, `is_finite`, `properties`

**Algebra (`lina/algebra/`):**
- `einsum` - Einstein summation notation
- `kron` - Kronecker product
- `contraction` - Tensor contractions

---

### Lie Groups Module (`lie/`) - 15 files, ~4,400 lines

**Core Groups (`lie/groups/`):**
| Group | DoF | Storage | Lines | Description |
|-------|-----|---------|-------|-------------|
| `SO2<T>` | 1 | unit complex | 335 | 2D rotations |
| `SE2<T>` | 3 | SO2 + Vector2 | 533 | 2D rigid transforms |
| `SO3<T>` | 3 | unit quaternion | 602 | 3D rotations |
| `SE3<T>` | 6 | SO3 + Vector3 | 700 | 3D rigid transforms |

**Similarity Groups (`lie/groups/`):**
| Group | DoF | Lines | Description |
|-------|-----|-------|-------------|
| `RxSO2<T>` | 2 | 260 | 2D rotation + scale |
| `RxSO3<T>` | 4 | 347 | 3D rotation + scale |
| `Sim2<T>` | 4 | 413 | 2D similarity transform |
| `Sim3<T>` | 7 | 490 | 3D similarity transform |

**Algorithms (`lie/algorithms/`):**
- `average.hpp` (449 lines) - Biinvariant/Frechet mean computation
- `spline.hpp` (332 lines) - Lie group splines for smooth trajectories

**Batched SIMD (`lie/batch/`):**
- `SO3Batch<T,N>` (347 lines) - N rotations processed in parallel
- `SE3Batch<T,N>` (434 lines) - N poses processed in parallel
- Uses `quaternion_view` for transparent SIMD acceleration

**All groups implement:**
- `exp()` / `log()` - Exponential/logarithm maps
- `hat()` / `vee()` - Lie algebra ↔ vector conversion
- `Adj()` - Adjoint representation
- `inverse()`, `operator*` - Group operations
- `matrix()` - Matrix representation
- Jacobians for optimization

---

### Optimization Module (`opti/`) - 25 files, ~3,500 lines

**First-Order Optimizers (`opti/gradient/update_policies/`):**
| Optimizer | File | Lines | Reference |
|-----------|------|-------|-----------|
| Vanilla GD | `vanilla_update.hpp` | 38 | - |
| Momentum | `momentum_update.hpp` | 85 | Polyak 1964 |
| Nesterov | `nesterov_update.hpp` | 100 | Nesterov 1983 |
| AdaGrad | `adagrad_update.hpp` | 90 | Duchi 2011 |
| AdaDelta | `adadelta_update.hpp` | 130 | Zeiler 2012 |
| RMSprop | `rmsprop_update.hpp` | 90 | Hinton 2012 |
| Adam | `adam_update.hpp` | 125 | Kingma 2014 |
| AMSGrad | `amsgrad_update.hpp` | 135 | Reddi 2018 |
| NAdam | `nadam_update.hpp` | 90 | Dozat 2016 |
| AdaBound | `adabound_update.hpp` | 160 | Luo 2019 |
| Yogi | `yogi_update.hpp` | 135 | Zaheer 2018 |

**Second-Order Optimizers (`opti/quasi_newton/`):**
| Optimizer | File | Lines | Reference |
|-----------|------|-------|-----------|
| L-BFGS | `lbfgs.hpp` | 400 | Liu & Nocedal 1989 |
| Gauss-Newton | `gauss_newton.hpp` | 609 | - |
| Levenberg-Marquardt | `levenberg_marquardt.hpp` | 452 | - |

**Line Search (`opti/line_search/`):**
| Algorithm | Description |
|-----------|-------------|
| Armijo | Backtracking with sufficient decrease |
| Wolfe | Strong Wolfe conditions |
| Weak Wolfe | Weak Wolfe conditions |
| Goldstein | Goldstein conditions |

**Decay Policies (`opti/decay/`):**
| Policy | Description |
|--------|-------------|
| `StepDecay` | Drop by factor at fixed intervals |
| `ExponentialDecay` | Smooth exponential decay |
| `CosineAnnealing` | Cosine curve (SGDR style) |
| `LinearDecay` | Linear decrease to min_lr |
| `InverseTimeDecay` | 1/(1 + decay_rate * t) |
| `WarmupDecay` | Linear warmup then constant |
| `PolynomialDecay` | Flexible power-based decay |

**Test Problems (`opti/problem/`):**
| Problem | Description |
|---------|-------------|
| `Sphere` | Simple convex quadratic |
| `Rosenbrock` | Classic banana function |
| `Rastrigin` | Highly multimodal |
| `Ackley` | Flat outer region |

---

## TODO - Metaheuristic Module (`meta/`)

**Location:** `include/optinum/meta/`

Population-based and sampling-based optimization methods for non-convex, black-box, and stochastic optimization problems.

### Priority 1 (P1) - Core Methods

| ID | Task | Difficulty | Est. Lines | Use Case |
|----|------|------------|------------|----------|
| `meta-mppi` | **MPPI** (Model Predictive Path Integral) | Medium | ~300 | Robotics trajectory optimization |
| `meta-pso` | **PSO** (Particle Swarm Optimization) | Medium | ~250 | Global optimization, swarm intelligence |
| `meta-cem` | **CEM** (Cross-Entropy Method) | Medium | ~200 | Importance sampling, policy search |

### Priority 2 (P2) - Classical Methods

| ID | Task | Difficulty | Est. Lines | Use Case |
|----|------|------------|------------|----------|
| `meta-sa` | **SA** (Simulated Annealing) | Easy | ~150 | Combinatorial optimization |
| `meta-de` | **DE** (Differential Evolution) | Medium | ~200 | Continuous global optimization |

### Priority 3 (P3) - Advanced Methods

| ID | Task | Difficulty | Est. Lines | Use Case |
|----|------|------------|------------|----------|
| `meta-cmaes` | **CMA-ES** (Covariance Matrix Adaptation) | Hard | ~400 | State-of-the-art evolutionary strategy |
| `meta-ga` | **GA** (Genetic Algorithm) | Medium | ~300 | General evolutionary optimization |

### Meta-Optimizers (Wrappers)

| ID | Task | Difficulty | Est. Lines | Description |
|----|------|------------|------------|-------------|
| `meta-lookahead` | **Lookahead** | Easy | ~100 | Slow weights + fast weights (Zhang 2019) |
| `meta-swats` | **SWATS** | Medium | ~150 | Switch from Adam to SGD (Keskar 2017) |

---

## Implementation Guidelines

### SIMD Requirements

All metaheuristic methods MUST use SIMD acceleration for population operations:

```cpp
// Example: PSO velocity update for population of particles
template <typename T, std::size_t N>
void update_velocities(
    simd::Matrix<T, simd::Dynamic, N>& velocities,  // [num_particles x dim]
    const simd::Matrix<T, simd::Dynamic, N>& positions,
    const simd::Matrix<T, simd::Dynamic, N>& personal_best,
    const simd::Vector<T, N>& global_best,
    T w, T c1, T c2
) {
    // Process all particles in parallel using SIMD
    // Use simd::pack for vectorized random number generation
    // Avoid for-loops over particles - use matrix operations
}
```

### File Structure

```
include/optinum/meta/
├── mppi.hpp           # Model Predictive Path Integral
├── pso.hpp            # Particle Swarm Optimization
├── cem.hpp            # Cross-Entropy Method
├── sa.hpp             # Simulated Annealing
├── de.hpp             # Differential Evolution
├── cmaes.hpp          # CMA-ES
├── ga.hpp             # Genetic Algorithm
├── lookahead.hpp      # Lookahead meta-optimizer
├── swats.hpp          # SWATS meta-optimizer
└── meta.hpp           # Module header (includes all)
```

### Common Interface

All metaheuristic optimizers should follow this pattern:

```cpp
template <typename T = double>
class PSO {
public:
    struct Config {
        std::size_t population_size = 50;
        std::size_t max_iterations = 1000;
        T tolerance = 1e-6;
        // Method-specific parameters...
    };

    struct Result {
        simd::Vector<T, simd::Dynamic> best_position;
        T best_value;
        std::size_t iterations;
        bool converged;
        std::vector<T> history;  // Best value per iteration
    };

    explicit PSO(Config config = {});

    template <typename F>
    Result optimize(F&& objective, const simd::Vector<T, simd::Dynamic>& initial);

    template <typename F>
    Result optimize(F&& objective, 
                    const simd::Vector<T, simd::Dynamic>& lower_bounds,
                    const simd::Vector<T, simd::Dynamic>& upper_bounds);
};
```

### Testing

- Add tests in `test/meta/`
- Test on standard benchmark functions (Sphere, Rosenbrock, Rastrigin, Ackley)
- Verify convergence to global optimum
- Test with different population sizes
- Benchmark SIMD vs scalar performance

### API Exposure

1. Add include to `meta/meta.hpp`
2. Add `#include "meta/meta.hpp"` to `optinum.hpp`
3. Add using declarations for public types

---

## Local Reference Implementations

The following reference implementations are available in `xtra/` for guidance:

| Algorithm | Location | Notes |
|-----------|----------|-------|
| **PSO** | `xtra/ensmallen/include/ensmallen_bits/pso/` | Policy-based, lbest/gbest |
| **CMA-ES** | `xtra/ensmallen/include/ensmallen_bits/cmaes/` | Full algorithm with policies |
| **DE** | `xtra/ensmallen/include/ensmallen_bits/de/` | Simple best/1/bin strategy |
| **SA** | `xtra/ensmallen/include/ensmallen_bits/sa/` | Cooling schedules, move control |
| **MPPI** | `../drivekit/src/drivekit/pred/mppi.cpp` | Robotics trajectory optimization |

**Key Patterns from ensmallen:**
- Template on policies (UpdatePolicy, InitPolicy, CoolingSchedule)
- Callbacks: `BeginOptimization`, `Evaluate`, `StepTaken`, `EndOptimization`
- Uses Armadillo matrices (we use `simd::Matrix`/`simd::Vector`)
- Convergence via horizon-based improvement tracking

---

## References

**Metaheuristics:**
- MPPI: Williams et al. (2017) https://arxiv.org/abs/1707.02342
- PSO: Kennedy & Eberhart (1995)
- CEM: Rubinstein (1999)
- CMA-ES: Hansen & Ostermeier (2001)
- Lookahead: Zhang et al. (2019) https://arxiv.org/abs/1907.08610
- SWATS: Keskar & Socher (2017) https://arxiv.org/abs/1712.07628

**Optimization:**
- Adam: Kingma & Ba (2014) https://arxiv.org/abs/1412.6980
- L-BFGS: Liu & Nocedal (1989)
- ensmallen: `xtra/ensmallen/` (local copy)

**Lie Groups:**
- Sophus: `xtra/Sophus/` (local copy)
- micro-lie: https://github.com/artivis/manif

**SIMD:**
- Intel Intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
- SLEEF: `xtra/sleef/` (local copy)

---

**Last Updated:** December 28, 2025
**Beads Tracking:** Run `bd ready` to see available tasks
