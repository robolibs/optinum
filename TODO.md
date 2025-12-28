# OPTINUM - Optimization & Numerics Library

> **HEADER-ONLY C++20 SIMD-ACCELERATED LIBRARY**
> Just `#include <optinum/optinum.hpp>` - No compilation required

---

## Module Status Overview

| Module | Status | Lines | Description |
|--------|--------|-------|-------------|
| `simd/` | **COMPLETE** | ~20,000 | SIMD pack<T,W>, views, 40+ math functions (SSE/AVX/AVX-512/NEON) |
| `lina/` | **COMPLETE** | ~2,800 | Linear algebra: 5 decompositions, solvers, DARE, Jacobian |
| `lie/` | **COMPLETE** | ~4,400 | Lie groups: SO2/SE2/SO3/SE3, Sim2/Sim3, batched SIMD |
| `opti/` | **PARTIAL** | ~2,400 | 7 optimizers done, 9 remaining |

**Test Status:** 74/74 test suites passing (350+ test cases)

---

## Core Architecture: datapod + SIMD

**This library has a strict separation of concerns:**

1. **Data Storage** → `datapod::mat::*` (POD types, own memory, serializable)
2. **SIMD Operations** → `optinum::simd::*` (wrap datapod, provide SIMD acceleration)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER CODE                                       │
│         optinum::Vector<float, 3> v;   // SIMD-accelerated type             │
│         optinum::matmul(A, B);         // SIMD operation                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         optinum::simd:: (SIMD Layer)                         │
│                                                                              │
│   simd::Vector<T,N>  ──wraps──>  dp::mat::vector<T,N>  (internal pod_)      │
│   simd::Matrix<T,R,C> ─wraps──>  dp::mat::matrix<T,R,C> (internal pod_)     │
│   simd::Scalar<T>    ──wraps──>  dp::mat::scalar<T>    (internal pod_)      │
│                                                                              │
│   All operations use SIMD intrinsics (AVX/SSE/NEON) under the hood          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         datapod::mat:: (Storage Layer)                       │
│                                                                              │
│   dp::mat::scalar<T>      - rank-0 tensor (single value)                    │
│   dp::mat::vector<T,N>    - rank-1 tensor (1D array)                        │
│   dp::mat::matrix<T,R,C>  - rank-2 tensor (2D array, column-major)          │
│   dp::mat::quaternion<T>  - unit quaternion [w,x,y,z]                       │
│                                                                              │
│   ✓ POD (Plain Old Data) - trivially copyable                               │
│   ✓ Cache-aligned for SIMD                                                  │
│   ✓ Serializable for ROS2/network                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Design Principle:**
- **datapod owns the data** - POD structs, serializable, ROS2-compatible
- **optinum provides SIMD operations** - wraps datapod, accelerates computation
- **Zero-copy access** - `v.pod()` returns reference to underlying datapod type

```cpp
// Example: datapod storage + optinum SIMD
optinum::Vector<float, 3> v;           // SIMD wrapper
v[0] = 1.0f; v[1] = 2.0f; v[2] = 3.0f;

datapod::mat::vector<float, 3>& pod = v.pod();  // Access underlying POD
// pod is serializable, can be sent over ROS2, saved to disk, etc.

auto norm = optinum::norm(v);          // SIMD-accelerated operation
```

---

## SIMD Acceleration Stack

All operations use vectorized instructions via the `simd::pack<T,W>` abstraction:

```
┌─────────────────────────────────────────────────────────────────┐
│  User API:  optinum::matmul(), optinum::exp(), optinum::SO3<>   │
├─────────────────────────────────────────────────────────────────┤
│  Wrappers:  simd::Vector<T,N>, simd::Matrix<T,R,C>              │
│             (wrap datapod types, delegate to backend)           │
├─────────────────────────────────────────────────────────────────┤
│  Pack:      simd::pack<T,W> - W elements in SIMD registers      │
├─────────────────────────────────────────────────────────────────┤
│  Backend:   AVX-512 (W=8/16) | AVX (W=4/8) | SSE (W=2/4) | NEON │
└─────────────────────────────────────────────────────────────────┘
```

**Key SIMD Features:**
- `pack<T,W>` - SIMD register abstraction (float×8, double×4, etc.)
- Dual-path execution: compile-time (fixed N) + runtime (Dynamic)
- 40+ vectorized math functions (exp, log, sin, cos, tanh, erf, etc.)
- 2-27x speedup over scalar operations

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
- `jacobian`, `gradient` - numerical differentiation
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

### Optimization Module (`opti/`) - 13 files, ~2,200 lines

**Implemented Optimizers:**

| Optimizer | Type | File | Lines | SIMD Speedup |
|-----------|------|------|-------|--------------|
| Vanilla GD | First-order | `vanilla_update.hpp` | 38 | baseline |
| Momentum | First-order | `momentum_update.hpp` | 85 | 2.1x |
| Nesterov | First-order | `nesterov_update.hpp` | 100 | ~2x |
| RMSprop | First-order | `rmsprop_update.hpp` | 90 | 5.8x |
| Adam | First-order | `adam_update.hpp` | 125 | 3.6x |
| AMSGrad | First-order | `amsgrad_update.hpp` | 135 | ~3.5x |
| Yogi | First-order | `yogi_update.hpp` | 135 | ~3.5x |
| Gauss-Newton | Second-order | `gauss_newton.hpp` | 609 | - |
| Levenberg-Marquardt | Second-order | `levenberg_marquardt.hpp` | 452 | - |

**Infrastructure:**
- `GradientDescent<UpdatePolicy>` - Template optimizer
- `OptimizationResult<T>` - Result type with convergence info
- Callback system for monitoring
- Decay policies (NoDecay, extensible)
- Test problems (Sphere function)

---

## TODO - Remaining Work

### Priority 1 (P1) - Essential Optimizers

| ID | Task | Difficulty | Est. Lines | Reference |
|----|------|------------|------------|-----------|
| ~~`optinum-0kz`~~ | ~~**Nesterov Momentum**~~ | ~~Easy~~ | ~~100~~ | **DONE** - Nesterov 1983 |
| ~~`optinum-b3o`~~ | ~~**AdaGrad**~~ | ~~Easy~~ | ~~90~~ | **DONE** - Duchi 2011 |
| ~~`optinum-rd9`~~ | ~~**AdaDelta**~~ | ~~Easy~~ | ~~130~~ | **DONE** - Zeiler 2012 |
| ~~`optinum-9qi`~~ | ~~**AMSGrad**~~ | ~~Trivial~~ | ~~230~~ | **DONE** - Reddi 2018 |
| `optinum-mde` | **L-BFGS** | Hard | ~400 | Liu & Nocedal 1989 |

**Implementation Pattern (for update policies):**
```cpp
// File: opti/gradient/update_policies/nesterov_update.hpp
struct NesterovUpdate {
    simd::Vector<double, simd::Dynamic> velocity;  // Use double precision for state
    double momentum = 0.9;

    template <typename T, std::size_t N>
    void update(simd::Vector<T, N>& x, T lr, const simd::Vector<T, N>& grad) {
        // SIMD dual-path: if constexpr (N == Dynamic) { runtime } else { compile-time }
        // Lookahead: v = μv - α∇f(x + μv); x += v
    }
};
```

### Priority 2 (P2) - Important Extensions

| ID | Task | Difficulty | Est. Lines |
|----|------|------------|------------|
| `optinum-bbn` | **Line Search** (Armijo, Wolfe) | Medium | ~200 |
| `optinum-8zj` | **Lookahead** meta-optimizer | Easy | ~80 |
| ~~`optinum-hi6`~~ | ~~**AdaBound**~~ | ~~Easy~~ | ~~160~~ | **DONE** - Luo 2019 |
| ~~`optinum-aak`~~ | ~~**Yogi**~~ | ~~Easy~~ | ~~90~~ | **DONE** - Zaheer 2018 |
| `optinum-m0v` | **Metaheuristic Module** (MPPI, PSO, CEM) | Large | ~1500 |

### Priority 3 (P3) - Nice to Have

| ID | Task | Difficulty | Est. Lines |
|----|------|------------|------------|
| `optinum-5v1` | NAdam | Easy | ~90 |
| `optinum-gck` | SWATS meta-optimizer | Medium | ~150 |
| `optinum-nfp` | Test problems (Rosenbrock, Rastrigin) | Easy | ~200 |
| `optinum-puk` | Learning rate schedulers | Easy | ~150 |
| `optinum-p6b` | Hessian computation | Medium | ~150 |

### Dependencies

```
optinum-mde (L-BFGS) ──depends-on──> optinum-bbn (Line Search)
```

---

## Implementation Guidelines

### SIMD Requirements

All optimizers MUST use SIMD acceleration:

```cpp
template <typename T, std::size_t N>
void update(simd::Vector<T, N>& x, T lr, const simd::Vector<T, N>& grad) {
    const std::size_t n = x.size();  // Runtime size (works for Dynamic)

    if constexpr (N == simd::Dynamic) {
        // Runtime SIMD path
        constexpr std::size_t W = std::is_same_v<T, double> ? 4 : 8;
        using pack_t = simd::pack<T, W>;
        for (std::size_t i = 0; i + W <= n; i += W) {
            pack_t px = pack_t::load(&x[i]);
            pack_t pg = pack_t::load(&grad[i]);
            // ... SIMD operations ...
            px.store(&x[i]);
        }
        // Handle remainder
    } else {
        // Compile-time SIMD path
        constexpr std::size_t W = simd::backend::preferred_simd_lanes<T, N>();
        // ... unrolled SIMD ...
    }
}
```

### File Locations

- Update policies: `opti/gradient/update_policies/`
- Meta-optimizers: `opti/meta/`
- Quasi-Newton: `opti/quasi_newton/`
- Test problems: `opti/problem/`

### Testing

- Add tests in `test/opti/`
- Test both fixed-size and Dynamic vectors
- Verify convergence on Sphere function
- Compare SIMD vs scalar performance

### API Exposure

1. Add include to `opti/opti.hpp`
2. Add using declaration to `optinum.hpp`

---

## Metaheuristic Module (Future)

**Location:** `include/optinum/meta/`

**Planned Methods:**
| Method | Priority | Est. Hours | Use Case |
|--------|----------|------------|----------|
| MPPI | High | 4h | Model Predictive Path Integral (robotics) |
| PSO | High | 4h | Particle Swarm Optimization |
| CEM | Medium | 3h | Cross-Entropy Method |
| SA | Medium | 3h | Simulated Annealing |
| GA | Low | 6h | Genetic Algorithm |
| CMA-ES | Low | 8h | Covariance Matrix Adaptation |

---

## Quick Reference

### datapod + SIMD Pattern

**The fundamental pattern - datapod stores, optinum accelerates:**
```cpp
// optinum types wrap datapod internally
optinum::Vector<float, 3> v;           // Contains dp::mat::vector<float, 3> internally
v[0] = 1.0f; v[1] = 2.0f; v[2] = 3.0f;

// Access underlying POD for serialization/ROS2
datapod::mat::vector<float, 3>& pod = v.pod();

// SIMD-accelerated operations
auto n = optinum::norm(v);             // Uses AVX/SSE under the hood
auto w = optinum::normalized(v);       // Returns new optinum::Vector
```

**Creating from existing datapod:**
```cpp
datapod::mat::matrix<double, 3, 3> raw_data;  // POD storage
optinum::Matrix<double, 3, 3> A(raw_data);    // Wrap for SIMD ops
auto det = optinum::determinant(A);            // SIMD-accelerated
```

### SIMD Pack Operations

```cpp
simd::pack<float, 8> a, b;
auto c = a + b;           // SIMD add (8 floats at once)
auto d = simd::exp(a);    // SIMD exp
auto s = a.reduce_add();  // Horizontal sum
```

### Non-owning Views

```cpp
dp::mat::vector<float, 16> data;       // datapod owns memory
auto v = simd::view<8>(data);          // Non-owning SIMD view
simd::scale(2.0f, v);                  // In-place SIMD scale
// data is modified, no copy made
```

### Lie Groups

```cpp
lie::SO3<double> R = lie::SO3<double>::exp(omega);
lie::Vector<double, 3> p_rot = R * p;
lie::Vector<double, 3> omega_back = R.log();
```

### Linear Algebra

```cpp
auto [L, U, P] = lina::lu(A);
auto x = lina::solve(A, b);
auto J = lina::jacobian(f, x);
```

---

## References

**Optimization:**
- Adam: Kingma & Ba (2014) https://arxiv.org/abs/1412.6980
- L-BFGS: Liu & Nocedal (1989)
- ensmallen: https://github.com/mlpack/ensmallen

**Lie Groups:**
- Sophus: https://github.com/strasdat/Sophus
- micro-lie: https://github.com/artivis/manif

**SIMD:**
- Intel Intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
- SLEEF: https://sleef.org/

---

**Last Updated:** December 28, 2025
**Beads Tracking:** Run `bd ready` to see available tasks
