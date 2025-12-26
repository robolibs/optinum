# OPTINUM - Optimization & Numerics Library

> **HEADER-ONLY C++20 LIBRARY** - No compilation required, just `#include <optinum/optinum.hpp>`

---

## Module Status

| Module | Status | Description |
|--------|--------|-------------|
| `simd/` | **✅ COMPLETE** | SIMD operations, views, pack<T,W>, math functions |
| `lina/` | **✅ COMPLETE** | Linear algebra (matmul, decompose, solve, einsum) |
| `opti/` | **📋 PLANNED** | Numerical optimization (not started) |
| **API** | **✅ COMPLETE** | Unified optinum:: namespace (Armadillo-style!) |

---

## 🎯 PLAN: API UNIFICATION

### Goal: Clean Armadillo-Style Public API

**Current state:** Users must mix `simd::` and `lina::` namespaces
```cpp
auto A = simd::Matrix<double, 3, 3>::random();  // simd namespace
auto x = lina::solve(A, b);                      // lina namespace
```

**Target state:** Single `optinum::` namespace (like Armadillo's `arma::`)
```cpp
auto A = optinum::Matrix<double, 3, 3>::random();
auto x = optinum::solve(A, b);
// Or with SHORT_NAMESPACE: on::Matrix, on::solve
```

---

### Implementation Tasks

**Phase 1: Create Unified Namespace** ✅ **COMPLETE**
- [x] Update `include/optinum/optinum.hpp` with namespace aliases
- [x] Expose types: `optinum::Matrix`, `optinum::Vector`, `optinum::Tensor`, `optinum::Scalar`, `optinum::Complex`
- [x] Expose all lina:: functions: `optinum::solve`, `optinum::determinant`, etc. (20+ functions)
- [x] Expose all simd:: math functions: `optinum::exp`, `optinum::sin`, etc. (40+ functions)
- [x] Expose all simd:: algorithms: `optinum::sum`, `optinum::add`, etc. (10+ functions)
- [x] Expose utility functions: `optinum::view`, `optinum::noalias`, layout conversion, Voigt
- [x] Keep simd:: and lina:: accessible for power users
- [x] Create demo examples showing unified API
- [x] All 80+ functions exposed through optinum::

**Phase 2: Documentation**
- [ ] Update README with optinum:: as primary API
- [ ] Document that `simd::`/`lina::` are implementation details (still accessible)
- [ ] Create API reference documentation
- [ ] Add migration guide if needed

---

### Current API Exposure (optinum::)

**✅ COMPLETE - 80+ Functions Exposed:**

```cpp
// File: include/optinum/optinum.hpp
// Everything users need in one namespace!

namespace optinum {
    // === TYPES (5) ===
    Matrix<T,R,C>, Vector<T,N>, Tensor<T,Dims...>, Scalar<T>, Complex<T,N>
    
    // === LINEAR ALGEBRA (20+) ===
    // Basic: solve, determinant, inverse, matmul, transpose, adjoint, cofactor
    // Decompositions: lu, qr, svd, cholesky, eigen_sym
    // Solvers: solve, lstsq
    // Norms: dot, norm, norm_fro, cross
    // Tensor: einsum, inner, outer, hadamard
    // BLAS: scale, axpy
    
    // === SIMD MATH (40+) ===
    // Exponential/Log: exp, log, sqrt, pow, exp2, expm1, log2, log10, log1p, cbrt
    // Trigonometric: sin, cos, tan, asin, acos, atan, atan2
    // Hyperbolic: sinh, cosh, tanh, asinh, acosh, atanh
    // Rounding: ceil, floor, round, trunc
    // Utility: abs, clamp, hypot
    // Boolean: isnan, isinf, isfinite
    // Special: erf, tgamma, lgamma
    
    // === ALGORITHMS (10+) ===
    // Reduce: sum, min, max
    // Elementwise: add, sub, mul, div, fill, copy
    
    // === UTILITIES (5+) ===
    view, noalias, torowmajor, tocolumnmajor, to_voigt, from_voigt
}
```

**See:** `examples/unified_api_demo.cpp` and `examples/math_functions_demo.cpp`

---

### Future Module Integration Rules

**⚠️ CRITICAL:** When adding new features to ANY module:

1. **For simd:: additions:**
   - Implement in `simd/` (infrastructure layer)
   - Add type alias or `using` declaration in `optinum.hpp`
   - Example: New `simd::Complex` → Add `using Complex = simd::Complex;` to optinum::

2. **For lina:: additions:**
   - Implement in `lina/` (algorithm layer)
   - Add `using` declaration in `optinum.hpp`
   - Example: New `lina::pinv()` → Add `using lina::pinv;` to optinum::

3. **For opti:: additions (future):**
   - Implement in `opti/` (optimization layer)
   - Add types and functions to `optinum::` namespace
   - Example: `opti::Adam` → Add `using Adam = opti::Adam;` to optinum::
   - Example: `opti::minimize()` → Add `using opti::minimize;` to optinum::

**Rule of thumb:** Any public-facing feature MUST be exposed in `optinum::` namespace.

---

### Design Principles for API Exposure

✅ **DO expose in optinum::**
- All types users create (Matrix, Vector, Tensor, optimizers)
- All functions users call (solve, determinant, minimize)
- Public-facing utilities (factory functions, conversions)

❌ **DON'T expose in optinum::**
- Internal implementation details (backend kernels, pack<T,W> details)
- Advanced/power-user features (view manipulation internals)
- Debug/development utilities

**When in doubt:** If a user would reasonably want to call it, expose it in `optinum::`.

---

---

## Architecture

```
datapod (dp::)                       # DATA OWNERSHIP (external library v0.0.9)
├── mat::scalar<T>                   # rank-0 (single value)
├── mat::vector<T, N>                # rank-1 (1D array, aligned)
├── mat::matrix<T, R, C>             # rank-2 (column-major)
└── mat::tensor<T, Dims...>          # rank-N (N-D array)

optinum (on::)                       # SIMD OPERATIONS (this library)
├── simd/                            # Non-owning SIMD views + algorithms
│   ├── pack<T, W>                   # SIMD register (SSE/AVX/AVX-512/NEON)
│   ├── view<W>(dp_obj)              # Factory: dp type → simd view
│   ├── algo::                       # BLAS-like + math transforms
│   └── math::                       # Vectorized math (40+ functions)
│
├── lina/                            # Linear algebra
│   ├── basic/                       # matmul, transpose, inverse, det, adj, cof
│   ├── decompose/                   # LU, QR, SVD, Cholesky, Eigen
│   ├── solve/                       # solve, lstsq
│   └── algebra/                     # einsum, contraction
│
└── opti/                            # Optimization (PLANNED)
    ├── gradient/                    # GD, SGD
    ├── adaptive/                    # Adam, AdaGrad, RMSProp
    ├── quasi_newton/                # L-BFGS
    └── evolutionary/                # CMA-ES, PSO, DE
```

**Data Flow:**
```
dp::mat::vector<float, N>     (owns memory)
         ↓
simd::view<W>(dp_vector)      (non-owning SIMD view)
         ↓
simd::exp(view)               (algorithm layer)
         ↓
simd::exp(pack<float,8>)      (intrinsic layer - AVX)
```

---

## CRITICAL IMPLEMENTATION RULE

⚠️ **ALL operations MUST use SIMD** (except constexpr contexts, tail loops, or scalar fallbacks)

**Pattern:**
```cpp
constexpr std::size_t W = preferred_simd_lanes<T, N>();
constexpr std::size_t main = main_loop_count<N, W>();

// SIMD main loop
for (std::size_t i = 0; i < main; i += W) {
    auto p = pack<T, W>::loadu(data + i);
    // ... SIMD operations ...
    p.storeu(result + i);
}

// Scalar tail loop
for (std::size_t i = main; i < N; ++i) {
    result[i] = scalar_operation(data[i]);
}
```

**Fallback chain:** AVX-512 → AVX → SSE → NEON → `pack<T,W>` scalar

---

## Module 1: SIMD - ✅ COMPLETE

### Implementation Summary

**Core Types:**
- ✅ `pack<T,W>` - SSE/AVX/AVX-512/NEON implementations
- ✅ `mask<T,W>` - Comparison results, blend/select
- ✅ `Kernel<T,W,Rank>` - Memory layout descriptor
- ✅ Views: scalar, vector, matrix, tensor
- ✅ Slicing: `seq()`, `fseq<>()`, `all`, `fix<>()`
- ✅ Special views: diagonal, filter, random_access

**SIMD Math (40+ functions, float & double):**
- ✅ Basic: exp, log, sin, cos, tan, sqrt, tanh
- ✅ Inverse trig: asin, acos, atan, atan2
- ✅ Hyperbolic: sinh, cosh, asinh, acosh, atanh
- ✅ Power: pow, exp2, expm1, log2, log10, log1p, cbrt
- ✅ Rounding: ceil, floor, round, trunc
- ✅ Utility: abs, clamp, hypot
- ✅ Tests: isnan, isinf, isfinite
- ✅ Special: erf, tgamma, lgamma

**Algorithms:**
- ✅ Elementwise: axpy, scale, add, sub, mul, div, fill, copy
- ✅ Reductions: sum, min, max, dot, norm
- ✅ Transforms: exp, log, sin, cos, etc. (on views)

**Wrapper Types (Vector, Matrix, Tensor):**
- ✅ Factories: zeros, ones, iota, random, randint
- ✅ Operations: fill, reverse, cast, flatten, reshape, squeeze
- ✅ Layout: torowmajor, tocolumnmajor
- ✅ Mechanics: Voigt notation conversion
- ✅ Optimization: noalias() hints

**Backend:**
- ✅ Specialized 2x2/3x3/4x4 det/inverse (32-243x speedup)
- ✅ SIMD matmul, transpose, dot, norm kernels

**Platform Support:**
- ✅ AVX-512: pack<float,16>, pack<double,8>, pack<int32_t,16>, pack<int64_t,8>
- ✅ AVX: pack<float,8>, pack<double,4>
- ✅ SSE: pack<float,4>, pack<double,2>
- ✅ NEON: pack<float,4>, pack<double,2>, pack<int32_t,4>, pack<int64_t,2>

**Debug:**
- ✅ Bounds checking, shape checking (OPTINUM_ENABLE_RUNTIME_CHECKS)
- ✅ Timing utilities
- ✅ `pack<std::complex<T>, W>` SIMD
- ✅ Complex math: real, imag, conj, magnitude, arg

### Missing (Optional/Future)
- [ ] Hardware gather/scatter (AVX-512 k-registers)
- [ ] SLEEF/MKL/libXSMM backends
- [ ] Sparse matrix support

---

## Module 2: Linear Algebra - ✅ COMPLETE

### Implementation Summary

**Basic Operations:**
- ✅ matmul, transpose, inverse, determinant, norm, trace
- ✅ Specialized 2x2/3x3/4x4 kernels (direct formulas)
- ✅ adjoint/adjugate matrix
- ✅ cofactor matrix

**Decompositions:**
- ✅ LU with partial pivoting
- ✅ QR (Householder reflections)
- ✅ SVD (one-sided Jacobi)
- ✅ Cholesky (SPD matrices)
- ✅ Eigendecomposition (power iteration, symmetric)

**Solvers:**
- ✅ solve (Ax = b via LU)
- ✅ lstsq (least squares via QR)

**Tensor Algebra:**
- ✅ einsum (rank-1/2)
- ✅ contraction (inner, outer, hadamard)

**Expression Templates:**
- ✅ Lazy evaluation (VecAdd, VecScale, MatAdd, MatScale)
- ✅ SIMD backend integration

**SIMD Integration:**
- ✅ All reductions use `backend::dot`, `backend::reduce_sum`
- ✅ Expression templates use SIMD elementwise ops
- ✅ Cholesky/QR/Lstsq use SIMD for inner products
- ✅ Column operations vectorized (column-major layout)

### Missing Features - SIMD-Realistic Assessment

**✅ IMPLEMENT - High SIMD Benefit (80-95% SIMD coverage):**
- [ ] **pinv()** - Pseudo-inverse (wraps SIMD SVD) - CRITICAL
- [ ] **rank()** - Matrix rank (wraps SIMD SVD) - CRITICAL
- [ ] **cond()** / **rcond()** - Condition number (wraps SIMD SVD) - CRITICAL
- [ ] **kron()** - Kronecker product (80% SIMD elementwise ops)
- [ ] **null()** - Null space (wraps SIMD SVD)
- [ ] **orth()** - Orthonormal basis (wraps SIMD QR)
- [ ] **is_finite()** - Finite check (95% SIMD via isfinite + reductions)
- [ ] **log_det()** - Log determinant (wraps SIMD LU + log reduction)

**⚠️ DEFER - Complex or Limited SIMD (<50% coverage):**
- [ ] expmat() - Matrix exponential (complex Padé, ~70% SIMD)
- [ ] Schur decomposition (iterative QR, ~60% SIMD, niche)
- [ ] is_symmetric() / is_hermitian() (~30% SIMD, strided access)
- [ ] sqrtmat() / logmat() / powmat() (complex, ~50% SIMD)

**❌ DON'T IMPLEMENT - Poor SIMD Fit (<30% coverage):**
- [ ] ~~Sylvester solver~~ - Sequential back-sub, <20% SIMD
- [ ] ~~Lyapunov solver~~ - Same as Sylvester
- [ ] ~~balance()~~ - Strided row ops, ~30% SIMD
- [ ] ~~Hessenberg~~ - Preprocessing only, limited value

**Optional/Future:**
- [ ] Double contraction A:B
- [ ] Tensor cross product (beyond 3D)
- [ ] Extend einsum to rank > 2
- [ ] Network einsum (multi-tensor optimization)
- [ ] MatSub, MatMul (lazy operations)
- [ ] Unary math expressions (sin, cos, exp on matrices)
- [ ] Lazy decompositions (SVD, LU, QR)
- [ ] Blocked/tiled algorithms
- [ ] Complex matrix support
- [ ] Sparse matrices (CSR/CSC/COO)
- [ ] BLAS/MKL backend switching

**⚠️ REMINDER:** When implementing any of these, add to `optinum::` namespace in `optinum.hpp`!

---

## Module 3: Optimization - 📋 PLANNED

### Planned Structure

```
include/optinum/opti/
├── opti.hpp                     # Module header
├── core/
│   ├── traits.hpp               # Function type traits
│   ├── callback.hpp             # Callback infrastructure
│   └── log.hpp                  # Logging utilities
├── gradient/
│   ├── gd.hpp                   # Gradient Descent
│   └── sgd.hpp                  # SGD + Momentum + Nesterov
├── adaptive/
│   ├── adam.hpp                 # Adam + AdaMax + AMSGrad + NAdam
│   ├── adagrad.hpp              # AdaGrad
│   └── rmsprop.hpp              # RMSProp
├── quasi_newton/
│   └── lbfgs.hpp                # L-BFGS
├── evolutionary/
│   ├── cmaes.hpp                # CMA-ES
│   ├── de.hpp                   # Differential Evolution
│   ├── pso.hpp                  # Particle Swarm
│   └── sa.hpp                   # Simulated Annealing
├── schedule/
│   ├── cyclical.hpp             # Cyclical learning rate
│   └── warmup.hpp               # Warm restarts
└── problem/
    ├── sphere.hpp               # ✅ DONE (test problem)
    ├── rosenbrock.hpp           # Rosenbrock function
    ├── rastrigin.hpp            # Rastrigin function
    └── ackley.hpp               # Ackley function
```

### Implementation Phases

**Phase 1: Core Infrastructure**
- [ ] `core/traits.hpp` - Type traits for objective functions
- [ ] `core/callback.hpp` - Callback system (early stop, print, timer)

**Phase 2: First-Order Methods**
- [ ] `gradient/gd.hpp` - Gradient Descent
- [ ] `gradient/sgd.hpp` - SGD with momentum/Nesterov
- [ ] `adaptive/adam.hpp` - Adam optimizer

**Phase 3: Second-Order Methods**
- [ ] `quasi_newton/lbfgs.hpp` - L-BFGS

**Phase 4: Derivative-Free Methods**
- [ ] `evolutionary/cmaes.hpp` - CMA-ES
- [ ] `evolutionary/de.hpp` - Differential Evolution
- [ ] `evolutionary/pso.hpp` - Particle Swarm

**Phase 5: Utilities**
- [ ] More test problems (rosenbrock, rastrigin, ackley)
- [ ] Learning rate schedulers
- [ ] Advanced callbacks (gradient clipping, progress bars)

**⚠️ CRITICAL REMINDER:** 
- When implementing opti:: features, expose them in `optinum::` namespace!
- Example: `opti::GradientDescent` → Add `using GradientDescent = opti::GradientDescent;`
- Example: `opti::minimize()` → Add `using opti::minimize;`
- See "PLAN: API UNIFICATION" section above for rules

---

## Design Principles

1. **Header-only** - No compilation needed
2. **Non-owning views** - Zero-copy over datapod types
3. **SIMD-first** - All hot paths vectorized
4. **Zero-cost abstractions** - Expression templates, compile-time dims
5. **Constexpr friendly** - Scalar fallback for compile-time eval
6. **datapod integration** - Prefer `dp::` over `std::` types
7. **Modern C++20** - Concepts, constexpr, fold expressions
8. **Real-time safe** - No dynamic allocation in critical paths

---

## Datapod Type Usage

**In `lina::` and `opti::` modules, prefer datapod types:**

| Use | Instead of |
|-----|------------|
| `dp::Result<T, dp::Error>` | exceptions, error codes |
| `dp::Optional<T>` | `std::optional<T>` |
| `dp::Vector<T>` | `std::vector<T>` |
| `dp::Array<T, N>` | `std::array<T, N>` |
| `dp::String` | `std::string` |

**Error handling pattern:**
```cpp
dp::Result<dp::mat::vector<T, N>, dp::Error> solve(const Matrix& A, const Vector& b) {
    if (is_singular(A)) {
        return dp::Result<...>::err(dp::Error::invalid_argument("matrix is singular"));
    }
    return dp::Result<...>::ok(solution);
}
```

---

## Build & Test

```bash
make config    # Configure (preserves cache)
make build     # Build examples and tests
make test      # Run all tests
make clean     # Clean build artifacts
```

**Test count:** 53 tests (all passing)

---

## Performance Highlights

**SIMD Math Speedups (vs scalar):**
- sin/cos: 22x (float), 6x (double)
- tanh: 27x (float), 7x (double)
- exp: 8x (float), 5x (double)
- sinh/cosh: 19x (float), 18x (double)
- atan: 11x (float), 5x (double)

**Small Matrix Kernels:**
- 2x2 det: 32x faster than LU
- 3x3 det: 140x faster than LU
- 4x4 det: 243x faster than LU
- 2x2/3x3/4x4 inverse: < 0.001ms for 1M operations

---

## What Optinum Does Better Than Fastor

| Feature | Advantage |
|---------|-----------|
| Modern C++20 | Concepts, constexpr, cleaner syntax |
| LU with pivoting | Numerically stable (Fastor lacks pivot) |
| QR/Cholesky/Eigen | Fastor lacks these decompositions |
| Result<T, Error> | Safe error handling (vs exceptions) |
| datapod integration | Clean ownership model |
| Real-time friendly | No dynamic allocation in hot paths |
| Focused scope | Less bloat, easier to maintain |

---

## Dependencies

- **C++20** or later
- **datapod** v0.0.9 (fetched automatically via xmake)
- **doctest** (for tests, fetched automatically)
- **Optional:** AVX2/AVX-512 for best performance

---

## References

- Intel Intrinsics Guide: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
- Fastor: https://github.com/romeric/Fastor
- ensmallen: https://github.com/mlpack/ensmallen
- SLEEF: https://sleef.org/
- libXSMM: https://github.com/libxsmm/libxsmm
