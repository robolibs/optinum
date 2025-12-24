# OPTINUM - Optimization & Numerics Library

> **HEADER-ONLY LIBRARY** - No compilation required, just `#include <optinum/optinum.hpp>`

---

## Quick Status

| Module | Status | Description |
|--------|--------|-------------|
| `simd/` | **DONE** | Non-owning views, pack<T,W>, SIMD math, algorithms |
| `lina/` | **DONE** | Linear algebra (matmul, decompose, solve, einsum) |
| `opti/` | **PLANNED** | Numerical optimization (GD, Adam, CMA-ES, etc.) |

---

## Architecture Overview

```
datapod (dp::)                       # DATA OWNERSHIP (external library)
├── mat::scalar<T>                   # rank-0 (single value)
├── mat::vector<T, N>                # rank-1 (1D array, aligned)
├── mat::matrix<T, R, C>             # rank-2 (2D, column-major, aligned)
└── mat::tensor<T, Dims...>          # rank-N (N-D array)

optinum (on::)                       # SIMD OPERATIONS (this library)
├── simd/        # Non-owning SIMD views + algorithms
│   ├── pack<T, W>                   # SIMD register abstraction (W lanes)
│   ├── mask<T, W>                   # comparison results, blend/select
│   ├── Kernel<T, W, Rank>           # ptr + extents + strides + load/store
│   ├── scalar_view<T, W>            # view over dp::mat::scalar
│   ├── vector_view<T, W>            # view over dp::mat::vector
│   ├── matrix_view<T, W>            # view over dp::mat::matrix
│   ├── tensor_view<T, W, Rank>      # view over dp::mat::tensor
│   ├── view<W>(dp_obj)              # factory: dp type -> simd view
│   ├── algo::axpy, dot, norm, ...   # BLAS-like algorithms on views
│   ├── algo::exp, sin, cos, ...     # math transforms on views
│   └── math::exp, sin, cos, ...     # vectorized math on pack<T,W>
│
├── lina/        # Linear algebra operations
│   ├── matmul, transpose, inverse   # matrix operations
│   ├── lu, qr, svd, cholesky, eigen # decompositions
│   ├── solve, lstsq                 # linear solvers
│   ├── einsum, contraction          # tensor algebra
│   └── expr/                        # expression templates
│
└── opti/        # Numerical optimization (PLANNED)
    ├── GradientDescent, SGD, Adam...
    ├── LBFGS, CMA-ES, PSO...
    └── callbacks, schedulers...
```

**Data Flow:**
```
dp::mat::vector<float, N>   (owns memory)
         ↓
simd::view<W>(dp_vector)    (non-owning view)
         ↓
simd::exp(view)             (algorithm layer)
         ↓
simd::exp(pack<float,8>)    (intrinsic layer - AVX/SSE)
```

---

## Module 1: SIMD (`on::simd`) - COMPLETE

### Folder Structure

```
include/optinum/simd/
├── simd.hpp                     # Module header
├── arch/
│   ├── arch.hpp                 # SSE/AVX/AVX512/NEON detection
│   └── macros.hpp               # OPTINUM_INLINE, alignment, etc.
├── pack/
│   ├── pack.hpp                 # pack<T,W> primary template + scalar fallback
│   ├── sse.hpp                  # SSE: pack<float,4>, pack<double,2>
│   ├── avx.hpp                  # AVX: pack<float,8>, pack<double,4>
│   ├── avx512.hpp               # AVX-512 (stub)
│   └── neon.hpp                 # ARM NEON (stub)
├── view/
│   ├── view.hpp                 # Module header
│   ├── scalar_view.hpp          # Non-owning view over dp::mat::scalar
│   ├── vector_view.hpp          # Non-owning view over dp::mat::vector
│   ├── matrix_view.hpp          # Non-owning view over dp::mat::matrix
│   └── tensor_view.hpp          # Non-owning view over dp::mat::tensor
├── algo/
│   ├── traits.hpp               # is_packable_view, view_value_t, etc.
│   ├── elementwise.hpp          # axpy, scale, add, sub, mul, div, fill, copy
│   ├── reduce.hpp               # sum, min, max, dot, norm
│   └── transform.hpp            # exp, log, sin, cos, tanh, sqrt
├── math/
│   ├── simd_math.hpp            # Module header
│   ├── detail/constants.hpp     # Math constants (LN2, PI, coefficients)
│   ├── exp.hpp                  # exp() for pack<float,W>
│   ├── log.hpp                  # log() for pack<float,W>
│   ├── sin.hpp                  # sin() for pack<float,W>
│   ├── cos.hpp                  # cos() for pack<float,W>
│   ├── tanh.hpp                 # tanh() for pack<float,W>
│   └── sqrt.hpp                 # sqrt() for pack<float,W>
├── backend/
│   ├── backend.hpp              # Module header
│   ├── elementwise.hpp          # Low-level SIMD elementwise ops
│   ├── reduce.hpp               # Low-level SIMD reductions
│   ├── dot.hpp                  # Dot product kernel
│   ├── norm.hpp                 # L2 norm kernel
│   ├── matmul.hpp               # Matrix multiplication kernel
│   └── transpose.hpp            # Matrix transpose kernel
├── bridge.hpp                   # view<W>(dp_obj) factory functions
├── kernel.hpp                   # Kernel<T,W,Rank> - memory layout descriptor
├── mask.hpp                     # mask<T,W> - comparison results
├── scalar.hpp                   # Scalar<T> wrapper (legacy, uses dp internally)
├── vector.hpp                   # Vector<T,N> wrapper (legacy, uses dp internally)
├── matrix.hpp                   # Matrix<T,R,C> wrapper (legacy, uses dp internally)
└── tensor.hpp                   # Tensor<T,Dims...> wrapper (legacy, uses dp internally)
```

### Implementation Status

#### Core Types - DONE
- [x] `pack<T,W>` - SIMD register abstraction
  - [x] Scalar fallback (primary template)
  - [x] SSE: `pack<float,4>`, `pack<double,2>`
  - [x] AVX: `pack<float,8>`, `pack<double,4>`
  - [ ] AVX-512: stub only
  - [ ] ARM NEON: stub only
- [x] `mask<T,W>` - Comparison results, blend/select
- [x] `Kernel<T,W,Rank>` - Memory layout descriptor

#### Views - DONE
- [x] `scalar_view<T,W>` - View over dp::mat::scalar
- [x] `vector_view<T,W>` - View over dp::mat::vector
- [x] `matrix_view<T,W>` - View over dp::mat::matrix
- [x] `tensor_view<T,W,Rank>` - View over dp::mat::tensor
- [x] `view<W>(dp_obj)` - Factory functions in bridge.hpp

#### Algorithms - DONE
- [x] `algo/elementwise.hpp` - axpy, scale, add, sub, mul, div, fill, copy
- [x] `algo/reduce.hpp` - sum, min, max, dot, norm
- [x] `algo/transform.hpp` - exp, log, sin, cos, tanh, sqrt (works with any packable view)
- [x] `algo/traits.hpp` - is_packable_view, view_value_t, view_width_v

#### SIMD Math - DONE (float only)
- [x] `exp.hpp` - 7.94x speedup vs scalar
- [x] `log.hpp` - 4.80x speedup vs scalar
- [x] `sin.hpp` - 22.94x speedup vs scalar
- [x] `cos.hpp` - 22.02x speedup vs scalar
- [x] `tanh.hpp` - 27.55x speedup vs scalar
- [x] `sqrt.hpp` - 4.03x speedup vs scalar

### Future SIMD Work (Lower Priority)

#### Double Precision Math
- [ ] `exp.hpp` - double precision
- [ ] `log.hpp` - double precision
- [ ] `sin.hpp` / `cos.hpp` - double precision
- [ ] `tanh.hpp` - double precision

#### Additional Math Functions
- [ ] `pow.hpp` - pow(x, y) = exp(y * log(x))
- [ ] `asin.hpp`, `acos.hpp`, `atan.hpp`, `atan2.hpp` - Inverse trig
- [ ] `sinh.hpp`, `cosh.hpp` - Remaining hyperbolic
- [ ] `asinh.hpp`, `acosh.hpp`, `atanh.hpp` - Inverse hyperbolic

#### Platform Extensions
- [ ] AVX-512 specializations (pack/avx512.hpp is a stub)
- [ ] ARM NEON specializations (pack/neon.hpp is a stub)
- [ ] Integer SIMD for NEON

---

## Module 2: Linear Algebra (`on::lina`) - COMPLETE

### Folder Structure

```
include/optinum/lina/
├── lina.hpp                     # Module header
├── basic/
│   ├── matmul.hpp               # Matrix multiplication
│   ├── transpose.hpp            # Matrix transpose
│   ├── inverse.hpp              # Matrix inverse
│   ├── determinant.hpp          # Determinant
│   └── norm.hpp                 # Frobenius, L2, etc.
├── decompose/
│   ├── lu.hpp                   # LU factorization
│   ├── qr.hpp                   # QR factorization (Householder)
│   ├── svd.hpp                  # Singular Value Decomposition
│   ├── cholesky.hpp             # Cholesky decomposition
│   └── eigen.hpp                # Eigendecomposition (power iteration)
├── solve/
│   ├── solve.hpp                # Solve Ax = b (LU-based)
│   └── lstsq.hpp                # Least squares (QR-based)
├── algebra/
│   ├── einsum.hpp               # Einstein summation (rank-1/2)
│   └── contraction.hpp          # Tensor contraction
└── expr/
    └── expr.hpp                 # Expression templates (lazy evaluation)
```

### Implementation Status - ALL DONE

#### Basic Operations
- [x] `matmul.hpp` - Matrix multiplication (M×K × K×N → M×N)
- [x] `transpose.hpp` - Matrix transpose
- [x] `inverse.hpp` - Matrix inverse (LU-based)
- [x] `determinant.hpp` - Determinant (LU-based)
- [x] `norm.hpp` - Frobenius norm, L2 norm

#### Decompositions
- [x] `lu.hpp` - LU factorization with partial pivoting
- [x] `qr.hpp` - QR factorization (Householder reflections)
- [x] `svd.hpp` - Singular Value Decomposition (one-sided Jacobi)
- [x] `cholesky.hpp` - Cholesky decomposition (for SPD matrices)
- [x] `eigen.hpp` - Eigendecomposition (power iteration for symmetric)

#### Solvers
- [x] `solve.hpp` - Solve Ax = b using LU decomposition
- [x] `lstsq.hpp` - Least squares using QR decomposition

#### Tensor Algebra
- [x] `einsum.hpp` - Einstein summation for rank-1/2 tensors
- [x] `contraction.hpp` - Tensor contraction

#### Expression Templates
- [x] `expr.hpp` - Lazy evaluation with CRTP

### Future lina/ Work

#### SIMD Acceleration
- [ ] Integrate SIMD views into lina operations (currently uses scalar loops)
- [ ] SIMD-accelerated matmul for small fixed-size matrices
- [ ] SIMD-accelerated decompositions

#### Rank-N Tensor Algebra
- [ ] Extend einsum beyond rank-2
- [ ] General tensor contractions for rank > 2
- [ ] Strided/network einsum

#### Additional Features
- [ ] Blocked/tiled algorithms for cache efficiency
- [ ] Complex number support
- [ ] Sparse matrix support (future)

---

## Module 3: Optimization (`on::opti`) - PLANNED

### Folder Structure (Planned)

```
include/optinum/opti/
├── opti.hpp                     # Module header
├── core/
│   ├── function.hpp             # Function wrapper with mixins
│   ├── traits.hpp               # Function type traits
│   ├── checks.hpp               # Static interface checks
│   └── log.hpp                  # Logging utilities
├── callback/
│   ├── callback.hpp             # Base callback infrastructure
│   ├── early_stop.hpp           # Stop when loss plateaus
│   ├── grad_clip.hpp            # Gradient clipping
│   ├── print.hpp                # Print loss each iteration
│   ├── progress.hpp             # Progress bar
│   └── timer.hpp                # Time-based stopping
├── gradient/
│   ├── gd.hpp                   # Gradient Descent
│   └── sgd.hpp                  # Stochastic GD + Momentum + Nesterov
├── adaptive/
│   ├── adam.hpp                 # Adam + AdaMax + AMSGrad + NAdam
│   ├── adagrad.hpp              # AdaGrad
│   ├── rmsprop.hpp              # RMSProp
│   └── lookahead.hpp            # Lookahead wrapper
├── quasi_newton/
│   └── lbfgs.hpp                # L-BFGS
├── evolutionary/
│   ├── cmaes.hpp                # CMA-ES
│   ├── de.hpp                   # Differential Evolution
│   ├── pso.hpp                  # Particle Swarm Optimization
│   └── sa.hpp                   # Simulated Annealing
├── schedule/
│   ├── schedule.hpp             # Module header
│   ├── cyclical.hpp             # Cyclical LR
│   └── warmup.hpp               # Warm restarts
└── problem/
    ├── sphere.hpp               # Sphere function (DONE - test function exists)
    ├── rosenbrock.hpp           # Rosenbrock function
    ├── rastrigin.hpp            # Rastrigin function
    └── ackley.hpp               # Ackley function
```

### Implementation Status

#### Done
- [x] `problem/sphere.hpp` - Sphere benchmark function

#### Phase 1: Core Infrastructure
- [ ] `core/function.hpp` - Function wrapper
- [ ] `core/traits.hpp` - Type traits for objective functions
- [ ] `callback/callback.hpp` - Callback system

#### Phase 2: First-Order Methods
- [ ] `gradient/gd.hpp` - Gradient Descent
- [ ] `gradient/sgd.hpp` - SGD with momentum
- [ ] `adaptive/adam.hpp` - Adam optimizer

#### Phase 3: Second-Order Methods
- [ ] `quasi_newton/lbfgs.hpp` - L-BFGS

#### Phase 4: Derivative-Free Methods
- [ ] `evolutionary/cmaes.hpp` - CMA-ES
- [ ] `evolutionary/de.hpp` - Differential Evolution
- [ ] `evolutionary/pso.hpp` - Particle Swarm

---

## Testing

All tests use **doctest**. Run with `make test`.

### Test Coverage

```
test/simd/
├── arch/arch_test.cpp           # Architecture detection
├── pack/
│   ├── pack_test.cpp            # pack<T,W> operations
│   └── mask_test.cpp            # mask<T,W> operations
├── view/view_test.cpp           # All view types
├── algo/
│   ├── algo_elementwise_test.cpp # axpy, scale, add, etc.
│   └── transform_test.cpp       # exp, log, sin, cos, tanh, sqrt
├── backend/
│   ├── elementwise_test.cpp     # Low-level SIMD elementwise
│   ├── reduce_test.cpp          # Low-level SIMD reductions
│   ├── dot_test.cpp             # Dot product
│   ├── norm_test.cpp            # L2 norm
│   ├── matmul_test.cpp          # Matrix multiplication
│   └── transpose_test.cpp       # Matrix transpose
├── bridge_test.cpp              # view<W>() factory
├── scalar_test.cpp              # Scalar<T> wrapper
├── vector_test.cpp              # Vector<T,N> wrapper
└── matrix_test.cpp              # Matrix<T,R,C> wrapper

test/lina/
├── lina_test.cpp                # Module-level tests
├── basic/
│   ├── lina_matmul_test.cpp     # matmul
│   ├── lina_transpose_test.cpp  # transpose
│   ├── determinant_test.cpp     # determinant
│   ├── inverse_test.cpp         # inverse
│   └── lina_norm_test.cpp       # norm
├── decompose/
│   ├── lu_test.cpp              # LU decomposition
│   ├── qr_test.cpp              # QR decomposition
│   ├── svd_test.cpp             # SVD
│   ├── cholesky_test.cpp        # Cholesky
│   └── eigen_test.cpp           # Eigendecomposition
├── solve/
│   ├── solve_test.cpp           # Ax = b solver
│   └── lstsq_test.cpp           # Least squares
├── algebra/
│   ├── einsum_test.cpp          # Einstein summation
│   └── contraction_test.cpp     # Tensor contraction
└── expr/expr_test.cpp           # Expression templates

test/opti/
└── problem/sphere_test.cpp      # Sphere function
```

**Current test count: 33 tests, all passing**

---

## Examples

```
examples/
├── scalar_usage.cpp             # Scalar<T> usage
├── vector_usage.cpp             # Vector<T,N> usage
├── matrix_usage.cpp             # Matrix<T,R,C> usage
├── simd_views_usage.cpp         # Vector/Matrix/Tensor views + algorithms
├── simd_math_benchmark.cpp      # SIMD math performance
├── math_benchmark_all.cpp       # Comprehensive math benchmarks
├── fast_math_benchmark_new.cpp  # Fast math benchmarks
└── sphere_optimization.cpp      # Optimization example
```

---

## Build & Test

```bash
make config    # Configure (preserves cache)
make build     # Build examples and tests
make test      # Run all tests (33 tests)
make clean     # Clean build artifacts
```

---

## Design Principles

1. **Header-only**: No compilation, just include
2. **Non-owning views**: `simd::view<W>()` over `dp::mat::*` - zero copy
3. **Zero-cost abstractions**: Expression templates, compile-time dimensions
4. **SIMD everywhere**: All math operations vectorized when possible
5. **Constexpr friendly**: Scalar fallback for compile-time evaluation
6. **POD-friendly**: Easy serialization via `datapod`
7. **Modern C++**: Requires C++20 (concepts, constexpr, fold expressions)

---

## Dependencies

- **C++20** or later
- **datapod** library (fetched automatically via CMake/xmake)
- **Optional**: AVX2/AVX-512 for best SIMD performance

---

## References

- Cephes library: https://www.netlib.org/cephes/
- Intel Intrinsics Guide: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
- Fastor: https://github.com/romeric/Fastor
- ensmallen: https://github.com/mlpack/ensmallen
