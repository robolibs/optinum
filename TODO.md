# OPTINUM - Optimization & Numerics Library

> **HEADER-ONLY LIBRARY** - No compilation required, just `#include <optinum/optinum.hpp>`

---

## Quick Status

| Module | Status | Description |
|--------|--------|-------------|
| `simd/` | **IN PROGRESS** | Non-owning views, pack<T,W>, SIMD math, algorithms |
| `lina/` | **IN PROGRESS** | Linear algebra (matmul, decompose, solve, einsum) |
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
│   ├── avx512.hpp               # AVX-512 (PLANNED - currently stub)
│   ├── neon.hpp                 # ARM NEON (PLANNED - currently stub)
│   └── complex.hpp              # Complex number packs (PLANNED)
├── view/
│   ├── view.hpp                 # Module header
│   ├── scalar_view.hpp          # Non-owning view over dp::mat::scalar
│   ├── vector_view.hpp          # Non-owning view over dp::mat::vector
│   ├── matrix_view.hpp          # Non-owning view over dp::mat::matrix
│   ├── tensor_view.hpp          # Non-owning view over dp::mat::tensor
│   ├── slice.hpp                # seq(), fseq<>(), all, fix<>() (PLANNED)
│   ├── diagonal_view.hpp        # Diagonal view (PLANNED)
│   └── filter_view.hpp          # Masked/filtered view (PLANNED)
├── algo/
│   ├── traits.hpp               # is_packable_view, view_value_t, etc.
│   ├── elementwise.hpp          # axpy, scale, add, sub, mul, div, fill, copy
│   ├── reduce.hpp               # sum, min, max, dot, norm
│   └── transform.hpp            # exp, log, sin, cos, tanh, sqrt (+ more PLANNED)
├── math/
│   ├── simd_math.hpp            # Module header
│   ├── detail/constants.hpp     # Math constants (LN2, PI, coefficients)
│   ├── exp.hpp                  # exp() for pack<float,W> (double PLANNED)
│   ├── log.hpp                  # log() for pack<float,W> (double PLANNED)
│   ├── sin.hpp                  # sin() for pack<float,W> (double PLANNED)
│   ├── cos.hpp                  # cos() for pack<float,W> (double PLANNED)
│   ├── tan.hpp                  # tan() (PLANNED)
│   ├── tanh.hpp                 # tanh() for pack<float,W> (double PLANNED)
│   ├── sqrt.hpp                 # sqrt() for pack<float,W>
│   ├── pow.hpp                  # pow() (PLANNED)
│   ├── asin.hpp                 # asin() (PLANNED)
│   ├── acos.hpp                 # acos() (PLANNED)
│   ├── atan.hpp                 # atan() (PLANNED)
│   ├── atan2.hpp                # atan2() (PLANNED)
│   ├── sinh.hpp                 # sinh() (PLANNED)
│   ├── cosh.hpp                 # cosh() (PLANNED)
│   ├── asinh.hpp                # asinh() (PLANNED)
│   ├── acosh.hpp                # acosh() (PLANNED)
│   ├── atanh.hpp                # atanh() (PLANNED)
│   ├── exp2.hpp                 # exp2() (PLANNED)
│   ├── expm1.hpp                # expm1() (PLANNED)
│   ├── log2.hpp                 # log2() (PLANNED)
│   ├── log10.hpp                # log10() (PLANNED)
│   ├── log1p.hpp                # log1p() (PLANNED)
│   ├── cbrt.hpp                 # cbrt() (PLANNED)
│   ├── hypot.hpp                # hypot() (PLANNED)
│   ├── ceil.hpp                 # ceil() (PLANNED)
│   ├── floor.hpp                # floor() (PLANNED)
│   ├── round.hpp                # round() (PLANNED)
│   ├── trunc.hpp                # trunc() (PLANNED)
│   ├── erf.hpp                  # erf() (PLANNED)
│   ├── tgamma.hpp               # tgamma() (PLANNED)
│   └── lgamma.hpp               # lgamma() (PLANNED)
├── backend/
│   ├── backend.hpp              # Module header
│   ├── elementwise.hpp          # Low-level SIMD elementwise ops
│   ├── reduce.hpp               # Low-level SIMD reductions
│   ├── dot.hpp                  # Dot product kernel
│   ├── norm.hpp                 # L2 norm kernel
│   ├── matmul.hpp               # Matrix multiplication kernel
│   ├── transpose.hpp            # Matrix transpose kernel
│   ├── inverse_small.hpp        # Specialized 2x2, 3x3, 4x4 inverse (PLANNED)
│   ├── det_small.hpp            # Specialized 2x2, 3x3, 4x4 determinant (PLANNED)
│   └── gather_scatter.hpp       # Gather/scatter operations (PLANNED)
├── bridge.hpp                   # view<W>(dp_obj) factory functions
├── kernel.hpp                   # Kernel<T,W,Rank> - memory layout descriptor
├── mask.hpp                     # mask<T,W> - comparison results
├── io.hpp                       # operator<<, print(), write() (PLANNED)
├── debug.hpp                    # Bounds/shape checking (PLANNED)
├── scalar.hpp                   # Scalar<T> wrapper (uses dp internally)
├── vector.hpp                   # Vector<T,N> wrapper (uses dp internally)
├── matrix.hpp                   # Matrix<T,R,C> wrapper (uses dp internally)
└── tensor.hpp                   # Tensor<T,Dims...> wrapper (uses dp internally)
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

#### Views - DONE (Core)
- [x] `scalar_view<T,W>` - View over dp::mat::scalar
- [x] `vector_view<T,W>` - View over dp::mat::vector
- [x] `matrix_view<T,W>` - View over dp::mat::matrix
- [x] `tensor_view<T,W,Rank>` - View over dp::mat::tensor
- [x] `view<W>(dp_obj)` - Factory functions in bridge.hpp

#### Views - Missing Slicing/Indexing Features
- [ ] `seq(start, end)` - Runtime slicing
- [ ] `fseq<start, end>()` - Compile-time fixed slicing
- [ ] `fseq<start, end, step>()` - Compile-time strided slicing
- [ ] `all` - Select all elements in dimension
- [ ] `fix<I>()` - Fixed index (reduce dimension)
- [ ] Diagonal view - View over matrix diagonal
- [ ] Filter view - Masked/conditional view
- [ ] Random access view - Non-contiguous element access
- [ ] Block/submatrix views with arbitrary strides

#### Algorithms - DONE (Core)
- [x] `algo/elementwise.hpp` - axpy, scale, add, sub, mul, div, fill, copy
- [x] `algo/reduce.hpp` - sum, min, max, dot, norm
- [x] `algo/transform.hpp` - exp, log, sin, cos, tanh, sqrt (works with any packable view)
- [x] `algo/traits.hpp` - is_packable_view, view_value_t, view_width_v

#### Algorithms - Missing Transform Functions
- [ ] `tan(view)` - Tangent
- [ ] `asin(view)`, `acos(view)`, `atan(view)`, `atan2(view)` - Inverse trig
- [ ] `sinh(view)`, `cosh(view)` - Hyperbolic
- [ ] `asinh(view)`, `acosh(view)`, `atanh(view)` - Inverse hyperbolic
- [ ] `pow(view, exp)` - Power
- [ ] `ceil(view)`, `floor(view)`, `round(view)`, `trunc(view)` - Rounding
- [ ] `abs(view)` - Absolute value
- [ ] `clamp(view, lo, hi)` - Clamp to range

#### SIMD Math - DONE (float only)
- [x] `exp.hpp` - 7.94x speedup vs scalar
- [x] `log.hpp` - 4.80x speedup vs scalar
- [x] `sin.hpp` - 22.94x speedup vs scalar
- [x] `cos.hpp` - 22.02x speedup vs scalar
- [x] `tanh.hpp` - 27.55x speedup vs scalar
- [x] `sqrt.hpp` - 4.03x speedup vs scalar

#### Wrapper Types (Scalar, Vector, Matrix, Tensor) - Missing Features
- [ ] `fill(value)` - Fill all elements with value
- [ ] `iota(start)` / `arange(start)` - Fill with sequential values
- [ ] `zeros()` - Fill with zeros (static factory)
- [ ] `ones()` - Fill with ones (static factory)
- [ ] `random()` - Fill with random values [0, 1)
- [ ] `randint(lo, hi)` - Fill with random integers
- [ ] `reverse()` - Reverse element order
- [ ] `cast<U>()` - Type conversion
- [ ] `noalias()` - Hint for no aliasing (optimization)
- [ ] `squeeze()` - Remove dimensions of size 1
- [ ] `reshape<Dims...>()` - Reshape tensor
- [ ] `flatten()` - Flatten to 1D vector
- [ ] `tocolumnmajor()` / `torowmajor()` - Layout conversion
- [ ] Voigt notation conversion for mechanics

#### I/O and Debugging - Missing
- [ ] `operator<<` - Stream output for all types
- [ ] `print()` - Formatted printing
- [ ] `write(filename)` - Write to file
- [ ] Timing utilities for benchmarking

#### Debug Mode Features - Missing
- [ ] `OPTINUM_BOUNDS_CHECK` - Enable bounds checking
- [ ] `OPTINUM_SHAPE_CHECK` - Enable shape compatibility checking
- [ ] `OPTINUM_ENABLE_RUNTIME_CHECKS` - Master switch for all checks
- [ ] Assertion macros with informative messages

### Missing SIMD Features (High Priority)

#### Platform Extensions - CRITICAL
- [ ] AVX-512 full implementation (pack/avx512.hpp is a stub)
  - [ ] `pack<float,16>` - 16x32-bit float
  - [ ] `pack<double,8>` - 8x64-bit double
  - [ ] `pack<int32_t,16>` - 16x32-bit int
  - [ ] `pack<int64_t,8>` - 8x64-bit int
  - [ ] All arithmetic, comparison, reduction operations
  - [ ] Mask operations with AVX-512 mask registers
- [ ] ARM NEON full implementation (pack/neon.hpp is a stub)
  - [ ] `pack<float,4>` - 4x32-bit float (float32x4_t)
  - [ ] `pack<double,2>` - 2x64-bit double (float64x2_t, ARM64 only)
  - [ ] `pack<int32_t,4>` - 4x32-bit int (int32x4_t)
  - [ ] `pack<int64_t,2>` - 2x64-bit int (int64x2_t)
  - [ ] All arithmetic, comparison, reduction operations

#### Pack Operations - Missing
- [ ] `set(values...)` - Set individual lane values
- [ ] `set_sequential(start)` - Fill with sequential values (start, start+1, ...)
- [ ] `reverse()` - Reverse lane order
- [ ] `shift(i)` - Shift lanes left/right
- [ ] `cast<U>()` - Type conversion between pack types
- [ ] `rotate(i)` - Rotate lanes
- [ ] Gather/Scatter operations
  - [ ] `gather(base_ptr, index_pack)` - Gather from non-contiguous memory
  - [ ] `scatter(base_ptr, index_pack, values)` - Scatter to non-contiguous memory

#### Complex Number Support
- [ ] `pack<std::complex<float>, W>` - Complex float SIMD
- [ ] `pack<std::complex<double>, W>` - Complex double SIMD
- [ ] `real()`, `imag()` - Extract real/imaginary parts
- [ ] `conj()` - Complex conjugate
- [ ] `magnitude()`, `arg()` - Polar form operations

#### Double Precision Math - CRITICAL
- [ ] `exp.hpp` - double precision (exp for pack<double,2/4>)
- [ ] `log.hpp` - double precision
- [ ] `sin.hpp` / `cos.hpp` - double precision
- [ ] `tanh.hpp` - double precision
- [ ] `sqrt.hpp` - double precision (Newton-Raphson refinement)

#### Additional Math Functions - HIGH PRIORITY
- [ ] `tan.hpp` - Tangent (sin/cos based)
- [ ] `asin.hpp` - Arc sine
- [ ] `acos.hpp` - Arc cosine
- [ ] `atan.hpp` - Arc tangent
- [ ] `atan2.hpp` - Two-argument arc tangent
- [ ] `pow.hpp` - Power function pow(x, y) = exp(y * log(x))
- [ ] `sinh.hpp` - Hyperbolic sine
- [ ] `cosh.hpp` - Hyperbolic cosine
- [ ] `asinh.hpp` - Inverse hyperbolic sine
- [ ] `acosh.hpp` - Inverse hyperbolic cosine
- [ ] `atanh.hpp` - Inverse hyperbolic tangent

#### Additional Math Functions - MEDIUM PRIORITY
- [ ] `exp2.hpp` - Base-2 exponential
- [ ] `expm1.hpp` - exp(x) - 1 (accurate for small x)
- [ ] `log2.hpp` - Base-2 logarithm
- [ ] `log10.hpp` - Base-10 logarithm
- [ ] `log1p.hpp` - log(1 + x) (accurate for small x)
- [ ] `cbrt.hpp` - Cube root
- [ ] `hypot.hpp` - Hypotenuse sqrt(x^2 + y^2) without overflow

#### Rounding Functions
- [ ] `ceil.hpp` - Ceiling (round up)
- [ ] `floor.hpp` - Floor (round down)
- [ ] `round.hpp` - Round to nearest
- [ ] `trunc.hpp` - Truncate toward zero

#### Special Functions
- [ ] `erf.hpp` - Error function
- [ ] `tgamma.hpp` - Gamma function
- [ ] `lgamma.hpp` - Log gamma function

#### Boolean/Status Functions
- [ ] `isinf(pack)` - Test for infinity
- [ ] `isnan(pack)` - Test for NaN
- [ ] `isfinite(pack)` - Test for finite values

#### Extended Intrinsics / Helpers
- [ ] Element extraction helpers - `get<I>(pack)` compile-time lane access
- [ ] Register reverse intrinsics
- [ ] Complex arrangement functions for interleaved data

---

## Module 2: Linear Algebra (`on::lina`) - IN PROGRESS

### Folder Structure

```
include/optinum/lina/
├── lina.hpp                     # Module header
├── basic/
│   ├── matmul.hpp               # Matrix multiplication
│   ├── transpose.hpp            # Matrix transpose
│   ├── inverse.hpp              # Matrix inverse
│   ├── determinant.hpp          # Determinant
│   ├── adjoint.hpp              # Adjoint/Adjugate matrix (PLANNED)
│   ├── cofactor.hpp             # Cofactor matrix (PLANNED)
│   ├── trace.hpp                # Matrix trace (PLANNED - currently inline)
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

### Implementation Status - CORE DONE

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

### SIMD Integration - DONE

The following operations now use SIMD backend for acceleration:

| Operation | File | SIMD Usage |
|-----------|------|------------|
| `inner` (Frobenius) | `contraction.hpp` | `backend::dot` for flattened matrix |
| `hadamard` | `contraction.hpp` | `backend::mul` for elementwise |
| `outer` | `contraction.hpp` | `backend::mul_scalar` per column |
| LU solve | `lu.hpp` | SIMD dot for forward/back substitution |
| QR decomposition | `qr.hpp` | SIMD dot/axpy for Householder ops |

Note: Some operations remain scalar due to strided memory access patterns:
- Row operations in column-major matrices (LU elimination, QR right-multiply)
- These would require gather/scatter which may not be faster for small matrices

### Missing lina/ Features

#### Basic Operations - Missing
- [ ] `adjoint.hpp` - Adjoint/Adjugate matrix (transpose of cofactor matrix)
- [ ] `cofactor.hpp` - Cofactor matrix
- [ ] `trace.hpp` - Dedicated trace function (SIMD optimized)
- [ ] Specialized 2x2, 3x3, 4x4 SIMD kernels for determinant
- [ ] Specialized 2x2, 3x3, 4x4 SIMD kernels for inverse (direct formulas)
- [ ] Double contraction A:B (Frobenius inner product for tensors)
- [ ] Tensor cross product

#### Tensor Algebra - Missing
- [ ] Extend einsum beyond rank-2 (arbitrary rank tensors)
- [ ] General tensor contractions for rank > 2
- [ ] Network einsum (multi-tensor contraction with optimization)
- [ ] Compile-time contraction order optimization
- [ ] `Index<i,j,k>` type system for compile-time index specification
- [ ] Voigt notation conversion for mechanics tensors
- [ ] Cyclic contractions

#### Expression Templates - Missing
- [ ] Subtraction expression `MatSub<L, R>`
- [ ] Multiplication expression `MatMul<L, R>` (lazy matmul)
- [ ] Unary math expressions (sin, cos, exp, log on matrices)
- [ ] Transpose expression (lazy transpose)
- [ ] Determinant expression (lazy determinant)
- [ ] Inverse expression (lazy inverse)
- [ ] Solve expression (lazy linear solve)
- [ ] SVD/LU/QR expressions (lazy decompositions)
- [ ] Norm expression (lazy norm computation)

#### Additional Features - Missing
- [ ] Blocked/tiled algorithms for cache efficiency
- [ ] Complex number support (complex<T> matrices)
- [ ] Sparse matrix support (CSR, CSC, COO formats)
- [ ] SIMD gather/scatter for strided row operations
- [ ] BLAS/MKL backend switching for large matrices
- [ ] libXSMM integration for small matrix optimization
- [ ] Block size tuning macros

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
7. **Use datapod types**: Prefer `dp::` types over `std::` equivalents (see below)
8. **Modern C++**: Requires C++20 (concepts, constexpr, fold expressions)

---

## Datapod Type Usage

**In `lina::` and `opti::` modules, always prefer datapod types over std equivalents:**

### Error Handling
| Use | Instead of |
|-----|------------|
| `dp::Result<T, dp::Error>` | exceptions, error codes |
| `dp::Res<T>` | (alias for `Result<T, Error>`) |
| `dp::Optional<T>` | `std::optional<T>` |
| `dp::Error` | custom error types |

### Containers
| Use | Instead of |
|-----|------------|
| `dp::Vector<T>` | `std::vector<T>` |
| `dp::Array<T, N>` | `std::array<T, N>` |
| `dp::String` | `std::string` |
| `dp::Pair<K, V>` | `std::pair<K, V>` |
| `dp::Tuple<Ts...>` | `std::tuple<Ts...>` |
| `dp::Variant<Ts...>` | `std::variant<Ts...>` |

### Matrix Types (for data ownership)
| Use | Description |
|-----|-------------|
| `dp::mat::scalar<T>` | Rank-0 (single value) |
| `dp::mat::vector<T, N>` | Rank-1 (1D array) |
| `dp::mat::matrix<T, R, C>` | Rank-2 (2D, column-major) |
| `dp::mat::tensor<T, Dims...>` | Rank-N (N-dimensional) |

### Error Factory Methods
```cpp
dp::Error::ok()                    // Success (code 0)
dp::Error::invalid_argument(msg)   // Invalid input
dp::Error::out_of_range(msg)       // Index out of bounds
dp::Error::not_found(msg)          // Item not found
// ... and more
```

### Usage Pattern
```cpp
// Failable operations return dp::Result
dp::Result<dp::mat::vector<T, N>, dp::Error> solve(const Matrix& A, const Vector& b) {
    if (is_singular(A)) {
        return dp::Result<...>::err(dp::Error::invalid_argument("matrix is singular"));
    }
    // ... compute solution ...
    return dp::Result<...>::ok(solution);
}

// Caller handles result
auto result = solve(A, b);
if (result.is_ok()) {
    auto x = result.unwrap();
} else {
    auto err = result.unwrap_err();
    // handle error
}
```

---

## Dependencies

- **C++20** or later
- **datapod** library (fetched automatically via CMake/xmake)
- **Optional**: AVX2/AVX-512 for best SIMD performance

---

---

## Feature Gap Summary (vs Fastor)

This section tracks features present in Fastor that are missing in optinum.

### Priority Levels

| Priority | Description |
|----------|-------------|
| **P0** | Critical - Core functionality gaps |
| **P1** | High - Important for usability |
| **P2** | Medium - Nice to have |
| **P3** | Low - Optional/advanced |

### P0 - Critical (Core Functionality)

| Feature | Category | Status |
|---------|----------|--------|
| AVX-512 full implementation | SIMD | Stub only |
| ARM NEON full implementation | SIMD | Stub only |
| Double precision math (exp, log, sin, cos, tanh) | SIMD Math | Missing |
| `tan()` function | SIMD Math | Missing |
| `asin()`, `acos()`, `atan()`, `atan2()` | SIMD Math | Missing |
| `pow()` function | SIMD Math | Missing |
| `ceil()`, `floor()`, `round()`, `trunc()` | SIMD Math | Missing |
| Specialized 2x2, 3x3, 4x4 inverse kernels | Backend | Missing |
| Gather/Scatter operations | SIMD | Missing |

### P1 - High (Usability)

| Feature | Category | Status |
|---------|----------|--------|
| `sinh()`, `cosh()` | SIMD Math | Missing |
| `asinh()`, `acosh()`, `atanh()` | SIMD Math | Missing |
| `exp2()`, `log2()`, `log10()` | SIMD Math | Missing |
| `expm1()`, `log1p()` | SIMD Math | Missing |
| `isinf()`, `isnan()`, `isfinite()` | SIMD Math | Missing |
| `zeros()`, `ones()`, `iota()` factories | Tensor | Missing |
| `random()`, `randint()` factories | Tensor | Missing |
| `reshape()`, `flatten()`, `squeeze()` | Tensor | Missing |
| View slicing (`seq()`, `fseq()`, `all`) | Views | Missing |
| Stream output `operator<<` | I/O | Missing |
| Debug bounds/shape checking | Debug | Missing |
| Adjoint/Adjugate matrix | LinAlg | Missing |
| Cofactor matrix | LinAlg | Missing |

### P2 - Medium (Nice to Have)

| Feature | Category | Status |
|---------|----------|--------|
| Complex number support | SIMD | Missing |
| `cbrt()` cube root | SIMD Math | Missing |
| `hypot()` | SIMD Math | Missing |
| `erf()` error function | SIMD Math | Missing |
| `tgamma()`, `lgamma()` | SIMD Math | Missing |
| Pack `set()`, `set_sequential()` | SIMD | Missing |
| Pack `reverse()`, `shift()`, `rotate()` | SIMD | Missing |
| Pack `cast<U>()` | SIMD | Missing |
| Diagonal view | Views | Missing |
| Filter view | Views | Missing |
| Network einsum | Tensor | Missing |
| More expression template ops | Expr | Missing |
| Double contraction | LinAlg | Missing |
| Tensor cross product | LinAlg | Missing |
| Voigt notation | LinAlg | Missing |

### P3 - Low (Advanced/Optional)

| Feature | Category | Status |
|---------|----------|--------|
| Intel MIC support | SIMD | Not planned |
| SLEEF backend | SIMD Math | Optional |
| libXSMM backend | Backend | Optional |
| Intel MKL backend | Backend | Optional |
| BLAS size threshold switching | Backend | Optional |
| Block size tuning macros | Backend | Optional |
| Continuum mechanics tags | LinAlg | Not needed |
| Plane strain/stress modes | LinAlg | Not needed |
| Sparse matrix support | LinAlg | Future |

### What Optinum Does Better Than Fastor

| Feature | Description |
|---------|-------------|
| Modern C++20 | Concepts, constexpr, cleaner syntax |
| LU with pivoting | Numerically stable (Fastor has no pivot) |
| QR decomposition | Fastor lacks this |
| Cholesky decomposition | Fastor lacks this |
| Eigendecomposition | Fastor lacks this (for symmetric matrices) |
| Result<T, Error> | Safe error handling (Fastor uses exceptions) |
| datapod integration | Clean ownership model |
| Focused scope | Less bloat, easier to maintain |
| Real-time friendly | No dynamic allocation in hot paths |

---

## References

- Cephes library: https://www.netlib.org/cephes/
- Intel Intrinsics Guide: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
- Fastor: https://github.com/romeric/Fastor
- ensmallen: https://github.com/mlpack/ensmallen
- SLEEF: https://sleef.org/ (vectorized math library)
- libXSMM: https://github.com/libxsmm/libxsmm (small matrix library)
