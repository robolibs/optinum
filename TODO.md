# OPTINUM - Optimization & Numerics Library

> **HEADER-ONLY LIBRARY** - No compilation required, just `#include <optinum/optinum.hpp>`

---

## Native SIMD Math Implementation (Complete)

### Overview
High-performance vectorized math functions with no external dependencies.
Accuracy: ~3-5 ULP (good for ML/graphics, not scientific computing).

### Benchmark Results (Current)
```
| Function  | SLEEF Speedup | Our fast_* Speedup | Our Advantage |
|-----------|---------------|---------------------|---------------|
| exp       | 3.65x         | 6.33x               | 1.7x faster   |
| log       | 1.85x         | 4.32x               | 2.3x faster   |
| sin       | 8.89x         | 16.93x              | 1.9x faster   |
| cos       | 7.48x         | 15.81x              | 2.1x faster   |
| tanh      | 7.28x         | 27.16x              | 3.7x faster   |
| pow       | 1.07x         | 2.90x               | 2.7x faster   |
| sqrt      | N/A           | 3.48x               | Native SIMD   |
```

### Implementation Plan

#### Phase A: Core Functions (Priority Order)

| Function | Difficulty | Algorithm | Notes |
|----------|------------|-----------|-------|
| **exp** | Easy | Range reduction + polynomial | Already done: `fast_exp.hpp` |
| **log** | Easy | Range reduction + polynomial | Use `log(x) = log(2^n * m) = n*ln2 + log(m)` |
| **sin/cos** | Medium | Payne-Hanek reduction + polynomial | Reduce to [-pi/4, pi/4], use Taylor/Chebyshev |
| **tan** | Medium | `sin/cos` ratio | After sin/cos work |
| **tanh** | Easy | `(exp(2x)-1)/(exp(2x)+1)` or polynomial | Can use fast_exp |
| **pow** | Medium | `exp(y * log(x))` | Compose exp + log |
| **sqrt** | Easy | Newton-Raphson + `_mm256_rsqrt_ps` | Hardware has rsqrt seed |
| **cbrt** | Medium | Newton-Raphson | Similar to sqrt |

#### Phase B: Inverse Trig (Lower Priority)
| Function | Difficulty | Notes |
|----------|------------|-------|
| asin | Medium | Polynomial on reduced range |
| acos | Easy | `pi/2 - asin(x)` |
| atan | Medium | Polynomial + range reduction |
| atan2 | Medium | Quadrant handling + atan |

#### Phase C: Hyperbolic (Use exp-based)
| Function | Algorithm |
|----------|-----------|
| sinh | `(exp(x) - exp(-x)) / 2` |
| cosh | `(exp(x) + exp(-x)) / 2` |
| asinh | `log(x + sqrt(x^2 + 1))` |
| acosh | `log(x + sqrt(x^2 - 1))` |
| atanh | `0.5 * log((1+x)/(1-x))` |

### File Structure
```
include/optinum/simd/math/
├── simd_math.hpp          # Public API (includes all headers)
├── detail/
│   ├── map.hpp            # Scalar fallback (existing)
│   └── constants.hpp      # DONE: Math constants (LN2, PI, coefficients)
├── fast_exp.hpp           # DONE: exp for float (AVX + SSE4.1)
├── fast_log.hpp           # DONE: log for float (AVX + SSE4.1)
├── fast_trig.hpp          # DONE: sin, cos for float (AVX + SSE4.1)
├── fast_hyp.hpp           # DONE: tanh, sinh, cosh (AVX + SSE4.1)
├── fast_pow.hpp           # DONE: pow, sqrt, rsqrt, cbrt, powi (AVX + SSE4.1)
├── fast_inv_trig.hpp      # TODO: asin, acos, atan, atan2
├── exponential.hpp        # Scalar fallback (template)
├── trig.hpp               # Scalar fallback (template)
├── hyperbolic.hpp         # Scalar fallback (template)
└── pow.hpp                # Scalar fallback (template)
```

### Algorithm Details

#### exp(x) - DONE
```
1. Clamp x to [-88, 88] (avoid overflow/underflow)
2. n = round(x / ln2)           -- integer part
3. r = x - n * ln2              -- fractional part in [-ln2/2, ln2/2]  
4. exp(r) ≈ 1 + r + r²/2 + r³/6 + r⁴/24 + r⁵/120  (polynomial)
5. result = exp(r) * 2^n        -- scale by power of 2 (bit manipulation)
```

#### log(x) - TODO
```
1. Extract exponent: x = 2^n * m, where m in [1, 2)
2. Normalize: m' = (m - 1) / (m + 1), maps to [-1/3, 1/3]
3. log(m) ≈ 2 * (m' + m'^3/3 + m'^5/5 + ...)  (polynomial in m'^2)
4. result = n * ln2 + log(m)
```

#### sin(x) / cos(x) - TODO  
```
1. Range reduction: x' = x mod 2π, then to [-π, π]
2. Further reduce to [-π/4, π/4] using symmetry:
   - sin(x) = cos(π/2 - x)
   - sin(x + π) = -sin(x)
   - cos(x + π/2) = -sin(x)
3. Polynomial approximation (minimax coefficients):
   sin(x) ≈ x - x³/6 + x⁵/120 - x⁷/5040 + ...
   cos(x) ≈ 1 - x²/2 + x⁴/24 - x⁶/720 + ...
```

#### tanh(x) - TODO (Easy using exp)
```
For |x| < 0.625:
  tanh(x) ≈ x * (1 - x²/3 + 2x⁴/15 - ...)  (polynomial)
For |x| >= 0.625:
  tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
  Or: tanh(x) = 1 - 2/(exp(2x) + 1)  (more stable for large x)
```

### Implementation Checklist

- [x] `fast_exp.hpp` - exp for float (AVX + SSE4.1)
- [x] `detail/constants.hpp` - Centralize all magic numbers
- [x] `fast_log.hpp` - log for float (AVX + SSE4.1)
- [x] `fast_trig.hpp` - sin, cos for float (AVX + SSE4.1)
- [x] `fast_hyp.hpp` - tanh, sinh, cosh (using fast_exp)
- [x] `fast_pow.hpp` - pow, sqrt, rsqrt, cbrt, powi
- [ ] Add double precision to fast_* functions
- [ ] `detail/poly.hpp` - Horner's method, Estrin's method helpers
- [ ] Add accuracy tests (compare against std:: with tolerance)
- [ ] Add AVX-512 variants
- [ ] Add NEON (ARM) variants
- [ ] Implement `fast_inv_trig.hpp` (asin, acos, atan, atan2)

### Testing Strategy
```cpp
// For each function, test:
// 1. Accuracy: |fast_f(x) - std::f(x)| < tolerance (e.g., 1e-5 for float)
// 2. Edge cases: 0, ±inf, NaN, denormals
// 3. Range boundaries: overflow/underflow thresholds
// 4. Performance: benchmark vs scalar
```

### References
- Cephes library: https://www.netlib.org/cephes/
- "Elementary Functions" by Muller (textbook)
- Intel Intrinsics Guide: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/

---

## Overview

**optinum** is a high-performance C++ header-only library combining:
- **SIMD-accelerated tensor operations** (inspired by [Fastor](https://github.com/romeric/Fastor))
- **Numerical optimization algorithms** (inspired by [ensmallen](https://github.com/mlpack/ensmallen))
- **POD data storage** via [datapod](../datapod) (`dp::` namespace)

```cpp
namespace optinum { ... }
namespace on = optinum;  // alias
```

---

## Dependencies

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Code                                   │
│                                                                     │
│   #include <datapod/matrix.hpp>                                     │
│   #include <optinum/optinum.hpp>                                    │
│                                                                     │
│   dp::mat::matrix<float, 4, 4> A, B, C;                             │
│   on::lina::matmul(A, B, C);              // C = A * B              │
│   // OR with explicit SIMD views:                                   │
│   auto vA = on::simd::view<8>(A);                                   │
│   auto vB = on::simd::view<8>(B);                                   │
│   auto vC = on::simd::view<8>(C);                                   │
│   on::lina::matmul(vA, vB, vC);                                     │
│                                                                     │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          optinum                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │
│  │   on::opti   │─▶│   on::lina   │─▶│        on::simd          │   │
│  │ (optimizers) │  │ (linear alg) │  │  (views + algorithms)    │   │
│  └──────────────┘  └──────────────┘  └────────────┬─────────────┘   │
│                                                   │                 │
│                                          ┌────────┴────────┐        │
│                                          │  simd::view<W>  │        │
│                                          │  (non-owning)   │        │
│                                          └────────┬────────┘        │
└───────────────────────────────────────────────────┼─────────────────┘
                                                    │ points to
                                                    ▼
                              ┌─────────────────────────────────────┐
                              │            datapod                  │
                              │  dp::mat::vector, matrix, tensor    │
                              │  (owns memory, POD, serializable)   │
                              └─────────────────────────────────────┘
```

---

## Namespace Structure

```
datapod (dp)                         # DATA OWNERSHIP (external library)
├── mat::scalar<T>                   # rank-0 (single value)
├── mat::vector<T, N>                # rank-1 (1D array, aligned)
├── mat::matrix<T, R, C>             # rank-2 (2D array, column-major, aligned)
└── mat::tensor<T, Dims...>          # rank-N (N-D array)

optinum (on)                         # SIMD OPERATIONS (this library)
├── simd        # Non-owning SIMD views + algorithms
│   ├── pack<T, W>                   # SIMD register abstraction (W lanes)
│   ├── mask<T, W>                   # comparison results, blend/select
│   ├── Kernel<T, W, Rank>           # ptr + extents + strides + load/store
│   ├── scalar_view<T, W>            # view over dp::mat::scalar
│   ├── vector_view<T, W>            # view over dp::mat::vector
│   ├── matrix_view<T, W>            # view over dp::mat::matrix
│   ├── tensor_view<T, W, Rank>      # view over dp::mat::tensor
│   ├── view<W>(dp_obj)              # factory: dp type -> simd view
│   ├── algo::axpy, dot, norm, ...   # algorithms on views
│   ├── math::exp, sin, cos, ...     # vectorized math
│   └── arch/                        # platform detection
│
├── lina        # Linear algebra operations (operate on dp types via views)
│   ├── matmul, transpose, inverse   # matrix operations
│   ├── lu, qr, svd, cholesky        # decompositions
│   ├── solve, lstsq                 # linear solvers
│   ├── einsum, contraction          # tensor algebra
│   └── norm, dot, cross             # vector operations
│
└── opti        # Numerical optimization
    ├── GradientDescent, SGD, Adam...
    ├── LBFGS, CMA-ES, PSO...
    └── callbacks, schedulers...
```

---

## Architecture Graph

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              on::opti                                        │
│                         (Optimization Layer)                                 │
│                                                                              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────────────────────┐  │
│  │  gradient/  │ │  adaptive/  │ │quasi_newton/│ │     evolutionary/      │  │
│  │  gd, sgd    │ │ adam, rmsp  │ │   lbfgs     │ │  cmaes, de, pso, sa    │  │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └───────────┬────────────┘  │
│         │               │               │                    │               │
│  ┌──────┴───────────────┴───────────────┴────────────────────┴─────────┐     │
│  │                            core/                                    │     │
│  │            function, traits, callbacks, schedule, search            │     │
│  └─────────────────────────────────┬───────────────────────────────────┘     │
└────────────────────────────────────┼─────────────────────────────────────────┘
                                     │
                                     │ uses
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              on::lina                                        │
│                       (Linear Algebra Layer)                                 │
│                                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   basic/    │  │  decompose/ │  │   solve/    │  │      algebra/       │  │
│  │matmul, trans│  │ lu, qr, svd │  │ solve, lstsq│  │ einsum, contraction │  │
│  │ inv, det    │  │cholesky, eig│  │             │  │ inner, outer, perm  │  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
│         │                │               │                     │             │
│  ┌──────┴────────────────┴───────────────┴─────────────────────┴───────────┐ │
│  │                            expr/                                        │ │
│  │              Expression templates, lazy evaluation, views               │ │
│  └─────────────────────────────────┬───────────────────────────────────────┘ │
└────────────────────────────────────┼─────────────────────────────────────────┘
                                     │
                                     │ operates on
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              on::simd                                        │
│                      (SIMD Types + Primitives)                               │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                         User-facing Types                            │    │
│  │    Scalar<T>           Vector<T, N>           Matrix<T, R, C>        │    │
│  │      (rank-0)            (rank-1)                (rank-2)            │    │
│  └──────────────────────────────────┬───────────────────────────────────┘    │
│                                     │                                        │
│  ┌──────────────────────────────────┴───────────────────────────────────┐    │
│  │                          backend/                                    │    │
│  │     elementwise (add, sub, mul, div), reduce (sum, min, max)         │    │
│  │     dot, norm, matmul, transpose                                     │    │
│  └──────────────────────────────────┬───────────────────────────────────┘    │
│                                     │                                        │
│  ┌──────────────────────────────────┴───────────────────────────────────┐    │
│  │                         intrinsic/                                   │    │
│  │            SIMDVec<T, Width> - CPU register abstraction              │    │
│  │                  SSE / AVX / AVX-512 / NEON                          │    │
│  └──────────────────────────────────┬───────────────────────────────────┘    │
│                                     │                                        │
│  ┌─────────────────┐  ┌─────────────┴─────────────┐  ┌───────────────────┐   │
│  │     math/       │  │          arch/            │  │      meta/        │   │
│  │ sin,cos,exp,log │  │  platform, cpuid, macros  │  │ metaprogramming   │   │
│  └─────────────────┘  └───────────────────────────┘  └───────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     │ wraps (composition)
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              datapod (dp::)                                  │
│                            (POD Data Storage)                                │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                         dp::mat:: types                              │    │
│  │  dp::mat::scalar<T>    dp::mat::vector<T,N>    dp::mat::matrix<T,R,C>│    │
│  │       (rank-0)               (rank-1)                 (rank-2)       │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐   │
│  │  sequential.hpp │  │  adapters.hpp   │  │       spatial.hpp           │   │
│  │ Vector, String, │  │ Optional,Result │  │  Point, Pose, Quaternion    │   │
│  │ Queue, Stack    │  │ Variant, Pair   │  │  Velocity, State, Geo       │   │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐                                    │
│  │ associative.hpp │  │  temporal.hpp   │                                    │
│  │   Map, Set      │  │ Stamp,TimeSeries│                                    │
│  └─────────────────┘  └─────────────────┘                                    │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## SIMD Architecture (Bottom-Up)

This is how SIMD operations flow through the library layers:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           User API Layer                                    │
│                                                                             │
│   simd::Tensor<float, 8> a, b;                                              │
│   auto c = a + b;                    // Element-wise add                    │
│   auto d = lina::dot(a, b);          // Dot product                         │
│   simd::Matrix<float,4,4> M1, M2;                                           │
│   auto M3 = lina::matmul(M1, M2);    // Matrix multiply                     │
│                                                                             │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │ calls
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Backend Layer                                     │
│                       (simd/backend/*.hpp)                                  │
│                                                                             │
│   backend::add<float, 8>(dst, src1, src2);    // Element-wise ops           │
│   backend::reduce_sum<float, 8>(src);          // Reductions                │
│   backend::dot<float, 8>(src1, src2);          // Dot product               │
│   backend::matmul<float,4,4,4>(dst, A, B);     // Matrix multiply           │
│                                                                             │
│   - Chooses best implementation based on size                               │
│   - Handles alignment, loop tiling, remainder                               │
│                                                                             │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │ uses
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Intrinsic Layer                                     │
│                     (simd/intrinsic/*.hpp)                                  │
│                                                                             │
│   SIMDVec<float, 4> v1, v2;     // Wraps __m128 (SSE)                       │
│   SIMDVec<float, 8> v3, v4;     // Wraps __m256 (AVX)                       │
│   SIMDVec<float, 16> v5, v6;    // Wraps __m512 (AVX-512)                   │
│                                                                             │
│   auto v = SIMDVec<float,4>::load(ptr);                                     │
│   auto w = v1 + v2;                                                         │
│   auto s = v.hsum();            // Horizontal sum                           │
│   w.store(ptr);                                                             │
│                                                                             │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │ wraps
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Arch Layer                                        │
│                       (simd/arch/*.hpp)                                     │
│                                                                             │
│   OPTINUM_HAS_SSE, OPTINUM_HAS_AVX, OPTINUM_HAS_AVX512, OPTINUM_HAS_NEON    │
│   OPTINUM_SIMD_LEVEL = 128 / 256 / 512                                      │
│   OPTINUM_INLINE, OPTINUM_SIMD_ALIGN                                        │
│                                                                             │
│   #include <immintrin.h>  // x86                                            │
│   #include <arm_neon.h>   // ARM                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Current vs Target Implementation

### Current (Scalar Loops)
```cpp
// simd/tensor.hpp - Current implementation
constexpr Tensor& operator+=(const Tensor& rhs) noexcept {
    for (size_type i = 0; i < N; ++i)
        pod_[i] += rhs.pod_[i];  // One element at a time
    return *this;
}
```

### Target (SIMD Backend)
```cpp
// simd/tensor.hpp - Target implementation
Tensor& operator+=(const Tensor& rhs) noexcept {
    if constexpr (std::is_constant_evaluated()) {
        // Compile-time: scalar fallback (constexpr)
        for (size_type i = 0; i < N; ++i)
            pod_[i] += rhs.pod_[i];
    } else {
        // Runtime: SIMD backend
        backend::add<T, N>(data(), data(), rhs.data());
    }
    return *this;
}
```

---

## SIMDVec Type Mapping

| Type   | SSE (128-bit) | AVX (256-bit) | AVX-512 (512-bit) | NEON (128-bit) |
|--------|---------------|---------------|-------------------|----------------|
| float  | `__m128` (4)  | `__m256` (8)  | `__m512` (16)     | `float32x4_t` (4) |
| double | `__m128d` (2) | `__m256d` (4) | `__m512d` (8)     | `float64x2_t` (2) |
| int32  | `__m128i` (4) | `__m256i` (8) | `__m512i` (16)    | `int32x4_t` (4) |

---

## Folder Structure

```
include/optinum/
├── optinum.hpp                      # Master header + namespace alias
│
├── simd/                            # on::simd namespace (Types + SIMD primitives)
│   ├── simd.hpp                     # simd module header
│   │
│   ├── arch/                        # Architecture detection [DONE]
│   │   ├── arch.hpp                 #   ✓ SSE/AVX/AVX512/NEON detection
│   │   └── macros.hpp               #   ✓ OPTINUM_INLINE, alignment, etc.
│   │
│   ├── meta/                        # Template metaprogramming
│   │   └── meta.hpp                 #   pack_prod, type traits
│   │
│   ├── intrinsic/                   # SIMD register abstraction
│   │   ├── simd_vec.hpp             #   SIMDVec<T, Width> main template
│   │   ├── sse.hpp                  #   SSE: float(__m128), double(__m128d)
│   │   ├── avx.hpp                  #   AVX: float(__m256), double(__m256d)
│   │   ├── avx512.hpp               #   AVX-512: float(__m512), double(__m512d)
│   │   └── neon.hpp                 #   ARM NEON: float32x4_t, float64x2_t
│   │
│   ├── math/                        # Vectorized math functions
│   │   └── math.hpp                 #   sin, cos, exp, log, pow, etc.
│   │
│   ├── backend/                     # SIMD operation implementations
│   │   ├── backend.hpp              #   Dispatcher header
│   │   ├── elementwise.hpp          #   add, sub, mul, div (vector ops)
│   │   ├── reduce.hpp               #   sum, min, max (reductions)
│   │   ├── dot.hpp                  #   Dot product
│   │   ├── norm.hpp                 #   L2 norm, normalize
│   │   ├── matmul.hpp               #   Matrix multiplication
│   │   └── transpose.hpp            #   Matrix transpose
│   │
│   ├── scalar.hpp                   # ✓ Scalar<T> wraps dp::mat::scalar<T>
│   ├── vector.hpp                   # ✓ Vector<T,N> wraps dp::mat::vector<T,N> (1D)
│   ├── matrix.hpp                   # ✓ Matrix<T,R,C> wraps dp::mat::matrix<T,R,C>
│   ├── tensor.hpp                   # ✓ Tensor<T,Dims...> wraps dp::mat::tensor (N-D, rank >= 3)
│   └── traits.hpp                   #   Type traits for vectors/matrices
│
├── lina/                            # on::lina namespace (Linear Algebra Operations)
│   ├── lina.hpp                     # lina module header
│   │
│   ├── basic/                       # Basic matrix operations
│   │   ├── matmul.hpp               #   Matrix multiplication
│   │   ├── transpose.hpp            #   Transpose
│   │   ├── inverse.hpp              #   Matrix inverse
│   │   ├── determinant.hpp          #   Determinant
│   │   ├── trace.hpp                #   Trace
│   │   └── norm.hpp                 #   Frobenius, L2, etc.
│   │
│   ├── decompose/                   # Matrix decompositions
│   │   ├── lu.hpp                   #   LU factorization
│   │   ├── qr.hpp                   #   QR factorization
│   │   ├── svd.hpp                  #   Singular value decomposition
│   │   ├── cholesky.hpp             #   Cholesky decomposition
│   │   └── eig.hpp                  #   Eigendecomposition
│   │
│   ├── solve/                       # Linear solvers
│   │   ├── solve.hpp                #   Solve Ax = b
│   │   └── lstsq.hpp                #   Least squares
│   │
│   ├── algebra/                     # Tensor algebra
│   │   ├── einsum.hpp               #   Einstein summation
│   │   ├── contraction.hpp          #   Tensor contraction
│   │   ├── permute.hpp              #   Tensor permutation
│   │   ├── inner.hpp                #   Inner product
│   │   └── outer.hpp                #   Outer product
│   │
│   └── expr/                        # Expression templates
│       ├── abstract.hpp             #   CRTP base class
│       ├── binary/                  #   Binary operations
│       │   ├── arithmetic.hpp       #     +, -, *, /
│       │   ├── compare.hpp          #     <, >, ==, !=
│       │   └── math.hpp             #     pow, atan2, hypot
│       ├── unary/                   #   Unary operations
│       │   ├── math.hpp             #     sqrt, abs, sin, cos
│       │   └── bool.hpp             #     !, all_of, any_of
│       └── views/                   #   Tensor views/slices
│           ├── view.hpp             #     1D/2D/ND views
│           ├── diag.hpp             #     Diagonal view
│           └── filter.hpp           #     Boolean mask filter
│
└── opti/                            # on::opti namespace (Optimization)
    ├── opti.hpp                     # opti module header
    │
    ├── core/                        # Core infrastructure
    │   ├── function.hpp             #   Function wrapper with mixins
    │   ├── traits.hpp               #   Function type traits
    │   ├── checks.hpp               #   Static interface checks
    │   └── log.hpp                  #   Logging utilities
    │
    ├── callback/                    # Optimization callbacks
    │   ├── callback.hpp             #   Base callback infrastructure
    │   ├── early_stop.hpp           #   Stop when loss plateaus
    │   ├── grad_clip.hpp            #   Gradient clipping
    │   ├── print.hpp                #   Print loss each iteration
    │   ├── progress.hpp             #   Progress bar
    │   └── timer.hpp                #   Time-based stopping
    │
    ├── gradient/                    # First-order methods
    │   ├── gradient.hpp             #   Module header
    │   ├── gd.hpp                   #   Gradient Descent
    │   ├── sgd/                     #   SGD family
    │   │   ├── sgd.hpp              #     Stochastic GD
    │   │   ├── momentum.hpp         #     Momentum SGD
    │   │   └── nesterov.hpp         #     Nesterov Momentum
    │   ├── coordinate/              #   Coordinate descent
    │   │   ├── random.hpp           #     Random coordinate
    │   │   ├── cyclic.hpp           #     Cyclic coordinate
    │   │   └── greedy.hpp           #     Greedy coordinate
    │   └── parallel/                #   Parallel methods
    │       └── hogwild.hpp          #     Hogwild! (lock-free SGD)
    │
    ├── adaptive/                    # Adaptive learning rate
    │   ├── adaptive.hpp             #   Module header
    │   ├── adam/                    #   Adam family
    │   │   ├── adam.hpp             #     Adam
    │   │   ├── adamax.hpp           #     AdaMax
    │   │   ├── amsgrad.hpp          #     AMSGrad
    │   │   ├── nadam.hpp            #     Nadam
    │   │   └── padam.hpp            #     PAdam
    │   ├── adagrad.hpp              #   AdaGrad
    │   ├── adadelta.hpp             #   AdaDelta
    │   ├── rmsprop.hpp              #   RMSProp
    │   ├── adabelief.hpp            #   AdaBelief
    │   ├── adabound.hpp             #   AdaBound
    │   ├── yogi.hpp                 #   Yogi
    │   ├── eve.hpp                  #   Eve
    │   ├── swats.hpp                #   SWATS (Adam-to-SGD)
    │   └── lookahead.hpp            #   Lookahead wrapper
    │
    ├── variance/                    # Variance reduction
    │   ├── variance.hpp             #   Module header
    │   ├── svrg.hpp                 #   SVRG
    │   ├── sarah.hpp                #   SARAH / SARAH+
    │   └── katyusha.hpp             #   Katyusha
    │
    ├── quasi_newton/                # Second-order methods
    │   ├── quasi_newton.hpp         #   Module header
    │   ├── lbfgs.hpp                #   L-BFGS
    │   └── iqn.hpp                  #   Incremental Quasi-Newton
    │
    ├── proximal/                    # Proximal methods
    │   ├── proximal.hpp             #   Module header
    │   ├── fbs.hpp                  #   Forward-Backward Splitting
    │   ├── fista.hpp                #   FISTA
    │   ├── fasta.hpp                #   FASTA
    │   └── frankwolfe/              #   Frank-Wolfe / Conditional gradient
    │       ├── frankwolfe.hpp       #     Frank-Wolfe optimizer
    │       ├── atoms.hpp            #     Atom dictionary
    │       └── constraint.hpp       #     Constraint types
    │
    ├── constrained/                 # Constrained optimization
    │   ├── constrained.hpp          #   Module header
    │   ├── augmented.hpp            #   Augmented Lagrangian
    │   └── sdp/                     #   Semidefinite programming
    │       ├── primal_dual.hpp      #     Primal-dual solver
    │       └── lrsdp.hpp            #     Low-rank SDP
    │
    ├── evolutionary/                # Derivative-free / evolutionary
    │   ├── evolutionary.hpp         #   Module header
    │   ├── cmaes/                   #   CMA-ES family
    │   │   ├── cmaes.hpp            #     CMA-ES
    │   │   ├── active.hpp           #     Active CMA-ES
    │   │   ├── bipop.hpp            #     BIPOP-CMA-ES
    │   │   └── ipop.hpp             #     IPOP-CMA-ES
    │   ├── de.hpp                   #   Differential Evolution
    │   ├── pso.hpp                  #   Particle Swarm Optimization
    │   ├── sa.hpp                   #   Simulated Annealing
    │   ├── spsa.hpp                 #   SPSA
    │   └── cne.hpp                  #   Conventional Neural Evolution
    │
    ├── multiobjective/              # Multi-objective optimization
    │   ├── multiobjective.hpp       #   Module header
    │   ├── nsga2.hpp                #   NSGA-II
    │   ├── agemoea.hpp              #   AGE-MOEA
    │   ├── moead.hpp                #   MOEA/D
    │   └── indicator/               #   Quality indicators
    │       ├── epsilon.hpp          #     Epsilon indicator
    │       ├── igd.hpp              #     Inverted Generational Distance
    │       └── hypervolume.hpp      #     Hypervolume indicator
    │
    ├── schedule/                    # Learning rate scheduling
    │   ├── schedule.hpp             #   Module header
    │   ├── cyclical.hpp             #   Cyclical LR (SGDR)
    │   ├── warmup.hpp               #   Warm restarts
    │   └── adaptive.hpp             #   SPALeRA, Big Batch
    │
    ├── search/                      # Hyperparameter search
    │   ├── search.hpp               #   Module header
    │   └── grid.hpp                 #   Grid search
    │
    └── problem/                     # Benchmark functions
        ├── problem.hpp              #   Module header
        ├── unconstrained/           #   Single-objective test functions
        │   ├── rosenbrock.hpp       #     Rosenbrock function
        │   ├── sphere.hpp           #     ✓ Sphere function
        │   ├── rastrigin.hpp        #     Rastrigin function
        │   └── ackley.hpp           #     Ackley function
        └── multiobjective/          #   Multi-objective test functions
            ├── dtlz/                #     DTLZ test suite
            ├── zdt/                 #     ZDT test suite
            └── schaffer.hpp         #     Schaffer functions
```

---

## Implementation Roadmap

### Phase 0: SIMD Architecture Refactor - Non-Owning Views [TODO]

**Goal:** Refactor SIMD layer from owning wrappers to non-owning views over `dp::mat::*` types.

#### 0.0 Design Philosophy

**Current (problematic):**
```cpp
dp::mat::vector<float, 3> point;           // dp owns data
simd::Vector<float, 3> simd_point(point);  // COPIES data into simd wrapper
// ... do SIMD operations ...
point = simd_point.pod();                  // COPY back to dp
```

**New (zero-copy views):**
```cpp
dp::mat::vector<float, 1024> a, b, c;      // dp owns ALL data
simd::vector<float, 8> va = simd::view<8>(a);  // non-owning view (just ptr + metadata)
simd::vector<float, 8> vb = simd::view<8>(b);
simd::axpy(2.0f, va, vb, simd::view<8>(c)); // operates directly on dp memory
// No copy ever happens - c is already updated
```

**Type symmetry:**
```
dp::mat::scalar<T>       →  simd::scalar<T, W>       (non-owning view)
dp::mat::vector<T, N>    →  simd::vector<T, W>       (non-owning view)
dp::mat::matrix<T, R, C> →  simd::matrix<T, W>       (non-owning view)
dp::mat::tensor<T, ...>  →  simd::tensor<T, W, Rank> (non-owning view)
```

**Key principles:**
- `dp::mat::*` types own memory (POD, serializable, cache-aligned)
- `simd::*` types (scalar/vector/matrix/tensor) are non-owning views with SIMD operations
- Algorithms accept views, operate directly on dp memory
- Views are cheap, trivially copyable, stack-allocated

#### 0.1 New File Structure

```
simd/
├── arch/                    # [KEEP] Platform detection (unchanged)
│   ├── arch.hpp
│   └── macros.hpp
│
├── pack/                    # [NEW] SIMD register abstraction
│   ├── pack.hpp             #   pack<T,W> - primary template + scalar fallback
│   ├── pack_sse.hpp         #   SSE specializations (float4, double2, int32x4, int64x2)
│   ├── pack_avx.hpp         #   AVX/AVX2 specializations
│   ├── pack_avx512.hpp      #   AVX-512 specializations
│   └── pack_neon.hpp        #   ARM NEON specializations
│
├── mask.hpp                 # [NEW] mask<T,W> - comparison results, any/all, blend/select
│
├── kernel.hpp               # [NEW] Kernel<T,W,Rank> - ptr + extents + strides + load/store
│
├── scalar.hpp               # [REPLACE] simd::scalar<T,W> - non-owning view (rank-0)
├── vector.hpp               # [REPLACE] simd::vector<T,W> - non-owning view (rank-1)
├── matrix.hpp               # [REPLACE] simd::matrix<T,W> - non-owning view (rank-2)
├── tensor.hpp               # [REPLACE] simd::tensor<T,W,Rank> - non-owning view (rank-N)
│
├── bridge.hpp               # [NEW] simd::view<W>(dp_obj) factory functions
│
├── algo/                    # [NEW] Algorithms on views
│   ├── algo.hpp             #   Module header
│   ├── elementwise.hpp      #   add, sub, mul, div, axpy, scale
│   ├── reduce.hpp           #   sum, min, max, prod, dot, norm
│   ├── transform.hpp        #   apply(f, view), map
│   └── linalg.hpp           #   matmul, transpose, solve (delegating to lina/)
│
├── math/                    # [KEEP] SIMD math functions (updated to use pack<T,W>)
│   ├── simd_math.hpp
│   ├── fast_exp.hpp
│   ├── fast_log.hpp
│   ├── fast_trig.hpp
│   ├── fast_hyp.hpp
│   ├── fast_pow.hpp
│   └── detail/
│       ├── constants.hpp
│       └── map.hpp
│
└── simd.hpp                 # [UPDATE] Module header - exports views + bridge + algo
```

**Files to DELETE:**
- `simd/intrinsic/simd_vec.hpp` (replaced by pack/)
- `simd/intrinsic/sse.hpp` (replaced by pack_sse.hpp)
- `simd/intrinsic/avx.hpp` (replaced by pack_avx.hpp)
- `simd/intrinsic/avx512.hpp` (replaced by pack_avx512.hpp)
- `simd/intrinsic/neon.hpp` (replaced by pack_neon.hpp)
- `simd/backend/` (merged into algo/)

**Files to REPLACE (same name, new content):**
- `simd/scalar.hpp` - was owning Scalar<T>, becomes non-owning simd::scalar<T,W>
- `simd/vector.hpp` - was owning Vector<T,N>, becomes non-owning simd::vector<T,W>
- `simd/matrix.hpp` - was owning Matrix<T,R,C>, becomes non-owning simd::matrix<T,W>
- `simd/tensor.hpp` - was owning Tensor<T,Dims...>, becomes non-owning simd::tensor<T,W,Rank>

#### 0.2 Core Types

##### pack<T, W> - SIMD Register Abstraction
```cpp
namespace simd {

// Primary template (scalar fallback)
template <typename T, std::size_t W>
struct pack {
    static_assert(W > 0 && std::is_arithmetic_v<T>);
    using value_type = T;
    static constexpr std::size_t width = W;
    
    alignas(W * sizeof(T)) T data[W];
    
    // Construction
    pack() = default;
    explicit pack(T val);                    // broadcast
    
    // Load/Store
    static pack load(const T* p);            // aligned
    static pack loadu(const T* p);           // unaligned
    void store(T* p) const;
    void storeu(T* p) const;
    
    // Arithmetic
    pack operator+(pack rhs) const;
    pack operator-(pack rhs) const;
    pack operator*(pack rhs) const;
    pack operator/(pack rhs) const;
    pack operator-() const;
    
    // FMA
    static pack fma(pack a, pack b, pack c);  // a*b + c
    static pack fms(pack a, pack b, pack c);  // a*b - c
    
    // Math
    pack sqrt() const;
    pack rsqrt() const;
    pack abs() const;
    static pack min(pack a, pack b);
    static pack max(pack a, pack b);
    
    // Reductions
    T hsum() const;
    T hmin() const;
    T hmax() const;
    T hprod() const;
    
    // Bitwise (integer only)
    pack operator&(pack rhs) const;
    pack operator|(pack rhs) const;
    pack operator^(pack rhs) const;
    pack operator~() const;
    pack operator<<(int n) const;
    pack operator>>(int n) const;
    
    // Element access
    T operator[](std::size_t i) const;
};

// Specializations in pack_sse.hpp, pack_avx.hpp, etc.
// pack<float, 4>   -> __m128
// pack<float, 8>   -> __m256
// pack<float, 16>  -> __m512
// pack<double, 2>  -> __m128d
// pack<double, 4>  -> __m256d
// pack<double, 8>  -> __m512d
// pack<int32_t, 4/8/16> -> __m128i/__m256i/__m512i
// pack<int64_t, 2/4/8>  -> __m128i/__m256i/__m512i

} // namespace simd
```

##### mask<T, W> - Comparison Results
```cpp
namespace simd {

template <typename T, std::size_t W>
struct mask {
    // Internal representation varies by ISA:
    // - SSE/AVX: __m128/__m256 (all bits set per lane)
    // - AVX-512: __mmask8/__mmask16
    // - Scalar: std::array<bool, W>
    
    // Factory
    static mask all_true();
    static mask all_false();
    static mask first_n(std::size_t n);      // first n lanes true
    
    // Combine
    mask operator&(mask rhs) const;
    mask operator|(mask rhs) const;
    mask operator^(mask rhs) const;
    mask operator!() const;
    
    // Query
    bool all() const;                         // all lanes true
    bool any() const;                         // any lane true
    bool none() const;                        // no lane true
    int popcount() const;                     // count true lanes
    
    // Element access
    bool operator[](std::size_t i) const;
};

// Comparisons return mask
template <typename T, std::size_t W>
mask<T,W> cmp_eq(pack<T,W> a, pack<T,W> b);
mask<T,W> cmp_ne(pack<T,W> a, pack<T,W> b);
mask<T,W> cmp_lt(pack<T,W> a, pack<T,W> b);
mask<T,W> cmp_le(pack<T,W> a, pack<T,W> b);
mask<T,W> cmp_gt(pack<T,W> a, pack<T,W> b);
mask<T,W> cmp_ge(pack<T,W> a, pack<T,W> b);

// Masked operations
template <typename T, std::size_t W>
pack<T,W> blend(pack<T,W> a, pack<T,W> b, mask<T,W> m);  // m ? b : a
pack<T,W> maskload(const T* p, mask<T,W> m);
void maskstore(T* p, pack<T,W> v, mask<T,W> m);

} // namespace simd
```

##### Kernel<T, W, Rank> - Memory Layout Descriptor
```cpp
namespace simd {

template <typename T, std::size_t W, std::size_t Rank>
struct Kernel {
    T* ptr = nullptr;
    std::array<std::size_t, Rank> extents{};
    std::array<std::size_t, Rank> strides{};
    
    // Metadata
    constexpr std::size_t extent(std::size_t d) const { return extents[d]; }
    constexpr std::size_t stride(std::size_t d) const { return strides[d]; }
    constexpr std::size_t linear_size() const;
    constexpr std::size_t num_packs() const { return (linear_size() + W - 1) / W; }
    constexpr std::size_t tail_size() const { return linear_size() % W; }
    
    // Linear (contiguous) access
    pack<T,W> load_pack(std::size_t pack_idx) const;
    void store_pack(std::size_t pack_idx, pack<T,W> v) const;
    
    // Tail handling (last partial pack)
    pack<T,W> load_pack_tail(std::size_t pack_idx, std::size_t valid) const;
    void store_pack_tail(std::size_t pack_idx, pack<T,W> v, std::size_t valid) const;
    
    // Scalar access
    T& at_linear(std::size_t i) const { return ptr[i]; }
};

} // namespace simd
```

##### Views - Typed Wrappers Around Kernel
```cpp
namespace simd {

template <typename T, std::size_t W>
struct scalar_view {
    Kernel<T, W, 0> k;  // Rank-0: single element
    T& get() const { return *k.ptr; }
    operator T&() const { return get(); }
};

template <typename T, std::size_t W>
struct vector_view {
    Kernel<T, W, 1> k;  // Rank-1: 1D array
    
    std::size_t size() const { return k.extent(0); }
    T& operator[](std::size_t i) const { return k.at_linear(i); }
    
    // Pack access
    pack<T,W> load_pack(std::size_t i) const { return k.load_pack(i); }
    void store_pack(std::size_t i, pack<T,W> v) const { k.store_pack(i, v); }
    std::size_t num_packs() const { return k.num_packs(); }
    std::size_t tail_size() const { return k.tail_size(); }
};

template <typename T, std::size_t W>
struct matrix_view {
    Kernel<T, W, 2> k;  // Rank-2: 2D array (column-major)
    
    std::size_t rows() const { return k.extent(0); }
    std::size_t cols() const { return k.extent(1); }
    T& operator()(std::size_t r, std::size_t c) const;
    
    // Row/column views
    vector_view<T, W> row(std::size_t r) const;
    vector_view<T, W> col(std::size_t c) const;
};

template <typename T, std::size_t W, std::size_t Rank>
struct tensor_view {
    Kernel<T, W, Rank> k;
    
    template <typename... Idx>
    T& operator()(Idx... idx) const;
    
    std::size_t extent(std::size_t d) const { return k.extent(d); }
};

} // namespace simd
```

##### Bridge - Factory Functions for dp Types
```cpp
// bridge.hpp
namespace simd {

// Detect SIMD width for type
template <typename T>
constexpr std::size_t default_width() {
    if constexpr (std::is_same_v<T, float>) {
#if defined(OPTINUM_HAS_AVX)
        return 8;
#elif defined(OPTINUM_HAS_SSE)
        return 4;
#else
        return 4;  // scalar fallback, 4-wide
#endif
    } else if constexpr (std::is_same_v<T, double>) {
#if defined(OPTINUM_HAS_AVX)
        return 4;
#elif defined(OPTINUM_HAS_SSE)
        return 2;
#else
        return 2;
#endif
    }
    // ... int32_t, int64_t
}

// View factory: explicit width
template <std::size_t W, typename T, std::size_t N>
vector_view<T, W> view(dp::mat::vector<T, N>& v) {
    return vector_view<T, W>{
        Kernel<T, W, 1>{v.data(), {N}, {1}}
    };
}

template <std::size_t W, typename T, std::size_t N>
vector_view<const T, W> view(const dp::mat::vector<T, N>& v) {
    return vector_view<const T, W>{
        Kernel<const T, W, 1>{v.data(), {N}, {1}}
    };
}

template <std::size_t W, typename T, std::size_t R, std::size_t C>
matrix_view<T, W> view(dp::mat::matrix<T, R, C>& m) {
    // Column-major: stride(0)=1, stride(1)=R
    return matrix_view<T, W>{
        Kernel<T, W, 2>{m.data(), {R, C}, {1, R}}
    };
}

// View factory: auto width
template <typename T, std::size_t N>
auto view(dp::mat::vector<T, N>& v) {
    return view<default_width<T>()>(v);
}

} // namespace simd
```

#### 0.3 Algorithm Examples

```cpp
// algo/elementwise.hpp
namespace simd {

// axpy: y = a*x + y
template <typename T, std::size_t W>
void axpy(T alpha, vector_view<const T, W> x, vector_view<T, W> y) {
    const std::size_t n = x.size();
    const std::size_t npacks = n / W;
    const std::size_t tail = n % W;
    
    pack<T, W> a(alpha);
    
    for (std::size_t i = 0; i < npacks; ++i) {
        pack<T, W> xi = x.load_pack(i);
        pack<T, W> yi = y.load_pack(i);
        y.store_pack(i, pack<T,W>::fma(a, xi, yi));
    }
    
    if (tail > 0) {
        pack<T, W> xi = x.k.load_pack_tail(npacks, tail);
        pack<T, W> yi = y.k.load_pack_tail(npacks, tail);
        y.k.store_pack_tail(npacks, pack<T,W>::fma(a, xi, yi), tail);
    }
}

// dot: sum(x[i] * y[i])
template <typename T, std::size_t W>
T dot(vector_view<const T, W> x, vector_view<const T, W> y) {
    const std::size_t n = x.size();
    const std::size_t npacks = n / W;
    const std::size_t tail = n % W;
    
    pack<T, W> acc(T{0});
    
    for (std::size_t i = 0; i < npacks; ++i) {
        pack<T, W> xi = x.load_pack(i);
        pack<T, W> yi = y.load_pack(i);
        acc = pack<T,W>::fma(xi, yi, acc);
    }
    
    T result = acc.hsum();
    
    // Scalar tail
    for (std::size_t i = npacks * W; i < n; ++i) {
        result += x[i] * y[i];
    }
    
    return result;
}

} // namespace simd
```

#### 0.4 Usage Example (End-to-End)

```cpp
#include <datapod/matrix.hpp>
#include <optinum/simd/simd.hpp>

namespace dp = datapod;
namespace on = optinum;

int main() {
    // dp owns all data
    dp::mat::vector<float, 1024> x, y, z;
    dp::mat::matrix<float, 64, 64> A;
    
    // Initialize...
    for (int i = 0; i < 1024; ++i) {
        x[i] = static_cast<float>(i);
        y[i] = 1.0f;
    }
    
    // Get SIMD views (auto-detect width: 8 for float on AVX)
    auto vx = on::simd::view(x);  // vector_view<float, 8>
    auto vy = on::simd::view(y);
    auto vz = on::simd::view(z);
    
    // SIMD operations - directly on dp memory
    on::simd::axpy(2.0f, vx, vy);           // y = 2*x + y
    float d = on::simd::dot(vx, vy);        // dot product
    on::simd::copy(vx, vz);                 // z = x
    on::simd::scale(0.5f, vz);              // z *= 0.5
    
    // Results are already in dp containers - no copy needed
    std::cout << "y[0] = " << y[0] << "\n";  // 1.0 (original) + 2*0 = 1.0
    std::cout << "y[1] = " << y[1] << "\n";  // 1.0 + 2*1 = 3.0
    std::cout << "dot  = " << d << "\n";
    
    return 0;
}
```

#### 0.5 Implementation Checklist

##### Step 1: pack<T,W> Infrastructure
- [ ] `simd/pack/pack.hpp` - Primary template (scalar fallback)
- [ ] `simd/pack/pack_sse.hpp` - SSE specializations
  - [ ] `pack<float, 4>`, `pack<double, 2>`
  - [ ] `pack<int32_t, 4>`, `pack<int64_t, 2>`
- [ ] `simd/pack/pack_avx.hpp` - AVX/AVX2 specializations
  - [ ] `pack<float, 8>`, `pack<double, 4>`
  - [ ] `pack<int32_t, 8>`, `pack<int64_t, 4>`
- [ ] `simd/pack/pack_avx512.hpp` - AVX-512 specializations
- [ ] `simd/pack/pack_neon.hpp` - ARM NEON specializations
- [ ] Tests for pack<T,W>

##### Step 2: mask<T,W> and Comparisons
- [ ] `simd/mask.hpp` - mask type + comparison functions
- [ ] Masked load/store
- [ ] blend/select
- [ ] Tests for mask operations

##### Step 3: Kernel and Views
- [ ] `simd/kernel.hpp` - Kernel<T,W,Rank>
- [ ] `simd/view/scalar_view.hpp`
- [ ] `simd/view/vector_view.hpp`
- [ ] `simd/view/matrix_view.hpp`
- [ ] `simd/view/tensor_view.hpp`
- [ ] `simd/view/view.hpp` - module header
- [ ] Tests for views

##### Step 4: Bridge to datapod
- [ ] `simd/bridge.hpp` - view<W>() factory functions
- [ ] Auto-width detection
- [ ] Tests for bridge

##### Step 5: Algorithms
- [ ] `simd/algo/elementwise.hpp` - add, sub, mul, div, axpy, scale, copy
- [ ] `simd/algo/reduce.hpp` - sum, min, max, prod, dot, norm
- [ ] `simd/algo/transform.hpp` - apply, map
- [ ] `simd/algo/algo.hpp` - module header
- [ ] Tests for algorithms

##### Step 6: Update math/ to use pack<T,W>
- [ ] Update `fast_exp.hpp` to use pack<T,W> instead of raw intrinsics
- [ ] Update `fast_log.hpp`
- [ ] Update `fast_trig.hpp`
- [ ] Update `fast_hyp.hpp`
- [ ] Update `fast_pow.hpp`
- [ ] Update `simd_math.hpp`

##### Step 7: Update lina/ to use views
- [ ] Update `lina/basic/matmul.hpp`
- [ ] Update `lina/basic/transpose.hpp`
- [ ] Update `lina/basic/norm.hpp`
- [ ] Update `lina/solve/solve.hpp`
- [ ] Update `lina/solve/lstsq.hpp`
- [ ] Update decomposition algorithms
- [ ] Update tests

##### Step 8: Cleanup
- [ ] Delete old owning types (scalar.hpp, vector.hpp, matrix.hpp, tensor.hpp)
- [ ] Delete old intrinsic/ folder
- [ ] Delete old backend/ folder
- [ ] Update simd.hpp module header
- [ ] Update examples
- [ ] Update documentation

#### 0.6 Migration Notes

**Breaking changes:**
- `simd::Scalar<T>`, `simd::Vector<T,N>`, `simd::Matrix<T,R,C>` are removed
- Use `dp::mat::*` for data ownership
- Use `simd::view<W>(dp_obj)` to get SIMD views
- `SIMDVec<T,W>` renamed to `pack<T,W>`

**lina/ changes:**
- Functions like `lina::matmul(A, B)` will accept either:
  - `dp::mat::matrix` directly (creates views internally), OR
  - `simd::matrix_view` explicitly
- Return types change from `simd::Matrix` to `void` (result passed as out param)
  or return `dp::mat::matrix` directly

---

### Phase 1: SIMD Foundation [DONE]

#### 1.1 Architecture Detection [DONE]
- [x] `simd/arch/arch.hpp` - Platform & SIMD capability detection
  - [x] Compiler detection (GCC, Clang, MSVC, Intel)
  - [x] Platform detection (Windows, Linux, macOS)
  - [x] x86 SIMD: SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2
  - [x] x86 SIMD: AVX, AVX2, AVX-512 (F, VL, BW, DQ, CD)
  - [x] x86 SIMD: FMA, F16C, BMI1, BMI2
  - [x] ARM SIMD: NEON, SVE, SVE2
  - [x] SIMD width constants
  - [x] Feature query functions: `has_sse()`, `has_avx()`, etc.
- [x] `simd/arch/macros.hpp` - Utility macros
  - [x] `OPTINUM_INLINE` / `OPTINUM_NOINLINE`
  - [x] `OPTINUM_SIMD_ALIGN` / `OPTINUM_SIMD_ALIGNMENT`
  - [x] `OPTINUM_RESTRICT`
  - [x] `OPTINUM_LIKELY` / `OPTINUM_UNLIKELY`
  - [x] `OPTINUM_PREFETCH_READ` / `OPTINUM_PREFETCH_WRITE`
  - [x] `OPTINUM_UNREACHABLE` / `OPTINUM_ASSUME`
  - [x] `OPTINUM_VECTORIZE` / `OPTINUM_UNROLL`

#### 1.2 SIMD Register Abstraction [DONE]
- [x] `simd/intrinsic/simd_vec.hpp` - Main `SIMDVec<T, Width>` template (scalar fallback)
- [x] `simd/intrinsic/sse.hpp` - SSE specializations
  - [x] `SIMDVec<float, 4>` wrapping `__m128`
  - [x] `SIMDVec<double, 2>` wrapping `__m128d`
  - [x] Operations: load, loadu, store, storeu, broadcast
  - [x] Operations: add, sub, mul, div, fma/fms
  - [x] Operations: sqrt, rsqrt, min, max
  - [x] Operations: hsum/hmin/hmax (horizontal reductions)
- [x] `simd/intrinsic/avx.hpp` - AVX specializations
- [x] `simd/intrinsic/avx512.hpp` - AVX-512 specializations
- [x] `simd/intrinsic/neon.hpp` - ARM NEON specializations

#### 1.3 Backend Operations [DONE]
- [x] `simd/backend/backend.hpp` - Common utilities
- [x] `simd/backend/elementwise.hpp` - add, sub, mul, div (+ scalar variants)
- [x] `simd/backend/reduce.hpp` - sum, min, max
- [x] `simd/backend/dot.hpp` - Dot product
- [x] `simd/backend/norm.hpp` - L2 norm, normalize
- [x] `simd/backend/matmul.hpp` - Matrix multiplication + matvec (column-major)
- [x] `simd/backend/transpose.hpp` - Matrix transpose (column-major)

#### 1.4 Update Tensor/Matrix to Use Backend [DONE]
- [x] Update `simd/tensor.hpp` to use backend
- [x] Update `simd/matrix.hpp` to use backend
- [x] Maintain `constexpr` for compile-time evaluation (`std::is_constant_evaluated()` scalar fallback)

#### 1.5 SIMD Math (Fastor-like) [DONE]
- [x] `simd/math/simd_math.hpp` - Module header (public SIMD math API)
- [x] `simd/math/elementary.hpp` - min/max/clamp, abs, sign, ceil/floor/round/trunc
- [x] `simd/math/exponential.hpp` - exp/exp2/expm1, log/log2/log10/log1p
- [x] `simd/math/trig.hpp` - sin/cos/tan, asin/acos/atan/atan2
- [x] `simd/math/hyperbolic.hpp` - sinh/cosh/tanh, asinh/acosh/atanh
- [x] `simd/math/pow.hpp` - pow, cbrt
- [x] `simd/math/special.hpp` - erf/erfc, hypot, tgamma/lgamma

- [x] Policy: always provide scalar fallback (`std::`), SIMD paths use intrinsics and/or poly approximations

#### 1.6 SIMD Coverage Breadth (beyond float/double) [IN PROGRESS]
- [x] Integer SIMD: `SIMDVec<int32_t, 4/8/16>`, `SIMDVec<int64_t, 2/4/8>` (x86)
- [ ] Integer SIMD: NEON equivalents
- [ ] Comparisons and masks: `cmp_eq/ne/lt/le/gt/ge` returning mask + `any/all`, `blend/select`
- [ ] Masked load/store + remainder-safe kernels (optional gather/scatter later)
- [x] Horizontal reductions: added `hprod` for all types
- [x] Bitwise ops for integer SIMD (and/or/xor, shifts, shr_logical)
- [ ] Complex SIMD (TBD): depends on datapod complex type decision; keep API aligned with dp types

### Phase 2: Linear Algebra [DONE]
- [x] `lina/basic/` - matmul, transpose, inverse, determinant
- [x] `lina/decompose/` - LU, QR, SVD, Cholesky, Eigen
- [x] `lina/solve/` - Linear solvers + lstsq
- [x] `lina/algebra/` - einsum, contraction, inner/outer
- [x] `lina/expr/` - Minimal expression templates

### Phase 3: Optimization Core [FUTURE]
- [ ] `opti/core/` - Function traits, interface
- [ ] `opti/callback/` - Callback system
- [ ] `opti/gradient/` - GD, SGD

### Phase 4: Optimizers [FUTURE]
- [ ] `opti/adaptive/` - Adam, RMSProp, etc.
- [ ] `opti/quasi_newton/` - L-BFGS
- [ ] `opti/evolutionary/` - CMA-ES, DE, PSO

### Phase 5: Advanced [FUTURE]
- [ ] `opti/proximal/` - FISTA, Frank-Wolfe
- [ ] `opti/constrained/` - Augmented Lagrangian
- [ ] `opti/multiobjective/` - NSGA-II, MOEA/D

---

## Component Details

### SIMDVec<T, Width> Interface

```cpp
namespace optinum::simd {

template <typename T, std::size_t Width>
class SIMDVec {
public:
    using value_type = T;
    static constexpr std::size_t width = Width;
    
    // Construction
    SIMDVec() = default;
    explicit SIMDVec(T val);                    // Broadcast scalar
    
    // Load/Store (aligned)
    static SIMDVec load(const T* ptr);
    void store(T* ptr) const;
    
    // Load/Store (unaligned)
    static SIMDVec loadu(const T* ptr);
    void storeu(T* ptr) const;
    
    // Arithmetic
    SIMDVec operator+(SIMDVec rhs) const;
    SIMDVec operator-(SIMDVec rhs) const;
    SIMDVec operator*(SIMDVec rhs) const;
    SIMDVec operator/(SIMDVec rhs) const;
    SIMDVec operator-() const;                  // Negation
    
    // FMA (Fused Multiply-Add)
    static SIMDVec fma(SIMDVec a, SIMDVec b, SIMDVec c);  // a*b + c
    static SIMDVec fms(SIMDVec a, SIMDVec b, SIMDVec c);  // a*b - c
    
    // Math
    SIMDVec sqrt() const;
    SIMDVec rsqrt() const;                      // 1/sqrt(x) approximate
    SIMDVec abs() const;
    
    // Min/Max
    static SIMDVec min(SIMDVec a, SIMDVec b);
    static SIMDVec max(SIMDVec a, SIMDVec b);
    
    // Reductions
    T hsum() const;                             // Horizontal sum
    T hmin() const;                             // Horizontal min
    T hmax() const;                             // Horizontal max
    
private:
    native_type data_;  // __m128, __m256, etc.
};

} // namespace optinum::simd
```

### Backend Usage Pattern

```cpp
// simd/backend/elementwise.hpp
namespace optinum::simd::backend {

template<typename T, std::size_t N>
OPTINUM_INLINE void add(T* OPTINUM_RESTRICT dst, 
                        const T* OPTINUM_RESTRICT src1, 
                        const T* OPTINUM_RESTRICT src2) {
    constexpr std::size_t W = arch::simd_width<T>();
    constexpr std::size_t main_loop = (N / W) * W;
    
    // Main SIMD loop
    for (std::size_t i = 0; i < main_loop; i += W) {
        auto a = SIMDVec<T, W>::loadu(src1 + i);
        auto b = SIMDVec<T, W>::loadu(src2 + i);
        (a + b).storeu(dst + i);
    }
    
    // Remainder (scalar)
    for (std::size_t i = main_loop; i < N; ++i) {
        dst[i] = src1[i] + src2[i];
    }
}

} // namespace optinum::simd::backend
```

---

## Testing

**Every header file in `include/` must have a corresponding test file in `test/`.**

```
include/optinum/simd/arch/arch.hpp       ->  test/simd/arch/arch_test.cpp       ✓
include/optinum/simd/scalar.hpp          ->  test/simd/scalar_test.cpp          ✓
include/optinum/simd/vector.hpp          ->  test/simd/vector_test.cpp          ✓
include/optinum/simd/matrix.hpp          ->  test/simd/matrix_test.cpp          ✓
include/optinum/simd/intrinsic/simd_vec.hpp ->  test/simd/intrinsic/simd_vec_test.cpp ✓
include/optinum/simd/intrinsic/sse.hpp   ->  test/simd/intrinsic/sse_test.cpp   ✓
include/optinum/simd/intrinsic/avx.hpp   ->  test/simd/intrinsic/avx_test.cpp   ✓
include/optinum/simd/intrinsic/avx512.hpp -> test/simd/intrinsic/avx512_test.cpp ✓
include/optinum/simd/intrinsic/neon.hpp  ->  test/simd/intrinsic/neon_test.cpp  ✓
include/optinum/simd/backend/elementwise.hpp -> test/simd/backend/elementwise_test.cpp ✓
include/optinum/simd/backend/reduce.hpp  ->  test/simd/backend/reduce_test.cpp  ✓
include/optinum/simd/backend/dot.hpp     ->  test/simd/backend/dot_test.cpp     ✓
include/optinum/simd/backend/norm.hpp    ->  test/simd/backend/norm_test.cpp    ✓
include/optinum/simd/backend/matmul.hpp  ->  test/simd/backend/matmul_test.cpp  ✓
include/optinum/simd/backend/transpose.hpp -> test/simd/backend/transpose_test.cpp ✓
include/optinum/simd/math/simd_math.hpp        ->  test/simd/math/simd_math_test.cpp
include/optinum/simd/math/elementary.hpp       ->  test/simd/math/elementary_test.cpp
include/optinum/simd/math/exponential.hpp      ->  test/simd/math/exponential_test.cpp
include/optinum/simd/math/trig.hpp             ->  test/simd/math/trig_test.cpp
include/optinum/simd/math/hyperbolic.hpp       ->  test/simd/math/hyperbolic_test.cpp
include/optinum/simd/math/pow.hpp              ->  test/simd/math/pow_test.cpp
include/optinum/simd/math/special.hpp          ->  test/simd/math/special_test.cpp

include/optinum/lina/lina.hpp                  ->  test/lina/lina_test.cpp ✓
include/optinum/lina/basic/matmul.hpp          ->  test/lina/basic/lina_matmul_test.cpp ✓
include/optinum/lina/basic/transpose.hpp       ->  test/lina/basic/lina_transpose_test.cpp ✓
include/optinum/lina/basic/determinant.hpp     ->  test/lina/basic/determinant_test.cpp ✓
include/optinum/lina/basic/inverse.hpp         ->  test/lina/basic/inverse_test.cpp ✓
include/optinum/lina/basic/norm.hpp            ->  test/lina/basic/lina_norm_test.cpp ✓
include/optinum/lina/decompose/lu.hpp          ->  test/lina/decompose/lu_test.cpp ✓
include/optinum/lina/decompose/qr.hpp          ->  test/lina/decompose/qr_test.cpp ✓
include/optinum/lina/decompose/cholesky.hpp    ->  test/lina/decompose/cholesky_test.cpp ✓
include/optinum/lina/decompose/eigen.hpp       ->  test/lina/decompose/eigen_test.cpp ✓
include/optinum/lina/decompose/svd.hpp         ->  test/lina/decompose/svd_test.cpp ✓
include/optinum/lina/solve/solve.hpp           ->  test/lina/solve/solve_test.cpp ✓
include/optinum/lina/solve/lstsq.hpp           ->  test/lina/solve/lstsq_test.cpp ✓
include/optinum/lina/algebra/einsum.hpp        ->  test/lina/algebra/einsum_test.cpp ✓
include/optinum/lina/algebra/contraction.hpp   ->  test/lina/algebra/contraction_test.cpp ✓
include/optinum/lina/expr/expr.hpp             ->  test/lina/expr/expr_test.cpp ✓
include/optinum/opti/problem/sphere.hpp  ->  test/opti/problem/sphere_test.cpp  ✓
```

Tests use **doctest**. Do NOT add `DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN`.

---

## Design Principles

1. **Header-only**: No compilation, just include
2. **Composition over inheritance**: `on::simd::*` wraps `dp::*` via `.pod()`
3. **Zero-cost abstractions**: Expression templates, compile-time dimensions
4. **SIMD everywhere**: All math operations vectorized when possible
5. **Constexpr friendly**: Scalar fallback for compile-time evaluation
6. **POD-friendly**: Easy serialization via `datapod`
7. **Modern C++**: Requires C++20 (concepts, constexpr, fold expressions)
8. **Use datapod over std**: Prefer `dp::` types over `std::` equivalents

---

## datapod Type Mapping

**Always prefer datapod types over std equivalents:**

| Instead of | Use |
|------------|-----|
| `std::vector<T>` | `dp::Vector<T>` |
| `std::array<T,N>` | `dp::Array<T,N>` |
| `std::string` | `dp::String` |
| `std::optional<T>` | `dp::Optional<T>` |
| `std::variant<Ts...>` | `dp::Variant<Ts...>` |
| `std::pair<K,V>` | `dp::Pair<K,V>` |
| `std::tuple<Ts...>` | `dp::Tuple<Ts...>` |
| `std::unordered_map<K,V>` | `dp::Map<K,V>` |
| `std::unordered_set<T>` | `dp::Set<T>` |
| Exceptions | `dp::Result<T,E>` |

---

## Requirements

- **C++20** or later
- **datapod** library (fetched automatically via CMake/xmake)
- **Optional**: AVX2/AVX-512 for best SIMD performance

---

## Build & Test

```bash
make config    # Configure (preserves cache)
make build     # Build examples and tests
make test      # Run all tests
make clean     # Clean build artifacts
```

---

## License

[TBD]
