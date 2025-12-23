# OPTINUM - Optimization & Numerics Library

> **HEADER-ONLY LIBRARY** - No compilation required, just `#include <optinum/optinum.hpp>`

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
│   #include <optinum/optinum.hpp>                                    │
│   on::simd::Matrix<float, 4, 4> A, B;                               │
│   auto C = on::lina::matmul(A, B);                                  │
│   on::opti::Adam optimizer;                                         │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          optinum                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │
│  │   on::opti   │─▶│   on::lina   │─▶│        on::simd          │   │
│  │ (optimizers) │  │ (linear alg) │  │ (types + SIMD primitives)│   │
│  └──────────────┘  └──────────────┘  └────────────┬─────────────┘   │
│                                                   │                 │
└───────────────────────────────────────────────────┼─────────────────┘
                                                    │
                                                    ▼
                              ┌─────────────────────────────────────┐
                              │            datapod                  │
                              │   dp::scalar, dp::tensor, dp::matrix│
                              │   dp::Vector, dp::Optional, etc.    │
                              └─────────────────────────────────────┘
```

---

## Namespace Structure

```
optinum (on)
├── simd        # SIMD types + primitives
│   ├── Scalar<T>                    # wraps dp::scalar<T>
│   ├── Tensor<T, N>                 # wraps dp::tensor<T, N>
│   ├── Matrix<T, R, C>              # wraps dp::matrix<T, R, C>
│   ├── SIMDVec<T, Width>            # CPU register abstraction
│   └── arch, backend                # low-level SIMD infrastructure
│
├── lina        # Linear algebra operations
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
│  │    Scalar<T>           Tensor<T, N>           Matrix<T, R, C>        │    │
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
│  │                           matrix.hpp                                 │    │
│  │    dp::scalar<T>          dp::tensor<T,N>         dp::matrix<T,R,C>  │    │
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
│   ├── scalar.hpp                   # ✓ Scalar<T> wraps dp::scalar<T>
│   ├── tensor.hpp                   # ✓ Tensor<T,N> wraps dp::tensor<T,N>
│   ├── matrix.hpp                   # ✓ Matrix<T,R,C> wraps dp::matrix<T,R,C>
│   └── traits.hpp                   #   Type traits for tensors
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

### Phase 1: SIMD Foundation [IN PROGRESS]

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

#### 1.2 SIMD Register Abstraction [TODO]
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

#### 1.3 Backend Operations [TODO]
- [x] `simd/backend/backend.hpp` - Common utilities
- [x] `simd/backend/elementwise.hpp` - add, sub, mul, div (+ scalar variants)
- [x] `simd/backend/reduce.hpp` - sum, min, max
- [x] `simd/backend/dot.hpp` - Dot product
- [x] `simd/backend/norm.hpp` - L2 norm, normalize
- [x] `simd/backend/matmul.hpp` - Matrix multiplication + matvec (column-major)
- [x] `simd/backend/transpose.hpp` - Matrix transpose (column-major)

#### 1.4 Update Tensor/Matrix to Use Backend [TODO]
- [x] Update `simd/tensor.hpp` to use backend
- [x] Update `simd/matrix.hpp` to use backend
- [x] Maintain `constexpr` for compile-time evaluation (`std::is_constant_evaluated()` scalar fallback)

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
include/optinum/simd/tensor.hpp          ->  test/simd/tensor_test.cpp          ✓
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
