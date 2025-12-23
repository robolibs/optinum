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
│   on::simd::Matrix<float, 4, 4> A;                                  │
│   on::calc::Adam optimizer;                                         │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          optinum                                    │
│  ┌─────────────────────────┐   ┌─────────────────────────────────┐  │
│  │      on::calc           │──▶│         on::simd                │  │
│  │   (optimization)        │   │   (SIMD tensor math)            │  │
│  └─────────────────────────┘   └───────────────┬─────────────────┘  │
│                                                │                    │
└────────────────────────────────────────────────┼────────────────────┘
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
├── simd        # SIMD-accelerated tensor math
│   ├── Scalar<T>                    # wraps dp::scalar<T>
│   ├── Tensor<T, N>                 # wraps dp::tensor<T, N>
│   ├── Matrix<T, R, C>              # wraps dp::matrix<T, R, C>
│   ├── SIMDVec<T, Width>            # CPU register abstraction
│   └── einsum, matmul, inverse...   # operations
│
└── calc        # Numerical optimization
    ├── GradientDescent, SGD, Adam...
    ├── LBFGS, CMA-ES, PSO...
    └── callbacks, schedulers...
```

---

## Architecture Graph

```
                              ┌──────────────────────────────────────────────┐
                              │              on::calc                        │
                              │         (Optimization Layer)                 │
                              │                                              │
                              │  ┌─────────┐ ┌─────────┐ ┌─────────────────┐ │
                              │  │gradient/│ │adaptive/│ │  evolutionary/  │ │
                              │  │ gd, sgd │ │adam,rmsp│ │ cmaes, de, pso  │ │
                              │  └────┬────┘ └────┬────┘ └───────┬─────────┘ │
                              │       │           │               │          │
                              │  ┌────┴───────────┴───────────────┴────┐     │
                              │  │             core/                   │     │
                              │  │   function, traits, callbacks       │     │
                              │  └─────────────────┬───────────────────┘     │
                              └────────────────────┼─────────────────────────┘
                                                   │
                                                   │ uses
                                                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                                on::simd                                      │
│                          (SIMD Math Layer)                                   │
│                                                                              │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐  │
│  │  linalg/   │  │  algebra/  │  │   expr/    │  │       tensor/          │  │
│  │matmul, inv │  │  einsum,   │  │ lazy eval, │  │ Scalar<T>              │  │
│  │ lu, qr,svd │  │contraction │  │ views      │  │ Tensor<T,N>            │  │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  │ Matrix<T,R,C>          │  │
│        │               │               │         └───────────┬────────────┘  │
│        └───────────────┴───────────────┴─────────────────────┤               │
│                                                              │               │
│  ┌───────────────────────────────────────────────────────────┴─────────────┐ │
│  │                         intrinsic/                                      │ │
│  │              SIMDVec<T, Width> - CPU register abstraction               │ │
│  │                    SSE / AVX / AVX-512 / NEON                           │ │
│  └───────────────────────────────────────────────────────────┬─────────────┘ │
│                                                              │               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┴─────────────┐ │
│  │     math/       │  │     meta/       │  │          config/              │ │
│  │ sin,cos,exp,log │  │  metaprogramming│  │   platform, cpuid, macros     │ │
│  └─────────────────┘  └─────────────────┘  └───────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         │ wraps (composition)
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                                datapod (dp::)                                │
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

## Folder Structure

```
include/optinum/
├── optinum.hpp                      # Master header + namespace alias
│
├── simd/                            # on::simd namespace
│   ├── simd.hpp                     # simd module header
│   │
│   ├── config/                      # Platform detection
│   │   ├── config.hpp               #   compiler, OS, C++ version
│   │   ├── cpuid.hpp                #   CPU feature detection
│   │   └── macros.hpp               #   FASTOR_INLINE, etc.
│   │
│   ├── meta/                        # Template metaprogramming
│   │   └── meta.hpp                 #   pack_prod, type traits
│   │
│   ├── intrinsic/                   # SIMD register abstraction
│   │   ├── simd_vec.hpp             #   SIMDVec<T, Width>
│   │   ├── abi.hpp                  #   sse, avx, avx512, neon
│   │   ├── float.hpp                #   float specializations
│   │   ├── double.hpp               #   double specializations
│   │   └── ops.hpp                  #   load, store, add, mul, fma
│   │
│   ├── math/                        # Vectorized math functions
│   │   └── math.hpp                 #   sin, cos, exp, log, pow, etc.
│   │
│   ├── tensor/                      # Wrappers over dp:: types
│   │   ├── scalar.hpp               #   Scalar<T> wraps dp::scalar<T>
│   │   ├── tensor.hpp               #   Tensor<T,N> wraps dp::tensor<T,N>
│   │   ├── matrix.hpp               #   Matrix<T,R,C> wraps dp::matrix<T,R,C>
│   │   └── traits.hpp               #   type traits for tensors
│   │
│   ├── expr/                        # Expression templates
│   │   ├── abstract.hpp             #   CRTP base class
│   │   ├── binary/                  #   binary operations
│   │   │   ├── arithmetic.hpp       #     +, -, *, /
│   │   │   ├── compare.hpp          #     <, >, ==, !=
│   │   │   └── math.hpp             #     pow, atan2, hypot
│   │   ├── unary/                   #   unary operations
│   │   │   ├── math.hpp             #     sqrt, abs, sin, cos
│   │   │   └── bool.hpp             #     !, all_of, any_of
│   │   └── views/                   #   tensor views/slices
│   │       ├── view.hpp             #     1D/2D/ND views
│   │       ├── diag.hpp             #     diagonal view
│   │       └── filter.hpp           #     boolean mask filter
│   │
│   ├── linalg/                      # Linear algebra
│   │   ├── linalg.hpp               #   module header
│   │   ├── matmul.hpp               #   matrix multiplication
│   │   ├── transpose.hpp            #   transpose
│   │   ├── inverse.hpp              #   matrix inverse
│   │   ├── determinant.hpp          #   determinant
│   │   ├── trace.hpp                #   trace
│   │   ├── norm.hpp                 #   frobenius, L2, etc.
│   │   ├── solve.hpp                #   linear solve Ax=b
│   │   └── decompose/               #   matrix decompositions
│   │       ├── lu.hpp               #     LU factorization
│   │       ├── qr.hpp               #     QR factorization
│   │       └── svd.hpp              #     singular value decomposition
│   │
│   ├── algebra/                     # Tensor algebra
│   │   ├── algebra.hpp              #   module header
│   │   ├── einsum.hpp               #   Einstein summation
│   │   ├── contraction.hpp          #   tensor contraction
│   │   ├── permute.hpp              #   tensor permutation
│   │   ├── inner.hpp                #   inner product
│   │   └── outer.hpp                #   outer product
│   │
│   └── backend/                     # Internal optimized kernels
│       ├── matmul/                  #   SIMD matmul kernels
│       ├── transpose/               #   SIMD transpose kernels
│       └── reduce/                  #   SIMD reduction kernels
│
└── calc/                            # on::calc namespace
    ├── calc.hpp                     # calc module header
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
    │   ├── gradient.hpp             #   module header
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
    │   ├── adaptive.hpp             #   module header
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
    │   ├── variance.hpp             #   module header
    │   ├── svrg.hpp                 #   SVRG
    │   ├── sarah.hpp                #   SARAH / SARAH+
    │   └── katyusha.hpp             #   Katyusha
    │
    ├── quasi_newton/                # Second-order methods
    │   ├── quasi_newton.hpp         #   module header
    │   ├── lbfgs.hpp                #   L-BFGS
    │   └── iqn.hpp                  #   Incremental Quasi-Newton
    │
    ├── proximal/                    # Proximal methods
    │   ├── proximal.hpp             #   module header
    │   ├── fbs.hpp                  #   Forward-Backward Splitting
    │   ├── fista.hpp                #   FISTA
    │   ├── fasta.hpp                #   FASTA
    │   └── frankwolfe/              #   Frank-Wolfe / Conditional gradient
    │       ├── frankwolfe.hpp       #     Frank-Wolfe optimizer
    │       ├── atoms.hpp            #     Atom dictionary
    │       └── constraint.hpp       #     Constraint types
    │
    ├── constrained/                 # Constrained optimization
    │   ├── constrained.hpp          #   module header
    │   ├── augmented.hpp            #   Augmented Lagrangian
    │   └── sdp/                     #   Semidefinite programming
    │       ├── primal_dual.hpp      #     Primal-dual solver
    │       └── lrsdp.hpp            #     Low-rank SDP
    │
    ├── evolutionary/                # Derivative-free / evolutionary
    │   ├── evolutionary.hpp         #   module header
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
    │   ├── multiobjective.hpp       #   module header
    │   ├── nsga2.hpp                #   NSGA-II
    │   ├── agemoea.hpp              #   AGE-MOEA
    │   ├── moead.hpp                #   MOEA/D
    │   └── indicator/               #   Quality indicators
    │       ├── epsilon.hpp          #     Epsilon indicator
    │       ├── igd.hpp              #     Inverted Generational Distance
    │       └── hypervolume.hpp      #     Hypervolume indicator
    │
    ├── schedule/                    # Learning rate scheduling
    │   ├── schedule.hpp             #   module header
    │   ├── cyclical.hpp             #   Cyclical LR (SGDR)
    │   ├── warmup.hpp               #   Warm restarts
    │   └── adaptive.hpp             #   SPALeRA, Big Batch
    │
    ├── search/                      # Hyperparameter search
    │   ├── search.hpp               #   module header
    │   └── grid.hpp                 #   Grid search
    │
    └── problem/                     # Benchmark functions
        ├── problem.hpp              #   module header
        ├── unconstrained/           #   Single-objective test functions
        │   ├── rosenbrock.hpp       #     Rosenbrock function
        │   ├── sphere.hpp           #     Sphere function
        │   ├── rastrigin.hpp        #     Rastrigin function
        │   └── ackley.hpp           #     Ackley function
        └── multiobjective/          #   Multi-objective test functions
            ├── dtlz/                #     DTLZ test suite
            ├── zdt/                 #     ZDT test suite
            └── schaffer.hpp         #     Schaffer functions
```

---

## Component Details

### on::simd - SIMD Math Layer

#### Tensor Wrappers (Composition over dp::)

```cpp
namespace optinum::simd {
namespace dp = ::datapod;

template <typename T>
class Scalar {
    dp::scalar<T> pod_{};
public:
    constexpr dp::scalar<T>& pod() noexcept { return pod_; }
    // SIMD ops...
};

template <typename T, std::size_t N>
class Tensor {
    dp::tensor<T, N> pod_{};
public:
    constexpr dp::tensor<T, N>& pod() noexcept { return pod_; }
    // SIMD ops: dot, norm, +, -, *, /
};

template <typename T, std::size_t R, std::size_t C>
class Matrix {
    dp::matrix<T, R, C> pod_{};
public:
    constexpr dp::matrix<T, R, C>& pod() noexcept { return pod_; }
    // SIMD ops: matmul, transpose, inverse, det, trace
};

} // namespace optinum::simd
```

#### SIMD Intrinsics Abstraction

```cpp
namespace optinum::simd {

template <typename T, std::size_t Width>
class SIMDVec {
    // Wraps __m128, __m256, __m512, float32x4_t, etc.
public:
    static SIMDVec load(const T* ptr);
    static SIMDVec loadu(const T* ptr);  // unaligned
    void store(T* ptr) const;
    void storeu(T* ptr) const;           // unaligned

    SIMDVec operator+(SIMDVec rhs) const;
    SIMDVec operator-(SIMDVec rhs) const;
    SIMDVec operator*(SIMDVec rhs) const;
    SIMDVec operator/(SIMDVec rhs) const;

    static SIMDVec fma(SIMDVec a, SIMDVec b, SIMDVec c);  // a*b+c
    T hsum() const;  // horizontal sum
};

} // namespace optinum::simd
```

---

### on::calc - Optimization Layer

#### Function Interface

```cpp
namespace optinum::calc {

// User provides a function with these methods:
// - T Evaluate(const MatType& x)
// - void Gradient(const MatType& x, GradType& g)
// - T EvaluateWithGradient(const MatType& x, GradType& g)

template <typename FunctionType, typename MatType = on::simd::Tensor<double, 0>>
class Optimizer {
public:
    MatType Optimize(FunctionType& fn, MatType& x);
};

} // namespace optinum::calc
```

#### Optimizer Example

```cpp
namespace optinum::calc {

class Adam {
public:
    Adam(double lr = 0.001, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8);

    template <typename FunctionType, typename MatType>
    MatType Optimize(FunctionType& fn, MatType& x);

private:
    double lr_, beta1_, beta2_, eps_;
};

} // namespace optinum::calc
```

---

## Usage Examples

### Basic SIMD Operations

```cpp
#include <optinum/optinum.hpp>

namespace dp = datapod;

int main() {
    // Create from datapod
    dp::matrix<float, 4, 4> raw{};
    raw.set_identity();

    // Wrap in optinum for SIMD operations
    on::simd::Matrix<float, 4, 4> A(raw);
    on::simd::Matrix<float, 4, 4> B;
    B.fill(2.0f);

    // SIMD-accelerated operations
    auto C = A * B;           // matmul
    auto D = transpose(C);    // transpose
    auto det = determinant(A);

    // Get back to datapod for serialization/storage
    dp::matrix<float, 4, 4>& result = C.pod();
}
```

### Optimization

```cpp
#include <optinum/optinum.hpp>

// Define objective function
struct Rosenbrock {
    double Evaluate(const on::simd::Tensor<double, 2>& x) {
        double a = 1.0 - x[0];
        double b = x[1] - x[0] * x[0];
        return a * a + 100.0 * b * b;
    }

    void Gradient(const on::simd::Tensor<double, 2>& x,
                  on::simd::Tensor<double, 2>& g) {
        g[0] = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
        g[1] = 200.0 * (x[1] - x[0] * x[0]);
    }
};

int main() {
    Rosenbrock fn;
    on::simd::Tensor<double, 2> x;
    x[0] = -1.0;
    x[1] = 1.0;

    on::calc::Adam optimizer(0.01);
    auto result = optimizer.Optimize(fn, x);
    // result ≈ [1.0, 1.0]
}
```

### Using Callbacks

```cpp
#include <optinum/optinum.hpp>

int main() {
    Rosenbrock fn;
    on::simd::Tensor<double, 2> x;

    on::calc::Adam optimizer;
    on::calc::PrintLoss printer;
    on::calc::EarlyStop stopper(1e-6, 10);

    optimizer.Optimize(fn, x, printer, stopper);
}
```

---

## Implementation Roadmap

### Phase 1: Foundation
- [ ] `simd/config/` - Platform detection
- [ ] `simd/meta/` - Metaprogramming utilities
- [ ] `simd/intrinsic/` - SIMDVec abstraction
- [ ] `simd/tensor/` - Scalar, Tensor, Matrix wrappers

### Phase 2: SIMD Math
- [ ] `simd/math/` - Vectorized transcendentals
- [ ] `simd/expr/` - Expression templates
- [ ] `simd/linalg/` - matmul, inverse, decompose
- [ ] `simd/algebra/` - einsum, contraction

### Phase 3: Optimization Core
- [ ] `calc/core/` - Function traits, interface
- [ ] `calc/callback/` - Callback system
- [ ] `calc/gradient/` - GD, SGD

### Phase 4: Optimizers
- [ ] `calc/adaptive/` - Adam, RMSProp, etc.
- [ ] `calc/quasi_newton/` - L-BFGS
- [ ] `calc/evolutionary/` - CMA-ES, DE, PSO

### Phase 5: Advanced
- [ ] `calc/proximal/` - FISTA, Frank-Wolfe
- [ ] `calc/constrained/` - Augmented Lagrangian
- [ ] `calc/multiobjective/` - NSGA-II, MOEA/D

---

## Testing

**Every header file in `include/` must have a corresponding test file in `test/`.**

The `test/` folder structure mirrors `include/`:

```
include/optinum/simd/scalar.hpp  ->  test/simd/scalar_test.cpp
include/optinum/simd/tensor.hpp  ->  test/simd/tensor_test.cpp
include/optinum/simd/matrix.hpp  ->  test/simd/matrix_test.cpp
include/optinum/calc/core/function.hpp  ->  test/calc/core/function_test.cpp
```

Tests use **doctest**. Do NOT add `DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN` - it's already configured in CMake/xmake.

Example test file:

```cpp
#include <doctest/doctest.h>
#include <optinum/simd/scalar.hpp>

TEST_CASE("Scalar basic operations") {
    optinum::simd::Scalar<float> a(3.0f);
    optinum::simd::Scalar<float> b(2.0f);

    CHECK(a.get() == 3.0f);
    CHECK((a + b).get() == 5.0f);
}
```

---

## Design Principles

1. **Header-only**: No compilation, just include
2. **Composition over inheritance**: `on::simd::*` wraps `dp::*` via `.pod()`
3. **Zero-cost abstractions**: Expression templates, compile-time dimensions
4. **SIMD everywhere**: All math operations vectorized when possible
5. **POD-friendly**: Easy serialization via `datapod`
6. **Modern C++**: Requires C++20 (concepts, constexpr, fold expressions)

---

## Requirements

- **C++20** or later
- **datapod** library (sibling directory)
- **Optional**: AVX2/AVX-512 for best SIMD performance

---

## License

[TBD]
