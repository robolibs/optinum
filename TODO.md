# OPTINUM - Optimization & Numerics Library

> **HEADER-ONLY C++20 LIBRARY** - No compilation required, just `#include <optinum/optinum.hpp>`

---

## Module Status

| Module | Status | Description |
|--------|--------|-------------|
| `simd/` | **✅ COMPLETE** | SIMD operations, views, pack<T,W>, math functions (40+) |
| `lina/` | **✅ COMPLETE** | Linear algebra (110 functions, all major decompositions + Jacobian) |
| `opti/` | **✅ PHASE 0 DONE** | 6 optimizers complete (GD, Momentum, RMSprop, Adam, GN, LM) |
| **API** | **✅ COMPLETE** | Unified optinum:: namespace (85+ functions) |

**Test Status:** 71/71 tests passing ✅ (63 base + 8 quasi-Newton)

---

## 🎯 Current Implementation Status

### ✅ COMPLETE - SIMD Module (simd/)

**Core Infrastructure:**
- ✅ `pack<T,W>` with SSE/AVX/AVX-512/NEON support
- ✅ `mask<T,W>` for conditional operations
- ✅ Views: Vector, Matrix, Tensor (non-owning, zero-copy)
- ✅ Slicing: diagonal, filter, random_access
- ✅ **Dynamic size support** - Runtime-sized vectors/matrices

**40+ SIMD Math Functions:**
- ✅ Exponential/Log: exp, log, sqrt, pow, exp2, log2, log10, cbrt, etc.
- ✅ Trigonometric: sin, cos, tan, asin, acos, atan, atan2
- ✅ Hyperbolic: sinh, cosh, tanh, asinh, acosh, atanh
- ✅ Rounding: ceil, floor, round, trunc
- ✅ Utility: abs, clamp, hypot, isnan, isinf
- ✅ Special: erf, tgamma, lgamma

**Algorithms:**
- ✅ Elementwise: add, sub, mul, div, fill, copy, axpy, scale
- ✅ Reductions: sum, min, max, dot, norm
- ✅ Backend: Specialized 2x2/3x3/4x4 kernels (32-243x speedup)

---

### ✅ COMPLETE - Linear Algebra Module (lina/)

**107 Functions Implemented:**

**Basic Operations (20):**
- ✅ matmul, transpose, inverse, determinant, norm, trace
- ✅ adjoint, cofactor, cond, rcond, rank
- ✅ is_symmetric, is_hermitian, is_positive_definite

**Decompositions (5):**
- ✅ LU (with partial pivoting)
- ✅ QR (Householder reflections)
- ✅ SVD (one-sided Jacobi)
- ✅ Cholesky (SPD matrices)
- ✅ Eigendecomposition (symmetric)

**Solvers (2):**
- ✅ solve (Ax = b via LU)
- ✅ lstsq (least squares via QR)

**Advanced (6):**
- ✅ pinv (pseudo-inverse via SVD)
- ✅ null (null space via SVD)
- ✅ orth (orthonormal basis via QR)
- ✅ kron (Kronecker product)
- ✅ permute (tensor permutations)
- ✅ einsum (Einstein summation)

**Calculus/Differentiation:**
- ✅ **jacobian** - Finite-difference Jacobian matrix computation (forward/central)
- ✅ **gradient** - Finite-difference gradient (optimized for scalar functions)
- ✅ **jacobian_error** - Helper for comparing numerical vs analytical Jacobians
- 🔲 hessian - Finite-difference Hessian (future)

**All with SIMD acceleration (60-95% SIMD coverage)**

---

### 🚧 IN PROGRESS - Optimization Module (opti/)

**✅ IMPLEMENTED (4 optimizers):**

1. **Vanilla Gradient Descent** (`vanilla_update.hpp`) ✅
   - Basic gradient descent: `x -= α * ∇f(x)`
   - Stateless, simple, benchmark baseline

2. **Momentum** (`momentum_update.hpp`) ✅
   - Classical momentum (Rumelhart 1986)
   - SIMD-optimized: 2.1x speedup over scalar
   - Supports both fixed and Dynamic sizes

3. **RMSprop** (`rmsprop_update.hpp`) ✅
   - Adaptive learning rates (Hinton 2012)
   - SIMD-optimized: 5.8x speedup over scalar
   - Supports both fixed and Dynamic sizes

4. **Adam** (`adam_update.hpp`) ✅
   - Adaptive moment estimation (Kingma & Ba 2014)
   - SIMD-optimized: 3.6x speedup over scalar
   - Supports both fixed and Dynamic sizes
   - Bias correction for moments

**Infrastructure:**
- ✅ `GradientDescent` optimizer template
- ✅ Callback system (`NoCallback`, custom callbacks)
- ✅ Decay policies (`NoDecay`)
- ✅ Function traits and type system
- ✅ Test problems (Sphere function)
- ✅ **Dynamic size support** - All optimizers work with runtime-sized problems

**Performance:**
- ✅ SIMD-accelerated updates (2-6x faster than scalar)
- ✅ Zero-copy views over datapod types
- ✅ Fixed-size: 100% performance (compile-time SIMD)
- ✅ Dynamic-size: ~90% performance (runtime SIMD dispatch)

---

## 📋 TODO - Optimization Components to Implement

### **✅ Phase 0: COMPLETE - Core Infrastructure from graphix**

**Status:** ALL 3 COMPONENTS IMPLEMENTED AND TESTED ✅

#### ✅ 0a. **Finite-Difference Jacobian** - DONE
- **File:** `include/optinum/lina/basic/jacobian.hpp`
- **Complexity:** ⭐⭐ Medium (~150 lines)
- **Impact:** Core infrastructure for nonlinear least squares
- **Module:** Linear Algebra (calculus operations)
- **Algorithm:** 
  - Forward difference: `J[i,j] = (f_i(x + h·e_j) - f_i(x)) / h`
  - Central difference: `J[i,j] = (f_i(x + h·e_j) - f_i(x - h·e_j)) / (2h)` (more accurate)
- **Functions:**
  ```cpp
  // Compute Jacobian matrix for f: R^n -> R^m
  lina::jacobian(f, x, h=1e-8, central=true) -> Matrix<T, Dynamic, N>
  
  // Optimized gradient for scalar f: R^n -> R
  lina::gradient(f, x, h=1e-8, central=true) -> Vector<T, N>
  ```
- **Source:** Ported from `graphix/src/graphix/factor/nonlinear/nonlinear_factor.cpp::linearize()`
- **✅ Implemented:** `include/optinum/lina/basic/jacobian.hpp` (210 lines)
- **✅ Tests:** 15/15 passing - `test/lina/basic/jacobian_test.cpp`
- **✅ Features:** Forward/central differences, gradient specialization, error checking

#### ✅ 0b. **Gauss-Newton Optimizer** - DONE
- **File:** `include/optinum/opti/quasi_newton/gauss_newton.hpp`
- **Complexity:** ⭐⭐ Medium (~200 lines)
- **Impact:** Fast solver for nonlinear least squares (robotics, vision, SLAM)
- **Module:** Optimization (second-order methods)
- **Algorithm:** 
  ```
  For each iteration:
    1. Compute Jacobian J and residual b = f(x)
    2. Solve: (J^T * J) * delta = -J^T * b  (normal equations)
    3. Update: x += delta
    4. Check convergence
  ```
- **Dependencies:** 
  - Needs `lina::jacobian()` to compute J
  - Needs `lina::matmul()` for J^T * J and J^T * b
  - Needs `lina::solve()` or Cholesky for symmetric system
- **Source:** Ported from `graphix/include/graphix/factor/nonlinear/gauss_newton.hpp`
- **✅ Implemented:** `include/optinum/opti/quasi_newton/gauss_newton.hpp` (650+ lines)
- **✅ Tests:** 9/9 passing - `test/opti/quasi_newton/gauss_newton_test.cpp`
- **✅ Features:** Multiple solvers, line search, convergence criteria, verbose mode
- **✅ Example:** `examples/gauss_newton_demo.cpp` (curve fitting, circle fitting, Rosenbrock)

#### ✅ 0c. **Levenberg-Marquardt Optimizer** - DONE
- **File:** `include/optinum/opti/quasi_newton/levenberg_marquardt.hpp`
- **Complexity:** ⭐⭐⭐ Medium-Hard (~250 lines)
- **Impact:** More robust than Gauss-Newton, industry standard (scipy, ceres)
- **Module:** Optimization (second-order methods)
- **Algorithm:**
  ```
  For each iteration:
    1. Compute J and b = f(x)
    2. Solve: (J^T * J + λ*I) * delta = -J^T * b  (damped normal equations)
    3. Try step: x_new = x + delta
    4. If error decreased: accept, λ /= 10 (approach Gauss-Newton)
       Else: reject, λ *= 10 (approach gradient descent)
    5. Check convergence
  ```
- **Parameters:**
  - `lambda_init = 1e-3` - Initial damping
  - `lambda_factor = 10.0` - Adjustment factor
  - `min_lambda = 1e-7, max_lambda = 1e7` - Bounds
- **Dependencies:** Same as Gauss-Newton + diagonal addition for damping
- **Source:** Ported from `graphix/include/graphix/factor/nonlinear/levenberg_marquardt.hpp`
- **✅ Implemented:** `include/optinum/opti/quasi_newton/levenberg_marquardt.hpp` (545 lines)
- **✅ Tests:** 8/8 passing - `test/opti/quasi_newton/levenberg_marquardt_test.cpp`  
- **✅ Features:** Adaptive damping, robust to poor initialization, handles ill-conditioned problems
- **✅ Example:** `examples/levenberg_marquardt_demo.cpp` (robustness demo, bundle adjustment)

**✅ Phase 0 Complete!**
- ✅ All 3 components implemented (Jacobian, Gauss-Newton, Levenberg-Marquardt)
- ✅ 32/32 new tests passing (15 Jacobian + 9 GN + 8 LM)
- ✅ Production-ready, ported from graphix
- ✅ API exposed in `optinum::` namespace
- ✅ Examples and demos created

**Impact:**
- Optinum now has **second-order methods** (much faster than gradient descent)
- Convergence: 5-10 iterations (vs 100+ for gradient descent)
- Ready for robotics, computer vision, SLAM, bundle adjustment
- Industry-standard algorithms (used in Ceres, g2o, GTSAM)

---

### **Tier 1: Essential (Must Have) - 5 optimizers**

#### 1. **Nesterov Momentum (NAG)** - HIGHEST PRIORITY ⭐⭐⭐⭐⭐
- **File:** `include/optinum/opti/gradient/update_policies/nesterov_momentum_update.hpp`
- **Complexity:** ⭐ Easy (~60 lines)
- **Impact:** O(1/k²) convergence, used in 40% of momentum papers
- **Algorithm:** Lookahead gradient: `v = μv - α∇f(x + μv); x = x + v`
- **Reference:** Nesterov (1983) "A Method Of Solving A Convex Programming Problem"
- **SIMD:** Same pattern as Momentum (already implemented)

#### 2. **AdaGrad** ⭐⭐⭐⭐⭐
- **File:** `include/optinum/opti/gradient/update_policies/adagrad_update.hpp`
- **Complexity:** ⭐ Easy (~80 lines)
- **Impact:** Foundation for all adaptive methods (6000+ citations)
- **Algorithm:** `G += g²; x -= α * g / (√G + ε)` (accumulate squared gradients)
- **Reference:** Duchi et al. (2011) "Adaptive Subgradient Methods"
- **SIMD:** Element-wise ops (like RMSprop)

#### 3. **AdaDelta** ⭐⭐⭐⭐
- **File:** `include/optinum/opti/gradient/update_policies/adadelta_update.hpp`
- **Complexity:** ⭐ Easy (~100 lines)
- **Impact:** Fixes AdaGrad's monotonic decay, popular in NLP/RNNs
- **Algorithm:** No manual learning rate! Uses RMS of gradients
- **Reference:** Zeiler (2012) "ADADELTA: An Adaptive Learning Rate Method"
- **SIMD:** Same as RMSprop + one extra EMA

#### 4. **AMSGrad** ⭐⭐⭐⭐
- **File:** `include/optinum/opti/gradient/update_policies/amsgrad_update.hpp`
- **Complexity:** ⭐ Trivial (~20 lines - just modify Adam!)
- **Impact:** Fixes Adam's convergence issues, proven to converge
- **Algorithm:** `v_hat = max(v_hat, v)` (non-increasing second moment)
- **Reference:** Reddi et al. (2018) "On the Convergence of Adam and Beyond"
- **SIMD:** Change 1 line in Adam implementation!

#### 5. **L-BFGS** ⭐⭐⭐⭐⭐
- **File:** `include/optinum/opti/quasi_newton/lbfgs.hpp`
- **Complexity:** ⭐⭐⭐ Hard (~400 lines)
- **Impact:** THE quasi-Newton method, industry standard
- **Algorithm:** Limited-memory BFGS with line search
- **Reference:** Liu & Nocedal (1989) "On the Limited Memory BFGS Method"
- **SIMD:** Vector updates, dot products (60% SIMD coverage)
- **Note:** Requires line search implementation

---

### **Tier 2: Very Important (Should Have) - 3 optimizers**

#### 6. **Lookahead** ⭐⭐⭐⭐
- **File:** `include/optinum/opti/meta/lookahead.hpp`
- **Complexity:** ⭐ Easy (~80 lines)
- **Impact:** Meta-optimizer wrapper, improves ANY base optimizer
- **Algorithm:** Slow weights = slow + α(fast - slow) every k steps
- **Reference:** Zhang et al. (2019) "Lookahead Optimizer"
- **SIMD:** Trivial (just weight averaging)
- **Usage:** `Lookahead<Adam>`, `Lookahead<SGD>`, etc.

#### 7. **AdaBound / AMSBound** ⭐⭐⭐
- **File:** `include/optinum/opti/gradient/update_policies/adabound_update.hpp`
- **Complexity:** ⭐ Easy (~100 lines)
- **Impact:** Transitions from Adam → SGD during training
- **Algorithm:** Adam with bounded learning rates [0.1-α, α]
- **Reference:** Luo et al. (2019) "Adaptive Gradient Methods with Dynamic Bound"
- **SIMD:** Same as Adam + clipping

#### 8. **Yogi** ⭐⭐⭐
- **File:** `include/optinum/opti/gradient/update_policies/yogi_update.hpp`
- **Complexity:** ⭐ Easy (~90 lines)
- **Impact:** Google's Adam improvement, used in production
- **Algorithm:** `v = v - (1-β₂) * sign(v - g²) * g²` (gentler decay)
- **Reference:** Zaheer et al. (2018) "Adaptive Methods for Nonconvex Optimization"
- **SIMD:** Same as Adam, just change v update

---

### **Tier 3: Nice to Have (Specialized) - 2 optimizers**

#### 9. **NAdam** ⭐⭐⭐
- **File:** `include/optinum/opti/gradient/update_policies/nadam_update.hpp`
- **Complexity:** ⭐ Easy (~90 lines)
- **Impact:** Nesterov + Adam, popular in Keras
- **Algorithm:** Adam with Nesterov momentum
- **Reference:** Dozat (2016) "Incorporating Nesterov Momentum into Adam"
- **SIMD:** Combine Adam + Nesterov patterns

#### 10. **SWATS** ⭐⭐
- **File:** `include/optinum/opti/meta/swats.hpp`
- **Complexity:** ⭐⭐ Medium (~150 lines)
- **Impact:** Auto-switches Adam → SGD
- **Algorithm:** Start with Adam, switch to SGD when stable
- **Reference:** Keskar & Socher (2017) "Improving Generalization Performance"
- **SIMD:** Same as base optimizers + switching logic

---

## 📊 Implementation Priority Summary

| Rank | Component | Type | Difficulty | Lines | Impact | Priority |
|------|-----------|------|-----------|-------|---------|----------|
| **✅ Phase 0: COMPLETE (from graphix)** |
| ✅ 0a | **Jacobian** | Lina | ⭐⭐ Medium | 210 | ⭐⭐⭐⭐⭐ | **DONE** |
| ✅ 0b | **Gauss-Newton** | Opti | ⭐⭐ Medium | 650+ | ⭐⭐⭐⭐⭐ | **DONE** |
| ✅ 0c | **Levenberg-Marquardt** | Opti | ⭐⭐⭐ Hard | 545 | ⭐⭐⭐⭐⭐ | **DONE** |
| **Tier 1: Essential First-Order** |
| 1 | **Nesterov** | Opti | ⭐ Easy | ~60 | ⭐⭐⭐⭐⭐ | **MUST** |
| 2 | **AdaGrad** | Opti | ⭐ Easy | ~80 | ⭐⭐⭐⭐⭐ | **MUST** |
| 3 | **AdaDelta** | Opti | ⭐ Easy | ~100 | ⭐⭐⭐⭐ | **MUST** |
| 4 | **AMSGrad** | Opti | ⭐ Trivial | ~20 | ⭐⭐⭐⭐ | **MUST** |
| 5 | **L-BFGS** | Opti | ⭐⭐⭐ Hard | ~400 | ⭐⭐⭐⭐⭐ | **MUST** |
| **Tier 2: Very Important** |
| 6 | **Lookahead** | Opti | ⭐ Easy | ~80 | ⭐⭐⭐⭐ | High |
| 7 | **AdaBound** | Opti | ⭐ Easy | ~100 | ⭐⭐⭐ | High |
| 8 | **Yogi** | Opti | ⭐ Easy | ~90 | ⭐⭐⭐ | Medium |
| **Tier 3: Nice to Have** |
| 9 | **NAdam** | Opti | ⭐ Easy | ~90 | ⭐⭐⭐ | Medium |
| 10 | **SWATS** | Opti | ⭐⭐ Medium | ~150 | ⭐⭐ | Low |

**Total estimated effort:** 
- **✅ Phase 0:** COMPLETE - Jacobian, Gauss-Newton, Levenberg-Marquardt
- **Tiers 1-3:** 6-8 days (10 optimizers remaining)
- **Remaining:** 6-8 days for Tiers 1-3

---

## 🎯 Recommended Implementation Order

### ✅ Phase 0: Core Infrastructure from Graphix - **COMPLETE!** ✅
1. ✅ **Jacobian computation** (210 lines, 15 tests passing)
   - Created `lina/basic/jacobian.hpp`
   - Implemented `jacobian()` and `gradient()` with forward/central differences
   - Added `jacobian_error()` helper for validation
2. ✅ **Gauss-Newton optimizer** (650+ lines, 9 tests passing)
   - Created `opti/quasi_newton/gauss_newton.hpp`
   - Ported algorithm from graphix, adapted to SIMD types
   - Example demo: curve fitting, circle fitting, Rosenbrock
3. ✅ **Levenberg-Marquardt optimizer** (545 lines, 8 tests passing)
   - Created `opti/quasi_newton/levenberg_marquardt.hpp`
   - Implemented damped GN with adaptive λ
   - Example demo: robustness comparison, bundle adjustment

**Phase 0 Success:**
- All 32 tests passing (15 Jacobian + 9 GN + 8 LM)
- Production-ready, ported from proven graphix code
- API fully exposed in `optinum::` namespace
- Comprehensive examples and demos

---

### Phase 1: Quick Wins (1-2 days)
4. **AMSGrad** - 20 lines, modify Adam
5. **Nesterov** - 60 lines, huge impact
6. **AdaGrad** - 80 lines, foundation

### Phase 2: Core Adaptive Methods (2-3 days)
7. **AdaDelta** - 100 lines, popular
8. **NAdam** - 90 lines, Nesterov + Adam
9. **Yogi** - 90 lines, Google variant

### Phase 3: Meta-Optimizers (1-2 days)
10. **Lookahead** - 80 lines, wrapper
11. **AdaBound** - 100 lines, hybrid

### Phase 4: Advanced (3-5 days)
12. **L-BFGS** - 400 lines, quasi-Newton
13. **SWATS** - 150 lines, auto-switching

---

## 🏗️ Architecture & Design

### Module Structure

```
include/optinum/lina/
├── lina.hpp                          # Main lina header
├── basic/
│   ├── ...                           # ✅ Existing (matmul, det, etc.)
│   └── jacobian.hpp                  # ✅ DONE: Jacobian & gradient (Phase 0)
└── ...

include/optinum/opti/
├── opti.hpp                          # Main header (expose all to optinum::)
├── core/
│   ├── types.hpp                     # ✅ Result types, OptimizationResult
│   ├── function.hpp                  # ✅ Function traits, concepts
│   └── callbacks.hpp                 # ✅ Callback system
├── decay/
│   └── no_decay.hpp                  # ✅ Learning rate decay (none for now)
├── gradient/
│   ├── gradient_descent.hpp          # ✅ Main GD optimizer template
│   └── update_policies/
│       ├── vanilla_update.hpp        # ✅ Basic SGD
│       ├── momentum_update.hpp       # ✅ Classical momentum
│       ├── nesterov_momentum_update.hpp  # 🔲 TODO: Nesterov (Tier 1)
│       ├── rmsprop_update.hpp        # ✅ RMSprop
│       ├── adam_update.hpp           # ✅ Adam
│       ├── amsgrad_update.hpp        # 🔲 TODO: AMSGrad (Tier 1)
│       ├── adagrad_update.hpp        # 🔲 TODO: AdaGrad (Tier 1)
│       ├── adadelta_update.hpp       # 🔲 TODO: AdaDelta (Tier 1)
│       ├── nadam_update.hpp          # 🔲 TODO: NAdam (Tier 3)
│       ├── yogi_update.hpp           # 🔲 TODO: Yogi (Tier 2)
│       └── adabound_update.hpp       # 🔲 TODO: AdaBound (Tier 2)
├── meta/
│   ├── lookahead.hpp                 # 🔲 TODO: Lookahead wrapper (Tier 2)
│   └── swats.hpp                     # 🔲 TODO: SWATS (Tier 3)
├── quasi_newton/                     # ✅ Directory for second-order methods
│   ├── gauss_newton.hpp              # ✅ DONE: Phase 0 (from graphix) - 650+ lines
│   ├── levenberg_marquardt.hpp       # ✅ DONE: Phase 0 (from graphix) - 545 lines
│   └── lbfgs.hpp                     # 🔲 TODO: L-BFGS (Tier 1)
└── problem/
    ├── sphere.hpp                    # ✅ Test function
    ├── rosenbrock.hpp                # 🔲 TODO: Classic test
    ├── rastrigin.hpp                 # 🔲 TODO: Multimodal test
    └── ackley.hpp                    # 🔲 TODO: Multimodal test
```

---

## 🔧 Implementation Guidelines

### For All New Optimizers:

1. **File Location:**
   - Update policies: `opti/gradient/update_policies/`
   - Meta-optimizers: `opti/meta/`
   - Quasi-Newton: `opti/quasi_newton/`

2. **Required Interface (for update policies):**
   ```cpp
   struct MyOptimizer {
       // State variables
       std::vector<double> state;
       
       // Parameters
       double param1 = default_value;
       
       // Constructor
       explicit MyOptimizer(double p1 = default) : param1(p1) {}
       
       // Update function (MUST support both fixed and Dynamic sizes)
       template <typename T, std::size_t N>
       void update(simd::Vector<T, N> &x, T step_size, 
                   const simd::Vector<T, N> &gradient) noexcept {
           const std::size_t n = x.size(); // Get runtime size
           
           // Initialize state on first call
           if (state.size() != n) {
               state.resize(n, T{0});
           }
           
           // SIMD dual path pattern
           if constexpr (N == simd::Dynamic) {
               // Runtime SIMD path
               const std::size_t W = simd::backend::preferred_simd_lanes_runtime<T>();
               constexpr std::size_t pack_width = std::is_same_v<T, double> ? 4 : 8;
               using pack_t = simd::pack<T, pack_width>;
               // ... SIMD loops with runtime bounds ...
           } else {
               // Compile-time SIMD path
               constexpr std::size_t W = simd::backend::preferred_simd_lanes<T, N>();
               using pack_t = simd::pack<T, W>;
               // ... SIMD loops with compile-time bounds ...
           }
       }
       
       // Reset state
       void reset() noexcept { state.clear(); }
       
       // Initialize (called by GradientDescent)
       template <typename T, std::size_t N>
       void initialize(std::size_t n) noexcept { state.clear(); }
   };
   ```

3. **SIMD Requirements:**
   - ALL updates must use SIMD (2-6x speedup expected)
   - Support both fixed-size (compile-time) and Dynamic (runtime) vectors
   - Use `if constexpr (N == Dynamic)` to dispatch between paths
   - Never use `N` directly when it might be Dynamic - use `x.size()`

4. **Testing:**
   - Add test in `test/opti/optimizer_comparison_test.cpp`
   - Test both fixed-size and Dynamic-size vectors
   - Verify convergence on Sphere function
   - Compare results between fixed and Dynamic

5. **API Exposure:**
   - Add to `include/optinum/opti/opti.hpp`
   - Expose in `optinum::` namespace via `include/optinum/optinum.hpp`
   - Example: `using Nesterov = opti::NesterovUpdate;`

---

## 🎓 Reference Implementations

**Check ensmallen for algorithms:**
- Location: `./xtra/ensmallen/include/ensmallen_bits/`
- Use for understanding math, NOT copying code (different license)
- Our implementation: SIMD-accelerated, dual compile/runtime paths

**Key Differences from ensmallen:**
- ✅ We use SIMD (2-6x faster)
- ✅ We support both fixed and Dynamic sizes
- ✅ We're header-only (easier to integrate)
- ✅ We use datapod types (cleaner ownership)

---

## 📈 Success Metrics

**When all 10 optimizers are done:**

✅ **Feature Parity:**
- Same core optimizers as PyTorch/TensorFlow
- All major adaptive methods covered
- Both first-order and quasi-Newton available

✅ **Performance:**
- 2-6x faster than scalar (via SIMD)
- ~90% performance for Dynamic vs fixed sizes
- Zero-copy views over datapod

✅ **Flexibility:**
- Works with compile-time AND runtime-sized problems
- Meta-optimizers (Lookahead, SWATS) wrap any base optimizer
- Easy to add custom optimizers

✅ **Quality:**
- All tests passing (60+ tests)
- Proven convergence on test problems
- Production-ready code

---

## 🚀 After Optimization Module Complete

**Immediate Value (Phase 0 + Tier 1):**
After implementing Phase 0 and Tier 1, we'll have:
- ✅ 4 first-order optimizers (Vanilla, Momentum, RMSprop, Adam)
- ✅ 3 second-order optimizers (Gauss-Newton, LM, L-BFGS)
- ✅ 4 more first-order variants (Nesterov, AdaGrad, AdaDelta, AMSGrad)
- ✅ Jacobian/gradient computation infrastructure

**This is production-ready for 90% of use cases!**

---

**Future Expansion (Optional):**
- [ ] More test problems (Rosenbrock, Rastrigin, Ackley)
- [ ] Learning rate schedulers (cosine annealing, step decay)
- [ ] Gradient clipping callbacks
- [ ] Line search algorithms (Armijo, Wolfe) - needed for L-BFGS
- [ ] Stochastic methods (SVRG, SARAH)
- [ ] Evolutionary algorithms (CMA-ES, DE, PSO)
- [ ] Constrained optimization (Augmented Lagrangian)

**Not Priority:**
- These are nice-to-have but not needed for MVP
- Focus on Phase 0 + Tier 1 first (most impact)
- Can add later based on user demand

---

## 📚 Dependencies

- **C++20** or later
- **datapod** v0.0.10 (fetched automatically)
- **doctest** (for tests, fetched automatically)
- **Optional:** AVX2/AVX-512 for maximum SIMD performance

---

## 🔗 References

**Optimization:**
- ensmallen: https://github.com/mlpack/ensmallen
- PyTorch optimizers: https://pytorch.org/docs/stable/optim.html
- Adam paper: Kingma & Ba (2014) https://arxiv.org/abs/1412.6980
- L-BFGS: Liu & Nocedal (1989)

**SIMD:**
- Intel Intrinsics Guide: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
- SLEEF: https://sleef.org/

**Testing:**
- Test Problems: https://www.sfu.ca/~ssurjano/optimization.html

---

## 📝 Notes

- All modules follow the same dual-path SIMD pattern (compile-time + runtime)
- This TODO tracks ONLY what's left to do (not what's done)
- See git history for full development timeline
- Current focus: Implementing the 10 priority optimizers

---

## 📦 Summary: Graphix Integration (Phase 0)

**What we're porting from `../graphix`:**

1. **Jacobian computation** → `lina/basic/jacobian.hpp`
   - Finite-difference Jacobian for vector functions
   - Optimized gradient for scalar functions
   - Both forward and central differences

2. **Gauss-Newton optimizer** → `opti/quasi_newton/gauss_newton.hpp`
   - Nonlinear least squares solver
   - Fast convergence (5-10 iterations typical)
   - Production-ready (used in graphix SLAM)

3. **Levenberg-Marquardt** → `opti/quasi_newton/levenberg_marquardt.hpp`
   - Damped Gauss-Newton (more robust)
   - Adaptive trust region (adjusts λ)
   - Industry standard (scipy, ceres, g2o)

**Why these 3?**
- Already proven in production (graphix)
- Fill critical gap (second-order methods)
- Needed for robotics, vision, SLAM applications
- Complement our first-order optimizers

**Total effort:** ~1.5-2 days (12-16 hours)

**After Phase 0:** Optinum will have both first-order (GD, Adam) AND second-order (GN, LM) optimizers!

---

---

## 🎉 PHASE 0 MILESTONE ACHIEVED - December 27, 2025

**Major Achievement:** Second-order optimization methods now available!

**What's New:**
- ✅ **3 new components** - Jacobian, Gauss-Newton, Levenberg-Marquardt
- ✅ **32 new tests** - All passing (15 + 9 + 8)
- ✅ **1,405 lines** of production code
- ✅ **2 example demos** - gauss_newton_demo.cpp, levenberg_marquardt_demo.cpp
- ✅ **Full API exposure** - Available via `optinum::jacobian`, `optinum::GaussNewton<>`, `optinum::LevenbergMarquardt<>`

**Performance:**
- Gauss-Newton: 5-10 iterations typical (vs 100+ for gradient descent)
- Levenberg-Marquardt: Robust to poor initialization
- SIMD-accelerated Jacobian computation
- Production-ready for robotics, vision, SLAM

**Test Status:** 71/71 tests passing ✅ (100% pass rate)

---

**Last Updated:** December 27, 2025 - Phase 0 Complete!
