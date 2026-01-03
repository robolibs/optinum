<img align="right" width="26%" src="./misc/logo.png">

# Optinum

SIMD-accelerated tensor operations and numerical optimization for high-performance robotics applications.

## Development Status

See [TODO.md](./TODO.md) for the complete development plan and current progress.

## Overview

**Optinum** is a header-only C++20 library that combines SIMD-accelerated tensor operations with numerical optimization algorithms, specifically designed for applications requiring real-time performance and deterministic behavior.

The library provides five integrated modules:
- **`simd/`** - SIMD-accelerated operations (SSE/AVX/AVX-512/NEON) with 39 vectorized math functions
- **`lina/`** - Linear algebra (LU, QR, SVD, Cholesky, eigendecomposition, solvers)
- **`lie/`** - Lie groups (SO2, SE2, SO3, SE3, Sim2, Sim3, RxSO2, RxSO3) with batched SIMD operations
- **`opti/`** - Gradient-based optimization (11 update policies, L-BFGS, Gauss-Newton, Levenberg-Marquardt)
- **`meta/`** - Metaheuristic optimization (PSO, CEM, CMA-ES, DE, GA, SA, MPPI, Lookahead, SWATS)

Key design principles:
- **Header-only** - Zero compilation, just include and use
- **Non-owning views** - Zero-copy SIMD operations over existing data
- **Real-time friendly** - No dynamic allocation in critical paths
- **POD-compatible** - Easy serialization for ROS2 message passing
- **Deterministic** - Predictable performance for control loops

Built on top of [datapod](https://codeberg.org/robolibs/datapod) for POD data ownership. Uses `on::` as short namespace alias (enabled by default in examples/tests via `-DSHORT_NAMESPACE`).

### Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              APPLICATION LAYER                               │
│                    (SLAM, Navigation, Control, Planning)                     │
└────────────────────────────────────┬─────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              optinum (on::)                                  │
│                                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌───────────────┐  │
│  │on::meta  │  │on::opti  │  │on::lina  │  │ on::lie   │  │   on::simd    │  │
│  │(meta-    │─▶│(gradient │─▶│(linear   │─▶│(Lie       │─▶│(SIMD views +  │  │
│  │heuristic)│  │  based)  │  │  algebra)│  │ groups)   │  │  algorithms)  │  │
│  │          │  │          │  │          │  │           │  │               │  │
│  │• PSO     │  │• Adam    │  │• LU, QR  │  │• SO2/SO3  │  │• pack<T,W>    │  │
│  │• CEM     │  │• L-BFGS  │  │• SVD     │  │• SE2/SE3  │  │• 39 math      │  │
│  │• CMA-ES  │  │• Gauss-  │  │• Cholesky│  │• Sim2/3   │  │• views        │  │
│  │• MPPI    │  │  Newton  │  │• solve   │  │• batched  │  │• algorithms   │  │
│  └──────────┘  └──────────┘  └──────────┘  └───────────┘  └───────────────┘  │
└────────────────────────────────────┬─────────────────────────────────────────┘
                                     │ wraps (zero-copy)
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                            datapod (dp::)                                    │
│                      (POD data storage - owns memory)                        │
│                                                                              │
│  dp::mat::scalar<T>      dp::mat::vector<T,N>    dp::mat::matrix<T,R,C>      │
│       (rank-0)                (rank-1)                 (rank-2)              │
│                                                                              │
│• Serializable for ROS2    • Cache-aligned      • Column-major (BLAS-like)    │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Data flow:**
```
dp::mat::vector<float, N>   (owns memory - serializable for ROS2)
         ↓
on::Vector<float, N>           (a view over on::simd::pack<float,W>)
         ↓
on::simd::view<W>(dp_vector)    (non-owning view - zero copy)
         ↓
on::simd::exp(view)             (algorithm layer - generic over view types)
         ↓
on::simd::exp(pack<float,8>)    (intrinsic layer - AVX/SSE/NEON)
```

## Installation

### Quick Start (CMake FetchContent)

```cmake
include(FetchContent)
FetchContent_Declare(
  optinum
  GIT_REPOSITORY https://codeberg.org/robolibs/optinum
  GIT_TAG main
)
FetchContent_MakeAvailable(optinum)

target_link_libraries(your_target PRIVATE optinum)
```

### Recommended: XMake

[XMake](https://xmake.io/) is a modern, fast, and cross-platform build system.

**Install XMake:**
```bash
curl -fsSL https://xmake.io/shget.text | bash
```

**Add to your xmake.lua:**
```lua
add_requires("optinum")

target("your_target")
    set_kind("binary")
    add_packages("optinum")
    add_files("src/*.cpp")
```

**Build:**
```bash
xmake
xmake run
```

### Complete Development Environment (Nix + Direnv + Devbox)

For the ultimate reproducible development environment:

**1. Install Nix (package manager from NixOS):**
```bash
# Determinate Nix Installer (recommended)
curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s -- install
```
[Nix](https://nixos.org/) - Reproducible, declarative package management

**2. Install direnv (automatic environment switching):**
```bash
sudo apt install direnv

# Add to your shell (~/.bashrc or ~/.zshrc):
eval "$(direnv hook bash)"  # or zsh
```
[direnv](https://direnv.net/) - Load environment variables based on directory

**3. Install Devbox (Nix-powered development environments):**
```bash
curl -fsSL https://get.jetpack.io/devbox | bash
```
[Devbox](https://www.jetpack.io/devbox/) - Portable, isolated dev environments

**4. Use the environment:**
```bash
cd optinum
direnv allow  # Allow .envrc (one-time)
# Environment automatically loaded! All dependencies available.

make build   # or xmake
make test
```

## Usage

### Basic Usage: SIMD-Accelerated Operations

```cpp
#include <optinum/optinum.hpp>

void process_sensor_data() {
    // State vector: [x, y, theta, vx, vy]
    dp::mat::vector<float, 5> state;
    dp::mat::matrix<float, 5, 5> covariance;

    // Create SIMD views (zero-copy, no allocation)
    auto x = on::simd::view<8>(state);
    auto P = on::simd::view<8>(covariance);

    // SIMD-accelerated operations
    on::simd::scale(0.99f, x);              // Prediction step
    on::simd::axpy(1.0f, sensor_data, x);   // Measurement update

    // Result already in 'state' - ready for serialization
}
```

### Linear Algebra: Solving Systems

```cpp
#include <optinum/lina/lina.hpp>

void solve_dynamics() {
    on::Matrix<double, 6, 6> A;  // Dynamics matrix
    on::Vector<double, 6> b;     // Target state

    // Solve Ax = b using LU decomposition (SIMD-accelerated)
    auto result = on::lina::try_solve(A, b);

    if (result.is_ok()) {
        auto x = result.unwrap();
        // Apply solution
    }
}
```

### Lie Groups: 3D Transformations

```cpp
#include <optinum/lie/lie.hpp>

void transform_points() {
    // Create SE3 pose from rotation and translation
    on::lie::SE3d pose = on::lie::SE3d::exp({0.1, 0.2, 0.3, 1.0, 2.0, 3.0});

    // Transform a point
    on::Vector<double, 3> point{1.0, 0.0, 0.0};
    auto transformed = pose.act(point);

    // Batched operations for point clouds
    on::lie::SE3Batch<double, 100> poses;  // 100 poses processed in parallel
}
```

### Optimization: Gradient-Based

```cpp
#include <optinum/opti/opti.hpp>

void optimize_trajectory() {
    // Define objective function
    auto objective = [](const auto& x) {
        return on::lina::dot(x, x);  // Sphere function
    };

    // Configure Adam optimizer
    on::opti::Adam<double> optimizer({
        .learning_rate = 0.01,
        .beta1 = 0.9,
        .beta2 = 0.999
    });

    on::Vector<double, 10> x;  // Initial guess
    auto result = optimizer.optimize(objective, x);
}
```

### Metaheuristic: Global Optimization

```cpp
#include <optinum/meta/meta.hpp>

void global_search() {
    // CMA-ES for non-convex optimization
    on::meta::CMAES<double> optimizer({
        .population_size = 50,
        .max_iterations = 1000
    });

    auto result = optimizer.optimize(rastrigin_function, lower_bounds, upper_bounds);
}
```

## Features

- **SIMD Math Functions** - 39 vectorized functions (exp, log, sin, cos, tanh, sqrt, erf, gamma, hypot)
  ```cpp
  auto x = on::simd::view<8>(data);  // AVX: 8 floats at once
  on::simd::exp(x);   // 7.94x speedup
  on::simd::tanh(x);  // 27.55x speedup
  ```

- **Linear Algebra Suite** - LU, QR, SVD, Cholesky, eigendecomposition with SIMD-accelerated solvers
  ```cpp
  auto [L, U, P] = on::lina::lu(A);   // LU decomposition with pivoting
  auto [Q, R] = on::lina::qr(A);      // QR decomposition
  auto [U, S, V] = on::lina::svd(A);  // Singular value decomposition
  ```

- **Lie Groups** - SO2, SE2, SO3, SE3, Sim2, Sim3, RxSO2, RxSO3 with exp/log maps, adjoints, and Jacobians
  ```cpp
  auto rotation = on::lie::SO3d::exp({0.1, 0.2, 0.3});
  auto pose = on::lie::SE3d::from_rotation_translation(rotation, translation);
  ```

- **11 Gradient Update Policies** - Adam, AdaGrad, AdaDelta, RMSprop, NAdam, AdaBound, Yogi, Nesterov, Momentum, AMSGrad, Vanilla

- **8 Decay Policies** - Cosine annealing, exponential, inverse time, linear, polynomial, step, warmup, no decay

- **Quasi-Newton Methods** - L-BFGS, Gauss-Newton, Levenberg-Marquardt for nonlinear least squares

- **9 Metaheuristics** - PSO, CEM, CMA-ES, DE, GA, SA, MPPI, Lookahead, SWATS for global and black-box optimization

- **Non-Owning Views** - Zero-copy SIMD operations over `dp::mat::*` types

- **Type-Safe Error Handling** - Uses `dp::Result<T, dp::Error>` instead of exceptions

- **Platform SIMD Support** - Automatic detection: SSE, AVX, AVX-512 (x86), NEON (ARM), scalar fallback

- **Real-Time Characteristics:**
  - Deterministic SIMD paths (no dynamic dispatch in hot loops)
  - Fixed-size containers (compile-time dimensions)
  - No hidden allocations (views are non-owning)
  - Cache-friendly column-major layout (BLAS/LAPACK compatible)

## Error Handling Strategy

Optinum uses a consistent error handling approach designed for real-time and embedded systems:

1. **Fallible operations** use `dp::Result<T, dp::Error>`:
   - `try_solve()`, `try_inverse()`, `try_lstsq()`, `try_dare()` - return Result
   - `solve()`, `inverse()`, `lstsq()`, `dare()` - wrapper that returns zero/identity on error
   ```cpp
   auto result = on::lina::try_solve(A, b);
   if (result.is_ok()) {
       auto x = result.unwrap();
   } else {
       // Handle error: result.error().message()
   }
   ```

2. **Bounds checking** uses `std::out_of_range` (STL convention):
   - `at()` methods throw `std::out_of_range`
   - `operator[]` does debug-only bounds checking (via `assert`)

3. **Optimizers** use status field in result struct:
   - `OptimizationResult.converged = false` on failure
   - `OptimizationResult.status` contains error message
   ```cpp
   auto result = optimizer.optimize(objective, x);
   if (!result.converged) {
       std::cerr << "Optimization failed: " << result.status << "\n";
   }
   ```

4. **Never use exceptions** for recoverable errors in new code - prefer `dp::Result` for explicit error handling

## Module Summary

| Module | Files | Lines | Description |
|--------|-------|-------|-------------|
| `simd/` | 92 | ~23,000 | SIMD pack types, views, 39 math functions |
| `lina/` | 37 | ~5,400 | 7 decompositions, solvers, DARE, Jacobian, Hessian |
| `lie/` | 22 | ~9,600 | 12 Lie groups, batched SIMD, splines, averaging |
| `opti/` | 32 | ~5,400 | 11 update policies, 8 decay policies, line search |
| `meta/` | 10 | ~3,900 | 9 metaheuristics |

**Test Status:** 104/105 test suites passing (500+ test cases)

## License

MIT License - see [LICENSE](./LICENSE) for details.

## Acknowledgments

Made possible thanks to [these amazing projects](./ACKNOWLEDGMENTS.md).
