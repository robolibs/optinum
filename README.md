<img align="right" width="26%" src="./misc/logo.png">

# Optinum

SIMD-accelerated tensor operations and numerical optimization for high-performance robotics applications.

## Development Status

See [TODO.md](./TODO.md) for the complete development plan and current progress.

## Overview

**Optinum** is a header-only C++ library that combines SIMD-accelerated tensor operations with numerical optimization algorithms, specifically designed for robotics applications requiring real-time performance and deterministic behavior.

Perfect for robotics tasks such as:
- **Real-time sensor fusion** - Fast matrix operations for Kalman filters and SLAM
- **Trajectory optimization** - Efficient gradient-based optimization for motion planning
- **Model predictive control** - High-frequency linear algebra for control loops
- **Neural network inference** - SIMD-accelerated tensor operations for onboard AI

Key design principles:
- **Header-only** - Zero compilation, just include and use
- **Non-owning views** - Zero-copy SIMD operations over existing data
- **Real-time friendly** - No dynamic allocation in critical paths
- **POD-compatible** - Easy serialization for ROS2 message passing
- **Deterministic** - Predictable performance for control loops

Built on top of [datapod](https://github.com/robolibs/datapod) for POD data ownership and provides three main modules:
1. **`simd/`** - SIMD-accelerated operations (AVX/SSE/NEON)
2. **`lina/`** - Linear algebra (LU, QR, SVD, solvers)
3. **`opti/`** - Optimization algorithms (planned)

### Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           ROBOTICS APPLICATION                               │
│                    (SLAM, Navigation, Control, etc.)                         │
└────────────────────────────────┬─────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              optinum (on::)                                  │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────────────────┐    │
│  │   on::opti   │  │   on::lina   │  │         on::simd                │    │
│  │ (optimizers) │─▶│ (linear alg) │─▶│  (views + algorithms)           │    │
│  │              │  │              │  │                                 │    │
│  │ • GD, SGD    │  │ • LU, QR     │  │  • view<W>(dp_obj)  [factory]   │    │
│  │ • Adam       │  │ • SVD        │  │  • axpy, dot, norm  [BLAS-like] │    │
│  │ • L-BFGS     │  │ • solve      │  │  • exp, sin, cos    [math]      │    │
│  │ • CMA-ES     │  │ • lstsq      │  │  • pack<T,W>        [SIMD reg]  │    │
│  │ (PLANNED)    │  │ • einsum     │  │                                 │    │
│  └──────────────┘  └──────────────┘  └─────────────────────────────────┘    │
└────────────────────────────────┬─────────────────────────────────────────────┘
                                 │ wraps (zero-copy)
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                            datapod (dp::)                                    │
│                      (POD data storage - owns memory)                        │
│                                                                              │
│  dp::mat::scalar<T>    dp::mat::vector<T,N>    dp::mat::matrix<T,R,C>       │
│       (rank-0)              (rank-1)                 (rank-2)                │
│                                                                              │
│  • Serializable for ROS2    • Cache-aligned    • Column-major (BLAS-like)   │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Data flow:**
```
dp::mat::vector<float, N>   (owns memory - serializable for ROS2)
         ↓
simd::view<W>(dp_vector)    (non-owning view - zero copy)
         ↓
simd::exp(view)             (algorithm layer - generic over view types)
         ↓
simd::exp(pack<float,8>)    (intrinsic layer - AVX/SSE/NEON)
```

## Installation

### Quick Start (CMake FetchContent)

```cmake
include(FetchContent)
FetchContent_Declare(
  optinum
  GIT_REPOSITORY https://github.com/robolibs/optinum
  GIT_TAG main
)
FetchContent_MakeAvailable(optinum)

target_link_libraries(your_robot_node PRIVATE optinum)
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

target("your_robot_node")
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

### Basic Usage: Sensor Fusion with Kalman Filter

```cpp
#include <optinum/optinum.hpp>

namespace dp = datapod;
namespace on = optinum;

// Real-time Kalman filter update for robot odometry
void kalman_update() {
    // State vector: [x, y, theta, vx, vy]
    dp::mat::vector<float, 5> state;
    dp::mat::matrix<float, 5, 5> covariance;
    
    // Create SIMD views (zero-copy, no allocation)
    auto x = on::simd::view<8>(state);
    auto P = on::simd::view<8>(covariance);
    
    // SIMD-accelerated operations for sensor fusion
    on::simd::scale(0.99f, x);           // Prediction step
    on::simd::axpy(1.0f, sensor_data, x); // Measurement update
    
    // Result already in 'state' - ready for ROS2 publish
}
```

### Advanced Usage: Trajectory Optimization

```cpp
#include <optinum/lina/lina.hpp>
#include <optinum/simd/simd.hpp>

namespace on = optinum;

// Solve quadratic program for robot trajectory
void optimize_trajectory() {
    // Dynamics matrix A (linearized around operating point)
    on::simd::Matrix<double, 6, 6> A;  // 6-DOF state
    on::simd::Vector<double, 6> b;     // Target state
    
    // Solve Ax = b using LU decomposition (SIMD-accelerated)
    auto result = on::lina::try_solve(A, b);
    
    if (result.is_ok()) {
        auto x = result.unwrap();
        // Apply control to robot actuators
    } else {
        // Handle singular matrix (infeasible trajectory)
        auto err = result.unwrap_err();
        ROS_ERROR("Optimization failed: %s", err.message().c_str());
    }
}
```

### ROS2 Integration Example

```cpp
#include <optinum/optinum.hpp>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

class SLAMNode : public rclcpp::Node {
    dp::mat::matrix<float, 3, 3> rotation_;  // POD - can serialize
    dp::mat::vector<float, 3> translation_;
    
public:
    void process_lidar_scan(const sensor_msgs::msg::LaserScan& scan) {
        // Create SIMD views for fast ICP alignment
        auto R = on::simd::view<8>(rotation_);
        auto t = on::simd::view<8>(translation_);
        
        // SIMD-accelerated point cloud transformation
        for (const auto& point : scan.ranges) {
            // Fast matrix-vector multiply using SIMD
            on::lina::matmul(R, point_vec, transformed);
        }
    }
};
```

## Features

- **SIMD Math Functions** - Vectorized exp, log, sin, cos, tanh, sqrt (7-27x faster than scalar)
  ```cpp
  auto x = on::simd::view<8>(sensor_data);  // AVX: 8 floats at once
  on::simd::exp(x);  // 7.94x speedup - perfect for activation functions
  on::simd::tanh(x); // 27.55x speedup - fast neural network inference
  ```

- **Non-Owning Views** - Zero-copy SIMD operations over `dp::mat::*` types. Ideal for ROS2 callbacks where data arrives from publishers - no allocations in the critical path.

- **Linear Algebra Suite** - LU, QR, SVD, Cholesky decompositions with SIMD-accelerated solvers. Essential for sensor fusion (Kalman filters), SLAM (bundle adjustment), and control (MPC).

- **Type-Safe Error Handling** - Uses `dp::Result<T, dp::Error>` instead of exceptions. Perfect for real-time systems where exception overhead is unacceptable.

- **POD-Compatible** - All data types from datapod are Plain Old Data, making serialization trivial for:
  - ROS2 message passing (zero-copy shared memory transport)
  - Logging to disk (replay sensor data for testing)
  - Network transmission (multi-robot coordination)

- **Header-Only Design** - No compilation required, just `#include <optinum/optinum.hpp>`. Simplifies ROS2 package dependencies and cross-compilation for embedded targets.

- **Platform SIMD Support** - Automatic detection and dispatch:
  - x86: SSE, AVX, AVX-512
  - ARM: NEON (planned)
  - Scalar fallback for portability

- **Real-Time Performance Characteristics:**
  - Deterministic SIMD paths (no dynamic dispatch in hot loops)
  - Fixed-size containers (compile-time dimensions)
  - No hidden allocations (views are non-owning)
  - Cache-friendly column-major layout (BLAS/LAPACK compatible)

## License

MIT License - see [LICENSE](./LICENSE) for details.

## Acknowledgments

Made possible thanks to [these amazing projects](./ACKNOWLEDGMENTS.md).
