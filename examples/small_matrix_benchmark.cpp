// =============================================================================
// Small Matrix Operations Benchmark
// Compares specialized kernels vs general LU-based approach for 2x2, 3x3, 4x4
// =============================================================================

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>

#include <optinum/lina/basic/determinant.hpp>
#include <optinum/lina/basic/inverse.hpp>
#include <optinum/lina/decompose/lu.hpp>
#include <optinum/simd/backend/det_small.hpp>
#include <optinum/simd/backend/inverse_small.hpp>
#include <optinum/simd/matrix.hpp>

constexpr std::size_t NUM_ITERATIONS = 1000000;

namespace on = optinum;

// =============================================================================
// Timing Utilities
// =============================================================================

class Timer {
  public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
        return static_cast<double>(duration.count()) / 1000.0;
    }

  private:
    std::chrono::high_resolution_clock::time_point start_;
};

// =============================================================================
// Benchmark Functions
// =============================================================================

template <std::size_t N> double benchmark_specialized_determinant() {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    on::simd::Matrix<float, N, N> m;
    for (std::size_t i = 0; i < N * N; ++i) {
        m.data()[i] = dist(gen);
    }

    Timer timer;
    timer.start();

    volatile float result = 0.0f;
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        if constexpr (N == 2) {
            result = on::simd::backend::det_2x2(m.data());
        } else if constexpr (N == 3) {
            result = on::simd::backend::det_3x3(m.data());
        } else if constexpr (N == 4) {
            result = on::simd::backend::det_4x4(m.data());
        }
    }

    (void)result; // Prevent optimization
    return timer.elapsed_ms();
}

template <std::size_t N> double benchmark_lu_determinant() {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    on::simd::Matrix<float, N, N> m;
    for (std::size_t i = 0; i < N * N; ++i) {
        m.data()[i] = dist(gen);
    }

    Timer timer;
    timer.start();

    volatile float result = 0.0f;
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        const auto f = on::lina::lu<float, N>(m);
        float det = static_cast<float>(f.sign);
        for (std::size_t i = 0; i < N; ++i) {
            det *= f.u(i, i);
        }
        result = det;
    }

    (void)result;
    return timer.elapsed_ms();
}

template <std::size_t N> double benchmark_specialized_inverse() {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(1.0f, 10.0f); // Positive to avoid singular

    // Create a well-conditioned matrix
    on::simd::Matrix<float, N, N> m;
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            m(i, j) = (i == j) ? 10.0f : dist(gen);
        }
    }

    float result_data[N * N];

    Timer timer;
    timer.start();

    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        if constexpr (N == 2) {
            on::simd::backend::inverse_2x2(m.data(), result_data);
        } else if constexpr (N == 3) {
            on::simd::backend::inverse_3x3(m.data(), result_data);
        } else if constexpr (N == 4) {
            on::simd::backend::inverse_4x4(m.data(), result_data);
        }
    }

    volatile float prevent_opt = result_data[0];
    (void)prevent_opt;

    return timer.elapsed_ms();
}

template <std::size_t N> double benchmark_lu_inverse() {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(1.0f, 10.0f);

    on::simd::Matrix<float, N, N> m;
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            m(i, j) = (i == j) ? 10.0f : dist(gen);
        }
    }

    Timer timer;
    timer.start();

    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        const auto f = on::lina::lu<float, N>(m);
        on::simd::Matrix<float, N, N> inv;
        for (std::size_t col = 0; col < N; ++col) {
            on::simd::Vector<float, N> e;
            e.fill(0.0f);
            e[col] = 1.0f;
            const auto x = on::lina::lu_solve(f, e);
            for (std::size_t row = 0; row < N; ++row) {
                inv(row, col) = x[row];
            }
        }
    }

    return timer.elapsed_ms();
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║           Small Matrix Operations Benchmark                               ║\n";
    std::cout << "║           Specialized Kernels vs General LU Decomposition                 ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
    std::cout << "Iterations: " << NUM_ITERATIONS << "\n";
    std::cout << "\n";

    // Header
    std::cout << std::setw(15) << "Operation" << std::setw(20) << "Specialized (ms)" << std::setw(20) << "LU-based (ms)"
              << std::setw(15) << "Speedup"
              << "\n";
    std::cout << std::string(70, '-') << "\n";

    // 2x2 Determinant
    {
        double specialized = benchmark_specialized_determinant<2>();
        double lu_based = benchmark_lu_determinant<2>();
        std::cout << std::setw(15) << "2x2 det" << std::setw(20) << std::fixed << std::setprecision(3) << specialized
                  << std::setw(20) << lu_based << std::setw(15) << std::setprecision(2) << (lu_based / specialized)
                  << "x\n";
    }

    // 2x2 Inverse
    {
        double specialized = benchmark_specialized_inverse<2>();
        double lu_based = benchmark_lu_inverse<2>();
        std::cout << std::setw(15) << "2x2 inverse" << std::setw(20) << std::fixed << std::setprecision(3)
                  << specialized << std::setw(20) << lu_based << std::setw(15) << std::setprecision(2)
                  << (lu_based / specialized) << "x\n";
    }

    std::cout << std::string(70, '-') << "\n";

    // 3x3 Determinant
    {
        double specialized = benchmark_specialized_determinant<3>();
        double lu_based = benchmark_lu_determinant<3>();
        std::cout << std::setw(15) << "3x3 det" << std::setw(20) << std::fixed << std::setprecision(3) << specialized
                  << std::setw(20) << lu_based << std::setw(15) << std::setprecision(2) << (lu_based / specialized)
                  << "x\n";
    }

    // 3x3 Inverse
    {
        double specialized = benchmark_specialized_inverse<3>();
        double lu_based = benchmark_lu_inverse<3>();
        std::cout << std::setw(15) << "3x3 inverse" << std::setw(20) << std::fixed << std::setprecision(3)
                  << specialized << std::setw(20) << lu_based << std::setw(15) << std::setprecision(2)
                  << (lu_based / specialized) << "x\n";
    }

    std::cout << std::string(70, '-') << "\n";

    // 4x4 Determinant
    {
        double specialized = benchmark_specialized_determinant<4>();
        double lu_based = benchmark_lu_determinant<4>();
        std::cout << std::setw(15) << "4x4 det" << std::setw(20) << std::fixed << std::setprecision(3) << specialized
                  << std::setw(20) << lu_based << std::setw(15) << std::setprecision(2) << (lu_based / specialized)
                  << "x\n";
    }

    // 4x4 Inverse
    {
        double specialized = benchmark_specialized_inverse<4>();
        double lu_based = benchmark_lu_inverse<4>();
        std::cout << std::setw(15) << "4x4 inverse" << std::setw(20) << std::fixed << std::setprecision(3)
                  << specialized << std::setw(20) << lu_based << std::setw(15) << std::setprecision(2)
                  << (lu_based / specialized) << "x\n";
    }

    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                          Benchmark Complete                                ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    return 0;
}
