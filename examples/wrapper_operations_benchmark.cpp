// Wrapper Type Operations Benchmark
// Benchmarks dp::mat::vector, dp::mat::matrix, dp::mat::tensor operations: fill, iota, reverse, cast

#include <chrono>
#include <iomanip>
#include <iostream>

#include <optinum/simd/simd.hpp>

namespace dp = datapod;
namespace simd = optinum::simd;

constexpr std::size_t NUM_ITERATIONS = 10000;

class LocalTimer {
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

// ============================================================================
// Vector Operations
// ============================================================================

template <std::size_t N> double benchmark_vector_fill() {
    dp::mat::Vector<float, N> v;
    LocalTimer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        v.fill(3.14f);
    }
    return timer.elapsed_ms();
}

template <std::size_t N> double benchmark_vector_iota() {
    dp::mat::Vector<float, N> v;
    LocalTimer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        simd::backend::iota<float, N>(v.data(), 0.0f, 1.0f);
    }
    return timer.elapsed_ms();
}

template <std::size_t N> double benchmark_vector_iota_start_step() {
    dp::mat::Vector<float, N> v;
    LocalTimer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        simd::backend::iota<float, N>(v.data(), 10.0f, 2.0f);
    }
    return timer.elapsed_ms();
}

template <std::size_t N> double benchmark_vector_reverse() {
    dp::mat::Vector<float, N> v;
    simd::backend::iota<float, N>(v.data(), 0.0f, 1.0f);
    LocalTimer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        simd::backend::reverse<float, N>(v.data());
    }
    return timer.elapsed_ms();
}

template <std::size_t N> double benchmark_vector_zeros() {
    LocalTimer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        dp::mat::Vector<float, N> v;
        simd::backend::fill<float, N>(v.data(), 0.0f);
        (void)v;
    }
    return timer.elapsed_ms();
}

template <std::size_t N> double benchmark_vector_ones() {
    LocalTimer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        dp::mat::Vector<float, N> v;
        simd::backend::fill<float, N>(v.data(), 1.0f);
        (void)v;
    }
    return timer.elapsed_ms();
}

template <std::size_t N> double benchmark_vector_arange() {
    LocalTimer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        dp::mat::Vector<float, N> v;
        simd::backend::iota<float, N>(v.data(), 0.0f, 1.0f);
        (void)v;
    }
    return timer.elapsed_ms();
}

template <std::size_t N> double benchmark_vector_cast_int_to_float() {
    dp::mat::Vector<int, N> vi;
    for (std::size_t i = 0; i < N; ++i) {
        vi[i] = static_cast<int>(i);
    }
    LocalTimer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        dp::mat::Vector<float, N> vf;
        for (std::size_t i = 0; i < N; ++i) {
            vf[i] = static_cast<float>(vi[i]);
        }
        (void)vf;
    }
    return timer.elapsed_ms();
}

template <std::size_t N> double benchmark_vector_cast_float_to_double() {
    dp::mat::Vector<float, N> vf;
    for (std::size_t i = 0; i < N; ++i) {
        vf[i] = static_cast<float>(i);
    }
    LocalTimer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        dp::mat::Vector<double, N> vd;
        for (std::size_t i = 0; i < N; ++i) {
            vd[i] = static_cast<double>(vf[i]);
        }
        (void)vd;
    }
    return timer.elapsed_ms();
}

template <std::size_t N> double benchmark_vector_add() {
    dp::mat::Vector<float, N> v1, v2;
    simd::backend::iota<float, N>(v1.data(), 0.0f, 1.0f);
    simd::backend::iota<float, N>(v2.data(), 10.0f, 1.0f);
    LocalTimer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        dp::mat::Vector<float, N> v3;
        simd::backend::add<float, N>(v3.data(), v1.data(), v2.data());
        (void)v3;
    }
    return timer.elapsed_ms();
}

template <std::size_t N> double benchmark_vector_mul_scalar() {
    dp::mat::Vector<float, N> v;
    simd::backend::iota<float, N>(v.data(), 0.0f, 1.0f);
    LocalTimer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        dp::mat::Vector<float, N> v2;
        simd::backend::mul_scalar<float, N>(v2.data(), v.data(), 2.5f);
        (void)v2;
    }
    return timer.elapsed_ms();
}

// ============================================================================
// Matrix Operations
// ============================================================================

template <std::size_t R, std::size_t C> double benchmark_matrix_fill() {
    dp::mat::Matrix<float, R, C> m;
    LocalTimer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        m.fill(3.14f);
    }
    return timer.elapsed_ms();
}

template <std::size_t R, std::size_t C> double benchmark_matrix_iota() {
    dp::mat::Matrix<float, R, C> m;
    LocalTimer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        simd::backend::iota<float, R * C>(m.data(), 0.0f, 1.0f);
    }
    return timer.elapsed_ms();
}

template <std::size_t R, std::size_t C> double benchmark_matrix_reverse() {
    dp::mat::Matrix<float, R, C> m;
    simd::backend::iota<float, R * C>(m.data(), 0.0f, 1.0f);
    LocalTimer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        simd::backend::reverse<float, R * C>(m.data());
    }
    return timer.elapsed_ms();
}

template <std::size_t R, std::size_t C> double benchmark_matrix_flatten() {
    dp::mat::Matrix<float, R, C> m;
    simd::backend::iota<float, R * C>(m.data(), 0.0f, 1.0f);
    LocalTimer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        dp::mat::Vector<float, R * C> v;
        for (std::size_t i = 0; i < R * C; ++i) {
            v[i] = m[i];
        }
        (void)v;
    }
    return timer.elapsed_ms();
}

template <std::size_t R, std::size_t C> double benchmark_matrix_zeros() {
    LocalTimer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        dp::mat::Matrix<float, R, C> m;
        simd::backend::fill<float, R * C>(m.data(), 0.0f);
        (void)m;
    }
    return timer.elapsed_ms();
}

template <std::size_t R, std::size_t C> double benchmark_matrix_ones() {
    LocalTimer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        dp::mat::Matrix<float, R, C> m;
        simd::backend::fill<float, R * C>(m.data(), 1.0f);
        (void)m;
    }
    return timer.elapsed_ms();
}

template <std::size_t R, std::size_t C> double benchmark_matrix_cast_int_to_float() {
    dp::mat::Matrix<int, R, C> mi;
    for (std::size_t i = 0; i < R * C; ++i) {
        mi[i] = static_cast<int>(i);
    }
    LocalTimer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        dp::mat::Matrix<float, R, C> mf;
        for (std::size_t i = 0; i < R * C; ++i) {
            mf[i] = static_cast<float>(mi[i]);
        }
        (void)mf;
    }
    return timer.elapsed_ms();
}

template <std::size_t R, std::size_t C> double benchmark_matrix_add() {
    dp::mat::Matrix<float, R, C> m1, m2;
    simd::backend::iota<float, R * C>(m1.data(), 0.0f, 1.0f);
    simd::backend::iota<float, R * C>(m2.data(), 10.0f, 1.0f);
    LocalTimer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        dp::mat::Matrix<float, R, C> m3;
        simd::backend::add<float, R * C>(m3.data(), m1.data(), m2.data());
        (void)m3;
    }
    return timer.elapsed_ms();
}

template <std::size_t R, std::size_t C> double benchmark_matrix_mul_scalar() {
    dp::mat::Matrix<float, R, C> m;
    simd::backend::iota<float, R * C>(m.data(), 0.0f, 1.0f);
    LocalTimer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        dp::mat::Matrix<float, R, C> m2;
        simd::backend::mul_scalar<float, R * C>(m2.data(), m.data(), 2.5f);
        (void)m2;
    }
    return timer.elapsed_ms();
}

template <std::size_t R, std::size_t K, std::size_t C> double benchmark_matrix_matmul() {
    dp::mat::Matrix<float, R, K> m1;
    dp::mat::Matrix<float, K, C> m2;
    simd::backend::iota<float, R * K>(m1.data(), 0.0f, 1.0f);
    simd::backend::iota<float, K * C>(m2.data(), 0.0f, 1.0f);
    LocalTimer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        dp::mat::Matrix<float, R, C> m3;
        simd::backend::matmul<float, R, K, C>(m3.data(), m1.data(), m2.data());
        (void)m3;
    }
    return timer.elapsed_ms();
}

template <std::size_t R, std::size_t C> double benchmark_matrix_transpose() {
    dp::mat::Matrix<float, R, C> m;
    simd::backend::iota<float, R * C>(m.data(), 0.0f, 1.0f);
    LocalTimer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        dp::mat::Matrix<float, C, R> mt;
        simd::backend::transpose<float, R, C>(mt.data(), m.data());
        (void)mt;
    }
    return timer.elapsed_ms();
}

// ============================================================================
// Tensor Operations
// ============================================================================

template <std::size_t D1, std::size_t D2, std::size_t D3> double benchmark_tensor_fill() {
    dp::mat::Tensor<float, D1, D2, D3> t;
    LocalTimer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        t.fill(3.14f);
    }
    return timer.elapsed_ms();
}

template <std::size_t D1, std::size_t D2, std::size_t D3> double benchmark_tensor_cast_int_to_float() {
    dp::mat::Tensor<int, D1, D2, D3> ti;
    ti.fill(42);
    LocalTimer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        dp::mat::Tensor<float, D1, D2, D3> tf;
        for (std::size_t i = 0; i < D1 * D2 * D3; ++i) {
            tf[i] = static_cast<float>(ti[i]);
        }
        (void)tf;
    }
    return timer.elapsed_ms();
}

template <std::size_t D1, std::size_t D2, std::size_t D3> double benchmark_tensor_add() {
    dp::mat::Tensor<float, D1, D2, D3> t1, t2;
    t1.fill(1.0f);
    t2.fill(2.0f);
    LocalTimer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        dp::mat::Tensor<float, D1, D2, D3> t3;
        simd::backend::add<float, D1 * D2 * D3>(t3.data(), t1.data(), t2.data());
        (void)t3;
    }
    return timer.elapsed_ms();
}

template <std::size_t D1, std::size_t D2, std::size_t D3> double benchmark_tensor_mul_scalar() {
    dp::mat::Tensor<float, D1, D2, D3> t;
    t.fill(1.0f);
    LocalTimer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        dp::mat::Tensor<float, D1, D2, D3> t2;
        simd::backend::mul_scalar<float, D1 * D2 * D3>(t2.data(), t.data(), 2.5f);
        (void)t2;
    }
    return timer.elapsed_ms();
}

// ============================================================================
// Main Benchmark Runner
// ============================================================================

void print_result(const char *name, double time_ms, std::size_t size) {
    double ops_per_sec = (NUM_ITERATIONS * 1000.0) / time_ms;
    double throughput_mb = (NUM_ITERATIONS * size * sizeof(float)) / (time_ms * 1024.0);
    std::cout << std::setw(30) << std::left << name << std::setw(12) << std::right << std::fixed << std::setprecision(2)
              << time_ms << " ms" << std::setw(12) << ops_per_sec << " op/s" << std::setw(12) << throughput_mb
              << " MB/s\n";
}

int main() {
    std::cout << "\n=== Wrapper Type Operations Benchmark ===\n";
    std::cout << "Iterations: " << NUM_ITERATIONS << "\n\n";

    // Vector Operations
    std::cout << "--- dp::mat::Vector<float, 1024> Operations ---\n";
    std::cout << std::setw(30) << std::left << "Operation" << std::setw(12) << std::right << "Time" << std::setw(12)
              << "Ops/sec" << std::setw(12) << "Throughput\n";
    std::cout << std::string(66, '-') << "\n";

    constexpr std::size_t VEC_SIZE = 1024;
    print_result("fill", benchmark_vector_fill<VEC_SIZE>(), VEC_SIZE);
    print_result("iota", benchmark_vector_iota<VEC_SIZE>(), VEC_SIZE);
    print_result("iota(start, step)", benchmark_vector_iota_start_step<VEC_SIZE>(), VEC_SIZE);
    print_result("reverse", benchmark_vector_reverse<VEC_SIZE>(), VEC_SIZE);
    print_result("zeros", benchmark_vector_zeros<VEC_SIZE>(), VEC_SIZE);
    print_result("ones", benchmark_vector_ones<VEC_SIZE>(), VEC_SIZE);
    print_result("arange", benchmark_vector_arange<VEC_SIZE>(), VEC_SIZE);
    print_result("cast<float>(int)", benchmark_vector_cast_int_to_float<VEC_SIZE>(), VEC_SIZE);
    print_result("cast<double>(float)", benchmark_vector_cast_float_to_double<VEC_SIZE>(), VEC_SIZE);
    print_result("operator+", benchmark_vector_add<VEC_SIZE>(), VEC_SIZE);
    print_result("operator* (scalar)", benchmark_vector_mul_scalar<VEC_SIZE>(), VEC_SIZE);

    // Matrix Operations (64x64)
    std::cout << "\n--- dp::mat::Matrix<float, 64, 64> Operations ---\n";
    std::cout << std::setw(30) << std::left << "Operation" << std::setw(12) << std::right << "Time" << std::setw(12)
              << "Ops/sec" << std::setw(12) << "Throughput\n";
    std::cout << std::string(66, '-') << "\n";

    constexpr std::size_t MAT_R = 64;
    constexpr std::size_t MAT_C = 64;
    constexpr std::size_t MAT_SIZE = MAT_R * MAT_C;

    print_result("fill", benchmark_matrix_fill<MAT_R, MAT_C>(), MAT_SIZE);
    print_result("iota", benchmark_matrix_iota<MAT_R, MAT_C>(), MAT_SIZE);
    print_result("reverse", benchmark_matrix_reverse<MAT_R, MAT_C>(), MAT_SIZE);
    print_result("flatten", benchmark_matrix_flatten<MAT_R, MAT_C>(), MAT_SIZE);
    print_result("zeros", benchmark_matrix_zeros<MAT_R, MAT_C>(), MAT_SIZE);
    print_result("ones", benchmark_matrix_ones<MAT_R, MAT_C>(), MAT_SIZE);
    print_result("cast<float>(int)", benchmark_matrix_cast_int_to_float<MAT_R, MAT_C>(), MAT_SIZE);
    print_result("operator+", benchmark_matrix_add<MAT_R, MAT_C>(), MAT_SIZE);
    print_result("operator* (scalar)", benchmark_matrix_mul_scalar<MAT_R, MAT_C>(), MAT_SIZE);
    print_result("operator* (matmul)", benchmark_matrix_matmul<MAT_R, MAT_R, MAT_C>(), MAT_SIZE);
    print_result("transpose", benchmark_matrix_transpose<MAT_R, MAT_C>(), MAT_SIZE);

    // Smaller Vector for detailed comparison
    std::cout << "\n--- dp::mat::Vector<float, 16> Operations (Small Size) ---\n";
    std::cout << std::setw(30) << std::left << "Operation" << std::setw(12) << std::right << "Time" << std::setw(12)
              << "Ops/sec" << std::setw(12) << "Throughput\n";
    std::cout << std::string(66, '-') << "\n";

    constexpr std::size_t SMALL_VEC = 16;
    print_result("fill", benchmark_vector_fill<SMALL_VEC>(), SMALL_VEC);
    print_result("iota", benchmark_vector_iota<SMALL_VEC>(), SMALL_VEC);
    print_result("reverse", benchmark_vector_reverse<SMALL_VEC>(), SMALL_VEC);
    print_result("zeros", benchmark_vector_zeros<SMALL_VEC>(), SMALL_VEC);

    // Smaller Matrix (8x8)
    std::cout << "\n--- dp::mat::Matrix<float, 8, 8> Operations (Small Size) ---\n";
    std::cout << std::setw(30) << std::left << "Operation" << std::setw(12) << std::right << "Time" << std::setw(12)
              << "Ops/sec" << std::setw(12) << "Throughput\n";
    std::cout << std::string(66, '-') << "\n";

    constexpr std::size_t SMALL_MAT = 8;
    constexpr std::size_t SMALL_MAT_SIZE = SMALL_MAT * SMALL_MAT;
    print_result("fill", benchmark_matrix_fill<SMALL_MAT, SMALL_MAT>(), SMALL_MAT_SIZE);
    print_result("iota", benchmark_matrix_iota<SMALL_MAT, SMALL_MAT>(), SMALL_MAT_SIZE);
    print_result("matmul 8x8x8", benchmark_matrix_matmul<SMALL_MAT, SMALL_MAT, SMALL_MAT>(), SMALL_MAT_SIZE);

    // Tensor Operations (8x8x8)
    std::cout << "\n--- dp::mat::Tensor<float, 8, 8, 8> Operations ---\n";
    std::cout << std::setw(30) << std::left << "Operation" << std::setw(12) << std::right << "Time" << std::setw(12)
              << "Ops/sec" << std::setw(12) << "Throughput\n";
    std::cout << std::string(66, '-') << "\n";

    constexpr std::size_t TENSOR_D = 8;
    constexpr std::size_t TENSOR_SIZE = TENSOR_D * TENSOR_D * TENSOR_D;
    print_result("fill", benchmark_tensor_fill<TENSOR_D, TENSOR_D, TENSOR_D>(), TENSOR_SIZE);
    print_result("cast<float>(int)", benchmark_tensor_cast_int_to_float<TENSOR_D, TENSOR_D, TENSOR_D>(), TENSOR_SIZE);
    print_result("operator+", benchmark_tensor_add<TENSOR_D, TENSOR_D, TENSOR_D>(), TENSOR_SIZE);
    print_result("operator* (scalar)", benchmark_tensor_mul_scalar<TENSOR_D, TENSOR_D, TENSOR_D>(), TENSOR_SIZE);

    std::cout << "\n=== Benchmark Complete ===\n\n";
    return 0;
}
