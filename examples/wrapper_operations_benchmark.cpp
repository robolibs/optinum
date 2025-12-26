// Wrapper Type Operations Benchmark
// Benchmarks Vector, Matrix, Tensor wrapper operations: fill, iota, reverse, flatten, cast

#include <chrono>
#include <iomanip>
#include <iostream>

#include <optinum/simd/simd.hpp>

using namespace optinum::simd;

constexpr std::size_t NUM_ITERATIONS = 10000;

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

// ============================================================================
// Vector Operations
// ============================================================================

template <std::size_t N> double benchmark_vector_fill() {
    Vector<float, N> v;
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        v.fill(3.14f);
    }
    return timer.elapsed_ms();
}

template <std::size_t N> double benchmark_vector_iota() {
    Vector<float, N> v;
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        v.iota();
    }
    return timer.elapsed_ms();
}

template <std::size_t N> double benchmark_vector_iota_start_step() {
    Vector<float, N> v;
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        v.iota(10.0f, 2.0f);
    }
    return timer.elapsed_ms();
}

template <std::size_t N> double benchmark_vector_reverse() {
    Vector<float, N> v;
    v.iota();
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        v.reverse();
    }
    return timer.elapsed_ms();
}

template <std::size_t N> double benchmark_vector_zeros() {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        auto v = Vector<float, N>::zeros();
        (void)v;
    }
    return timer.elapsed_ms();
}

template <std::size_t N> double benchmark_vector_ones() {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        auto v = Vector<float, N>::ones();
        (void)v;
    }
    return timer.elapsed_ms();
}

template <std::size_t N> double benchmark_vector_arange() {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        auto v = Vector<float, N>::arange();
        (void)v;
    }
    return timer.elapsed_ms();
}

template <std::size_t N> double benchmark_vector_cast_int_to_float() {
    Vector<int, N> vi;
    vi.iota();
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        auto vf = cast<float>(vi);
        (void)vf;
    }
    return timer.elapsed_ms();
}

template <std::size_t N> double benchmark_vector_cast_float_to_double() {
    Vector<float, N> vf;
    vf.iota();
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        auto vd = cast<double>(vf);
        (void)vd;
    }
    return timer.elapsed_ms();
}

template <std::size_t N> double benchmark_vector_add() {
    Vector<float, N> v1, v2;
    v1.iota();
    v2.iota(10.0f);
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        auto v3 = v1 + v2;
        (void)v3;
    }
    return timer.elapsed_ms();
}

template <std::size_t N> double benchmark_vector_mul_scalar() {
    Vector<float, N> v;
    v.iota();
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        auto v2 = v * 2.5f;
        (void)v2;
    }
    return timer.elapsed_ms();
}

// ============================================================================
// Matrix Operations
// ============================================================================

template <std::size_t R, std::size_t C> double benchmark_matrix_fill() {
    Matrix<float, R, C> m;
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        m.fill(3.14f);
    }
    return timer.elapsed_ms();
}

template <std::size_t R, std::size_t C> double benchmark_matrix_iota() {
    Matrix<float, R, C> m;
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        m.iota();
    }
    return timer.elapsed_ms();
}

template <std::size_t R, std::size_t C> double benchmark_matrix_reverse() {
    Matrix<float, R, C> m;
    m.iota();
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        m.reverse();
    }
    return timer.elapsed_ms();
}

template <std::size_t R, std::size_t C> double benchmark_matrix_flatten() {
    Matrix<float, R, C> m;
    m.iota();
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        auto v = m.flatten();
        (void)v;
    }
    return timer.elapsed_ms();
}

template <std::size_t R, std::size_t C> double benchmark_matrix_zeros() {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        auto m = Matrix<float, R, C>::zeros();
        (void)m;
    }
    return timer.elapsed_ms();
}

template <std::size_t R, std::size_t C> double benchmark_matrix_ones() {
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        auto m = Matrix<float, R, C>::ones();
        (void)m;
    }
    return timer.elapsed_ms();
}

template <std::size_t R, std::size_t C> double benchmark_matrix_cast_int_to_float() {
    Matrix<int, R, C> mi;
    mi.iota();
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        auto mf = cast<float>(mi);
        (void)mf;
    }
    return timer.elapsed_ms();
}

template <std::size_t R, std::size_t C> double benchmark_matrix_add() {
    Matrix<float, R, C> m1, m2;
    m1.iota();
    m2.iota(10.0f);
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        auto m3 = m1 + m2;
        (void)m3;
    }
    return timer.elapsed_ms();
}

template <std::size_t R, std::size_t C> double benchmark_matrix_mul_scalar() {
    Matrix<float, R, C> m;
    m.iota();
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        auto m2 = m * 2.5f;
        (void)m2;
    }
    return timer.elapsed_ms();
}

template <std::size_t R, std::size_t K, std::size_t C> double benchmark_matrix_matmul() {
    Matrix<float, R, K> m1;
    Matrix<float, K, C> m2;
    m1.iota();
    m2.iota();
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        auto m3 = m1 * m2;
        (void)m3;
    }
    return timer.elapsed_ms();
}

template <std::size_t R, std::size_t C> double benchmark_matrix_transpose() {
    Matrix<float, R, C> m;
    m.iota();
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        auto mt = transpose(m);
        (void)mt;
    }
    return timer.elapsed_ms();
}

// ============================================================================
// Tensor Operations
// ============================================================================

template <std::size_t D1, std::size_t D2, std::size_t D3> double benchmark_tensor_fill() {
    Tensor<float, D1, D2, D3> t;
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        t.fill(3.14f);
    }
    return timer.elapsed_ms();
}

template <std::size_t D1, std::size_t D2, std::size_t D3> double benchmark_tensor_cast_int_to_float() {
    Tensor<int, D1, D2, D3> ti;
    ti.fill(42);
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        auto tf = cast<float>(ti);
        (void)tf;
    }
    return timer.elapsed_ms();
}

template <std::size_t D1, std::size_t D2, std::size_t D3> double benchmark_tensor_add() {
    Tensor<float, D1, D2, D3> t1, t2;
    t1.fill(1.0f);
    t2.fill(2.0f);
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        auto t3 = t1 + t2;
        (void)t3;
    }
    return timer.elapsed_ms();
}

template <std::size_t D1, std::size_t D2, std::size_t D3> double benchmark_tensor_mul_scalar() {
    Tensor<float, D1, D2, D3> t;
    t.fill(1.0f);
    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        auto t2 = t * 2.5f;
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
    std::cout << "--- Vector<float, 1024> Operations ---\n";
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
    std::cout << "\n--- Matrix<float, 64, 64> Operations ---\n";
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
    std::cout << "\n--- Vector<float, 16> Operations (Small Size) ---\n";
    std::cout << std::setw(30) << std::left << "Operation" << std::setw(12) << std::right << "Time" << std::setw(12)
              << "Ops/sec" << std::setw(12) << "Throughput\n";
    std::cout << std::string(66, '-') << "\n";

    constexpr std::size_t SMALL_VEC = 16;
    print_result("fill", benchmark_vector_fill<SMALL_VEC>(), SMALL_VEC);
    print_result("iota", benchmark_vector_iota<SMALL_VEC>(), SMALL_VEC);
    print_result("reverse", benchmark_vector_reverse<SMALL_VEC>(), SMALL_VEC);
    print_result("zeros", benchmark_vector_zeros<SMALL_VEC>(), SMALL_VEC);

    // Smaller Matrix (8x8)
    std::cout << "\n--- Matrix<float, 8, 8> Operations (Small Size) ---\n";
    std::cout << std::setw(30) << std::left << "Operation" << std::setw(12) << std::right << "Time" << std::setw(12)
              << "Ops/sec" << std::setw(12) << "Throughput\n";
    std::cout << std::string(66, '-') << "\n";

    constexpr std::size_t SMALL_MAT = 8;
    constexpr std::size_t SMALL_MAT_SIZE = SMALL_MAT * SMALL_MAT;
    print_result("fill", benchmark_matrix_fill<SMALL_MAT, SMALL_MAT>(), SMALL_MAT_SIZE);
    print_result("iota", benchmark_matrix_iota<SMALL_MAT, SMALL_MAT>(), SMALL_MAT_SIZE);
    print_result("matmul 8x8x8", benchmark_matrix_matmul<SMALL_MAT, SMALL_MAT, SMALL_MAT>(), SMALL_MAT_SIZE);

    // Tensor Operations (8x8x8)
    std::cout << "\n--- Tensor<float, 8, 8, 8> Operations ---\n";
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
