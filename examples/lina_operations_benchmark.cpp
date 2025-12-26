// Linear Algebra Operations Benchmark
// Benchmarks lina module: matmul, transpose, inverse, determinant, decompositions, solvers

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>

#include <optinum/lina/lina.hpp>
#include <optinum/simd/simd.hpp>

namespace lina = optinum::lina;
namespace simd = optinum::simd;

constexpr std::size_t NUM_ITERATIONS_SMALL = 1000; // For expensive ops like decompositions
constexpr std::size_t NUM_ITERATIONS_FAST = 10000; // For fast ops like transpose

template <typename T, std::size_t R, std::size_t C>
void fill_random_matrix(simd::Matrix<T, R, C> &m, T min_val = -10.0, T max_val = 10.0) {
    static std::mt19937 gen(42);
    std::uniform_real_distribution<T> dist(min_val, max_val);
    for (std::size_t i = 0; i < R * C; ++i) {
        m[i] = dist(gen);
    }
}

template <typename T, std::size_t N>
void fill_random_vector(simd::Vector<T, N> &v, T min_val = -10.0, T max_val = 10.0) {
    static std::mt19937 gen(42);
    std::uniform_real_distribution<T> dist(min_val, max_val);
    for (std::size_t i = 0; i < N; ++i) {
        v[i] = dist(gen);
    }
}

// Make SPD matrix for Cholesky
template <typename T, std::size_t N> simd::Matrix<T, N, N> make_spd_matrix() {
    simd::Matrix<T, N, N> m;
    fill_random_matrix(m, -1.0, 1.0);
    // A^T * A is SPD
    auto mt = lina::transpose(m);
    m = lina::matmul(mt, m);
    // Add diagonal dominance to ensure positive definiteness
    for (std::size_t i = 0; i < N; ++i) {
        m(i, i) += static_cast<T>(N);
    }
    return m;
}

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
// Basic Operations
// ============================================================================

template <std::size_t M, std::size_t K, std::size_t N> double benchmark_matmul(std::size_t iters) {
    simd::Matrix<double, M, K> a;
    simd::Matrix<double, K, N> b;
    fill_random_matrix(a);
    fill_random_matrix(b);

    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < iters; ++iter) {
        auto c = lina::matmul(a, b);
        (void)c;
    }
    return timer.elapsed_ms();
}

template <std::size_t R, std::size_t C> double benchmark_transpose(std::size_t iters) {
    simd::Matrix<double, R, C> m;
    fill_random_matrix(m);

    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < iters; ++iter) {
        auto mt = lina::transpose(m);
        (void)mt;
    }
    return timer.elapsed_ms();
}

template <std::size_t N> double benchmark_inverse(std::size_t iters) {
    simd::Matrix<double, N, N> m;
    fill_random_matrix(m, -5.0, 5.0);
    // Ensure invertibility by adding diagonal dominance
    for (std::size_t i = 0; i < N; ++i) {
        m(i, i) += 10.0;
    }

    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < iters; ++iter) {
        auto inv = lina::inverse(m);
        (void)inv;
    }
    return timer.elapsed_ms();
}

template <std::size_t N> double benchmark_determinant(std::size_t iters) {
    simd::Matrix<double, N, N> m;
    fill_random_matrix(m);

    Timer timer;
    timer.start();
    double result = 0.0;
    for (std::size_t iter = 0; iter < iters; ++iter) {
        result += lina::determinant(m);
    }
    (void)result;
    return timer.elapsed_ms();
}

template <std::size_t N> double benchmark_norm_frobenius(std::size_t iters) {
    simd::Matrix<double, N, N> m;
    fill_random_matrix(m);

    Timer timer;
    timer.start();
    double result = 0.0;
    for (std::size_t iter = 0; iter < iters; ++iter) {
        result += simd::frobenius_norm(m);
    }
    (void)result;
    return timer.elapsed_ms();
}

// ============================================================================
// Decompositions
// ============================================================================

template <std::size_t N> double benchmark_lu_decomposition(std::size_t iters) {
    simd::Matrix<double, N, N> m;
    fill_random_matrix(m);

    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < iters; ++iter) {
        auto result = lina::lu(m);
        (void)result.l;
        (void)result.u;
        (void)result.p;
    }
    return timer.elapsed_ms();
}

template <std::size_t M, std::size_t N> double benchmark_qr_decomposition(std::size_t iters) {
    simd::Matrix<double, M, N> m;
    fill_random_matrix(m);

    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < iters; ++iter) {
        auto [Q, R] = lina::qr(m);
        (void)Q;
        (void)R;
    }
    return timer.elapsed_ms();
}

template <std::size_t M, std::size_t N> double benchmark_svd(std::size_t iters) {
    simd::Matrix<double, M, N> m;
    fill_random_matrix(m);

    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < iters; ++iter) {
        auto result = lina::svd(m);
        (void)result.u;
        (void)result.s;
        (void)result.vt;
    }
    return timer.elapsed_ms();
}

template <std::size_t N> double benchmark_cholesky(std::size_t iters) {
    auto m = make_spd_matrix<double, N>();

    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < iters; ++iter) {
        auto L = lina::cholesky(m);
        (void)L;
    }
    return timer.elapsed_ms();
}

template <std::size_t N> double benchmark_eigendecomposition(std::size_t iters) {
    simd::Matrix<double, N, N> m;
    // Make symmetric for eigendecomposition
    fill_random_matrix(m, -1.0, 1.0);
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = i + 1; j < N; ++j) {
            m(j, i) = m(i, j);
        }
    }

    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < iters; ++iter) {
        auto result = lina::eigen_sym(m);
        (void)result.vectors;
        (void)result.values;
    }
    return timer.elapsed_ms();
}

// ============================================================================
// Solvers
// ============================================================================

template <std::size_t N> double benchmark_solve(std::size_t iters) {
    simd::Matrix<double, N, N> A;
    simd::Vector<double, N> b;
    fill_random_matrix(A);
    fill_random_vector(b);
    // Ensure invertibility
    for (std::size_t i = 0; i < N; ++i) {
        A(i, i) += 10.0;
    }

    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < iters; ++iter) {
        auto x = lina::solve(A, b);
        (void)x;
    }
    return timer.elapsed_ms();
}

template <std::size_t M, std::size_t N> double benchmark_lstsq(std::size_t iters) {
    simd::Matrix<double, M, N> A;
    simd::Vector<double, M> b;
    fill_random_matrix(A);
    fill_random_vector(b);

    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < iters; ++iter) {
        auto x = lina::lstsq(A, b);
        (void)x;
    }
    return timer.elapsed_ms();
}

// ============================================================================
// Tensor Algebra
// ============================================================================

template <std::size_t N> double benchmark_hadamard(std::size_t iters) {
    simd::Matrix<double, N, N> A, B;
    fill_random_matrix(A);
    fill_random_matrix(B);

    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < iters; ++iter) {
        auto C = lina::hadamard(A, B);
        (void)C;
    }
    return timer.elapsed_ms();
}

template <std::size_t N> double benchmark_outer_product(std::size_t iters) {
    simd::Vector<double, N> u, v;
    fill_random_vector(u);
    fill_random_vector(v);

    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < iters; ++iter) {
        auto C = lina::outer(u, v);
        (void)C;
    }
    return timer.elapsed_ms();
}

template <std::size_t N> double benchmark_inner_product(std::size_t iters) {
    simd::Matrix<double, N, N> A, B;
    fill_random_matrix(A);
    fill_random_matrix(B);

    Timer timer;
    timer.start();
    double result = 0.0;
    for (std::size_t iter = 0; iter < iters; ++iter) {
        result += lina::inner(A, B);
    }
    (void)result;
    return timer.elapsed_ms();
}

// ============================================================================
// Main Benchmark Runner
// ============================================================================

void print_result(const char *name, double time_ms, std::size_t iters) {
    double ops_per_sec = (iters * 1000.0) / time_ms;
    double time_per_op = time_ms / iters;
    std::cout << std::setw(30) << std::left << name << std::setw(12) << std::right << std::fixed << std::setprecision(3)
              << time_per_op << " ms/op" << std::setw(12) << ops_per_sec << " op/s" << std::setw(12) << time_ms
              << " ms\n";
}

int main() {
    std::cout << "\n=== Linear Algebra Operations Benchmark ===\n";
    std::cout << "Fast ops: " << NUM_ITERATIONS_FAST << " iterations\n";
    std::cout << "Slow ops: " << NUM_ITERATIONS_SMALL << " iterations\n\n";

    // Basic Operations
    std::cout << "--- Basic Operations ---\n";
    std::cout << std::setw(30) << std::left << "Operation" << std::setw(12) << std::right << "Time/Op" << std::setw(12)
              << "Ops/sec" << std::setw(12) << "Total Time\n";
    std::cout << std::string(66, '-') << "\n";

    print_result("matmul 8x8 * 8x8", benchmark_matmul<8, 8, 8>(NUM_ITERATIONS_FAST), NUM_ITERATIONS_FAST);
    print_result("matmul 16x16 * 16x16", benchmark_matmul<16, 16, 16>(NUM_ITERATIONS_FAST), NUM_ITERATIONS_FAST);
    print_result("matmul 32x32 * 32x32", benchmark_matmul<32, 32, 32>(NUM_ITERATIONS_SMALL), NUM_ITERATIONS_SMALL);
    print_result("matmul 64x64 * 64x64", benchmark_matmul<64, 64, 64>(NUM_ITERATIONS_SMALL), NUM_ITERATIONS_SMALL);

    print_result("transpose 8x8", benchmark_transpose<8, 8>(NUM_ITERATIONS_FAST), NUM_ITERATIONS_FAST);
    print_result("transpose 16x16", benchmark_transpose<16, 16>(NUM_ITERATIONS_FAST), NUM_ITERATIONS_FAST);
    print_result("transpose 32x32", benchmark_transpose<32, 32>(NUM_ITERATIONS_FAST), NUM_ITERATIONS_FAST);
    print_result("transpose 64x64", benchmark_transpose<64, 64>(NUM_ITERATIONS_FAST), NUM_ITERATIONS_FAST);

    print_result("inverse 4x4", benchmark_inverse<4>(NUM_ITERATIONS_FAST), NUM_ITERATIONS_FAST);
    print_result("inverse 8x8", benchmark_inverse<8>(NUM_ITERATIONS_SMALL), NUM_ITERATIONS_SMALL);
    print_result("inverse 16x16", benchmark_inverse<16>(NUM_ITERATIONS_SMALL), NUM_ITERATIONS_SMALL);

    print_result("determinant 4x4", benchmark_determinant<4>(NUM_ITERATIONS_FAST), NUM_ITERATIONS_FAST);
    print_result("determinant 8x8", benchmark_determinant<8>(NUM_ITERATIONS_SMALL), NUM_ITERATIONS_SMALL);
    print_result("determinant 16x16", benchmark_determinant<16>(NUM_ITERATIONS_SMALL), NUM_ITERATIONS_SMALL);

    print_result("frobenius_norm 32x32", benchmark_norm_frobenius<32>(NUM_ITERATIONS_FAST), NUM_ITERATIONS_FAST);

    // Decompositions
    std::cout << "\n--- Matrix Decompositions ---\n";
    std::cout << std::setw(30) << std::left << "Operation" << std::setw(12) << std::right << "Time/Op" << std::setw(12)
              << "Ops/sec" << std::setw(12) << "Total Time\n";
    std::cout << std::string(66, '-') << "\n";

    print_result("LU 8x8", benchmark_lu_decomposition<8>(NUM_ITERATIONS_SMALL), NUM_ITERATIONS_SMALL);
    print_result("LU 16x16", benchmark_lu_decomposition<16>(NUM_ITERATIONS_SMALL), NUM_ITERATIONS_SMALL);
    print_result("LU 32x32", benchmark_lu_decomposition<32>(NUM_ITERATIONS_SMALL), NUM_ITERATIONS_SMALL);

    print_result("QR 8x8", benchmark_qr_decomposition<8, 8>(NUM_ITERATIONS_SMALL), NUM_ITERATIONS_SMALL);
    print_result("QR 16x16", benchmark_qr_decomposition<16, 16>(NUM_ITERATIONS_SMALL), NUM_ITERATIONS_SMALL);
    print_result("QR 32x32", benchmark_qr_decomposition<32, 32>(NUM_ITERATIONS_SMALL), NUM_ITERATIONS_SMALL);

    print_result("SVD 4x4", benchmark_svd<4, 4>(NUM_ITERATIONS_SMALL), NUM_ITERATIONS_SMALL);
    print_result("SVD 8x8", benchmark_svd<8, 8>(NUM_ITERATIONS_SMALL), NUM_ITERATIONS_SMALL);
    print_result("SVD 16x16", benchmark_svd<16, 16>(NUM_ITERATIONS_SMALL), NUM_ITERATIONS_SMALL);

    print_result("Cholesky 8x8", benchmark_cholesky<8>(NUM_ITERATIONS_SMALL), NUM_ITERATIONS_SMALL);
    print_result("Cholesky 16x16", benchmark_cholesky<16>(NUM_ITERATIONS_SMALL), NUM_ITERATIONS_SMALL);
    print_result("Cholesky 32x32", benchmark_cholesky<32>(NUM_ITERATIONS_SMALL), NUM_ITERATIONS_SMALL);

    print_result("Eigen 4x4", benchmark_eigendecomposition<4>(NUM_ITERATIONS_SMALL), NUM_ITERATIONS_SMALL);
    print_result("Eigen 8x8", benchmark_eigendecomposition<8>(NUM_ITERATIONS_SMALL), NUM_ITERATIONS_SMALL);

    // Solvers
    std::cout << "\n--- Linear System Solvers ---\n";
    std::cout << std::setw(30) << std::left << "Operation" << std::setw(12) << std::right << "Time/Op" << std::setw(12)
              << "Ops/sec" << std::setw(12) << "Total Time\n";
    std::cout << std::string(66, '-') << "\n";

    print_result("solve 8x8", benchmark_solve<8>(NUM_ITERATIONS_SMALL), NUM_ITERATIONS_SMALL);
    print_result("solve 16x16", benchmark_solve<16>(NUM_ITERATIONS_SMALL), NUM_ITERATIONS_SMALL);
    print_result("solve 32x32", benchmark_solve<32>(NUM_ITERATIONS_SMALL), NUM_ITERATIONS_SMALL);

    print_result("lstsq 16x8", benchmark_lstsq<16, 8>(NUM_ITERATIONS_SMALL), NUM_ITERATIONS_SMALL);
    print_result("lstsq 32x16", benchmark_lstsq<32, 16>(NUM_ITERATIONS_SMALL), NUM_ITERATIONS_SMALL);
    print_result("lstsq 64x32", benchmark_lstsq<64, 32>(NUM_ITERATIONS_SMALL), NUM_ITERATIONS_SMALL);

    // Tensor Algebra
    std::cout << "\n--- Tensor Algebra ---\n";
    std::cout << std::setw(30) << std::left << "Operation" << std::setw(12) << std::right << "Time/Op" << std::setw(12)
              << "Ops/sec" << std::setw(12) << "Total Time\n";
    std::cout << std::string(66, '-') << "\n";

    print_result("hadamard 32x32", benchmark_hadamard<32>(NUM_ITERATIONS_FAST), NUM_ITERATIONS_FAST);
    print_result("outer product 64", benchmark_outer_product<64>(NUM_ITERATIONS_FAST), NUM_ITERATIONS_FAST);
    print_result("inner product 32x32", benchmark_inner_product<32>(NUM_ITERATIONS_FAST), NUM_ITERATIONS_FAST);

    std::cout << "\n=== Benchmark Complete ===\n\n";
    return 0;
}
