// Linear Algebra Operations Benchmark
// Benchmarks lina module: matmul, transpose, inverse, determinant, decompositions, solvers

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>

#include <optinum/lina/lina.hpp>
#include <optinum/simd/simd.hpp>

namespace dp = datapod;
namespace lina = optinum::lina;
namespace simd = optinum::simd;

constexpr std::size_t NUM_ITERATIONS_SMALL = 1000; // For expensive ops like decompositions
constexpr std::size_t NUM_ITERATIONS_FAST = 10000; // For fast ops like transpose

template <typename T, std::size_t R, std::size_t C>
void fill_random_matrix(dp::mat::Matrix<T, R, C> &m, T min_val = -10.0, T max_val = 10.0) {
    static std::mt19937 gen(42);
    std::uniform_real_distribution<T> dist(min_val, max_val);
    for (std::size_t i = 0; i < R * C; ++i) {
        m[i] = dist(gen);
    }
}

template <typename T, std::size_t N>
void fill_random_vector(dp::mat::Vector<T, N> &v, T min_val = -10.0, T max_val = 10.0) {
    static std::mt19937 gen(42);
    std::uniform_real_distribution<T> dist(min_val, max_val);
    for (std::size_t i = 0; i < N; ++i) {
        v[i] = dist(gen);
    }
}

// Make SPD matrix for Cholesky
template <typename T, std::size_t N> dp::mat::Matrix<T, N, N> make_spd_matrix() {
    dp::mat::Matrix<T, N, N> m;
    fill_random_matrix(m, static_cast<T>(-1.0), static_cast<T>(1.0));
    // A^T * A is SPD
    dp::mat::Matrix<T, N, N> mt;
    simd::backend::transpose<T, N, N>(mt.data(), m.data());
    simd::backend::matmul<T, N, N, N>(m.data(), mt.data(), m.data());
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
    dp::mat::Matrix<double, M, K> a;
    dp::mat::Matrix<double, K, N> b;
    fill_random_matrix(a);
    fill_random_matrix(b);

    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < iters; ++iter) {
        dp::mat::Matrix<double, M, N> c;
        simd::backend::matmul<double, M, K, N>(c.data(), a.data(), b.data());
        (void)c;
    }
    return timer.elapsed_ms();
}

template <std::size_t R, std::size_t C> double benchmark_transpose(std::size_t iters) {
    dp::mat::Matrix<double, R, C> m;
    fill_random_matrix(m);

    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < iters; ++iter) {
        dp::mat::Matrix<double, C, R> mt;
        simd::backend::transpose<double, R, C>(mt.data(), m.data());
        (void)mt;
    }
    return timer.elapsed_ms();
}

template <std::size_t N> double benchmark_inverse(std::size_t iters) {
    dp::mat::Matrix<double, N, N> m;
    fill_random_matrix(m, -5.0, 5.0);
    // Ensure invertibility by adding diagonal dominance
    for (std::size_t i = 0; i < N; ++i) {
        m(i, i) += 10.0;
    }

    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < iters; ++iter) {
        simd::Matrix<double, N, N> wrapper(m);
        auto inv = lina::inverse(wrapper);
        (void)inv;
    }
    return timer.elapsed_ms();
}

template <std::size_t N> double benchmark_determinant(std::size_t iters) {
    dp::mat::Matrix<double, N, N> m;
    fill_random_matrix(m);

    Timer timer;
    timer.start();
    double result = 0.0;
    for (std::size_t iter = 0; iter < iters; ++iter) {
        simd::Matrix<double, N, N> wrapper(m);
        result += lina::determinant(wrapper);
    }
    (void)result;
    return timer.elapsed_ms();
}

template <std::size_t N> double benchmark_norm_frobenius(std::size_t iters) {
    dp::mat::Matrix<double, N, N> m;
    fill_random_matrix(m);

    Timer timer;
    timer.start();
    double result = 0.0;
    for (std::size_t iter = 0; iter < iters; ++iter) {
        result += simd::backend::norm_l2<double, N * N>(m.data());
    }
    (void)result;
    return timer.elapsed_ms();
}

// ============================================================================
// Decompositions
// ============================================================================

template <std::size_t N> double benchmark_lu_decomposition(std::size_t iters) {
    dp::mat::Matrix<double, N, N> m;
    fill_random_matrix(m);

    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < iters; ++iter) {
        simd::Matrix<double, N, N> wrapper(m);
        auto result = lina::lu(wrapper);
        (void)result.l;
        (void)result.u;
        (void)result.p;
    }
    return timer.elapsed_ms();
}

template <std::size_t M, std::size_t N> double benchmark_qr_decomposition(std::size_t iters) {
    dp::mat::Matrix<double, M, N> m;
    fill_random_matrix(m);

    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < iters; ++iter) {
        simd::Matrix<double, M, N> wrapper(m);
        auto [Q, R] = lina::qr(wrapper);
        (void)Q;
        (void)R;
    }
    return timer.elapsed_ms();
}

template <std::size_t M, std::size_t N> double benchmark_svd(std::size_t iters) {
    dp::mat::Matrix<double, M, N> m;
    fill_random_matrix(m);

    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < iters; ++iter) {
        simd::Matrix<double, M, N> wrapper(m);
        auto result = lina::svd(wrapper);
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
        simd::Matrix<double, N, N> wrapper(m);
        auto L = lina::cholesky(wrapper);
        (void)L;
    }
    return timer.elapsed_ms();
}

template <std::size_t N> double benchmark_eigendecomposition(std::size_t iters) {
    dp::mat::Matrix<double, N, N> m;
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
        simd::Matrix<double, N, N> wrapper(m);
        auto result = lina::eigen_sym(wrapper);
        (void)result.vectors;
        (void)result.values;
    }
    return timer.elapsed_ms();
}

// ============================================================================
// Solvers
// ============================================================================

template <std::size_t N> double benchmark_solve(std::size_t iters) {
    dp::mat::Matrix<double, N, N> A;
    dp::mat::Vector<double, N> b;
    fill_random_matrix(A);
    fill_random_vector(b);
    // Ensure invertibility
    for (std::size_t i = 0; i < N; ++i) {
        A(i, i) += 10.0;
    }

    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < iters; ++iter) {
        simd::Matrix<double, N, N> wrapper_A(A);
        simd::Vector<double, N> wrapper_b(b);
        auto x = lina::solve(wrapper_A, wrapper_b);
        (void)x;
    }
    return timer.elapsed_ms();
}

template <std::size_t M, std::size_t N> double benchmark_lstsq(std::size_t iters) {
    dp::mat::Matrix<double, M, N> A;
    dp::mat::Vector<double, M> b;
    fill_random_matrix(A);
    fill_random_vector(b);

    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < iters; ++iter) {
        simd::Matrix<double, M, N> wrapper_A(A);
        simd::Vector<double, M> wrapper_b(b);
        auto x = lina::lstsq(wrapper_A, wrapper_b);
        (void)x;
    }
    return timer.elapsed_ms();
}

// ============================================================================
// Tensor Algebra
// ============================================================================

template <std::size_t N> double benchmark_hadamard(std::size_t iters) {
    dp::mat::Matrix<double, N, N> A, B;
    fill_random_matrix(A);
    fill_random_matrix(B);

    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < iters; ++iter) {
        simd::Matrix<double, N, N> wrapper_A(A);
        simd::Matrix<double, N, N> wrapper_B(B);
        auto C = lina::hadamard(wrapper_A, wrapper_B);
        (void)C;
    }
    return timer.elapsed_ms();
}

template <std::size_t N> double benchmark_outer_product(std::size_t iters) {
    dp::mat::Vector<double, N> u, v;
    fill_random_vector(u);
    fill_random_vector(v);

    Timer timer;
    timer.start();
    for (std::size_t iter = 0; iter < iters; ++iter) {
        simd::Vector<double, N> wrapper_u(u);
        simd::Vector<double, N> wrapper_v(v);
        auto C = lina::outer(wrapper_u, wrapper_v);
        (void)C;
    }
    return timer.elapsed_ms();
}

template <std::size_t N> double benchmark_inner_product(std::size_t iters) {
    dp::mat::Matrix<double, N, N> A, B;
    fill_random_matrix(A);
    fill_random_matrix(B);

    Timer timer;
    timer.start();
    double result = 0.0;
    for (std::size_t iter = 0; iter < iters; ++iter) {
        simd::Matrix<double, N, N> wrapper_A(A);
        simd::Matrix<double, N, N> wrapper_B(B);
        result += lina::inner(wrapper_A, wrapper_B);
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
