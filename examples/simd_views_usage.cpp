// =============================================================================
// examples/simd_views_usage.cpp
// Demonstrates SIMD operations on vector, matrix, and tensor using views
// =============================================================================

#include <cmath>
#include <iomanip>
#include <iostream>

#include <datapod/matrix.hpp>
#include <optinum/optinum.hpp>




namespace dp = datapod;
namespace on = optinum;

// Helper to print a vector
template <typename T, std::size_t N> void print_vector(const char *name, const dp::mat::Vector<T, N> &v) {
    std::cout << name << " = [";
    for (std::size_t i = 0; i < N; ++i) {
        std::cout << std::setw(8) << std::fixed << std::setprecision(3) << v[i];
        if (i < N - 1)
            std::cout << ", ";
    }
    std::cout << "]\n";
}

// Helper to print a matrix
template <typename T, std::size_t R, std::size_t C>
void print_matrix(const char *name, const dp::mat::Matrix<T, R, C> &m) {
    std::cout << name << " (" << R << "x" << C << ") =\n";
    for (std::size_t r = 0; r < R; ++r) {
        std::cout << "  [";
        for (std::size_t c = 0; c < C; ++c) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(3) << m(r, c);
            if (c < C - 1)
                std::cout << ", ";
        }
        std::cout << "]\n";
    }
}

// Helper to print a tensor (3D)
template <typename T, std::size_t D0, std::size_t D1, std::size_t D2>
void print_tensor(const char *name, const dp::mat::Tensor<T, D0, D1, D2> &t) {
    std::cout << name << " (" << D0 << "x" << D1 << "x" << D2 << ") =\n";
    for (std::size_t k = 0; k < D2; ++k) {
        std::cout << "  [:,:," << k << "] =\n";
        for (std::size_t i = 0; i < D0; ++i) {
            std::cout << "    [";
            for (std::size_t j = 0; j < D1; ++j) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(3) << t(i, j, k);
                if (j < D1 - 1)
                    std::cout << ", ";
            }
            std::cout << "]\n";
        }
    }
}

// =============================================================================
// Vector Examples
// =============================================================================

void vector_examples() {
    std::cout << "================================================================\n";
    std::cout << "                    VECTOR EXAMPLES\n";
    std::cout << "================================================================\n\n";

    // Create vectors
    dp::mat::Vector<float, 8> x{{0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f}};
    dp::mat::Vector<float, 8> y{{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}};
    dp::mat::Vector<float, 8> z;

    print_vector("x", x);
    print_vector("y", y);

    // Create SIMD views (auto-detect width based on architecture)
    auto vx = on::simd::view(x);
    auto vy = on::simd::view(y);
    auto vz = on::simd::view(z);

    std::cout << "\n--- Elementwise Operations ---\n";

    // axpy: y = alpha * x + y
    on::simd::axpy(2.0f, vx, vy);
    print_vector("y = 2*x + y", y);

    // Reset y
    on::simd::fill(vy, 10.0f);
    print_vector("fill(y, 10)", y);

    // add: z = x + y
    on::simd::add(vx, vy, vz);
    print_vector("z = x + y", z);

    // scale: y = alpha * y
    on::simd::scale(0.5f, vy);
    print_vector("y = 0.5 * y", y);

    std::cout << "\n--- Math Transforms ---\n";

    // exp
    dp::mat::Vector<float, 8> exp_result;
    auto v_exp = on::simd::view(exp_result);
    on::simd::exp(vx, v_exp);
    print_vector("exp(x)", exp_result);

    // sqrt (use positive values)
    dp::mat::Vector<float, 8> pos{{1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f, 49.0f, 64.0f}};
    dp::mat::Vector<float, 8> sqrt_result;
    on::simd::sqrt(on::simd::view(pos), on::simd::view(sqrt_result));
    print_vector("sqrt([1,4,9,16,25,36,49,64])", sqrt_result);

    // tanh
    dp::mat::Vector<float, 8> tanh_result;
    on::simd::tanh(vx, on::simd::view(tanh_result));
    print_vector("tanh(x)", tanh_result);

    std::cout << "\n";
}

// =============================================================================
// Matrix Examples
// =============================================================================

void matrix_examples() {
    std::cout << "================================================================\n";
    std::cout << "                    MATRIX EXAMPLES\n";
    std::cout << "================================================================\n\n";

    // Create matrices
    dp::mat::Matrix<float, 3, 4> A;
    dp::mat::Matrix<float, 3, 4> B;
    dp::mat::Matrix<float, 3, 4> C;

    // Initialize A with values 1..12
    for (std::size_t i = 0; i < 12; ++i) {
        A[i] = static_cast<float>(i + 1);
    }

    // Initialize B with constant
    for (std::size_t i = 0; i < 12; ++i) {
        B[i] = 0.1f;
    }

    print_matrix("A", A);
    print_matrix("B", B);

    // Create SIMD views
    auto vA = on::simd::view<4>(A);
    auto vB = on::simd::view<4>(B);
    auto vC = on::simd::view<4>(C);

    std::cout << "--- Elementwise Operations ---\n";

    // add: C = A + B
    on::simd::add(vA, vB, vC);
    print_matrix("C = A + B", C);

    // mul (Hadamard): C = A * B
    on::simd::mul(vA, vB, vC);
    print_matrix("C = A .* B (Hadamard)", C);

    // scale in-place
    on::simd::scale(10.0f, vC);
    print_matrix("C = 10 * C", C);

    std::cout << "\n--- Math Transforms ---\n";

    // Create a matrix with small values for exp
    dp::mat::Matrix<float, 3, 4> small;
    for (std::size_t i = 0; i < 12; ++i) {
        small[i] = static_cast<float>(i) * 0.1f;
    }
    print_matrix("small", small);

    // exp
    dp::mat::Matrix<float, 3, 4> exp_result;
    on::simd::exp(on::simd::view<4>(small), on::simd::view<4>(exp_result));
    print_matrix("exp(small)", exp_result);

    // log (use positive values)
    dp::mat::Matrix<float, 3, 4> log_input;
    for (std::size_t i = 0; i < 12; ++i) {
        log_input[i] = static_cast<float>(i + 1);
    }
    dp::mat::Matrix<float, 3, 4> log_result;
    on::simd::log(on::simd::view<4>(log_input), on::simd::view<4>(log_result));
    print_matrix("log([1..12])", log_result);

    // sqrt
    dp::mat::Matrix<float, 3, 4> sqrt_result;
    on::simd::sqrt(on::simd::view<4>(log_input), on::simd::view<4>(sqrt_result));
    print_matrix("sqrt([1..12])", sqrt_result);

    std::cout << "\n";
}

// =============================================================================
// Tensor Examples
// =============================================================================

void tensor_examples() {
    std::cout << "================================================================\n";
    std::cout << "                    TENSOR EXAMPLES\n";
    std::cout << "================================================================\n\n";

    // Create 2x3x2 tensors (total 12 elements)
    dp::mat::Tensor<float, 2, 3, 2> T1;
    dp::mat::Tensor<float, 2, 3, 2> T2;
    dp::mat::Tensor<float, 2, 3, 2> T3;

    // Initialize T1 with values
    for (std::size_t i = 0; i < 12; ++i) {
        T1[i] = static_cast<float>(i + 1) * 0.1f;
    }

    // Initialize T2 with constant
    for (std::size_t i = 0; i < 12; ++i) {
        T2[i] = 1.0f;
    }

    print_tensor("T1", T1);
    print_tensor("T2", T2);

    // Create SIMD views
    auto vT1 = on::simd::view<4>(T1);
    auto vT2 = on::simd::view<4>(T2);
    auto vT3 = on::simd::view<4>(T3);

    std::cout << "--- Elementwise Operations ---\n";

    // add: T3 = T1 + T2
    on::simd::add(vT1, vT2, vT3);
    print_tensor("T3 = T1 + T2", T3);

    // axpy: T2 = 2*T1 + T2
    on::simd::fill(vT2, 1.0f); // reset T2
    on::simd::axpy(2.0f, vT1, vT2);
    print_tensor("T2 = 2*T1 + T2", T2);

    std::cout << "\n--- Math Transforms ---\n";

    // exp (in-place)
    dp::mat::Tensor<float, 2, 3, 2> exp_tensor;
    for (std::size_t i = 0; i < 12; ++i) {
        exp_tensor[i] = static_cast<float>(i) * 0.2f - 1.0f; // values from -1 to ~1.2
    }
    print_tensor("input", exp_tensor);

    on::simd::exp(on::simd::view<4>(exp_tensor)); // in-place!
    print_tensor("exp(input) [in-place]", exp_tensor);

    // tanh
    dp::mat::Tensor<float, 2, 3, 2> tanh_input;
    dp::mat::Tensor<float, 2, 3, 2> tanh_result;
    for (std::size_t i = 0; i < 12; ++i) {
        tanh_input[i] = static_cast<float>(i) * 0.5f - 3.0f; // values from -3 to ~2.5
    }
    on::simd::tanh(on::simd::view<4>(tanh_input), on::simd::view<4>(tanh_result));
    print_tensor("tanh([-3..2.5])", tanh_result);

    std::cout << "\n";
}

// =============================================================================
// Performance Comparison: Scalar vs SIMD
// =============================================================================

void performance_demo() {
    std::cout << "================================================================\n";
    std::cout << "              SIMD vs SCALAR COMPARISON\n";
    std::cout << "================================================================\n\n";

    constexpr std::size_t N = 1024;
    dp::mat::Vector<float, N> input;
    dp::mat::Vector<float, N> output_simd;
    dp::mat::Vector<float, N> output_scalar;

    // Initialize
    for (std::size_t i = 0; i < N; ++i) {
        input[i] = static_cast<float>(i) * 0.01f;
    }

    // SIMD exp
    auto vi = on::simd::view(input);
    auto vo = on::simd::view(output_simd);
    on::simd::exp(vi, vo);

    // Scalar exp (for comparison)
    for (std::size_t i = 0; i < N; ++i) {
        output_scalar[i] = std::exp(input[i]);
    }

    // Compare results
    float max_error = 0.0f;
    for (std::size_t i = 0; i < N; ++i) {
        float err = std::abs(output_simd[i] - output_scalar[i]);
        if (err > max_error)
            max_error = err;
    }

    std::cout << "Computed exp() on " << N << " elements\n";
    std::cout << "Max absolute error vs std::exp: " << std::scientific << max_error << "\n";
    std::cout << "First 8 SIMD results:   [";
    for (int i = 0; i < 8; ++i)
        std::cout << std::fixed << std::setprecision(4) << output_simd[i] << (i < 7 ? ", " : "");
    std::cout << "]\n";
    std::cout << "First 8 scalar results: [";
    for (int i = 0; i < 8; ++i)
        std::cout << std::fixed << std::setprecision(4) << output_scalar[i] << (i < 7 ? ", " : "");
    std::cout << "]\n\n";
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "\n";
    std::cout << "################################################################\n";
    std::cout << "#          OPTINUM SIMD VIEWS USAGE EXAMPLES                  #\n";
    std::cout << "################################################################\n";
    std::cout << "\n";
    std::cout << "SIMD Architecture Info:\n";
    std::cout << "  Level: " << on::simd::arch::simd_level() << "-bit\n";
    std::cout << "  Float width:  " << on::simd::arch::SIMD_WIDTH_FLOAT << " elements\n";
    std::cout << "  Double width: " << on::simd::arch::SIMD_WIDTH_DOUBLE << " elements\n";
    std::cout << "\n";

    vector_examples();
    matrix_examples();
    tensor_examples();
    performance_demo();

    std::cout << "================================================================\n";
    std::cout << "                    ALL EXAMPLES COMPLETE\n";
    std::cout << "================================================================\n\n";

    return 0;
}
