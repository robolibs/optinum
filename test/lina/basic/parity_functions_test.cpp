// Test for 8 parity functions
#include <cmath>
#include <iostream>
#include <optinum/lina/lina.hpp>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/vector.hpp>

namespace lina = optinum::lina;
using optinum::simd::Matrix;

namespace dp = datapod;

constexpr double TOL = 1e-9;
bool approx_equal(double a, double b, double tol = TOL) { return std::abs(a - b) < tol; }

void test_rank() {
    dp::mat::matrix<double, 3, 3> A;
    for (std::size_t i = 0; i < 9; ++i)
        A[i] = 0.0;
    A(0, 0) = 1.0;
    A(1, 1) = 1.0;
    A(2, 2) = 1.0;
    auto r = lina::rank(Matrix<double, 3, 3>(A));
    if (r != 3) {
        std::cerr << "rank() failed: expected 3, got " << r << "\n";
        std::exit(1);
    }
    std::cout << "rank() test passed\n";
}

void test_cond() {
    dp::mat::matrix<double, 3, 3> I;
    for (std::size_t i = 0; i < 9; ++i)
        I[i] = 0.0;
    I(0, 0) = 1.0;
    I(1, 1) = 1.0;
    I(2, 2) = 1.0;
    auto c = lina::cond(Matrix<double, 3, 3>(I));
    if (!approx_equal(c, 1.0, 1e-6)) {
        std::cerr << "cond() failed: expected ~1.0, got " << c << "\n";
        std::exit(1);
    }
    std::cout << "cond() test passed\n";
}

void test_rcond() {
    dp::mat::matrix<double, 3, 3> I;
    for (std::size_t i = 0; i < 9; ++i)
        I[i] = 0.0;
    I(0, 0) = 1.0;
    I(1, 1) = 1.0;
    I(2, 2) = 1.0;
    auto rc = lina::rcond(Matrix<double, 3, 3>(I));
    if (!approx_equal(rc, 1.0, 1e-6)) {
        std::cerr << "rcond() failed: expected ~1.0, got " << rc << "\n";
        std::exit(1);
    }
    std::cout << "rcond() test passed\n";
}

void test_pinv() {
    dp::mat::matrix<double, 3, 3> A;
    A(0, 0) = 1.0;
    A(0, 1) = 0.0;
    A(0, 2) = 0.0;
    A(1, 0) = 0.0;
    A(1, 1) = 2.0;
    A(1, 2) = 0.0;
    A(2, 0) = 0.0;
    A(2, 1) = 0.0;
    A(2, 2) = 3.0;

    auto A_pinv = lina::pinv(Matrix<double, 3, 3>(A));
    auto I = lina::matmul(A, A_pinv);

    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            if (!approx_equal(I(i, j), expected, 1e-6)) {
                std::cerr << "pinv() failed at (" << i << "," << j << ")\n";
                std::exit(1);
            }
        }
    }
    std::cout << "pinv() test passed\n";
}

void test_null() {
    dp::mat::matrix<double, 3, 3> A;
    for (std::size_t i = 0; i < 9; ++i)
        A[i] = 0.0;
    A(0, 0) = 1.0;
    A(1, 1) = 1.0;
    A(2, 2) = 1.0;
    auto N = lina::null(Matrix<double, 3, 3>(A));
    auto r = lina::rank(N);
    if (r != 0) {
        std::cerr << "null() failed: expected rank 0, got " << r << "\n";
        std::exit(1);
    }
    std::cout << "null() test passed\n";
}

void test_orth() {
    dp::mat::matrix<double, 3, 3> A;
    for (std::size_t i = 0; i < 9; ++i)
        A[i] = 0.0;
    A(0, 0) = 1.0;
    A(1, 1) = 1.0;
    A(2, 2) = 1.0;
    auto Q = lina::orth(Matrix<double, 3, 3>(A));
    auto QtQ = lina::matmul(lina::transpose(Q), Q);

    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            if (!approx_equal(QtQ(i, j), expected, 1e-6)) {
                std::cerr << "orth() failed at (" << i << "," << j << ")\n";
                std::exit(1);
            }
        }
    }
    std::cout << "orth() test passed\n";
}

void test_kron() {
    dp::mat::matrix<double, 2, 2> A;
    A(0, 0) = 1.0;
    A(0, 1) = 2.0;
    A(1, 0) = 3.0;
    A(1, 1) = 4.0;

    dp::mat::matrix<double, 2, 2> B;
    B(0, 0) = 0.0;
    B(0, 1) = 5.0;
    B(1, 0) = 6.0;
    B(1, 1) = 7.0;

    auto C = lina::kron(Matrix<double, 2, 2>(A), Matrix<double, 2, 2>(B));
    if (C.rows() != 4 || C.cols() != 4) {
        std::cerr << "kron() shape failed: expected 4x4, got " << C.rows() << "x" << C.cols() << "\n";
        std::exit(1);
    }
    std::cout << "kron() test passed\n";
}

void test_is_symmetric() {
    dp::mat::matrix<double, 3, 3> A;
    A(0, 0) = 1.0;
    A(0, 1) = 2.0;
    A(0, 2) = 3.0;
    A(1, 0) = 2.0;
    A(1, 1) = 4.0;
    A(1, 2) = 5.0;
    A(2, 0) = 3.0;
    A(2, 1) = 5.0;
    A(2, 2) = 6.0;

    if (!lina::is_symmetric(Matrix<double, 3, 3>(A))) {
        std::cerr << "is_symmetric() failed: expected true\n";
        std::exit(1);
    }
    std::cout << "is_symmetric() test passed\n";
}

void test_is_positive_definite() {
    dp::mat::matrix<double, 3, 3> A;
    A(0, 0) = 4.0;
    A(0, 1) = 2.0;
    A(0, 2) = 1.0;
    A(1, 0) = 2.0;
    A(1, 1) = 5.0;
    A(1, 2) = 2.0;
    A(2, 0) = 1.0;
    A(2, 1) = 2.0;
    A(2, 2) = 6.0;

    if (!lina::is_positive_definite(Matrix<double, 3, 3>(A))) {
        std::cerr << "is_positive_definite() failed: expected true\n";
        std::exit(1);
    }
    std::cout << "is_positive_definite() test passed\n";
}

int main() {
    std::cout << "\n=== Testing 8 Parity Functions ===\n\n";
    test_rank();
    test_cond();
    test_rcond();
    test_pinv();
    test_null();
    test_orth();
    test_kron();
    test_is_symmetric();
    test_is_positive_definite();
    std::cout << "\n=== All tests passed! ===\n";
    return 0;
}
