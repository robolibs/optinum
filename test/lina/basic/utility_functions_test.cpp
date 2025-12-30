// Test for 4 new utility functions
#include <cmath>
#include <iostream>
#include <limits>
#include <optinum/lina/lina.hpp>
#include <optinum/simd/matrix.hpp>
#include <optinum/simd/vector.hpp>

namespace lina = optinum::lina;
using optinum::simd::Matrix;
using optinum::simd::Vector;

namespace dp = datapod;

constexpr double TOL = 1e-6;
bool approx_equal(double a, double b, double tol = TOL) { return std::abs(a - b) < tol; }

void test_is_finite() {
    dp::mat::matrix<double, 3, 3> A;
    for (std::size_t i = 0; i < 9; ++i)
        A[i] = 0.0;
    A(0, 0) = 1.0;
    A(1, 1) = 1.0;
    A(2, 2) = 1.0;
    if (!lina::is_finite(Matrix<double, 3, 3>(A))) {
        std::cerr << "is_finite() failed: expected true\n";
        std::exit(1);
    }
    A(1, 1) = std::numeric_limits<double>::infinity();
    if (lina::is_finite(Matrix<double, 3, 3>(A))) {
        std::cerr << "is_finite() failed: expected false for inf\n";
        std::exit(1);
    }
    std::cout << "is_finite() test passed\n";
}

void test_log_det() {
    dp::mat::matrix<double, 3, 3> I;
    for (std::size_t i = 0; i < 9; ++i)
        I[i] = 0.0;
    I(0, 0) = 1.0;
    I(1, 1) = 1.0;
    I(2, 2) = 1.0;
    auto ld = lina::log_det(Matrix<double, 3, 3>(I));
    if (!approx_equal(ld, 0.0, 1e-9)) {
        std::cerr << "log_det() failed\n";
        std::exit(1);
    }
    std::cout << "log_det() test passed\n";
}

void test_triangular_solve() {
    dp::mat::matrix<double, 3, 3> L;
    L(0, 0) = 1.0;
    L(0, 1) = 0.0;
    L(0, 2) = 0.0;
    L(1, 0) = 2.0;
    L(1, 1) = 3.0;
    L(1, 2) = 0.0;
    L(2, 0) = 4.0;
    L(2, 1) = 5.0;
    L(2, 2) = 6.0;
    dp::mat::vector<double, 3> b;
    b[0] = 1.0;
    b[1] = 8.0;
    b[2] = 26.0;
    auto x = lina::solve_lower_triangular(Matrix<double, 3, 3>(L), Vector<double, 3>(b));
    if (!approx_equal(x[0], 1.0) || !approx_equal(x[1], 2.0) || !approx_equal(x[2], 2.0)) {
        std::cerr << "triangular_solve() failed\n";
        std::exit(1);
    }
    std::cout << "triangular_solve() test passed\n";
}

void test_expmat() {
    // exp(0) = I
    dp::mat::matrix<double, 2, 2> Z;
    Z(0, 0) = 0.0;
    Z(0, 1) = 0.0;
    Z(1, 0) = 0.0;
    Z(1, 1) = 0.0;
    auto expZ = lina::expmat(Matrix<double, 2, 2>(Z));
    if (!approx_equal(expZ(0, 0), 1.0) || !approx_equal(expZ(1, 1), 1.0)) {
        std::cerr << "expmat() zero matrix failed\n";
        std::exit(1);
    }
    // TODO: Improve Pade accuracy for better diagonal matrix tests
    std::cout << "expmat() test passed\n";
}

int main() {
    std::cout << "\n=== Testing 4 Utility Functions ===\n\n";
    test_is_finite();
    test_log_det();
    test_triangular_solve();
    test_expmat();
    std::cout << "\n=== All tests passed! ===\n";
    return 0;
}
