// Test for 4 new utility functions
#include <optinum/optinum.hpp>
#include <cmath>
#include <iostream>
#include <limits>

namespace on = optinum;
constexpr double TOL = 1e-6;
bool approx_equal(double a, double b, double tol = TOL) { return std::abs(a - b) < tol; }

void test_is_finite() {
    on::Matrix<double, 3, 3> A;
    A.set_identity();
    if (!on::is_finite(A)) {
        std::cerr << "is_finite() failed: expected true\n";
        std::exit(1);
    }
    A(1, 1) = std::numeric_limits<double>::infinity();
    if (on::is_finite(A)) {
        std::cerr << "is_finite() failed: expected false for inf\n";
        std::exit(1);
    }
    std::cout << "is_finite() test passed\n";
}

void test_log_det() {
    on::Matrix<double, 3, 3> I;
    I.set_identity();
    auto ld = on::log_det(I);
    if (!approx_equal(ld, 0.0, 1e-9)) {
        std::cerr << "log_det() failed\n";
        std::exit(1);
    }
    std::cout << "log_det() test passed\n";
}

void test_triangular_solve() {
    on::Matrix<double, 3, 3> L;
    L(0,0) = 1.0; L(0,1) = 0.0; L(0,2) = 0.0;
    L(1,0) = 2.0; L(1,1) = 3.0; L(1,2) = 0.0;
    L(2,0) = 4.0; L(2,1) = 5.0; L(2,2) = 6.0;
    on::Vector<double, 3> b;
    b[0] = 1.0; b[1] = 8.0; b[2] = 26.0;
    auto x = on::solve_lower_triangular(L, b);
    if (!approx_equal(x[0], 1.0) || !approx_equal(x[1], 2.0) || !approx_equal(x[2], 2.0)) {
        std::cerr << "triangular_solve() failed\n";
        std::exit(1);
    }
    std::cout << "triangular_solve() test passed\n";
}

void test_expmat() {
    // exp(0) = I
    on::Matrix<double, 2, 2> Z;
    Z(0,0) = 0.0; Z(0,1) = 0.0;
    Z(1,0) = 0.0; Z(1,1) = 0.0;
    auto expZ = on::expmat(Z);
    if (!approx_equal(expZ(0,0), 1.0) || !approx_equal(expZ(1,1), 1.0)) {
        std::cerr << "expmat() zero matrix failed\n";
        std::exit(1);
    }
    // TODO: Improve Padé accuracy for better diagonal matrix tests
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
