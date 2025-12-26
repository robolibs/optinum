// Test for 8 parity functions
#include <optinum/optinum.hpp>
#include <cmath>
#include <iostream>

namespace on = optinum;
constexpr double TOL = 1e-9;
bool approx_equal(double a, double b, double tol = TOL) { return std::abs(a - b) < tol; }

void test_rank() {
    on::Matrix<double, 3, 3> A;
    A.set_identity();
    auto r = on::rank(A);
    if (r != 3) {
        std::cerr << "rank() failed: expected 3, got " << r << "\n";
        std::exit(1);
    }
    std::cout << "rank() test passed\n";
}

void test_cond() {
    on::Matrix<double, 3, 3> I;
    I.set_identity();
    auto c = on::cond(I);
    if (!approx_equal(c, 1.0, 1e-6)) {
        std::cerr << "cond() failed: expected ~1.0, got " << c << "\n";
        std::exit(1);
    }
    std::cout << "cond() test passed\n";
}

void test_rcond() {
    on::Matrix<double, 3, 3> I;
    I.set_identity();
    auto rc = on::rcond(I);
    if (!approx_equal(rc, 1.0, 1e-6)) {
        std::cerr << "rcond() failed: expected ~1.0, got " << rc << "\n";
        std::exit(1);
    }
    std::cout << "rcond() test passed\n";
}

void test_pinv() {
    on::Matrix<double, 3, 3> A;
    A(0,0)=1.0; A(0,1)=0.0; A(0,2)=0.0;
    A(1,0)=0.0; A(1,1)=2.0; A(1,2)=0.0;
    A(2,0)=0.0; A(2,1)=0.0; A(2,2)=3.0;
    
    auto A_pinv = on::pinv(A);
    auto I = on::matmul(A, A_pinv);
    
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
    on::Matrix<double, 3, 3> A;
    A.set_identity();
    auto N = on::null(A);
    auto r = on::rank(N);
    if (r != 0) {
        std::cerr << "null() failed: expected rank 0, got " << r << "\n";
        std::exit(1);
    }
    std::cout << "null() test passed\n";
}

void test_orth() {
    on::Matrix<double, 3, 3> A;
    A.set_identity();
    auto Q = on::orth(A);
    auto QtQ = on::matmul(on::transpose(Q), Q);
    
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
    on::Matrix<double, 2, 2> A;
    A(0,0)=1.0; A(0,1)=2.0;
    A(1,0)=3.0; A(1,1)=4.0;
    
    on::Matrix<double, 2, 2> B;
    B(0,0)=0.0; B(0,1)=5.0;
    B(1,0)=6.0; B(1,1)=7.0;
    
    auto C = on::kron(A, B);
    if (C.rows() != 4 || C.cols() != 4) {
        std::cerr << "kron() shape failed: expected 4x4, got " << C.rows() << "x" << C.cols() << "\n";
        std::exit(1);
    }
    std::cout << "kron() test passed\n";
}

void test_is_symmetric() {
    on::Matrix<double, 3, 3> A;
    A(0,0)=1.0; A(0,1)=2.0; A(0,2)=3.0;
    A(1,0)=2.0; A(1,1)=4.0; A(1,2)=5.0;
    A(2,0)=3.0; A(2,1)=5.0; A(2,2)=6.0;
    
    if (!on::is_symmetric(A)) {
        std::cerr << "is_symmetric() failed: expected true\n";
        std::exit(1);
    }
    std::cout << "is_symmetric() test passed\n";
}

void test_is_positive_definite() {
    on::Matrix<double, 3, 3> A;
    A(0,0)=4.0; A(0,1)=2.0; A(0,2)=1.0;
    A(1,0)=2.0; A(1,1)=5.0; A(1,2)=2.0;
    A(2,0)=1.0; A(2,1)=2.0; A(2,2)=6.0;
    
    if (!on::is_positive_definite(A)) {
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
