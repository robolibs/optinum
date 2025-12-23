#include <doctest/doctest.h>
#include <optinum/lina/decompose/lu.hpp>

using optinum::lina::lu;
using optinum::lina::lu_solve;
using optinum::lina::permutation_matrix;
using optinum::simd::Matrix;
using optinum::simd::Tensor;

TEST_CASE("lina::lu reconstructs P*A = L*U") {
    Matrix<double, 3, 3> a;
    a(0, 0) = 0.0;
    a(1, 0) = 1.0;
    a(2, 0) = 2.0;
    a(0, 1) = 2.0;
    a(1, 1) = 1.0;
    a(2, 1) = 0.0;
    a(0, 2) = 1.0;
    a(1, 2) = 0.0;
    a(2, 2) = 1.0;

    const auto f = lu(a);
    CHECK(!f.singular);

    const auto P = permutation_matrix<double, 3>(f.p);
    const auto lhs = P * a;
    const auto rhs = f.l * f.u;

    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            CHECK(lhs(i, j) == doctest::Approx(rhs(i, j)));
        }
    }
}

TEST_CASE("lina::lu_solve solves Ax=b") {
    Matrix<double, 2, 2> a;
    a(0, 0) = 4.0;
    a(1, 0) = 2.0;
    a(0, 1) = 1.0;
    a(1, 1) = 3.0;

    Tensor<double, 2> b;
    b[0] = 1.0;
    b[1] = 2.0;

    const auto f = lu(a);
    const auto x = lu_solve(f, b);
    // Verify A*x == b
    const auto bx = a * x;
    CHECK(bx[0] == doctest::Approx(b[0]));
    CHECK(bx[1] == doctest::Approx(b[1]));
}

