#include <doctest/doctest.h>
#include <optinum/lina/solve/solve.hpp>

using optinum::lina::solve;
using optinum::lina::try_solve;
using optinum::simd::Matrix;
using optinum::simd::Tensor;

TEST_CASE("lina::solve solves Ax=b") {
    Matrix<double, 3, 3> a;
    a(0, 0) = 3.0;
    a(1, 0) = 2.0;
    a(2, 0) = -1.0;
    a(0, 1) = 2.0;
    a(1, 1) = -2.0;
    a(2, 1) = 4.0;
    a(0, 2) = -1.0;
    a(1, 2) = 0.5;
    a(2, 2) = -1.0;

    Tensor<double, 3> b;
    b[0] = 1.0;
    b[1] = -2.0;
    b[2] = 0.0;

    const auto x = solve(a, b);
    const auto ax = a * x;
    CHECK(ax[0] == doctest::Approx(b[0]).epsilon(1e-9));
    CHECK(ax[1] == doctest::Approx(b[1]).epsilon(1e-9));
    CHECK(ax[2] == doctest::Approx(b[2]).epsilon(1e-9));
}

TEST_CASE("lina::try_solve detects singular") {
    Matrix<double, 2, 2> a;
    a(0, 0) = 1.0;
    a(1, 0) = 2.0;
    a(0, 1) = 2.0;
    a(1, 1) = 4.0;

    Tensor<double, 2> b;
    b[0] = 1.0;
    b[1] = 2.0;

    const auto r = try_solve(a, b);
    CHECK(r.is_err());
}

