#include <doctest/doctest.h>
#include <optinum/lina/basic/matmul.hpp>
#include <optinum/lina/solve/solve.hpp>

namespace lina = optinum::lina;
using optinum::simd::Matrix;
using optinum::simd::Vector;

namespace dp = datapod;

TEST_CASE("lina::solve solves Ax=b") {
    dp::mat::matrix<double, 3, 3> a;
    a(0, 0) = 3.0;
    a(1, 0) = 2.0;
    a(2, 0) = -1.0;
    a(0, 1) = 2.0;
    a(1, 1) = -2.0;
    a(2, 1) = 4.0;
    a(0, 2) = -1.0;
    a(1, 2) = 0.5;
    a(2, 2) = -1.0;

    dp::mat::vector<double, 3> b;
    b[0] = 1.0;
    b[1] = -2.0;
    b[2] = 0.0;

    const auto x = lina::solve(Matrix<double, 3, 3>(a), Vector<double, 3>(b));
    // x is dp::mat::vector, wrap in view for matmul
    const auto ax = lina::matmul(Matrix<double, 3, 3>(a), Vector<double, 3>(x));
    CHECK(ax[0] == doctest::Approx(b[0]).epsilon(1e-9));
    CHECK(ax[1] == doctest::Approx(b[1]).epsilon(1e-9));
    CHECK(ax[2] == doctest::Approx(b[2]).epsilon(1e-9));
}

TEST_CASE("lina::try_solve detects singular") {
    dp::mat::matrix<double, 2, 2> a;
    a(0, 0) = 1.0;
    a(1, 0) = 2.0;
    a(0, 1) = 2.0;
    a(1, 1) = 4.0;

    dp::mat::vector<double, 2> b;
    b[0] = 1.0;
    b[1] = 2.0;

    const auto r = lina::try_solve(Matrix<double, 2, 2>(a), Vector<double, 2>(b));
    CHECK(r.is_err());
}
