#include <doctest/doctest.h>
#include <optinum/lina/basic/matmul.hpp>

using optinum::lina::matmul;
using optinum::simd::Matrix;
using optinum::simd::Tensor;

TEST_CASE("lina::matmul matrix-matrix") {
    Matrix<float, 2, 2> a;
    a(0, 0) = 1.f;
    a(1, 0) = 3.f;
    a(0, 1) = 2.f;
    a(1, 1) = 4.f;

    Matrix<float, 2, 2> b;
    b(0, 0) = 5.f;
    b(1, 0) = 7.f;
    b(0, 1) = 6.f;
    b(1, 1) = 8.f;

    const auto c = matmul(a, b);
    CHECK(c(0, 0) == doctest::Approx(19.f));
    CHECK(c(0, 1) == doctest::Approx(22.f));
    CHECK(c(1, 0) == doctest::Approx(43.f));
    CHECK(c(1, 1) == doctest::Approx(50.f));
}

TEST_CASE("lina::matmul matrix-vector") {
    Matrix<float, 2, 3> m;
    m(0, 0) = 1.f;
    m(1, 0) = 4.f;
    m(0, 1) = 2.f;
    m(1, 1) = 5.f;
    m(0, 2) = 3.f;
    m(1, 2) = 6.f;

    Tensor<float, 3> x;
    x[0] = 1.f;
    x[1] = 2.f;
    x[2] = 3.f;

    const auto y = matmul(m, x);
    CHECK(y[0] == doctest::Approx(14.f));
    CHECK(y[1] == doctest::Approx(32.f));
}

