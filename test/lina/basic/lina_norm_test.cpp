#include <doctest/doctest.h>
#include <optinum/lina/basic/norm.hpp>

using optinum::lina::cross;
using optinum::lina::norm_fro;
using optinum::simd::Matrix;
using optinum::simd::Tensor;

TEST_CASE("lina::dot and lina::norm (vector)") {
    Tensor<float, 3> a;
    a[0] = 3.f;
    a[1] = 0.f;
    a[2] = 4.f;

    CHECK(optinum::lina::dot(a, a) == doctest::Approx(25.f));
    CHECK(optinum::lina::norm(a) == doctest::Approx(5.f));
}

TEST_CASE("lina::cross") {
    Tensor<int, 3> a;
    a[0] = 1;
    a[1] = 0;
    a[2] = 0;
    Tensor<int, 3> b;
    b[0] = 0;
    b[1] = 1;
    b[2] = 0;

    const auto c = cross(a, b);
    CHECK(c[0] == 0);
    CHECK(c[1] == 0);
    CHECK(c[2] == 1);
}

TEST_CASE("lina::norm_fro (matrix)") {
    Matrix<float, 2, 2> a;
    a(0, 0) = 1.f;
    a(1, 0) = 3.f;
    a(0, 1) = 2.f;
    a(1, 1) = 4.f;
    CHECK(norm_fro(a) == doctest::Approx(std::sqrt(30.f)));
}
