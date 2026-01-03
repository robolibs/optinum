#include <doctest/doctest.h>
#include <optinum/lina/basic/norm.hpp>

namespace lina = optinum::lina;
using optinum::simd::Matrix;
using optinum::simd::Vector;

namespace dp = datapod;

TEST_CASE("lina::dot and lina::norm (vector)") {
    dp::mat::Vector<float, 3> a;
    a[0] = 3.f;
    a[1] = 0.f;
    a[2] = 4.f;

    CHECK(lina::dot(Vector<float, 3>(a), Vector<float, 3>(a)) == doctest::Approx(25.f));
    CHECK(lina::norm(Vector<float, 3>(a)) == doctest::Approx(5.f));
}

TEST_CASE("lina::cross") {
    dp::mat::Vector<int, 3> a;
    a[0] = 1;
    a[1] = 0;
    a[2] = 0;
    dp::mat::Vector<int, 3> b;
    b[0] = 0;
    b[1] = 1;
    b[2] = 0;

    const auto c = lina::cross(Vector<int, 3>(a), Vector<int, 3>(b));
    CHECK(c[0] == 0);
    CHECK(c[1] == 0);
    CHECK(c[2] == 1);
}

TEST_CASE("lina::norm_fro (matrix)") {
    dp::mat::Matrix<float, 2, 2> a;
    a(0, 0) = 1.f;
    a(1, 0) = 3.f;
    a(0, 1) = 2.f;
    a(1, 1) = 4.f;
    CHECK(lina::norm_fro(Matrix<float, 2, 2>(a)) == doctest::Approx(std::sqrt(30.f)));
}
