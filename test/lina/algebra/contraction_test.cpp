#include <doctest/doctest.h>
#include <optinum/lina/algebra/contraction.hpp>

namespace lina = optinum::lina;
using optinum::simd::Matrix;
using optinum::simd::Vector;

namespace dp = datapod;

TEST_CASE("lina::inner (vector)") {
    dp::mat::Vector<float, 3> a;
    a[0] = 1.f;
    a[1] = 2.f;
    a[2] = 3.f;
    dp::mat::Vector<float, 3> b;
    b[0] = 4.f;
    b[1] = 5.f;
    b[2] = 6.f;
    CHECK(lina::inner(Vector<float, 3>(a), Vector<float, 3>(b)) == doctest::Approx(32.f));
}

TEST_CASE("lina::outer") {
    dp::mat::Vector<int, 2> a;
    a[0] = 2;
    a[1] = 3;
    dp::mat::Vector<int, 3> b;
    b[0] = 4;
    b[1] = 5;
    b[2] = 6;

    const auto m = lina::outer(Vector<int, 2>(a), Vector<int, 3>(b));
    CHECK(m(0, 0) == 8);
    CHECK(m(1, 2) == 18);
}

TEST_CASE("lina::inner (matrix)") {
    dp::mat::Matrix<int, 2, 2> a;
    a(0, 0) = 1;
    a(1, 0) = 3;
    a(0, 1) = 2;
    a(1, 1) = 4;

    dp::mat::Matrix<int, 2, 2> b;
    b(0, 0) = 5;
    b(1, 0) = 7;
    b(0, 1) = 6;
    b(1, 1) = 8;

    CHECK(lina::inner(Matrix<int, 2, 2>(a), Matrix<int, 2, 2>(b)) == 70);
}
