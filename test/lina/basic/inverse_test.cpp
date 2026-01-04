#include <doctest/doctest.h>
#include <optinum/lina/basic/inverse.hpp>

namespace lina = optinum::lina;
using optinum::simd::Matrix;


TEST_CASE("lina::inverse 2x2") {
    // A = [4 7; 2 6], A^{-1} = (1/10)*[6 -7; -2 4]
    dp::mat::Matrix<double, 2, 2> a;
    a(0, 0) = 4.0;
    a(1, 0) = 2.0;
    a(0, 1) = 7.0;
    a(1, 1) = 6.0;

    const auto inv = lina::inverse(Matrix<double, 2, 2>(a));
    CHECK(inv(0, 0) == doctest::Approx(0.6));
    CHECK(inv(0, 1) == doctest::Approx(-0.7));
    CHECK(inv(1, 0) == doctest::Approx(-0.2));
    CHECK(inv(1, 1) == doctest::Approx(0.4));
}

TEST_CASE("lina::try_inverse detects singular") {
    dp::mat::Matrix<double, 2, 2> a;
    a(0, 0) = 1.0;
    a(1, 0) = 2.0;
    a(0, 1) = 2.0;
    a(1, 1) = 4.0;

    const auto r = lina::try_inverse(Matrix<double, 2, 2>(a));
    CHECK(r.is_err());
}
