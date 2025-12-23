#include <doctest/doctest.h>
#include <optinum/lina/basic/transpose.hpp>

using optinum::simd::Matrix;

TEST_CASE("lina::transpose") {
    Matrix<int, 2, 3> a;
    a(0, 0) = 1;
    a(1, 0) = 4;
    a(0, 1) = 2;
    a(1, 1) = 5;
    a(0, 2) = 3;
    a(1, 2) = 6;

    const auto at = optinum::lina::transpose(a);
    CHECK(at.rows() == 3);
    CHECK(at.cols() == 2);
    CHECK(at(0, 0) == 1);
    CHECK(at(1, 0) == 2);
    CHECK(at(2, 0) == 3);
    CHECK(at(0, 1) == 4);
    CHECK(at(1, 1) == 5);
    CHECK(at(2, 1) == 6);
}
