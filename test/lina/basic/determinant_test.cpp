#include <doctest/doctest.h>
#include <optinum/lina/basic/determinant.hpp>

using optinum::lina::determinant;
using optinum::simd::Matrix;

TEST_CASE("lina::determinant 2x2") {
    // [1 2; 3 4] det = -2
    Matrix<double, 2, 2> a;
    a(0, 0) = 1.0;
    a(1, 0) = 3.0;
    a(0, 1) = 2.0;
    a(1, 1) = 4.0;
    CHECK(determinant(a) == doctest::Approx(-2.0));
}

TEST_CASE("lina::determinant 3x3") {
    // det of [[1,2,3],[0,1,4],[5,6,0]] = 1*(1*0-4*6)-2*(0*0-4*5)+3*(0*6-1*5) = -24+40-15=1
    Matrix<double, 3, 3> a;
    a(0, 0) = 1.0;
    a(1, 0) = 0.0;
    a(2, 0) = 5.0;
    a(0, 1) = 2.0;
    a(1, 1) = 1.0;
    a(2, 1) = 6.0;
    a(0, 2) = 3.0;
    a(1, 2) = 4.0;
    a(2, 2) = 0.0;
    CHECK(determinant(a) == doctest::Approx(1.0));
}

