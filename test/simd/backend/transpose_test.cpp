#include <doctest/doctest.h>
#include <optinum/simd/backend/transpose.hpp>

TEST_CASE("backend transpose column-major 2x3 -> 3x2") {
    // M = [1 2 3; 4 5 6] (2x3), column-major => [1,4,2,5,3,6]
    float M[6] = {1.f, 4.f, 2.f, 5.f, 3.f, 6.f};
    float MT[6] = {};

    optinum::simd::backend::transpose<float, 2, 3>(MT, M);

    // MT = [1 4; 2 5; 3 6] (3x2), column-major => [1,2,3,4,5,6]
    CHECK(MT[0] == doctest::Approx(1.f));
    CHECK(MT[1] == doctest::Approx(2.f));
    CHECK(MT[2] == doctest::Approx(3.f));
    CHECK(MT[3] == doctest::Approx(4.f));
    CHECK(MT[4] == doctest::Approx(5.f));
    CHECK(MT[5] == doctest::Approx(6.f));
}

