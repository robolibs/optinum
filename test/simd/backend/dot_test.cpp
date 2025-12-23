#include <doctest/doctest.h>
#include <optinum/simd/backend/dot.hpp>

TEST_CASE("backend dot") {
    float a[4] = {1.f, 2.f, 3.f, 4.f};
    float b[4] = {5.f, 6.f, 7.f, 8.f};
    CHECK(optinum::simd::backend::dot<float, 4>(a, b) == doctest::Approx(70.f));
}

