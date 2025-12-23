#include <doctest/doctest.h>
#include <optinum/simd/backend/norm.hpp>

TEST_CASE("backend norm_l2 and normalize") {
    float a[3] = {3.f, 0.f, 4.f};
    CHECK(optinum::simd::backend::norm_l2<float, 3>(a) == doctest::Approx(5.f));

    float out[3] = {};
    optinum::simd::backend::normalize<float, 3>(out, a);
    CHECK(out[0] == doctest::Approx(0.6f));
    CHECK(out[1] == doctest::Approx(0.f));
    CHECK(out[2] == doctest::Approx(0.8f));
}

