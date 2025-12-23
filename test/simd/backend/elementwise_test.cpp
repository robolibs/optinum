#include <doctest/doctest.h>
#include <optinum/simd/backend/elementwise.hpp>

TEST_CASE("backend elementwise add/sub/mul/div") {
    float a[8] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
    float b[8] = {8.f, 7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f};

    float out[8] = {};

    optinum::simd::backend::add<float, 8>(out, a, b);
    CHECK(out[0] == doctest::Approx(9.f));
    CHECK(out[7] == doctest::Approx(9.f));

    optinum::simd::backend::sub<float, 8>(out, a, b);
    CHECK(out[0] == doctest::Approx(-7.f));
    CHECK(out[7] == doctest::Approx(7.f));

    optinum::simd::backend::mul<float, 8>(out, a, b);
    CHECK(out[0] == doctest::Approx(8.f));
    CHECK(out[7] == doctest::Approx(8.f));

    optinum::simd::backend::div<float, 8>(out, a, b);
    CHECK(out[0] == doctest::Approx(1.f / 8.f));
    CHECK(out[7] == doctest::Approx(8.f / 1.f));
}

TEST_CASE("backend elementwise scalar multiply/divide") {
    double a[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double out[5] = {};

    optinum::simd::backend::mul_scalar<double, 5>(out, a, 2.0);
    CHECK(out[0] == doctest::Approx(2.0));
    CHECK(out[4] == doctest::Approx(10.0));

    optinum::simd::backend::div_scalar<double, 5>(out, a, 2.0);
    CHECK(out[0] == doctest::Approx(0.5));
    CHECK(out[4] == doctest::Approx(2.5));
}

