#include <doctest/doctest.h>
#include <optinum/simd/backend/dot.hpp>

#include <cmath>
#include <random>

TEST_CASE("backend dot - small arrays") {
    float a[4] = {1.f, 2.f, 3.f, 4.f};
    float b[4] = {5.f, 6.f, 7.f, 8.f};
    CHECK(optinum::simd::backend::dot<float, 4>(a, b) == doctest::Approx(70.f));
}

TEST_CASE("backend dot - 16 elements (SSE/NEON friendly)") {
    float a[16], b[16];
    float expected = 0.f;
    for (int i = 0; i < 16; ++i) {
        a[i] = static_cast<float>(i + 1);
        b[i] = static_cast<float>(i + 1);
        expected += a[i] * b[i];
    }
    CHECK(optinum::simd::backend::dot<float, 16>(a, b) == doctest::Approx(expected));
}

TEST_CASE("backend dot - 32 elements (AVX friendly)") {
    double a[32], b[32];
    double expected = 0.0;
    for (int i = 0; i < 32; ++i) {
        a[i] = static_cast<double>(i + 1);
        b[i] = static_cast<double>(16 - i);
        expected += a[i] * b[i];
    }
    CHECK(optinum::simd::backend::dot<double, 32>(a, b) == doctest::Approx(expected));
}

TEST_CASE("backend dot - 64 elements") {
    float a[64], b[64];
    float expected = 0.f;
    for (int i = 0; i < 64; ++i) {
        a[i] = std::sin(static_cast<float>(i) * 0.1f);
        b[i] = std::cos(static_cast<float>(i) * 0.1f);
        expected += a[i] * b[i];
    }
    CHECK(optinum::simd::backend::dot<float, 64>(a, b) == doctest::Approx(expected).epsilon(1e-5));
}

TEST_CASE("backend dot - 128 elements") {
    double a[128], b[128];
    double expected = 0.0;
    for (int i = 0; i < 128; ++i) {
        a[i] = static_cast<double>(i) * 0.01;
        b[i] = static_cast<double>(128 - i) * 0.01;
        expected += a[i] * b[i];
    }
    CHECK(optinum::simd::backend::dot<double, 128>(a, b) == doctest::Approx(expected));
}

TEST_CASE("backend dot - non-power-of-2 sizes (tail handling)") {
    SUBCASE("17 elements") {
        float a[17], b[17];
        float expected = 0.f;
        for (int i = 0; i < 17; ++i) {
            a[i] = static_cast<float>(i + 1);
            b[i] = 2.f;
            expected += a[i] * b[i];
        }
        CHECK(optinum::simd::backend::dot<float, 17>(a, b) == doctest::Approx(expected));
    }

    SUBCASE("33 elements") {
        double a[33], b[33];
        double expected = 0.0;
        for (int i = 0; i < 33; ++i) {
            a[i] = static_cast<double>(i);
            b[i] = 1.0;
            expected += a[i] * b[i];
        }
        CHECK(optinum::simd::backend::dot<double, 33>(a, b) == doctest::Approx(expected));
    }

    SUBCASE("65 elements") {
        float a[65], b[65];
        float expected = 0.f;
        for (int i = 0; i < 65; ++i) {
            a[i] = 1.f;
            b[i] = static_cast<float>(i);
            expected += a[i] * b[i];
        }
        CHECK(optinum::simd::backend::dot<float, 65>(a, b) == doctest::Approx(expected));
    }
}

TEST_CASE("backend dot_runtime - dynamic sizes") {
    SUBCASE("16 elements") {
        float a[16], b[16];
        float expected = 0.f;
        for (int i = 0; i < 16; ++i) {
            a[i] = static_cast<float>(i + 1);
            b[i] = static_cast<float>(i + 1);
            expected += a[i] * b[i];
        }
        CHECK(optinum::simd::backend::dot_runtime<float>(a, b, 16) == doctest::Approx(expected));
    }

    SUBCASE("33 elements (non-power-of-2)") {
        double a[33], b[33];
        double expected = 0.0;
        for (int i = 0; i < 33; ++i) {
            a[i] = static_cast<double>(i + 1);
            b[i] = 0.5;
            expected += a[i] * b[i];
        }
        CHECK(optinum::simd::backend::dot_runtime<double>(a, b, 33) == doctest::Approx(expected));
    }

    SUBCASE("100 elements") {
        float a[100], b[100];
        float expected = 0.f;
        for (int i = 0; i < 100; ++i) {
            a[i] = static_cast<float>(i % 10);
            b[i] = static_cast<float>((i + 5) % 10);
            expected += a[i] * b[i];
        }
        CHECK(optinum::simd::backend::dot_runtime<float>(a, b, 100) == doctest::Approx(expected));
    }
}
