#include <doctest/doctest.h>
#include <optinum/simd/backend/reduce.hpp>

#include <cmath>
#include <limits>

TEST_CASE("backend reduce_sum/min/max - small array") {
    int a[7] = {5, 1, 9, -2, 3, 3, 0};

    CHECK(optinum::simd::backend::reduce_sum<int, 7>(a) == 19);
    CHECK(optinum::simd::backend::reduce_min<int, 7>(a) == -2);
    CHECK(optinum::simd::backend::reduce_max<int, 7>(a) == 9);
}

TEST_CASE("backend reduce_sum - 16 elements") {
    float a[16];
    float expected = 0.f;
    for (int i = 0; i < 16; ++i) {
        a[i] = static_cast<float>(i + 1);
        expected += a[i];
    }
    CHECK(optinum::simd::backend::reduce_sum<float, 16>(a) == doctest::Approx(expected));
}

TEST_CASE("backend reduce_sum - 32 elements") {
    double a[32];
    double expected = 0.0;
    for (int i = 0; i < 32; ++i) {
        a[i] = static_cast<double>(i) * 0.5;
        expected += a[i];
    }
    CHECK(optinum::simd::backend::reduce_sum<double, 32>(a) == doctest::Approx(expected));
}

TEST_CASE("backend reduce_sum - 64 elements") {
    float a[64];
    float expected = 0.f;
    for (int i = 0; i < 64; ++i) {
        a[i] = std::sin(static_cast<float>(i) * 0.1f);
        expected += a[i];
    }
    CHECK(optinum::simd::backend::reduce_sum<float, 64>(a) == doctest::Approx(expected).epsilon(1e-5));
}

TEST_CASE("backend reduce_sum - 128 elements") {
    double a[128];
    double expected = 0.0;
    for (int i = 0; i < 128; ++i) {
        a[i] = 1.0;
        expected += a[i];
    }
    CHECK(optinum::simd::backend::reduce_sum<double, 128>(a) == doctest::Approx(128.0));
}

TEST_CASE("backend reduce_sum - non-power-of-2 sizes") {
    SUBCASE("17 elements") {
        float a[17];
        float expected = 0.f;
        for (int i = 0; i < 17; ++i) {
            a[i] = static_cast<float>(i);
            expected += a[i];
        }
        CHECK(optinum::simd::backend::reduce_sum<float, 17>(a) == doctest::Approx(expected));
    }

    SUBCASE("33 elements") {
        double a[33];
        double expected = 0.0;
        for (int i = 0; i < 33; ++i) {
            a[i] = static_cast<double>(i + 1);
            expected += a[i];
        }
        CHECK(optinum::simd::backend::reduce_sum<double, 33>(a) == doctest::Approx(expected));
    }

    SUBCASE("65 elements") {
        float a[65];
        float expected = 0.f;
        for (int i = 0; i < 65; ++i) {
            a[i] = 2.f;
            expected += a[i];
        }
        CHECK(optinum::simd::backend::reduce_sum<float, 65>(a) == doctest::Approx(130.f));
    }
}

TEST_CASE("backend reduce_min - larger arrays") {
    SUBCASE("16 elements") {
        float a[16];
        for (int i = 0; i < 16; ++i) {
            a[i] = static_cast<float>(i + 1);
        }
        a[7] = -5.f; // minimum
        CHECK(optinum::simd::backend::reduce_min<float, 16>(a) == doctest::Approx(-5.f));
    }

    SUBCASE("32 elements") {
        double a[32];
        for (int i = 0; i < 32; ++i) {
            a[i] = static_cast<double>(i);
        }
        a[15] = -100.0; // minimum
        CHECK(optinum::simd::backend::reduce_min<double, 32>(a) == doctest::Approx(-100.0));
    }

    SUBCASE("33 elements (non-power-of-2)") {
        float a[33];
        for (int i = 0; i < 33; ++i) {
            a[i] = static_cast<float>(i + 10);
        }
        a[32] = -1.f; // minimum at tail
        CHECK(optinum::simd::backend::reduce_min<float, 33>(a) == doctest::Approx(-1.f));
    }
}

TEST_CASE("backend reduce_max - larger arrays") {
    SUBCASE("16 elements") {
        float a[16];
        for (int i = 0; i < 16; ++i) {
            a[i] = static_cast<float>(i);
        }
        a[3] = 100.f; // maximum
        CHECK(optinum::simd::backend::reduce_max<float, 16>(a) == doctest::Approx(100.f));
    }

    SUBCASE("32 elements") {
        double a[32];
        for (int i = 0; i < 32; ++i) {
            a[i] = static_cast<double>(-i);
        }
        a[20] = 50.0; // maximum
        CHECK(optinum::simd::backend::reduce_max<double, 32>(a) == doctest::Approx(50.0));
    }

    SUBCASE("65 elements (non-power-of-2)") {
        float a[65];
        for (int i = 0; i < 65; ++i) {
            a[i] = static_cast<float>(i);
        }
        a[64] = 1000.f; // maximum at tail
        CHECK(optinum::simd::backend::reduce_max<float, 65>(a) == doctest::Approx(1000.f));
    }
}

TEST_CASE("backend reduce with negative values") {
    double a[16] = {-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16};

    CHECK(optinum::simd::backend::reduce_sum<double, 16>(a) == doctest::Approx(-136.0));
    CHECK(optinum::simd::backend::reduce_min<double, 16>(a) == doctest::Approx(-16.0));
    CHECK(optinum::simd::backend::reduce_max<double, 16>(a) == doctest::Approx(-1.0));
}
