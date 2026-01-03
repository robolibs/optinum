#include <doctest/doctest.h>
#include <optinum/simd/backend/norm.hpp>

#include <cmath>

TEST_CASE("backend norm_l2 and normalize - small array") {
    float a[3] = {3.f, 0.f, 4.f};
    CHECK(optinum::simd::backend::norm_l2<float, 3>(a) == doctest::Approx(5.f));

    float out[3] = {};
    optinum::simd::backend::normalize<float, 3>(out, a);
    CHECK(out[0] == doctest::Approx(0.6f));
    CHECK(out[1] == doctest::Approx(0.f));
    CHECK(out[2] == doctest::Approx(0.8f));
}

TEST_CASE("backend norm_l2 - 16 elements") {
    float a[16];
    float sum_sq = 0.f;
    for (int i = 0; i < 16; ++i) {
        a[i] = static_cast<float>(i + 1);
        sum_sq += a[i] * a[i];
    }
    float expected = std::sqrt(sum_sq);
    CHECK(optinum::simd::backend::norm_l2<float, 16>(a) == doctest::Approx(expected));
}

TEST_CASE("backend norm_l2 - 32 elements") {
    double a[32];
    double sum_sq = 0.0;
    for (int i = 0; i < 32; ++i) {
        a[i] = static_cast<double>(i) * 0.1;
        sum_sq += a[i] * a[i];
    }
    double expected = std::sqrt(sum_sq);
    CHECK(optinum::simd::backend::norm_l2<double, 32>(a) == doctest::Approx(expected));
}

TEST_CASE("backend norm_l2 - 64 elements") {
    float a[64];
    float sum_sq = 0.f;
    for (int i = 0; i < 64; ++i) {
        a[i] = std::sin(static_cast<float>(i) * 0.1f);
        sum_sq += a[i] * a[i];
    }
    float expected = std::sqrt(sum_sq);
    CHECK(optinum::simd::backend::norm_l2<float, 64>(a) == doctest::Approx(expected).epsilon(1e-5));
}

TEST_CASE("backend norm_l2 - 128 elements") {
    double a[128];
    double sum_sq = 0.0;
    for (int i = 0; i < 128; ++i) {
        a[i] = 1.0;
        sum_sq += a[i] * a[i];
    }
    double expected = std::sqrt(sum_sq); // sqrt(128)
    CHECK(optinum::simd::backend::norm_l2<double, 128>(a) == doctest::Approx(expected));
}

TEST_CASE("backend norm_l2 - non-power-of-2 sizes") {
    SUBCASE("17 elements") {
        float a[17];
        float sum_sq = 0.f;
        for (int i = 0; i < 17; ++i) {
            a[i] = 1.f;
            sum_sq += a[i] * a[i];
        }
        float expected = std::sqrt(sum_sq); // sqrt(17)
        CHECK(optinum::simd::backend::norm_l2<float, 17>(a) == doctest::Approx(expected));
    }

    SUBCASE("33 elements") {
        double a[33];
        double sum_sq = 0.0;
        for (int i = 0; i < 33; ++i) {
            a[i] = static_cast<double>(i + 1);
            sum_sq += a[i] * a[i];
        }
        double expected = std::sqrt(sum_sq);
        CHECK(optinum::simd::backend::norm_l2<double, 33>(a) == doctest::Approx(expected));
    }

    SUBCASE("65 elements") {
        float a[65];
        float sum_sq = 0.f;
        for (int i = 0; i < 65; ++i) {
            a[i] = 2.f;
            sum_sq += a[i] * a[i];
        }
        float expected = std::sqrt(sum_sq); // sqrt(65 * 4) = sqrt(260)
        CHECK(optinum::simd::backend::norm_l2<float, 65>(a) == doctest::Approx(expected));
    }
}

TEST_CASE("backend normalize - larger arrays") {
    SUBCASE("16 elements") {
        float a[16], out[16];
        float sum_sq = 0.f;
        for (int i = 0; i < 16; ++i) {
            a[i] = static_cast<float>(i + 1);
            sum_sq += a[i] * a[i];
        }
        float norm = std::sqrt(sum_sq);

        optinum::simd::backend::normalize<float, 16>(out, a);

        for (int i = 0; i < 16; ++i) {
            CHECK(out[i] == doctest::Approx(a[i] / norm));
        }
    }

    SUBCASE("32 elements") {
        double a[32], out[32];
        double sum_sq = 0.0;
        for (int i = 0; i < 32; ++i) {
            a[i] = static_cast<double>(i);
            sum_sq += a[i] * a[i];
        }
        double norm = std::sqrt(sum_sq);

        optinum::simd::backend::normalize<double, 32>(out, a);

        for (int i = 0; i < 32; ++i) {
            CHECK(out[i] == doctest::Approx(a[i] / norm));
        }
    }

    SUBCASE("33 elements (non-power-of-2)") {
        float a[33], out[33];
        float sum_sq = 0.f;
        for (int i = 0; i < 33; ++i) {
            a[i] = static_cast<float>(i + 1);
            sum_sq += a[i] * a[i];
        }
        float norm = std::sqrt(sum_sq);

        optinum::simd::backend::normalize<float, 33>(out, a);

        for (int i = 0; i < 33; ++i) {
            CHECK(out[i] == doctest::Approx(a[i] / norm));
        }
    }
}

TEST_CASE("backend norm with negative values") {
    double a[16] = {-1, -2, -3, -4, -5, -6, -7, -8, 1, 2, 3, 4, 5, 6, 7, 8};
    double sum_sq = 0.0;
    for (int i = 0; i < 16; ++i) {
        sum_sq += a[i] * a[i];
    }
    double expected = std::sqrt(sum_sq);
    CHECK(optinum::simd::backend::norm_l2<double, 16>(a) == doctest::Approx(expected));
}
