#include <doctest/doctest.h>
#include <optinum/simd/backend/elementwise.hpp>

#include <cmath>

TEST_CASE("backend elementwise add/sub/mul/div - 8 elements") {
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

TEST_CASE("backend elementwise scalar multiply/divide - 5 elements") {
    double a[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double out[5] = {};

    optinum::simd::backend::mul_scalar<double, 5>(out, a, 2.0);
    CHECK(out[0] == doctest::Approx(2.0));
    CHECK(out[4] == doctest::Approx(10.0));

    optinum::simd::backend::div_scalar<double, 5>(out, a, 2.0);
    CHECK(out[0] == doctest::Approx(0.5));
    CHECK(out[4] == doctest::Approx(2.5));
}

TEST_CASE("backend elementwise - 16 elements (SSE/NEON friendly)") {
    float a[16], b[16], out[16];
    for (int i = 0; i < 16; ++i) {
        a[i] = static_cast<float>(i + 1);
        b[i] = static_cast<float>(16 - i);
    }

    optinum::simd::backend::add<float, 16>(out, a, b);
    for (int i = 0; i < 16; ++i) {
        CHECK(out[i] == doctest::Approx(17.f));
    }

    optinum::simd::backend::mul<float, 16>(out, a, b);
    for (int i = 0; i < 16; ++i) {
        CHECK(out[i] == doctest::Approx(a[i] * b[i]));
    }
}

TEST_CASE("backend elementwise - 32 elements (AVX friendly)") {
    double a[32], b[32], out[32];
    for (int i = 0; i < 32; ++i) {
        a[i] = static_cast<double>(i);
        b[i] = 2.0;
    }

    optinum::simd::backend::add<double, 32>(out, a, b);
    for (int i = 0; i < 32; ++i) {
        CHECK(out[i] == doctest::Approx(a[i] + 2.0));
    }

    optinum::simd::backend::sub<double, 32>(out, a, b);
    for (int i = 0; i < 32; ++i) {
        CHECK(out[i] == doctest::Approx(a[i] - 2.0));
    }
}

TEST_CASE("backend elementwise - 64 elements") {
    float a[64], b[64], out[64];
    for (int i = 0; i < 64; ++i) {
        a[i] = std::sin(static_cast<float>(i) * 0.1f);
        b[i] = std::cos(static_cast<float>(i) * 0.1f);
    }

    optinum::simd::backend::add<float, 64>(out, a, b);
    for (int i = 0; i < 64; ++i) {
        CHECK(out[i] == doctest::Approx(a[i] + b[i]).epsilon(1e-5));
    }
}

TEST_CASE("backend elementwise - 128 elements") {
    double a[128], b[128], out[128];
    for (int i = 0; i < 128; ++i) {
        a[i] = static_cast<double>(i) * 0.01;
        b[i] = static_cast<double>(128 - i) * 0.01;
    }

    optinum::simd::backend::mul<double, 128>(out, a, b);
    for (int i = 0; i < 128; ++i) {
        CHECK(out[i] == doctest::Approx(a[i] * b[i]));
    }
}

TEST_CASE("backend elementwise - non-power-of-2 sizes (tail handling)") {
    SUBCASE("17 elements") {
        float a[17], b[17], out[17];
        for (int i = 0; i < 17; ++i) {
            a[i] = static_cast<float>(i + 1);
            b[i] = 2.f;
        }
        optinum::simd::backend::add<float, 17>(out, a, b);
        for (int i = 0; i < 17; ++i) {
            CHECK(out[i] == doctest::Approx(a[i] + 2.f));
        }
    }

    SUBCASE("33 elements") {
        double a[33], b[33], out[33];
        for (int i = 0; i < 33; ++i) {
            a[i] = static_cast<double>(i);
            b[i] = 1.0;
        }
        optinum::simd::backend::sub<double, 33>(out, a, b);
        for (int i = 0; i < 33; ++i) {
            CHECK(out[i] == doctest::Approx(a[i] - 1.0));
        }
    }

    SUBCASE("65 elements") {
        float a[65], b[65], out[65];
        for (int i = 0; i < 65; ++i) {
            a[i] = 1.f;
            b[i] = static_cast<float>(i);
        }
        optinum::simd::backend::mul<float, 65>(out, a, b);
        for (int i = 0; i < 65; ++i) {
            CHECK(out[i] == doctest::Approx(static_cast<float>(i)));
        }
    }
}

// ============================================================================
// Runtime function tests
// ============================================================================

TEST_CASE("backend add_runtime - dynamic sizes") {
    SUBCASE("16 elements") {
        float a[16], b[16], out[16];
        for (int i = 0; i < 16; ++i) {
            a[i] = static_cast<float>(i + 1);
            b[i] = static_cast<float>(i + 1);
        }
        optinum::simd::backend::add_runtime<float>(out, a, b, 16);
        for (int i = 0; i < 16; ++i) {
            CHECK(out[i] == doctest::Approx(2.f * (i + 1)));
        }
    }

    SUBCASE("33 elements (non-power-of-2)") {
        double a[33], b[33], out[33];
        for (int i = 0; i < 33; ++i) {
            a[i] = static_cast<double>(i + 1);
            b[i] = 0.5;
        }
        optinum::simd::backend::add_runtime<double>(out, a, b, 33);
        for (int i = 0; i < 33; ++i) {
            CHECK(out[i] == doctest::Approx(a[i] + 0.5));
        }
    }

    SUBCASE("100 elements") {
        float a[100], b[100], out[100];
        for (int i = 0; i < 100; ++i) {
            a[i] = static_cast<float>(i % 10);
            b[i] = static_cast<float>((i + 5) % 10);
        }
        optinum::simd::backend::add_runtime<float>(out, a, b, 100);
        for (int i = 0; i < 100; ++i) {
            CHECK(out[i] == doctest::Approx(a[i] + b[i]));
        }
    }
}

TEST_CASE("backend sub_runtime - dynamic sizes") {
    SUBCASE("32 elements") {
        double a[32], b[32], out[32];
        for (int i = 0; i < 32; ++i) {
            a[i] = static_cast<double>(i * 2);
            b[i] = static_cast<double>(i);
        }
        optinum::simd::backend::sub_runtime<double>(out, a, b, 32);
        for (int i = 0; i < 32; ++i) {
            CHECK(out[i] == doctest::Approx(static_cast<double>(i)));
        }
    }
}

TEST_CASE("backend mul_runtime - dynamic sizes") {
    SUBCASE("64 elements") {
        float a[64], b[64], out[64];
        for (int i = 0; i < 64; ++i) {
            a[i] = static_cast<float>(i + 1);
            b[i] = 0.5f;
        }
        optinum::simd::backend::mul_runtime<float>(out, a, b, 64);
        for (int i = 0; i < 64; ++i) {
            CHECK(out[i] == doctest::Approx(a[i] * 0.5f));
        }
    }
}

TEST_CASE("backend mul_scalar_runtime - dynamic sizes") {
    SUBCASE("50 elements") {
        double a[50], out[50];
        for (int i = 0; i < 50; ++i) {
            a[i] = static_cast<double>(i + 1);
        }
        optinum::simd::backend::mul_scalar_runtime<double>(out, a, 3.0, 50);
        for (int i = 0; i < 50; ++i) {
            CHECK(out[i] == doctest::Approx(3.0 * (i + 1)));
        }
    }
}

TEST_CASE("backend axpy_runtime - dst = x + alpha * d") {
    SUBCASE("32 elements") {
        float x[32], d[32], dst[32];
        for (int i = 0; i < 32; ++i) {
            x[i] = 10.f;
            d[i] = static_cast<float>(i + 1);
        }
        // dst = x + alpha * d
        optinum::simd::backend::axpy_runtime<float>(dst, x, 2.f, d, 32);
        for (int i = 0; i < 32; ++i) {
            CHECK(dst[i] == doctest::Approx(10.f + 2.f * (i + 1)));
        }
    }

    SUBCASE("17 elements (non-power-of-2)") {
        double x[17], d[17], dst[17];
        for (int i = 0; i < 17; ++i) {
            x[i] = 5.0;
            d[i] = static_cast<double>(i);
        }
        optinum::simd::backend::axpy_runtime<double>(dst, x, -1.0, d, 17);
        for (int i = 0; i < 17; ++i) {
            CHECK(dst[i] == doctest::Approx(5.0 - static_cast<double>(i)));
        }
    }
}

TEST_CASE("backend axpy_inplace_runtime - x += alpha * d") {
    SUBCASE("64 elements") {
        float x[64], d[64];
        for (int i = 0; i < 64; ++i) {
            x[i] = static_cast<float>(i);
            d[i] = 1.f;
        }
        // x += alpha * d
        optinum::simd::backend::axpy_inplace_runtime<float>(x, 5.f, d, 64);
        for (int i = 0; i < 64; ++i) {
            CHECK(x[i] == doctest::Approx(static_cast<float>(i) + 5.f));
        }
    }
}

TEST_CASE("backend scale_sub_runtime - x -= alpha * g") {
    SUBCASE("32 elements") {
        double x[32], g[32];
        for (int i = 0; i < 32; ++i) {
            x[i] = 100.0;
            g[i] = static_cast<double>(i + 1);
        }
        // x -= alpha * g
        optinum::simd::backend::scale_sub_runtime<double>(x, 2.0, g, 32);
        for (int i = 0; i < 32; ++i) {
            CHECK(x[i] == doctest::Approx(100.0 - 2.0 * (i + 1)));
        }
    }
}

TEST_CASE("backend negate_runtime") {
    SUBCASE("16 elements") {
        float a[16], out[16];
        for (int i = 0; i < 16; ++i) {
            a[i] = static_cast<float>(i + 1);
        }
        optinum::simd::backend::negate_runtime<float>(out, a, 16);
        for (int i = 0; i < 16; ++i) {
            CHECK(out[i] == doctest::Approx(-static_cast<float>(i + 1)));
        }
    }
}

TEST_CASE("backend copy_runtime") {
    SUBCASE("64 elements") {
        double src[64], dst[64];
        for (int i = 0; i < 64; ++i) {
            src[i] = static_cast<double>(i * i);
            dst[i] = 0.0;
        }
        optinum::simd::backend::copy_runtime<double>(dst, src, 64);
        for (int i = 0; i < 64; ++i) {
            CHECK(dst[i] == doctest::Approx(src[i]));
        }
    }
}

TEST_CASE("backend fill_runtime") {
    SUBCASE("32 elements") {
        float arr[32];
        optinum::simd::backend::fill_runtime<float>(arr, 32, 42.f);
        for (int i = 0; i < 32; ++i) {
            CHECK(arr[i] == doctest::Approx(42.f));
        }
    }

    SUBCASE("17 elements (non-power-of-2)") {
        double arr[17];
        optinum::simd::backend::fill_runtime<double>(arr, 17, 3.14159);
        for (int i = 0; i < 17; ++i) {
            CHECK(arr[i] == doctest::Approx(3.14159));
        }
    }
}
