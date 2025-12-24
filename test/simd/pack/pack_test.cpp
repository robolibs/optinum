// =============================================================================
// test/simd/pack/pack_test.cpp
// Tests for pack<T,W> SIMD register abstraction
// =============================================================================

#include <doctest/doctest.h>
#include <optinum/simd/pack/avx.hpp>
#include <optinum/simd/pack/pack.hpp>
#include <optinum/simd/pack/sse.hpp>

namespace on = optinum;

// =============================================================================
// Scalar Fallback Tests
// =============================================================================

TEST_CASE("pack<float, 2> - Scalar fallback") {
    using pack_t = on::simd::pack<float, 2>;

    SUBCASE("Construction") {
        pack_t a;        // default
        pack_t b(3.14f); // broadcast
        CHECK(b[0] == doctest::Approx(3.14f));
        CHECK(b[1] == doctest::Approx(3.14f));
    }

    SUBCASE("Load/Store") {
        alignas(16) float data[2] = {1.0f, 2.0f};
        auto p = pack_t::load(data);
        CHECK(p[0] == doctest::Approx(1.0f));
        CHECK(p[1] == doctest::Approx(2.0f));

        alignas(16) float output[2];
        p.store(output);
        CHECK(output[0] == doctest::Approx(1.0f));
        CHECK(output[1] == doctest::Approx(2.0f));
    }

    SUBCASE("Arithmetic") {
        pack_t a(2.0f);
        pack_t b(3.0f);
        auto c = a + b;
        CHECK(c[0] == doctest::Approx(5.0f));
        CHECK(c[1] == doctest::Approx(5.0f));

        auto d = a * b;
        CHECK(d[0] == doctest::Approx(6.0f));
        CHECK(d[1] == doctest::Approx(6.0f));
    }

    SUBCASE("Reductions") {
        alignas(16) float data[2] = {1.0f, 2.0f};
        auto p = pack_t::load(data);
        CHECK(p.hsum() == doctest::Approx(3.0f));
        CHECK(p.hmin() == doctest::Approx(1.0f));
        CHECK(p.hmax() == doctest::Approx(2.0f));
        CHECK(p.hprod() == doctest::Approx(2.0f));
    }
}

// =============================================================================
// SSE Tests
// =============================================================================

#ifdef OPTINUM_HAS_SSE2

TEST_CASE("pack<float, 4> - SSE") {
    using pack_t = on::simd::pack<float, 4>;

    SUBCASE("Construction and broadcast") {
        pack_t a(2.5f);
        CHECK(a[0] == doctest::Approx(2.5f));
        CHECK(a[1] == doctest::Approx(2.5f));
        CHECK(a[2] == doctest::Approx(2.5f));
        CHECK(a[3] == doctest::Approx(2.5f));
    }

    SUBCASE("Load/Store aligned") {
        alignas(16) float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        auto p = pack_t::load(data);
        CHECK(p[0] == doctest::Approx(1.0f));
        CHECK(p[1] == doctest::Approx(2.0f));
        CHECK(p[2] == doctest::Approx(3.0f));
        CHECK(p[3] == doctest::Approx(4.0f));

        alignas(16) float output[4];
        p.store(output);
        for (int i = 0; i < 4; ++i) {
            CHECK(output[i] == doctest::Approx(data[i]));
        }
    }

    SUBCASE("Arithmetic operations") {
        pack_t a(2.0f);
        pack_t b(3.0f);

        auto sum = a + b;
        CHECK(sum[0] == doctest::Approx(5.0f));

        auto diff = a - b;
        CHECK(diff[0] == doctest::Approx(-1.0f));

        auto prod = a * b;
        CHECK(prod[0] == doctest::Approx(6.0f));

        auto quot = a / b;
        CHECK(quot[0] == doctest::Approx(2.0f / 3.0f));
    }

    SUBCASE("FMA operations") {
        pack_t a(2.0f);
        pack_t b(3.0f);
        pack_t c(5.0f);

        auto result = pack_t::fma(a, b, c); // 2*3 + 5 = 11
        CHECK(result[0] == doctest::Approx(11.0f));

        auto result2 = pack_t::fms(a, b, c); // 2*3 - 5 = 1
        CHECK(result2[0] == doctest::Approx(1.0f));
    }

    SUBCASE("Horizontal reductions") {
        alignas(16) float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        auto p = pack_t::load(data);

        CHECK(p.hsum() == doctest::Approx(10.0f));
        CHECK(p.hmin() == doctest::Approx(1.0f));
        CHECK(p.hmax() == doctest::Approx(4.0f));
        CHECK(p.hprod() == doctest::Approx(24.0f));
    }

    SUBCASE("Math functions") {
        pack_t a(4.0f);
        auto s = a.sqrt();
        CHECK(s[0] == doctest::Approx(2.0f));

        auto ab = a.abs();
        CHECK(ab[0] == doctest::Approx(4.0f));
    }

    SUBCASE("Min/Max") {
        pack_t a(2.0f);
        pack_t b(3.0f);

        auto minval = pack_t::min(a, b);
        CHECK(minval[0] == doctest::Approx(2.0f));

        auto maxval = pack_t::max(a, b);
        CHECK(maxval[0] == doctest::Approx(3.0f));
    }

    SUBCASE("Dot product") {
        alignas(16) float data1[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        alignas(16) float data2[4] = {2.0f, 3.0f, 4.0f, 5.0f};
        auto p1 = pack_t::load(data1);
        auto p2 = pack_t::load(data2);

        // 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
        CHECK(p1.dot(p2) == doctest::Approx(40.0f));
    }
}

TEST_CASE("pack<double, 2> - SSE2") {
    using pack_t = on::simd::pack<double, 2>;

    SUBCASE("Construction") {
        pack_t a(3.14);
        CHECK(a[0] == doctest::Approx(3.14));
        CHECK(a[1] == doctest::Approx(3.14));
    }

    SUBCASE("Arithmetic") {
        pack_t a(2.0);
        pack_t b(3.0);
        auto c = a + b;
        CHECK(c[0] == doctest::Approx(5.0));
    }

    SUBCASE("Reductions") {
        alignas(16) double data[2] = {1.5, 2.5};
        auto p = pack_t::load(data);
        CHECK(p.hsum() == doctest::Approx(4.0));
        CHECK(p.hmin() == doctest::Approx(1.5));
        CHECK(p.hmax() == doctest::Approx(2.5));
    }
}

TEST_CASE("pack<int32_t, 4> - SSE2 integers") {
    using pack_t = on::simd::pack<int32_t, 4>;

    SUBCASE("Construction") {
        pack_t a(42);
        CHECK(a[0] == 42);
        CHECK(a[1] == 42);
        CHECK(a[2] == 42);
        CHECK(a[3] == 42);
    }

    SUBCASE("Arithmetic") {
        pack_t a(10);
        pack_t b(3);

        auto sum = a + b;
        CHECK(sum[0] == 13);

        auto diff = a - b;
        CHECK(diff[0] == 7);

        auto prod = a * b;
        CHECK(prod[0] == 30);
    }

    SUBCASE("Bitwise operations") {
        pack_t a(0xF0);
        pack_t b(0x0F);

        auto and_result = a & b;
        CHECK(and_result[0] == 0x00);

        auto or_result = a | b;
        CHECK(or_result[0] == 0xFF);

        auto xor_result = a ^ b;
        CHECK(xor_result[0] == 0xFF);
    }

    SUBCASE("Shifts") {
        pack_t a(8);
        auto left = a << 2;
        CHECK(left[0] == 32);

        auto right = a >> 2;
        CHECK(right[0] == 2);
    }

    SUBCASE("Reductions") {
        alignas(16) int32_t data[4] = {1, 2, 3, 4};
        auto p = pack_t::load(data);

        CHECK(p.hsum() == 10);
        CHECK(p.hmin() == 1);
        CHECK(p.hmax() == 4);
        CHECK(p.hprod() == 24);
    }

    SUBCASE("Abs") {
        alignas(16) int32_t data[4] = {-1, 2, -3, 4};
        auto p = pack_t::load(data);
        auto abs_p = p.abs();

        CHECK(abs_p[0] == 1);
        CHECK(abs_p[1] == 2);
        CHECK(abs_p[2] == 3);
        CHECK(abs_p[3] == 4);
    }
}

TEST_CASE("pack<int64_t, 2> - SSE2 integers") {
    using pack_t = on::simd::pack<int64_t, 2>;

    SUBCASE("Construction") {
        pack_t a(123LL);
        CHECK(a[0] == 123LL);
        CHECK(a[1] == 123LL);
    }

    SUBCASE("Arithmetic") {
        pack_t a(10LL);
        pack_t b(3LL);

        auto sum = a + b;
        CHECK(sum[0] == 13LL);

        auto diff = a - b;
        CHECK(diff[0] == 7LL);

        auto prod = a * b;
        CHECK(prod[0] == 30LL);
    }

    SUBCASE("Reductions") {
        alignas(16) int64_t data[2] = {100LL, 200LL};
        auto p = pack_t::load(data);

        CHECK(p.hsum() == 300LL);
        CHECK(p.hmin() == 100LL);
        CHECK(p.hmax() == 200LL);
        CHECK(p.hprod() == 20000LL);
    }
}

#endif // OPTINUM_HAS_SSE2

// =============================================================================
// AVX Tests
// =============================================================================

#ifdef OPTINUM_HAS_AVX

TEST_CASE("pack<float, 8> - AVX") {
    using pack_t = on::simd::pack<float, 8>;

    SUBCASE("Construction") {
        pack_t a(2.5f);
        for (int i = 0; i < 8; ++i) {
            CHECK(a[i] == doctest::Approx(2.5f));
        }
    }

    SUBCASE("Load/Store") {
        alignas(32) float data[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        auto p = pack_t::load(data);

        for (int i = 0; i < 8; ++i) {
            CHECK(p[i] == doctest::Approx(data[i]));
        }

        alignas(32) float output[8];
        p.store(output);
        for (int i = 0; i < 8; ++i) {
            CHECK(output[i] == doctest::Approx(data[i]));
        }
    }

    SUBCASE("Arithmetic") {
        pack_t a(2.0f);
        pack_t b(3.0f);

        auto sum = a + b;
        CHECK(sum[0] == doctest::Approx(5.0f));

        auto prod = a * b;
        CHECK(prod[0] == doctest::Approx(6.0f));
    }

    SUBCASE("Horizontal reductions") {
        alignas(32) float data[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        auto p = pack_t::load(data);

        CHECK(p.hsum() == doctest::Approx(36.0f)); // 1+2+3+4+5+6+7+8 = 36
        CHECK(p.hmin() == doctest::Approx(1.0f));
        CHECK(p.hmax() == doctest::Approx(8.0f));
    }
}

TEST_CASE("pack<double, 4> - AVX") {
    using pack_t = on::simd::pack<double, 4>;

    SUBCASE("Construction") {
        pack_t a(3.14);
        for (int i = 0; i < 4; ++i) {
            CHECK(a[i] == doctest::Approx(3.14));
        }
    }

    SUBCASE("Arithmetic") {
        pack_t a(2.0);
        pack_t b(3.0);
        auto c = a + b;
        CHECK(c[0] == doctest::Approx(5.0));
    }

    SUBCASE("Reductions") {
        alignas(32) double data[4] = {1.0, 2.0, 3.0, 4.0};
        auto p = pack_t::load(data);
        CHECK(p.hsum() == doctest::Approx(10.0));
        CHECK(p.hmin() == doctest::Approx(1.0));
        CHECK(p.hmax() == doctest::Approx(4.0));
    }
}

#endif // OPTINUM_HAS_AVX

#ifdef OPTINUM_HAS_AVX2

TEST_CASE("pack<int32_t, 8> - AVX2") {
    using pack_t = on::simd::pack<int32_t, 8>;

    SUBCASE("Construction") {
        pack_t a(42);
        for (int i = 0; i < 8; ++i) {
            CHECK(a[i] == 42);
        }
    }

    SUBCASE("Arithmetic") {
        pack_t a(10);
        pack_t b(3);

        auto sum = a + b;
        CHECK(sum[0] == 13);

        auto prod = a * b;
        CHECK(prod[0] == 30);
    }

    SUBCASE("Bitwise operations") {
        pack_t a(0xF0);
        pack_t b(0x0F);

        auto and_result = a & b;
        CHECK(and_result[0] == 0x00);

        auto or_result = a | b;
        CHECK(or_result[0] == 0xFF);
    }

    SUBCASE("Shifts") {
        pack_t a(8);
        auto left = a << 2;
        CHECK(left[0] == 32);

        auto right = a >> 2;
        CHECK(right[0] == 2);
    }

    SUBCASE("Reductions") {
        alignas(32) int32_t data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        auto p = pack_t::load(data);

        CHECK(p.hsum() == 36); // 1+2+3+4+5+6+7+8 = 36
        CHECK(p.hmin() == 1);
        CHECK(p.hmax() == 8);
    }

    SUBCASE("Abs") {
        alignas(32) int32_t data[8] = {-1, 2, -3, 4, -5, 6, -7, 8};
        auto p = pack_t::load(data);
        auto abs_p = p.abs();

        for (int i = 0; i < 8; ++i) {
            CHECK(abs_p[i] == (i + 1));
        }
    }
}

TEST_CASE("pack<int64_t, 4> - AVX2") {
    using pack_t = on::simd::pack<int64_t, 4>;

    SUBCASE("Construction") {
        pack_t a(123LL);
        for (int i = 0; i < 4; ++i) {
            CHECK(a[i] == 123LL);
        }
    }

    SUBCASE("Arithmetic") {
        pack_t a(10LL);
        pack_t b(3LL);

        auto sum = a + b;
        CHECK(sum[0] == 13LL);

        auto prod = a * b;
        CHECK(prod[0] == 30LL);
    }

    SUBCASE("Reductions") {
        alignas(32) int64_t data[4] = {10LL, 20LL, 30LL, 40LL};
        auto p = pack_t::load(data);

        CHECK(p.hsum() == 100LL);
        CHECK(p.hmin() == 10LL);
        CHECK(p.hmax() == 40LL);
        CHECK(p.hprod() == 240000LL); // 10*20*30*40 = 240000
    }
}

#endif // OPTINUM_HAS_AVX2
