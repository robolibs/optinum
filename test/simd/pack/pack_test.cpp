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

// =============================================================================
// Pack Utility Functions Tests
// =============================================================================

TEST_CASE("pack utilities - get<I>(), set(), set_sequential(), reverse()") {
    using namespace on::simd;

    SUBCASE("pack<float, 4> - SSE utilities") {
        // Test set()
        auto p1 = pack<float, 4>::set(1.0f, 2.0f, 3.0f, 4.0f);
        CHECK(p1[0] == doctest::Approx(1.0f));
        CHECK(p1[1] == doctest::Approx(2.0f));
        CHECK(p1[2] == doctest::Approx(3.0f));
        CHECK(p1[3] == doctest::Approx(4.0f));

        // Test get<I>() - compile-time lane extraction
        CHECK(get<0>(p1) == doctest::Approx(1.0f));
        CHECK(get<1>(p1) == doctest::Approx(2.0f));
        CHECK(get<2>(p1) == doctest::Approx(3.0f));
        CHECK(get<3>(p1) == doctest::Approx(4.0f));

        // Test set_sequential()
        auto p2 = pack<float, 4>::set_sequential(0.0f);
        CHECK(p2[0] == doctest::Approx(0.0f));
        CHECK(p2[1] == doctest::Approx(1.0f));
        CHECK(p2[2] == doctest::Approx(2.0f));
        CHECK(p2[3] == doctest::Approx(3.0f));

        // Test set_sequential() with custom step
        auto p3 = pack<float, 4>::set_sequential(10.0f, 5.0f);
        CHECK(p3[0] == doctest::Approx(10.0f));
        CHECK(p3[1] == doctest::Approx(15.0f));
        CHECK(p3[2] == doctest::Approx(20.0f));
        CHECK(p3[3] == doctest::Approx(25.0f));

        // Test reverse()
        auto p4 = p1.reverse();
        CHECK(p4[0] == doctest::Approx(4.0f));
        CHECK(p4[1] == doctest::Approx(3.0f));
        CHECK(p4[2] == doctest::Approx(2.0f));
        CHECK(p4[3] == doctest::Approx(1.0f));
    }

    SUBCASE("pack<double, 2> - SSE utilities") {
        // Test set()
        auto p1 = pack<double, 2>::set(1.5, 2.5);
        CHECK(p1[0] == doctest::Approx(1.5));
        CHECK(p1[1] == doctest::Approx(2.5));

        // Test get<I>()
        CHECK(get<0>(p1) == doctest::Approx(1.5));
        CHECK(get<1>(p1) == doctest::Approx(2.5));

        // Test set_sequential()
        auto p2 = pack<double, 2>::set_sequential(0.0);
        CHECK(p2[0] == doctest::Approx(0.0));
        CHECK(p2[1] == doctest::Approx(1.0));

        // Test set_sequential() with step
        auto p3 = pack<double, 2>::set_sequential(100.0, 50.0);
        CHECK(p3[0] == doctest::Approx(100.0));
        CHECK(p3[1] == doctest::Approx(150.0));

        // Test reverse()
        auto p4 = p1.reverse();
        CHECK(p4[0] == doctest::Approx(2.5));
        CHECK(p4[1] == doctest::Approx(1.5));
    }

#ifdef OPTINUM_HAS_AVX
    SUBCASE("pack<float, 8> - AVX utilities") {
        // Test set()
        auto p1 = pack<float, 8>::set(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f);
        for (int i = 0; i < 8; ++i) {
            CHECK(p1[i] == doctest::Approx(static_cast<float>(i + 1)));
        }

        // Test get<I>()
        CHECK(get<0>(p1) == doctest::Approx(1.0f));
        CHECK(get<1>(p1) == doctest::Approx(2.0f));
        CHECK(get<2>(p1) == doctest::Approx(3.0f));
        CHECK(get<3>(p1) == doctest::Approx(4.0f));
        CHECK(get<4>(p1) == doctest::Approx(5.0f));
        CHECK(get<5>(p1) == doctest::Approx(6.0f));
        CHECK(get<6>(p1) == doctest::Approx(7.0f));
        CHECK(get<7>(p1) == doctest::Approx(8.0f));

        // Test set_sequential()
        auto p2 = pack<float, 8>::set_sequential(0.0f);
        for (int i = 0; i < 8; ++i) {
            CHECK(p2[i] == doctest::Approx(static_cast<float>(i)));
        }

        // Test set_sequential() with step
        auto p3 = pack<float, 8>::set_sequential(10.0f, 10.0f);
        for (int i = 0; i < 8; ++i) {
            CHECK(p3[i] == doctest::Approx(10.0f + 10.0f * i));
        }

        // Test reverse()
        auto p4 = p1.reverse();
        for (int i = 0; i < 8; ++i) {
            CHECK(p4[i] == doctest::Approx(static_cast<float>(8 - i)));
        }
    }

    SUBCASE("pack<double, 4> - AVX utilities") {
        // Test set()
        auto p1 = pack<double, 4>::set(1.5, 2.5, 3.5, 4.5);
        CHECK(p1[0] == doctest::Approx(1.5));
        CHECK(p1[1] == doctest::Approx(2.5));
        CHECK(p1[2] == doctest::Approx(3.5));
        CHECK(p1[3] == doctest::Approx(4.5));

        // Test get<I>()
        CHECK(get<0>(p1) == doctest::Approx(1.5));
        CHECK(get<1>(p1) == doctest::Approx(2.5));
        CHECK(get<2>(p1) == doctest::Approx(3.5));
        CHECK(get<3>(p1) == doctest::Approx(4.5));

        // Test set_sequential()
        auto p2 = pack<double, 4>::set_sequential(0.0);
        for (int i = 0; i < 4; ++i) {
            CHECK(p2[i] == doctest::Approx(static_cast<double>(i)));
        }

        // Test set_sequential() with step
        auto p3 = pack<double, 4>::set_sequential(100.0, 25.0);
        CHECK(p3[0] == doctest::Approx(100.0));
        CHECK(p3[1] == doctest::Approx(125.0));
        CHECK(p3[2] == doctest::Approx(150.0));
        CHECK(p3[3] == doctest::Approx(175.0));

        // Test reverse()
        auto p4 = p1.reverse();
        CHECK(p4[0] == doctest::Approx(4.5));
        CHECK(p4[1] == doctest::Approx(3.5));
        CHECK(p4[2] == doctest::Approx(2.5));
        CHECK(p4[3] == doctest::Approx(1.5));
    }
#endif // OPTINUM_HAS_AVX
}

// =============================================================================
// Pack Advanced Utilities Tests (Tier 2, 3, 4)
// =============================================================================

TEST_CASE("pack advanced utilities - rotate, shift, cast, gather, scatter") {
    using namespace on::simd;

    SUBCASE("pack<float, 4> - rotate/shift") {
        auto p = pack<float, 4>::set(1.0f, 2.0f, 3.0f, 4.0f);

        // Test rotate
        auto r1 = p.template rotate<1>();
        CHECK(r1[0] == doctest::Approx(2.0f));
        CHECK(r1[1] == doctest::Approx(3.0f));
        CHECK(r1[2] == doctest::Approx(4.0f));
        CHECK(r1[3] == doctest::Approx(1.0f));

        auto r2 = p.template rotate<2>();
        CHECK(r2[0] == doctest::Approx(3.0f));
        CHECK(r2[3] == doctest::Approx(2.0f));

        auto r_1 = p.template rotate<-1>(); // Same as rotate<3>
        CHECK(r_1[0] == doctest::Approx(4.0f));
        CHECK(r_1[3] == doctest::Approx(3.0f));

        // Test shift
        auto s1 = p.template shift<1>();
        CHECK(s1[0] == doctest::Approx(2.0f));
        CHECK(s1[1] == doctest::Approx(3.0f));
        CHECK(s1[2] == doctest::Approx(4.0f));
        CHECK(s1[3] == doctest::Approx(0.0f)); // Filled with zero

        auto s_1 = p.template shift<-1>();
        CHECK(s_1[0] == doctest::Approx(0.0f)); // Filled with zero
        CHECK(s_1[1] == doctest::Approx(1.0f));
    }

    SUBCASE("pack<float, 4> - gather/scatter") {
        alignas(16) float data[10] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f, 90.0f, 100.0f};
        alignas(16) int32_t indices[4] = {1, 3, 5, 7};

        // Test gather
        auto p = pack<float, 4>::gather(data, indices);
        CHECK(p[0] == doctest::Approx(20.0f)); // data[1]
        CHECK(p[1] == doctest::Approx(40.0f)); // data[3]
        CHECK(p[2] == doctest::Approx(60.0f)); // data[5]
        CHECK(p[3] == doctest::Approx(80.0f)); // data[7]

        // Test scatter
        alignas(16) float output[10] = {0};
        auto values = pack<float, 4>::set(111.0f, 222.0f, 333.0f, 444.0f);
        values.scatter(output, indices);
        CHECK(output[1] == doctest::Approx(111.0f));
        CHECK(output[3] == doctest::Approx(222.0f));
        CHECK(output[5] == doctest::Approx(333.0f));
        CHECK(output[7] == doctest::Approx(444.0f));
        CHECK(output[0] == doctest::Approx(0.0f)); // Untouched
    }

#ifdef OPTINUM_HAS_AVX
    SUBCASE("pack<float, 8> - rotate/shift") {
        auto p = pack<float, 8>::set_sequential(0.0f); // {0,1,2,3,4,5,6,7}

        // Test rotate
        auto r1 = p.template rotate<1>();
        for (int i = 0; i < 8; ++i) {
            CHECK(r1[i] == doctest::Approx(static_cast<float>((i + 1) % 8)));
        }

        auto r2 = p.template rotate<2>();
        for (int i = 0; i < 8; ++i) {
            CHECK(r2[i] == doctest::Approx(static_cast<float>((i + 2) % 8)));
        }

        auto r4 = p.template rotate<4>();
        CHECK(r4[0] == doctest::Approx(4.0f));
        CHECK(r4[4] == doctest::Approx(0.0f));

        // Test shift
        auto s1 = p.template shift<1>();
        CHECK(s1[0] == doctest::Approx(1.0f));
        CHECK(s1[6] == doctest::Approx(7.0f));
        CHECK(s1[7] == doctest::Approx(0.0f)); // Zero

        auto s_2 = p.template shift<-2>();
        CHECK(s_2[0] == doctest::Approx(0.0f)); // Zero
        CHECK(s_2[1] == doctest::Approx(0.0f)); // Zero
        CHECK(s_2[2] == doctest::Approx(0.0f));
    }

    SUBCASE("pack<float, 8> - gather/scatter") {
        alignas(32) float data[20];
        for (int i = 0; i < 20; ++i)
            data[i] = static_cast<float>(i * 10);

        alignas(32) int32_t indices[8] = {1, 3, 5, 7, 9, 11, 13, 15};

        // Test gather
        auto p = pack<float, 8>::gather(data, indices);
        for (int i = 0; i < 8; ++i) {
            CHECK(p[i] == doctest::Approx(static_cast<float>(indices[i] * 10)));
        }

        // Test scatter
        alignas(32) float output[20] = {0};
        auto values = pack<float, 8>::set_sequential(100.0f, 100.0f);
        values.scatter(output, indices);
        CHECK(output[1] == doctest::Approx(100.0f));
        CHECK(output[15] == doctest::Approx(800.0f));
        CHECK(output[0] == doctest::Approx(0.0f)); // Untouched
    }

    SUBCASE("pack<double, 4> - rotate/shift/gather/scatter") {
        auto p = pack<double, 4>::set(1.5, 2.5, 3.5, 4.5);

        // Test rotate
        auto r2 = p.template rotate<2>();
        CHECK(r2[0] == doctest::Approx(3.5));
        CHECK(r2[1] == doctest::Approx(4.5));
        CHECK(r2[2] == doctest::Approx(1.5));
        CHECK(r2[3] == doctest::Approx(2.5));

        // Test shift
        auto s1 = p.template shift<1>();
        CHECK(s1[0] == doctest::Approx(2.5));
        CHECK(s1[3] == doctest::Approx(0.0));

        // Test gather
        alignas(32) double data[10];
        for (int i = 0; i < 10; ++i)
            data[i] = static_cast<double>(i * 100);
        alignas(32) int64_t indices[4] = {2, 4, 6, 8};

        auto gathered = pack<double, 4>::gather(data, indices);
        CHECK(gathered[0] == doctest::Approx(200.0));
        CHECK(gathered[3] == doctest::Approx(800.0));

        // Test scatter
        alignas(32) double output[10] = {0};
        gathered.scatter(output, indices);
        CHECK(output[2] == doctest::Approx(200.0));
        CHECK(output[8] == doctest::Approx(800.0));
    }
#endif // OPTINUM_HAS_AVX
}
