// =============================================================================
// test/simd/pack/neon_test.cpp
// Tests for ARM NEON pack specializations
// =============================================================================

#include <doctest/doctest.h>
#include <optinum/simd/pack/neon.hpp>
#include <optinum/simd/pack/pack.hpp>

namespace on = optinum;

#ifdef OPTINUM_HAS_NEON

// =============================================================================
// pack<float, 4> - ARM NEON Tests
// =============================================================================

TEST_CASE("NEON: pack<float, 4> - Construction") {
    using pack_t = on::simd::pack<float, 4>;

    SUBCASE("Default constructor") {
        pack_t a;
        CHECK(a[0] == doctest::Approx(0.0f));
        CHECK(a[1] == doctest::Approx(0.0f));
        CHECK(a[2] == doctest::Approx(0.0f));
        CHECK(a[3] == doctest::Approx(0.0f));
    }

    SUBCASE("Broadcast constructor") {
        pack_t a(2.5f);
        CHECK(a[0] == doctest::Approx(2.5f));
        CHECK(a[1] == doctest::Approx(2.5f));
        CHECK(a[2] == doctest::Approx(2.5f));
        CHECK(a[3] == doctest::Approx(2.5f));
    }

    SUBCASE("Element-wise constructor") {
        pack_t a(1.0f, 2.0f, 3.0f, 4.0f);
        CHECK(a[0] == doctest::Approx(1.0f));
        CHECK(a[1] == doctest::Approx(2.0f));
        CHECK(a[2] == doctest::Approx(3.0f));
        CHECK(a[3] == doctest::Approx(4.0f));
    }

    SUBCASE("Factory - set") {
        auto a = pack_t::set(1.5f, 2.5f, 3.5f, 4.5f);
        CHECK(a[0] == doctest::Approx(1.5f));
        CHECK(a[1] == doctest::Approx(2.5f));
        CHECK(a[2] == doctest::Approx(3.5f));
        CHECK(a[3] == doctest::Approx(4.5f));
    }

    SUBCASE("Factory - set_sequential") {
        auto a = pack_t::set_sequential(10.0f, 2.0f);
        CHECK(a[0] == doctest::Approx(10.0f));
        CHECK(a[1] == doctest::Approx(12.0f));
        CHECK(a[2] == doctest::Approx(14.0f));
        CHECK(a[3] == doctest::Approx(16.0f));
    }
}

TEST_CASE("NEON: pack<float, 4> - Load/Store") {
    using pack_t = on::simd::pack<float, 4>;

    SUBCASE("Load/Store aligned") {
        alignas(16) float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        auto p = pack_t::load_aligned(data);
        CHECK(p[0] == doctest::Approx(1.0f));
        CHECK(p[1] == doctest::Approx(2.0f));
        CHECK(p[2] == doctest::Approx(3.0f));
        CHECK(p[3] == doctest::Approx(4.0f));

        alignas(16) float output[4];
        p.store_aligned(output);
        for (int i = 0; i < 4; ++i) {
            CHECK(output[i] == doctest::Approx(data[i]));
        }
    }

    SUBCASE("Load/Store unaligned") {
        float data[4] = {5.0f, 6.0f, 7.0f, 8.0f};
        auto p = pack_t::load_unaligned(data);
        CHECK(p[0] == doctest::Approx(5.0f));
        CHECK(p[1] == doctest::Approx(6.0f));
        CHECK(p[2] == doctest::Approx(7.0f));
        CHECK(p[3] == doctest::Approx(8.0f));

        float output[4];
        p.store_unaligned(output);
        for (int i = 0; i < 4; ++i) {
            CHECK(output[i] == doctest::Approx(data[i]));
        }
    }
}

TEST_CASE("NEON: pack<float, 4> - Arithmetic") {
    using pack_t = on::simd::pack<float, 4>;

    SUBCASE("Addition") {
        pack_t a(2.0f, 3.0f, 4.0f, 5.0f);
        pack_t b(1.0f, 2.0f, 3.0f, 4.0f);
        auto c = a + b;
        CHECK(c[0] == doctest::Approx(3.0f));
        CHECK(c[1] == doctest::Approx(5.0f));
        CHECK(c[2] == doctest::Approx(7.0f));
        CHECK(c[3] == doctest::Approx(9.0f));
    }

    SUBCASE("Subtraction") {
        pack_t a(5.0f, 6.0f, 7.0f, 8.0f);
        pack_t b(1.0f, 2.0f, 3.0f, 4.0f);
        auto c = a - b;
        CHECK(c[0] == doctest::Approx(4.0f));
        CHECK(c[1] == doctest::Approx(4.0f));
        CHECK(c[2] == doctest::Approx(4.0f));
        CHECK(c[3] == doctest::Approx(4.0f));
    }

    SUBCASE("Multiplication") {
        pack_t a(2.0f, 3.0f, 4.0f, 5.0f);
        pack_t b(2.0f, 2.0f, 2.0f, 2.0f);
        auto c = a * b;
        CHECK(c[0] == doctest::Approx(4.0f));
        CHECK(c[1] == doctest::Approx(6.0f));
        CHECK(c[2] == doctest::Approx(8.0f));
        CHECK(c[3] == doctest::Approx(10.0f));
    }

    SUBCASE("Division") {
        pack_t a(8.0f, 12.0f, 16.0f, 20.0f);
        pack_t b(2.0f, 3.0f, 4.0f, 5.0f);
        auto c = a / b;
        CHECK(c[0] == doctest::Approx(4.0f));
        CHECK(c[1] == doctest::Approx(4.0f));
        CHECK(c[2] == doctest::Approx(4.0f));
        CHECK(c[3] == doctest::Approx(4.0f));
    }

    SUBCASE("Negation") {
        pack_t a(1.0f, -2.0f, 3.0f, -4.0f);
        auto c = -a;
        CHECK(c[0] == doctest::Approx(-1.0f));
        CHECK(c[1] == doctest::Approx(2.0f));
        CHECK(c[2] == doctest::Approx(-3.0f));
        CHECK(c[3] == doctest::Approx(4.0f));
    }

    SUBCASE("Compound assignment") {
        pack_t a(1.0f, 2.0f, 3.0f, 4.0f);
        pack_t b(5.0f);

        a += b;
        CHECK(a[0] == doctest::Approx(6.0f));

        a -= b;
        CHECK(a[0] == doctest::Approx(1.0f));

        a *= pack_t(2.0f);
        CHECK(a[0] == doctest::Approx(2.0f));

        a /= pack_t(2.0f);
        CHECK(a[0] == doctest::Approx(1.0f));
    }
}

TEST_CASE("NEON: pack<float, 4> - Horizontal Operations") {
    using pack_t = on::simd::pack<float, 4>;

    SUBCASE("Horizontal sum") {
        pack_t a(1.0f, 2.0f, 3.0f, 4.0f);
        CHECK(a.hsum() == doctest::Approx(10.0f));
    }

    SUBCASE("Horizontal min") {
        pack_t a(3.0f, 1.0f, 4.0f, 2.0f);
        CHECK(a.hmin() == doctest::Approx(1.0f));
    }

    SUBCASE("Horizontal max") {
        pack_t a(3.0f, 1.0f, 4.0f, 2.0f);
        CHECK(a.hmax() == doctest::Approx(4.0f));
    }

    SUBCASE("Horizontal product") {
        pack_t a(2.0f, 3.0f, 4.0f, 1.0f);
        CHECK(a.hprod() == doctest::Approx(24.0f));
    }
}

TEST_CASE("NEON: pack<float, 4> - Math Functions") {
    using pack_t = on::simd::pack<float, 4>;

    SUBCASE("Square root") {
        pack_t a(4.0f, 9.0f, 16.0f, 25.0f);
        auto s = a.sqrt();
        CHECK(s[0] == doctest::Approx(2.0f));
        CHECK(s[1] == doctest::Approx(3.0f));
        CHECK(s[2] == doctest::Approx(4.0f));
        CHECK(s[3] == doctest::Approx(5.0f));
    }

    SUBCASE("Reciprocal square root") {
        pack_t a(4.0f);
        auto rs = a.rsqrt();
        CHECK(rs[0] == doctest::Approx(0.5f).epsilon(0.01)); // NEON estimate with refinement
    }

    SUBCASE("Absolute value") {
        pack_t a(-1.0f, 2.0f, -3.0f, 4.0f);
        auto ab = a.abs();
        CHECK(ab[0] == doctest::Approx(1.0f));
        CHECK(ab[1] == doctest::Approx(2.0f));
        CHECK(ab[2] == doctest::Approx(3.0f));
        CHECK(ab[3] == doctest::Approx(4.0f));
    }

    SUBCASE("Reciprocal") {
        pack_t a(2.0f, 4.0f, 5.0f, 10.0f);
        auto r = a.rcp();
        CHECK(r[0] == doctest::Approx(0.5f).epsilon(0.01));
        CHECK(r[1] == doctest::Approx(0.25f).epsilon(0.01));
        CHECK(r[2] == doctest::Approx(0.2f).epsilon(0.01));
        CHECK(r[3] == doctest::Approx(0.1f).epsilon(0.01));
    }

    SUBCASE("Min/Max") {
        pack_t a(1.0f, 5.0f, 3.0f, 7.0f);
        pack_t b(2.0f, 4.0f, 6.0f, 8.0f);

        auto minval = a.min(b);
        CHECK(minval[0] == doctest::Approx(1.0f));
        CHECK(minval[1] == doctest::Approx(4.0f));
        CHECK(minval[2] == doctest::Approx(3.0f));
        CHECK(minval[3] == doctest::Approx(7.0f));

        auto maxval = a.max(b);
        CHECK(maxval[0] == doctest::Approx(2.0f));
        CHECK(maxval[1] == doctest::Approx(5.0f));
        CHECK(maxval[2] == doctest::Approx(6.0f));
        CHECK(maxval[3] == doctest::Approx(8.0f));
    }
}

TEST_CASE("NEON: pack<float, 4> - FMA/FMS") {
    using pack_t = on::simd::pack<float, 4>;

    SUBCASE("Fused multiply-add") {
        pack_t a(2.0f, 3.0f, 4.0f, 5.0f);
        pack_t b(3.0f, 4.0f, 5.0f, 6.0f);
        pack_t c(1.0f, 2.0f, 3.0f, 4.0f);

        auto result = a.fmadd(b, c);                // c + a * b
        CHECK(result[0] == doctest::Approx(7.0f));  // 1 + 2*3 = 7
        CHECK(result[1] == doctest::Approx(14.0f)); // 2 + 3*4 = 14
        CHECK(result[2] == doctest::Approx(23.0f)); // 3 + 4*5 = 23
        CHECK(result[3] == doctest::Approx(34.0f)); // 4 + 5*6 = 34
    }

    SUBCASE("Fused multiply-subtract") {
        pack_t a(2.0f, 3.0f, 4.0f, 5.0f);
        pack_t b(3.0f, 4.0f, 5.0f, 6.0f);
        pack_t c(10.0f, 20.0f, 30.0f, 40.0f);

        auto result = a.fmsub(b, c);                // c - a * b
        CHECK(result[0] == doctest::Approx(4.0f));  // 10 - 2*3 = 4
        CHECK(result[1] == doctest::Approx(8.0f));  // 20 - 3*4 = 8
        CHECK(result[2] == doctest::Approx(10.0f)); // 30 - 4*5 = 10
        CHECK(result[3] == doctest::Approx(10.0f)); // 40 - 5*6 = 10
    }
}

TEST_CASE("NEON: pack<float, 4> - Dot Product") {
    using pack_t = on::simd::pack<float, 4>;

    SUBCASE("Dot product") {
        pack_t a(1.0f, 2.0f, 3.0f, 4.0f);
        pack_t b(5.0f, 6.0f, 7.0f, 8.0f);

        float dot = a.dot(b); // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        CHECK(dot == doctest::Approx(70.0f));
    }
}

TEST_CASE("NEON: pack<float, 4> - Permutations") {
    using pack_t = on::simd::pack<float, 4>;

    SUBCASE("Reverse") {
        pack_t a(1.0f, 2.0f, 3.0f, 4.0f);
        auto r = a.reverse();
        CHECK(r[0] == doctest::Approx(4.0f));
        CHECK(r[1] == doctest::Approx(3.0f));
        CHECK(r[2] == doctest::Approx(2.0f));
        CHECK(r[3] == doctest::Approx(1.0f));
    }

    SUBCASE("Rotate") {
        pack_t a(1.0f, 2.0f, 3.0f, 4.0f);

        auto r1 = a.rotate<1>();
        CHECK(r1[0] == doctest::Approx(2.0f));
        CHECK(r1[1] == doctest::Approx(3.0f));
        CHECK(r1[2] == doctest::Approx(4.0f));
        CHECK(r1[3] == doctest::Approx(1.0f));

        auto r2 = a.rotate<2>();
        CHECK(r2[0] == doctest::Approx(3.0f));
        CHECK(r2[1] == doctest::Approx(4.0f));
        CHECK(r2[2] == doctest::Approx(1.0f));
        CHECK(r2[3] == doctest::Approx(2.0f));
    }

    SUBCASE("Shift") {
        pack_t a(1.0f, 2.0f, 3.0f, 4.0f);

        auto s1 = a.shift<1>();
        CHECK(s1[0] == doctest::Approx(0.0f));
        CHECK(s1[1] == doctest::Approx(1.0f));
        CHECK(s1[2] == doctest::Approx(2.0f));
        CHECK(s1[3] == doctest::Approx(3.0f));
    }
}

TEST_CASE("NEON: pack<float, 4> - Gather/Scatter") {
    using pack_t = on::simd::pack<float, 4>;
    using ipack_t = on::simd::pack<int32_t, 4>;

    SUBCASE("Gather") {
        alignas(16) float data[8] = {10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f};
        ipack_t indices(0, 2, 4, 6);

        auto gathered = pack_t::gather(data, indices);
        CHECK(gathered[0] == doctest::Approx(10.0f));
        CHECK(gathered[1] == doctest::Approx(12.0f));
        CHECK(gathered[2] == doctest::Approx(14.0f));
        CHECK(gathered[3] == doctest::Approx(16.0f));
    }

    SUBCASE("Scatter") {
        alignas(16) float data[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        pack_t values(100.0f, 200.0f, 300.0f, 400.0f);
        ipack_t indices(1, 3, 5, 7);

        values.scatter(data, indices);
        CHECK(data[0] == doctest::Approx(0.0f));
        CHECK(data[1] == doctest::Approx(100.0f));
        CHECK(data[2] == doctest::Approx(0.0f));
        CHECK(data[3] == doctest::Approx(200.0f));
        CHECK(data[4] == doctest::Approx(0.0f));
        CHECK(data[5] == doctest::Approx(300.0f));
        CHECK(data[6] == doctest::Approx(0.0f));
        CHECK(data[7] == doctest::Approx(400.0f));
    }
}

TEST_CASE("NEON: pack<float, 4> - Type Conversion") {
    using fpack_t = on::simd::pack<float, 4>;
    using ipack_t = on::simd::pack<int32_t, 4>;

    SUBCASE("Float to int32") {
        fpack_t a(1.9f, 2.1f, -3.7f, 4.5f);
        auto i = a.cast_to_int();
        CHECK(i[0] == 1);
        CHECK(i[1] == 2);
        CHECK(i[2] == -3);
        CHECK(i[3] == 4);
    }

    SUBCASE("Int32 to float") {
        ipack_t a(10, 20, -30, 40);
        auto f = a.cast_to_float();
        CHECK(f[0] == doctest::Approx(10.0f));
        CHECK(f[1] == doctest::Approx(20.0f));
        CHECK(f[2] == doctest::Approx(-30.0f));
        CHECK(f[3] == doctest::Approx(40.0f));
    }
}

// =============================================================================
// pack<int32_t, 4> - ARM NEON Tests
// =============================================================================

TEST_CASE("NEON: pack<int32_t, 4> - Construction") {
    using pack_t = on::simd::pack<int32_t, 4>;

    SUBCASE("Broadcast constructor") {
        pack_t a(42);
        CHECK(a[0] == 42);
        CHECK(a[1] == 42);
        CHECK(a[2] == 42);
        CHECK(a[3] == 42);
    }

    SUBCASE("Element-wise constructor") {
        pack_t a(10, 20, 30, 40);
        CHECK(a[0] == 10);
        CHECK(a[1] == 20);
        CHECK(a[2] == 30);
        CHECK(a[3] == 40);
    }

    SUBCASE("Factory - set_sequential") {
        auto a = pack_t::set_sequential(100, 10);
        CHECK(a[0] == 100);
        CHECK(a[1] == 110);
        CHECK(a[2] == 120);
        CHECK(a[3] == 130);
    }
}

TEST_CASE("NEON: pack<int32_t, 4> - Arithmetic") {
    using pack_t = on::simd::pack<int32_t, 4>;

    SUBCASE("Addition") {
        pack_t a(10, 20, 30, 40);
        pack_t b(1, 2, 3, 4);
        auto c = a + b;
        CHECK(c[0] == 11);
        CHECK(c[1] == 22);
        CHECK(c[2] == 33);
        CHECK(c[3] == 44);
    }

    SUBCASE("Multiplication") {
        pack_t a(2, 3, 4, 5);
        pack_t b(10, 10, 10, 10);
        auto c = a * b;
        CHECK(c[0] == 20);
        CHECK(c[1] == 30);
        CHECK(c[2] == 40);
        CHECK(c[3] == 50);
    }

    SUBCASE("Negation") {
        pack_t a(1, -2, 3, -4);
        auto c = -a;
        CHECK(c[0] == -1);
        CHECK(c[1] == 2);
        CHECK(c[2] == -3);
        CHECK(c[3] == 4);
    }
}

TEST_CASE("NEON: pack<int32_t, 4> - Bitwise Operations") {
    using pack_t = on::simd::pack<int32_t, 4>;

    SUBCASE("Bitwise AND") {
        pack_t a(0xFF, 0xF0, 0x0F, 0xAA);
        pack_t b(0x0F, 0x0F, 0x0F, 0x55);
        auto c = a & b;
        CHECK(c[0] == 0x0F);
        CHECK(c[1] == 0x00);
        CHECK(c[2] == 0x0F);
        CHECK(c[3] == 0x00);
    }

    SUBCASE("Bitwise OR") {
        pack_t a(0xF0, 0xF0, 0x0F, 0xAA);
        pack_t b(0x0F, 0x0F, 0x0F, 0x55);
        auto c = a | b;
        CHECK(c[0] == 0xFF);
        CHECK(c[1] == 0xFF);
        CHECK(c[2] == 0x0F);
        CHECK(c[3] == 0xFF);
    }

    SUBCASE("Bitwise XOR") {
        pack_t a(0xFF, 0xF0, 0x0F, 0xAA);
        pack_t b(0x0F, 0x0F, 0x0F, 0x55);
        auto c = a ^ b;
        CHECK(c[0] == 0xF0);
        CHECK(c[1] == 0xFF);
        CHECK(c[2] == 0x00);
        CHECK(c[3] == 0xFF);
    }

    SUBCASE("Bitwise NOT") {
        pack_t a(0);
        auto c = ~a;
        CHECK(c[0] == -1);
    }
}

TEST_CASE("NEON: pack<int32_t, 4> - Bit Shifts") {
    using pack_t = on::simd::pack<int32_t, 4>;

    SUBCASE("Left shift") {
        pack_t a(1, 2, 3, 4);
        auto c = a << 2;
        CHECK(c[0] == 4);
        CHECK(c[1] == 8);
        CHECK(c[2] == 12);
        CHECK(c[3] == 16);
    }

    SUBCASE("Arithmetic right shift") {
        pack_t a(-8, -16, 8, 16);
        auto c = a >> 2;
        CHECK(c[0] == -2);
        CHECK(c[1] == -4);
        CHECK(c[2] == 2);
        CHECK(c[3] == 4);
    }

    SUBCASE("Logical right shift") {
        pack_t a(-8, -16, 8, 16);
        auto c = a.shr_logical(2);
        CHECK(c[0] > 0); // Sign bit shifted out
        CHECK(c[1] > 0);
        CHECK(c[2] == 2);
        CHECK(c[3] == 4);
    }
}

TEST_CASE("NEON: pack<int32_t, 4> - Horizontal Operations") {
    using pack_t = on::simd::pack<int32_t, 4>;

    SUBCASE("Horizontal sum") {
        pack_t a(10, 20, 30, 40);
        CHECK(a.hsum() == 100);
    }

    SUBCASE("Horizontal min") {
        pack_t a(30, 10, 40, 20);
        CHECK(a.hmin() == 10);
    }

    SUBCASE("Horizontal max") {
        pack_t a(30, 10, 40, 20);
        CHECK(a.hmax() == 40);
    }

    SUBCASE("Horizontal product") {
        pack_t a(2, 3, 4, 5);
        CHECK(a.hprod() == 120);
    }
}

TEST_CASE("NEON: pack<int32_t, 4> - Math Operations") {
    using pack_t = on::simd::pack<int32_t, 4>;

    SUBCASE("Absolute value") {
        pack_t a(-10, 20, -30, 40);
        auto ab = a.abs();
        CHECK(ab[0] == 10);
        CHECK(ab[1] == 20);
        CHECK(ab[2] == 30);
        CHECK(ab[3] == 40);
    }

    SUBCASE("Min/Max") {
        pack_t a(10, 50, 30, 70);
        pack_t b(20, 40, 60, 80);

        auto minval = a.min(b);
        CHECK(minval[0] == 10);
        CHECK(minval[1] == 40);
        CHECK(minval[2] == 30);
        CHECK(minval[3] == 70);

        auto maxval = a.max(b);
        CHECK(maxval[0] == 20);
        CHECK(maxval[1] == 50);
        CHECK(maxval[2] == 60);
        CHECK(maxval[3] == 80);
    }
}

TEST_CASE("NEON: pack<int32_t, 4> - Dot Product") {
    using pack_t = on::simd::pack<int32_t, 4>;

    SUBCASE("Dot product") {
        pack_t a(1, 2, 3, 4);
        pack_t b(5, 6, 7, 8);

        int dot = a.dot(b); // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        CHECK(dot == 70);
    }
}

TEST_CASE("NEON: pack<int32_t, 4> - Permutations") {
    using pack_t = on::simd::pack<int32_t, 4>;

    SUBCASE("Reverse") {
        pack_t a(10, 20, 30, 40);
        auto r = a.reverse();
        CHECK(r[0] == 40);
        CHECK(r[1] == 30);
        CHECK(r[2] == 20);
        CHECK(r[3] == 10);
    }

    SUBCASE("Rotate") {
        pack_t a(1, 2, 3, 4);

        auto r1 = a.rotate<1>();
        CHECK(r1[0] == 2);
        CHECK(r1[1] == 3);
        CHECK(r1[2] == 4);
        CHECK(r1[3] == 1);
    }
}

// =============================================================================
// pack<double, 2> and pack<int64_t, 2> - ARM64 Only Tests
// =============================================================================

#ifdef __aarch64__

TEST_CASE("NEON: pack<double, 2> - ARM64 Only") {
    using pack_t = on::simd::pack<double, 2>;

    SUBCASE("Construction") {
        pack_t a(3.14);
        CHECK(a[0] == doctest::Approx(3.14));
        CHECK(a[1] == doctest::Approx(3.14));

        pack_t b(1.5, 2.5);
        CHECK(b[0] == doctest::Approx(1.5));
        CHECK(b[1] == doctest::Approx(2.5));
    }

    SUBCASE("Arithmetic") {
        pack_t a(10.0, 20.0);
        pack_t b(2.0, 4.0);

        auto sum = a + b;
        CHECK(sum[0] == doctest::Approx(12.0));
        CHECK(sum[1] == doctest::Approx(24.0));

        auto div = a / b;
        CHECK(div[0] == doctest::Approx(5.0));
        CHECK(div[1] == doctest::Approx(5.0));
    }

    SUBCASE("Horizontal operations") {
        pack_t a(3.5, 6.5);
        CHECK(a.hsum() == doctest::Approx(10.0));
        CHECK(a.hmin() == doctest::Approx(3.5));
        CHECK(a.hmax() == doctest::Approx(6.5));
    }

    SUBCASE("Math functions") {
        pack_t a(9.0, 16.0);
        auto s = a.sqrt();
        CHECK(s[0] == doctest::Approx(3.0));
        CHECK(s[1] == doctest::Approx(4.0));
    }

    SUBCASE("FMA") {
        pack_t a(2.0, 3.0);
        pack_t b(5.0, 7.0);
        pack_t c(1.0, 2.0);

        auto result = a.fmadd(b, c);               // c + a * b
        CHECK(result[0] == doctest::Approx(11.0)); // 1 + 2*5
        CHECK(result[1] == doctest::Approx(23.0)); // 2 + 3*7
    }

    SUBCASE("Reverse") {
        pack_t a(1.5, 2.5);
        auto r = a.reverse();
        CHECK(r[0] == doctest::Approx(2.5));
        CHECK(r[1] == doctest::Approx(1.5));
    }

    SUBCASE("Type conversion") {
        pack_t a(10.7, -20.3);
        auto i = a.cast_to_int();
        CHECK(i[0] == 10);
        CHECK(i[1] == -20);
    }
}

TEST_CASE("NEON: pack<int64_t, 2> - ARM64 Only") {
    using pack_t = on::simd::pack<int64_t, 2>;

    SUBCASE("Construction") {
        pack_t a(42);
        CHECK(a[0] == 42);
        CHECK(a[1] == 42);

        pack_t b(100, 200);
        CHECK(b[0] == 100);
        CHECK(b[1] == 200);
    }

    SUBCASE("Arithmetic") {
        pack_t a(100, 200);
        pack_t b(10, 20);

        auto sum = a + b;
        CHECK(sum[0] == 110);
        CHECK(sum[1] == 220);

        auto prod = a * b;
        CHECK(prod[0] == 1000);
        CHECK(prod[1] == 4000);
    }

    SUBCASE("Bitwise operations") {
        pack_t a(0xFF00, 0x00FF);
        pack_t b(0x0F0F, 0x0F0F);

        auto and_result = a & b;
        CHECK(and_result[0] == 0x0F00);
        CHECK(and_result[1] == 0x000F);

        auto or_result = a | b;
        CHECK(or_result[0] == 0xFF0F);
        CHECK(or_result[1] == 0x0FFF);
    }

    SUBCASE("Bit shifts") {
        pack_t a(8, 16);
        auto left = a << 2;
        CHECK(left[0] == 32);
        CHECK(left[1] == 64);

        pack_t b(-16, 32);
        auto right = b >> 2;
        CHECK(right[0] == -4);
        CHECK(right[1] == 8);
    }

    SUBCASE("Horizontal operations") {
        pack_t a(100, 200);
        CHECK(a.hsum() == 300);

        pack_t b(50, 30);
        CHECK(b.hmin() == 30);
        CHECK(b.hmax() == 50);
    }

    SUBCASE("Math operations") {
        pack_t a(-10, 20);
        auto ab = a.abs();
        CHECK(ab[0] == 10);
        CHECK(ab[1] == 20);

        pack_t b(100, 200);
        pack_t c(150, 50);
        auto minval = b.min(c);
        CHECK(minval[0] == 100);
        CHECK(minval[1] == 50);
    }

    SUBCASE("Reverse") {
        pack_t a(111, 222);
        auto r = a.reverse();
        CHECK(r[0] == 222);
        CHECK(r[1] == 111);
    }

    SUBCASE("Type conversion") {
        pack_t a(100, -200);
        auto f = a.cast_to_float();
        CHECK(f[0] == doctest::Approx(100.0));
        CHECK(f[1] == doctest::Approx(-200.0));
    }
}

#endif // __aarch64__

// =============================================================================
// Element Access Tests
// =============================================================================

TEST_CASE("NEON: pack<float, 4> - Element Access") {
    using pack_t = on::simd::pack<float, 4>;

    SUBCASE("Template get<I>()") {
        pack_t a(10.0f, 20.0f, 30.0f, 40.0f);
        CHECK(a.get<0>() == doctest::Approx(10.0f));
        CHECK(a.get<1>() == doctest::Approx(20.0f));
        CHECK(a.get<2>() == doctest::Approx(30.0f));
        CHECK(a.get<3>() == doctest::Approx(40.0f));
    }

    SUBCASE("Runtime operator[]") {
        pack_t a(1.5f, 2.5f, 3.5f, 4.5f);
        for (int i = 0; i < 4; ++i) {
            CHECK(a[i] == doctest::Approx(1.5f + i));
        }
    }
}

TEST_CASE("NEON: pack<int32_t, 4> - Element Access") {
    using pack_t = on::simd::pack<int32_t, 4>;

    SUBCASE("Template get<I>()") {
        pack_t a(100, 200, 300, 400);
        CHECK(a.get<0>() == 100);
        CHECK(a.get<1>() == 200);
        CHECK(a.get<2>() == 300);
        CHECK(a.get<3>() == 400);
    }
}

#endif // OPTINUM_HAS_NEON
