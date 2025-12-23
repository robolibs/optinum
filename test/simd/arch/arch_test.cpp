#include <doctest/doctest.h>
#include <optinum/simd/arch/arch.hpp>
#include <optinum/simd/arch/macros.hpp>

using namespace optinum::simd::arch;

TEST_CASE("arch: simd_level returns valid value") {
    constexpr int level = simd_level();
    // Level should be 0, 128, 256, or 512
    CHECK((level == 0 || level == 128 || level == 256 || level == 512));
}

TEST_CASE("arch: simd_width_bytes is consistent with level") {
    constexpr int level = simd_level();
    constexpr int width = simd_width_bytes();

    if (level == 512) {
        CHECK(width == 64);
    } else if (level == 256) {
        CHECK(width == 32);
    } else if (level == 128) {
        CHECK(width == 16);
    } else {
        CHECK(width == 0);
    }
}

TEST_CASE("arch: simd_width<T> returns correct count") {
    constexpr int float_width = simd_width<float>();
    constexpr int double_width = simd_width<double>();

    // float is 4 bytes, double is 8 bytes
    if constexpr (simd_level() == 512) {
        CHECK(float_width == 16);
        CHECK(double_width == 8);
    } else if constexpr (simd_level() == 256) {
        CHECK(float_width == 8);
        CHECK(double_width == 4);
    } else if constexpr (simd_level() == 128) {
        CHECK(float_width == 4);
        CHECK(double_width == 2);
    } else {
        CHECK(float_width == 0);
        CHECK(double_width == 0);
    }
}

TEST_CASE("arch: feature detection functions are consteval") {
    // These should all be compile-time constants
    constexpr bool sse = has_sse();
    constexpr bool sse2 = has_sse2();
    constexpr bool avx = has_avx();
    constexpr bool avx2 = has_avx2();
    constexpr bool avx512 = has_avx512f();
    constexpr bool fma = has_fma();
    constexpr bool neon = has_neon();

    // Just verify they compile as constexpr - values depend on target
    CHECK((sse || !sse)); // tautology to use the value
    CHECK((sse2 || !sse2));
    CHECK((avx || !avx));
    CHECK((avx2 || !avx2));
    CHECK((avx512 || !avx512));
    CHECK((fma || !fma));
    CHECK((neon || !neon));
}

TEST_CASE("arch: SIMD hierarchy is consistent") {
    // If we have AVX512, we should have AVX2, AVX, SSE4.2, etc.
    if constexpr (has_avx512f()) {
        CHECK(has_avx2());
        CHECK(has_avx());
    }

    if constexpr (has_avx2()) {
        CHECK(has_avx());
    }

    if constexpr (has_avx()) {
        CHECK(has_sse42());
        CHECK(has_sse41());
    }

    if constexpr (has_sse42()) {
        CHECK(has_sse41());
    }

    if constexpr (has_sse41()) {
        CHECK(has_ssse3());
    }

    if constexpr (has_ssse3()) {
        CHECK(has_sse3());
    }

    if constexpr (has_sse3()) {
        CHECK(has_sse2());
    }

    if constexpr (has_sse2()) {
        CHECK(has_sse());
    }
}

TEST_CASE("arch: SIMD_WIDTH constants are consistent") {
    CHECK(SIMD_WIDTH_FLOAT == SIMD_WIDTH_BYTES / sizeof(float));
    CHECK(SIMD_WIDTH_DOUBLE == SIMD_WIDTH_BYTES / sizeof(double));
}

TEST_CASE("macros: OPTINUM_SIMD_ALIGNMENT matches simd level") {
    if constexpr (simd_level() == 512) {
        CHECK(OPTINUM_SIMD_ALIGNMENT == 64);
    } else if constexpr (simd_level() == 256) {
        CHECK(OPTINUM_SIMD_ALIGNMENT == 32);
    } else if constexpr (simd_level() == 128) {
        CHECK(OPTINUM_SIMD_ALIGNMENT == 16);
    }
}

// Test that the macros compile correctly
OPTINUM_INLINE int inline_test_func() { return 42; }

TEST_CASE("macros: OPTINUM_INLINE compiles correctly") { CHECK(inline_test_func() == 42); }

TEST_CASE("macros: branch hints compile correctly") {
    int x = 5;
    if (OPTINUM_LIKELY(x > 0)) {
        CHECK(x > 0);
    }
    if (OPTINUM_UNLIKELY(x < 0)) {
        CHECK(false); // Should not reach here
    }
}
