#include <doctest/doctest.h>
#include <optinum/simd/arch/arch.hpp>
#include <optinum/simd/backend/backend.hpp>
#include <optinum/simd/backend/dot.hpp>
#include <optinum/simd/backend/elementwise.hpp>
#include <optinum/simd/backend/matmul.hpp>

using namespace optinum::simd;

// =============================================================================
// SIMD Platform Verification Tests
// These tests verify that SIMD is actually being used on the platform
// =============================================================================

TEST_CASE("SIMD is enabled on this platform") {
    // On modern x86_64 or ARM64, we expect at least 128-bit SIMD (SSE/NEON)
    constexpr int level = arch::simd_level();

    // This test documents what SIMD level is available
    MESSAGE("SIMD level: ", level, " bits");
    MESSAGE("SIMD width (bytes): ", arch::simd_width_bytes());
    MESSAGE("SIMD width (float): ", arch::simd_width<float>());
    MESSAGE("SIMD width (double): ", arch::simd_width<double>());

    // On any modern platform, we expect at least SSE/NEON (128-bit)
    // This test will fail on platforms without SIMD, which is intentional
    // to alert developers that SIMD optimizations won't be effective
    CHECK(level >= 128);
}

TEST_CASE("SIMD feature detection") {
    // Document which features are available
    MESSAGE("SSE: ", arch::has_sse());
    MESSAGE("SSE2: ", arch::has_sse2());
    MESSAGE("SSE3: ", arch::has_sse3());
    MESSAGE("SSSE3: ", arch::has_ssse3());
    MESSAGE("SSE4.1: ", arch::has_sse41());
    MESSAGE("SSE4.2: ", arch::has_sse42());
    MESSAGE("AVX: ", arch::has_avx());
    MESSAGE("AVX2: ", arch::has_avx2());
    MESSAGE("AVX-512F: ", arch::has_avx512f());
    MESSAGE("FMA: ", arch::has_fma());
    MESSAGE("NEON: ", arch::has_neon());
    MESSAGE("SVE: ", arch::has_sve());

    // At least one SIMD instruction set should be available
    bool has_simd = arch::has_sse() || arch::has_neon() || arch::has_sve();
    CHECK(has_simd);
}

TEST_CASE("Pack width matches architecture") {
    // Verify pack width is consistent with SIMD level
    constexpr int level = arch::simd_level();

    if constexpr (level >= 512) {
        CHECK(arch::simd_width<float>() == 16);
        CHECK(arch::simd_width<double>() == 8);
    } else if constexpr (level >= 256) {
        CHECK(arch::simd_width<float>() == 8);
        CHECK(arch::simd_width<double>() == 4);
    } else if constexpr (level >= 128) {
        CHECK(arch::simd_width<float>() == 4);
        CHECK(arch::simd_width<double>() == 2);
    }
}

// =============================================================================
// Tests that verify SIMD code paths are exercised
// These use array sizes that require SIMD processing
// =============================================================================

TEST_CASE("Large array operations use SIMD paths") {
    // Use 256 elements - large enough to exercise SIMD on any platform
    constexpr std::size_t N = 256;

    float a[N], b[N], out[N];
    for (std::size_t i = 0; i < N; ++i) {
        a[i] = static_cast<float>(i + 1);
        b[i] = static_cast<float>(N - i);
    }

    SUBCASE("add_runtime with 256 elements") {
        backend::add_runtime<float>(out, a, b, N);
        for (std::size_t i = 0; i < N; ++i) {
            CHECK(out[i] == doctest::Approx(static_cast<float>(N + 1)));
        }
    }

    SUBCASE("mul_runtime with 256 elements") {
        backend::mul_runtime<float>(out, a, b, N);
        for (std::size_t i = 0; i < N; ++i) {
            CHECK(out[i] == doctest::Approx(a[i] * b[i]));
        }
    }

    SUBCASE("dot_runtime with 256 elements") {
        float expected = 0.f;
        for (std::size_t i = 0; i < N; ++i) {
            expected += a[i] * b[i];
        }
        float result = backend::dot_runtime<float>(a, b, N);
        CHECK(result == doctest::Approx(expected).epsilon(1e-4));
    }
}

TEST_CASE("Double precision large array operations") {
    constexpr std::size_t N = 128;

    double a[N], b[N], out[N];
    for (std::size_t i = 0; i < N; ++i) {
        a[i] = static_cast<double>(i) * 0.01;
        b[i] = static_cast<double>(N - i) * 0.01;
    }

    SUBCASE("add_runtime with 128 double elements") {
        backend::add_runtime<double>(out, a, b, N);
        for (std::size_t i = 0; i < N; ++i) {
            CHECK(out[i] == doctest::Approx(a[i] + b[i]));
        }
    }

    SUBCASE("dot_runtime with 128 double elements") {
        double expected = 0.0;
        for (std::size_t i = 0; i < N; ++i) {
            expected += a[i] * b[i];
        }
        double result = backend::dot_runtime<double>(a, b, N);
        CHECK(result == doctest::Approx(expected));
    }
}

TEST_CASE("Matrix operations use SIMD paths") {
    SUBCASE("16x16 matmul") {
        float A[256], B[256], C[256];

        // Initialize A as identity
        for (int i = 0; i < 256; ++i)
            A[i] = 0.f;
        for (int i = 0; i < 16; ++i)
            A[i * 16 + i] = 1.f;

        // Initialize B with sequential values
        for (int i = 0; i < 256; ++i)
            B[i] = static_cast<float>(i + 1);

        backend::matmul<float, 16, 16, 16>(C, A, B);

        // C should equal B (identity * B = B)
        for (int i = 0; i < 256; ++i) {
            CHECK(C[i] == doctest::Approx(B[i]));
        }
    }

    SUBCASE("32x32 matvec") {
        float M[1024], x[32], y[32];

        // M = identity
        for (int i = 0; i < 1024; ++i)
            M[i] = 0.f;
        for (int i = 0; i < 32; ++i)
            M[i * 32 + i] = 1.f;

        for (int i = 0; i < 32; ++i)
            x[i] = static_cast<float>(i + 1);

        backend::matvec<float, 32, 32>(y, M, x);

        // y should equal x
        for (int i = 0; i < 32; ++i) {
            CHECK(y[i] == doctest::Approx(x[i]));
        }
    }
}

TEST_CASE("Optimizer-specific SIMD utilities") {
    constexpr std::size_t N = 64;

    SUBCASE("axpy_runtime - y = x + alpha * d") {
        float x[N], d[N], y[N];
        for (std::size_t i = 0; i < N; ++i) {
            x[i] = static_cast<float>(i);
            d[i] = 1.f;
        }

        backend::axpy_runtime<float>(y, x, 10.f, d, N);

        for (std::size_t i = 0; i < N; ++i) {
            CHECK(y[i] == doctest::Approx(static_cast<float>(i) + 10.f));
        }
    }

    SUBCASE("scale_sub_runtime - x -= alpha * g (gradient descent)") {
        double x[N], g[N];
        for (std::size_t i = 0; i < N; ++i) {
            x[i] = 100.0;
            g[i] = static_cast<double>(i + 1);
        }

        backend::scale_sub_runtime<double>(x, 0.1, g, N);

        for (std::size_t i = 0; i < N; ++i) {
            CHECK(x[i] == doctest::Approx(100.0 - 0.1 * (i + 1)));
        }
    }

    SUBCASE("axpy_inplace_runtime - x += alpha * d") {
        float x[N], d[N];
        for (std::size_t i = 0; i < N; ++i) {
            x[i] = static_cast<float>(i);
            d[i] = 2.f;
        }

        backend::axpy_inplace_runtime<float>(x, 5.f, d, N);

        for (std::size_t i = 0; i < N; ++i) {
            CHECK(x[i] == doctest::Approx(static_cast<float>(i) + 10.f));
        }
    }
}

TEST_CASE("Edge cases - arrays smaller than SIMD width") {
    // These should still work correctly via scalar fallback

    SUBCASE("3 elements (smaller than any SIMD width)") {
        float a[3] = {1.f, 2.f, 3.f};
        float b[3] = {4.f, 5.f, 6.f};
        float out[3];

        backend::add_runtime<float>(out, a, b, 3);
        CHECK(out[0] == doctest::Approx(5.f));
        CHECK(out[1] == doctest::Approx(7.f));
        CHECK(out[2] == doctest::Approx(9.f));
    }

    SUBCASE("1 element") {
        double a[1] = {42.0};
        double b[1] = {8.0};
        double out[1];

        backend::mul_runtime<double>(out, a, b, 1);
        CHECK(out[0] == doctest::Approx(336.0));
    }
}
