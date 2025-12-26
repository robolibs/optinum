// =============================================================================
// test/simd/math/special_functions_test.cpp
// Tests for special mathematical functions (erf, tgamma, lgamma)
// =============================================================================

#include <cmath>
#include <doctest/doctest.h>
#include <optinum/simd/math/erf.hpp>
#include <optinum/simd/math/lgamma.hpp>
#include <optinum/simd/math/simd_math.hpp>
#include <optinum/simd/math/tgamma.hpp>

using namespace optinum::simd;

// Helper to check approximate equality
template <typename T> bool approx_equal(T a, T b, T tolerance = T(0.01)) { return std::abs(a - b) < tolerance; }

TEST_CASE("simd::erf - Error Function") {
    SUBCASE("pack<float, 4>") {
        // Test known values
        auto p0 = pack<float, 4>::set(0.0f, 0.0f, 0.0f, 0.0f);
        auto r0 = erf(p0);
        CHECK(approx_equal(r0[0], 0.0f)); // erf(0) = 0
        CHECK(approx_equal(r0[1], 0.0f));
        CHECK(approx_equal(r0[2], 0.0f));
        CHECK(approx_equal(r0[3], 0.0f));

        // erf(1) ≈ 0.8427
        auto p1 = pack<float, 4>::set(1.0f, 1.0f, 1.0f, 1.0f);
        auto r1 = erf(p1);
        CHECK(approx_equal(r1[0], 0.8427f));
        CHECK(approx_equal(r1[1], 0.8427f));

        // erf(2) ≈ 0.9953
        auto p2 = pack<float, 4>::set(2.0f, 2.0f, 2.0f, 2.0f);
        auto r2 = erf(p2);
        CHECK(approx_equal(r2[0], 0.9953f, 0.01f));

        // erf(3) ≈ 0.9999779
        auto p3 = pack<float, 4>::set(3.0f, 3.0f, 3.0f, 3.0f);
        auto r3 = erf(p3);
        CHECK(approx_equal(r3[0], 0.9999f, 0.01f));

        // Test negative values: erf(-x) = -erf(x)
        auto pn1 = pack<float, 4>::set(-1.0f, -1.0f, -1.0f, -1.0f);
        auto rn1 = erf(pn1);
        CHECK(approx_equal(rn1[0], -0.8427f));

        // Test mixed values
        auto pmix = pack<float, 4>::set(-2.0f, -1.0f, 1.0f, 2.0f);
        auto rmix = erf(pmix);
        CHECK(approx_equal(rmix[0], -0.9953f, 0.01f)); // erf(-2)
        CHECK(approx_equal(rmix[1], -0.8427f));        // erf(-1)
        CHECK(approx_equal(rmix[2], 0.8427f));         // erf(1)
        CHECK(approx_equal(rmix[3], 0.9953f, 0.01f));  // erf(2)

        // Test small values
        auto psmall = pack<float, 4>::set(0.1f, 0.1f, 0.1f, 0.1f);
        auto rsmall = erf(psmall);
        CHECK(approx_equal(rsmall[0], 0.1125f)); // erf(0.1) ≈ 0.1125
    }

    SUBCASE("pack<float, 8> - AVX") {
        auto p = pack<float, 8>::set(0.0f, 0.5f, 1.0f, 1.5f, 2.0f, -0.5f, -1.0f, -1.5f);
        auto r = erf(p);

        CHECK(approx_equal(r[0], 0.0f));     // erf(0)
        CHECK(approx_equal(r[1], 0.5205f));  // erf(0.5)
        CHECK(approx_equal(r[2], 0.8427f));  // erf(1)
        CHECK(approx_equal(r[3], 0.9661f));  // erf(1.5)
        CHECK(approx_equal(r[4], 0.9953f));  // erf(2)
        CHECK(approx_equal(r[5], -0.5205f)); // erf(-0.5)
        CHECK(approx_equal(r[6], -0.8427f)); // erf(-1)
        CHECK(approx_equal(r[7], -0.9661f)); // erf(-1.5)
    }

    SUBCASE("pack<double, 2>") {
        auto p = pack<double, 2>::set(1.0, 2.0);
        auto r = erf(p);

        CHECK(approx_equal(r[0], 0.8427, 0.01));
        CHECK(approx_equal(r[1], 0.9953, 0.01));
    }

    SUBCASE("pack<double, 4> - AVX") {
        auto p = pack<double, 4>::set(0.0, 0.5, 1.0, 1.5);
        auto r = erf(p);

        CHECK(approx_equal(r[0], 0.0, 0.01));
        CHECK(approx_equal(r[1], 0.5205, 0.01));
        CHECK(approx_equal(r[2], 0.8427, 0.01));
        CHECK(approx_equal(r[3], 0.9661, 0.01));
    }
}

TEST_CASE("simd::tgamma - Gamma Function") {
    SUBCASE("pack<float, 4>") {
        // Test known values
        // Γ(1) = 0! = 1
        auto p1 = pack<float, 4>::set(1.0f, 1.0f, 1.0f, 1.0f);
        auto r1 = tgamma(p1);
        CHECK(approx_equal(r1[0], 1.0f, 0.05f));

        // Γ(2) = 1! = 1
        auto p2 = pack<float, 4>::set(2.0f, 2.0f, 2.0f, 2.0f);
        auto r2 = tgamma(p2);
        CHECK(approx_equal(r2[0], 1.0f, 0.05f));

        // Γ(3) = 2! = 2
        auto p3 = pack<float, 4>::set(3.0f, 3.0f, 3.0f, 3.0f);
        auto r3 = tgamma(p3);
        CHECK(approx_equal(r3[0], 2.0f, 0.1f));

        // Γ(4) = 3! = 6
        auto p4 = pack<float, 4>::set(4.0f, 4.0f, 4.0f, 4.0f);
        auto r4 = tgamma(p4);
        CHECK(approx_equal(r4[0], 6.0f, 0.3f));

        // Γ(5) = 4! = 24
        auto p5 = pack<float, 4>::set(5.0f, 5.0f, 5.0f, 5.0f);
        auto r5 = tgamma(p5);
        CHECK(approx_equal(r5[0], 24.0f, 2.0f));

        // Γ(0.5) = √π ≈ 1.7724538
        auto phalf = pack<float, 4>::set(0.5f, 0.5f, 0.5f, 0.5f);
        auto rhalf = tgamma(phalf);
        CHECK(approx_equal(rhalf[0], 1.7724f, 0.1f));

        // Test mixed factorial values
        auto pmix = pack<float, 4>::set(1.0f, 2.0f, 3.0f, 4.0f);
        auto rmix = tgamma(pmix);
        CHECK(approx_equal(rmix[0], 1.0f, 0.05f)); // Γ(1) = 1
        CHECK(approx_equal(rmix[1], 1.0f, 0.05f)); // Γ(2) = 1
        CHECK(approx_equal(rmix[2], 2.0f, 0.1f));  // Γ(3) = 2
        CHECK(approx_equal(rmix[3], 6.0f, 0.3f));  // Γ(4) = 6
    }

    SUBCASE("pack<float, 8> - AVX") {
        auto p = pack<float, 8>::set(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 0.5f, 1.5f, 2.5f);
        auto r = tgamma(p);

        CHECK(approx_equal(r[0], 1.0f, 0.05f));   // Γ(1)
        CHECK(approx_equal(r[1], 1.0f, 0.05f));   // Γ(2)
        CHECK(approx_equal(r[2], 2.0f, 0.1f));    // Γ(3)
        CHECK(approx_equal(r[3], 6.0f, 0.3f));    // Γ(4)
        CHECK(approx_equal(r[4], 24.0f, 2.0f));   // Γ(5)
        CHECK(approx_equal(r[5], 1.7724f, 0.1f)); // Γ(0.5) = √π
        CHECK(approx_equal(r[6], 0.8862f, 0.1f)); // Γ(1.5) = √π/2
        CHECK(approx_equal(r[7], 1.3293f, 0.1f)); // Γ(2.5) = 3√π/4
    }

    SUBCASE("pack<double, 2>") {
        auto p = pack<double, 2>::set(3.0, 4.0);
        auto r = tgamma(p);

        CHECK(approx_equal(r[0], 2.0, 0.01));
        CHECK(approx_equal(r[1], 6.0, 0.01));
    }

    SUBCASE("pack<double, 4> - AVX") {
        auto p = pack<double, 4>::set(1.0, 2.0, 3.0, 4.0);
        auto r = tgamma(p);

        CHECK(approx_equal(r[0], 1.0, 0.01));
        CHECK(approx_equal(r[1], 1.0, 0.01));
        CHECK(approx_equal(r[2], 2.0, 0.01));
        CHECK(approx_equal(r[3], 6.0, 0.01));
    }
}

TEST_CASE("simd::lgamma - Log Gamma Function") {
    SUBCASE("pack<float, 4>") {
        // Test known values
        // lgamma(1) = log(Γ(1)) = log(1) = 0
        auto p1 = pack<float, 4>::set(1.0f, 1.0f, 1.0f, 1.0f);
        auto r1 = lgamma(p1);
        CHECK(approx_equal(r1[0], 0.0f, 0.05f));

        // lgamma(2) = log(Γ(2)) = log(1) = 0
        auto p2 = pack<float, 4>::set(2.0f, 2.0f, 2.0f, 2.0f);
        auto r2 = lgamma(p2);
        CHECK(approx_equal(r2[0], 0.0f, 0.05f));

        // lgamma(3) = log(Γ(3)) = log(2) ≈ 0.693
        auto p3 = pack<float, 4>::set(3.0f, 3.0f, 3.0f, 3.0f);
        auto r3 = lgamma(p3);
        CHECK(approx_equal(r3[0], 0.693f, 0.1f));

        // lgamma(4) = log(Γ(4)) = log(6) ≈ 1.792
        auto p4 = pack<float, 4>::set(4.0f, 4.0f, 4.0f, 4.0f);
        auto r4 = lgamma(p4);
        CHECK(approx_equal(r4[0], 1.792f, 0.2f));

        // lgamma(5) = log(Γ(5)) = log(24) ≈ 3.178
        auto p5 = pack<float, 4>::set(5.0f, 5.0f, 5.0f, 5.0f);
        auto r5 = lgamma(p5);
        CHECK(approx_equal(r5[0], 3.178f, 0.3f));

        // lgamma(10) = log(9!) = log(362880) ≈ 12.801
        auto p10 = pack<float, 4>::set(10.0f, 10.0f, 10.0f, 10.0f);
        auto r10 = lgamma(p10);
        CHECK(approx_equal(r10[0], 12.801f, 1.0f));

        // Test mixed values
        auto pmix = pack<float, 4>::set(1.0f, 2.0f, 3.0f, 4.0f);
        auto rmix = lgamma(pmix);
        CHECK(approx_equal(rmix[0], 0.0f, 0.05f));  // lgamma(1)
        CHECK(approx_equal(rmix[1], 0.0f, 0.05f));  // lgamma(2)
        CHECK(approx_equal(rmix[2], 0.693f, 0.1f)); // lgamma(3)
        CHECK(approx_equal(rmix[3], 1.792f, 0.2f)); // lgamma(4)
    }

    SUBCASE("pack<float, 8> - AVX") {
        auto p = pack<float, 8>::set(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f);
        auto r = lgamma(p);

        CHECK(approx_equal(r[0], 0.0f, 0.05f));  // lgamma(1) = log(1) = 0
        CHECK(approx_equal(r[1], 0.0f, 0.05f));  // lgamma(2) = log(1) = 0
        CHECK(approx_equal(r[2], 0.693f, 0.1f)); // lgamma(3) = log(2)
        CHECK(approx_equal(r[3], 1.792f, 0.2f)); // lgamma(4) = log(6)
        CHECK(approx_equal(r[4], 3.178f, 0.3f)); // lgamma(5) = log(24)
        CHECK(approx_equal(r[5], 4.787f, 0.5f)); // lgamma(6) = log(120)
        CHECK(approx_equal(r[6], 6.579f, 0.7f)); // lgamma(7) = log(720)
        CHECK(approx_equal(r[7], 8.525f, 0.9f)); // lgamma(8) = log(5040)
    }

    SUBCASE("pack<double, 2>") {
        auto p = pack<double, 2>::set(3.0, 4.0);
        auto r = lgamma(p);

        CHECK(approx_equal(r[0], 0.693, 0.01));
        CHECK(approx_equal(r[1], 1.792, 0.01));
    }

    SUBCASE("pack<double, 4> - AVX") {
        auto p = pack<double, 4>::set(1.0, 2.0, 3.0, 4.0);
        auto r = lgamma(p);

        CHECK(approx_equal(r[0], 0.0, 0.01));
        CHECK(approx_equal(r[1], 0.0, 0.01));
        CHECK(approx_equal(r[2], 0.693, 0.01));
        CHECK(approx_equal(r[3], 1.792, 0.01));
    }

    SUBCASE("lgamma vs log(tgamma) consistency") {
        // For values where tgamma doesn't overflow, lgamma ≈ log(tgamma)
        auto p = pack<float, 4>::set(1.5f, 2.5f, 3.5f, 4.5f);
        auto r_lgamma = lgamma(p);
        auto r_tgamma = tgamma(p);

        // Manually compute log(tgamma) for comparison
        for (int i = 0; i < 4; i++) {
            float expected = std::log(r_tgamma[i]);
            CHECK(approx_equal(r_lgamma[i], expected, 0.2f));
        }
    }
}

TEST_CASE("Special functions - Edge Cases") {
    SUBCASE("erf - Saturation at large values") {
        // erf(x) → 1 as x → ∞
        auto plarge = pack<float, 4>::set(5.0f, 5.0f, 5.0f, 5.0f);
        auto rlarge = erf(plarge);
        CHECK(approx_equal(rlarge[0], 1.0f, 0.001f));

        // erf(x) → -1 as x → -∞
        auto pnlarge = pack<float, 4>::set(-5.0f, -5.0f, -5.0f, -5.0f);
        auto rnlarge = erf(pnlarge);
        CHECK(approx_equal(rnlarge[0], -1.0f, 0.001f));
    }

    SUBCASE("tgamma - Fractional values") {
        // Γ(0.5) = √π
        auto p = pack<float, 4>::set(0.5f, 0.5f, 0.5f, 0.5f);
        auto r = tgamma(p);
        float sqrt_pi = std::sqrt(3.14159265359f);
        CHECK(approx_equal(r[0], sqrt_pi, 0.1f));
    }

    SUBCASE("lgamma - Large values stability") {
        // lgamma should handle large values better than log(tgamma)
        auto p = pack<float, 4>::set(20.0f, 20.0f, 20.0f, 20.0f);
        auto r = lgamma(p);
        // lgamma(20) = log(19!) ≈ 39.34
        CHECK(approx_equal(r[0], 39.34f, 3.0f));
    }
}

TEST_CASE("Special functions - Comparison with std library") {
    SUBCASE("erf accuracy check") {
        float test_vals[] = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f, 1.5f, 2.0f};

        for (float val : test_vals) {
            auto p = pack<float, 4>::set(val, val, val, val);
            auto r = erf(p);
            float expected = std::erf(val);
            CHECK(approx_equal(r[0], expected, 0.02f));
        }
    }

    SUBCASE("tgamma accuracy check") {
        float test_vals[] = {1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f};

        for (float val : test_vals) {
            auto p = pack<float, 4>::set(val, val, val, val);
            auto r = tgamma(p);
            float expected = std::tgamma(val);
            CHECK(approx_equal(r[0], expected, expected * 0.1f)); // 10% tolerance
        }
    }

    SUBCASE("lgamma accuracy check") {
        float test_vals[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 10.0f};

        for (float val : test_vals) {
            auto p = pack<float, 4>::set(val, val, val, val);
            auto r = lgamma(p);
            float expected = std::lgamma(val);
            CHECK(approx_equal(r[0], expected, std::max(0.01f, std::abs(expected) * 0.2f))); // 20% tolerance or 0.01
        }
    }
}
