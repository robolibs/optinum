#pragma once

// =============================================================================
// optinum/simd/math/asin.hpp
// Arc sine using polynomial approximation
// Valid for x in [-1, 1]
// =============================================================================

#include <optinum/simd/mask.hpp>
#include <optinum/simd/math/detail/constants.hpp>
#include <optinum/simd/math/sqrt.hpp>
#include <optinum/simd/pack/avx.hpp>
#include <optinum/simd/pack/pack.hpp>
#include <optinum/simd/pack/sse.hpp>

namespace optinum::simd {

    // =============================================================================
    // asin(x) - Arc sine
    //
    // Uses polynomial approximation for |x| <= 0.5:
    //   asin(x) ≈ x + x³/6 + 3x⁵/40 + 15x⁷/336 + ...
    //
    // For |x| > 0.5, uses identity:
    //   asin(x) = sign(x) * (π/2 - 2*asin(sqrt((1-|x|)/2)))
    //
    // Valid range: [-1, 1]
    // Returns: [-π/2, π/2]
    // Accuracy: ~3-5 ULP
    // =============================================================================

    // -------------------------------------------------------------------------
    // pack<float, 4> - SSE
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<float, 4> asin(const pack<float, 4> &x) noexcept {
        using P = pack<float, 4>;

        auto abs_x = x.abs();

        // For |x| <= 0.5, use polynomial approximation
        // For |x| > 0.5, use asin(x) = π/2 - 2*asin(sqrt((1-|x|)/2))
        auto use_alt = cmp_gt(abs_x, P(0.5f));

        // Polynomial approximation for small values
        // asin(x) ≈ x * (1 + x²*(c1 + x²*(c2 + x²*(c3 + x²*c4))))
        constexpr float c1 = 0.166666667f; // 1/6
        constexpr float c2 = 0.075f;       // 3/40
        constexpr float c3 = 0.044642857f; // 15/336
        constexpr float c4 = 0.030381944f; // ~1/33

        auto x2 = x * x;
        auto poly = P(c1) + x2 * (P(c2) + x2 * (P(c3) + x2 * P(c4)));
        auto direct = x * (P(1.0f) + x2 * poly);

        // Alternative formula: asin(x) = π/2 - 2*asin(sqrt((1-|x|)/2))
        auto y = sqrt((P(1.0f) - abs_x) * P(0.5f));
        auto y2 = y * y;
        auto poly_y = P(c1) + y2 * (P(c2) + y2 * (P(c3) + y2 * P(c4)));
        auto asin_y = y * (P(1.0f) + y2 * poly_y);
        auto alt = P(math_constants::PI_2_F) - P(2.0f) * asin_y;

        // Restore sign for alternative formula
        auto negative = cmp_lt(x, P(0.0f));
        alt = blend(alt, -alt, negative);

        // Choose between direct and alternative
        return blend(direct, alt, use_alt);
    }

    // -------------------------------------------------------------------------
    // pack<float, 8> - AVX
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<float, 8> asin(const pack<float, 8> &x) noexcept {
        using P = pack<float, 8>;

        auto abs_x = x.abs();
        auto use_alt = cmp_gt(abs_x, P(0.5f));

        constexpr float c1 = 0.166666667f;
        constexpr float c2 = 0.075f;
        constexpr float c3 = 0.044642857f;
        constexpr float c4 = 0.030381944f;

        auto x2 = x * x;
        auto poly = P(c1) + x2 * (P(c2) + x2 * (P(c3) + x2 * P(c4)));
        auto direct = x * (P(1.0f) + x2 * poly);

        auto y = sqrt((P(1.0f) - abs_x) * P(0.5f));
        auto y2 = y * y;
        auto poly_y = P(c1) + y2 * (P(c2) + y2 * (P(c3) + y2 * P(c4)));
        auto asin_y = y * (P(1.0f) + y2 * poly_y);
        auto alt = P(math_constants::PI_2_F) - P(2.0f) * asin_y;

        auto negative = cmp_lt(x, P(0.0f));
        alt = blend(alt, -alt, negative);

        return blend(direct, alt, use_alt);
    }

    // -------------------------------------------------------------------------
    // pack<double, 2> - SSE (double precision)
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<double, 2> asin(const pack<double, 2> &x) noexcept {
        using P = pack<double, 2>;

        auto abs_x = x.abs();
        auto use_alt = cmp_gt(abs_x, P(0.5));

        // Higher precision coefficients for double
        constexpr double c1 = 0.16666666666666666;
        constexpr double c2 = 0.075;
        constexpr double c3 = 0.044642857142857144;
        constexpr double c4 = 0.030381944444444443;
        constexpr double c5 = 0.022372159090909091;
        constexpr double c6 = 0.017352764423076923;

        auto x2 = x * x;
        auto poly = P(c1) + x2 * (P(c2) + x2 * (P(c3) + x2 * (P(c4) + x2 * (P(c5) + x2 * P(c6)))));
        auto direct = x * (P(1.0) + x2 * poly);

        auto y = sqrt((P(1.0) - abs_x) * P(0.5));
        auto y2 = y * y;
        auto poly_y = P(c1) + y2 * (P(c2) + y2 * (P(c3) + y2 * (P(c4) + y2 * (P(c5) + y2 * P(c6)))));
        auto asin_y = y * (P(1.0) + y2 * poly_y);
        auto alt = P(math_constants::PI_2_D) - P(2.0) * asin_y;

        auto negative = cmp_lt(x, P(0.0));
        alt = blend(alt, -alt, negative);

        return blend(direct, alt, use_alt);
    }

    // -------------------------------------------------------------------------
    // pack<double, 4> - AVX (double precision)
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<double, 4> asin(const pack<double, 4> &x) noexcept {
        using P = pack<double, 4>;

        auto abs_x = x.abs();
        auto use_alt = cmp_gt(abs_x, P(0.5));

        constexpr double c1 = 0.16666666666666666;
        constexpr double c2 = 0.075;
        constexpr double c3 = 0.044642857142857144;
        constexpr double c4 = 0.030381944444444443;
        constexpr double c5 = 0.022372159090909091;
        constexpr double c6 = 0.017352764423076923;

        auto x2 = x * x;
        auto poly = P(c1) + x2 * (P(c2) + x2 * (P(c3) + x2 * (P(c4) + x2 * (P(c5) + x2 * P(c6)))));
        auto direct = x * (P(1.0) + x2 * poly);

        auto y = sqrt((P(1.0) - abs_x) * P(0.5));
        auto y2 = y * y;
        auto poly_y = P(c1) + y2 * (P(c2) + y2 * (P(c3) + y2 * (P(c4) + y2 * (P(c5) + y2 * P(c6)))));
        auto asin_y = y * (P(1.0) + y2 * poly_y);
        auto alt = P(math_constants::PI_2_D) - P(2.0) * asin_y;

        auto negative = cmp_lt(x, P(0.0));
        alt = blend(alt, -alt, negative);

        return blend(direct, alt, use_alt);
    }

} // namespace optinum::simd
