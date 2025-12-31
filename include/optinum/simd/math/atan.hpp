#pragma once

// =============================================================================
// optinum/simd/math/atan.hpp
// Arc tangent: atan(x)
// Using polynomial approximation and range reduction
// =============================================================================

#include <cmath>
#include <optinum/simd/mask.hpp>
#include <optinum/simd/math/detail/constants.hpp>
#if defined(OPTINUM_HAS_AVX)
#include <optinum/simd/pack/avx.hpp>
#endif
#include <optinum/simd/pack/pack.hpp>
#if defined(OPTINUM_HAS_SSE2)
#include <optinum/simd/pack/sse.hpp>
#endif

#if defined(OPTINUM_HAS_NEON)
#include <optinum/simd/pack/neon.hpp>
#endif

namespace optinum::simd {

    // =========================================================================
    // Generic scalar fallback - works for any pack<T, W>
    // =========================================================================
    template <typename T, std::size_t W> OPTINUM_INLINE pack<T, W> atan(const pack<T, W> &x) noexcept {
        pack<T, W> result;
        for (std::size_t i = 0; i < W; ++i) {
            result.data_[i] = std::atan(x.data_[i]);
        }
        return result;
    }

    // =============================================================================
    // atan() - Arc tangent
    //
    // For |x| <= 1: Use 11th-order polynomial approximation
    // For |x| > 1:  Use identity atan(x) = sign(x)*π/2 - atan(1/x)
    //
    // Polynomial coefficients from minimax approximation
    // =============================================================================

    // -------------------------------------------------------------------------
    // pack<float, 4> - SSE
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<float, 4> atan(const pack<float, 4> &x) noexcept {
        using P = pack<float, 4>;

        // Get absolute value
        auto abs_x = x.abs();

        // Check if |x| > 1
        auto gt_one = cmp_gt(abs_x, P(1.0f));

        // For |x| > 1, use reciprocal
        auto x_reduced = blend(abs_x, P(1.0f) / abs_x, gt_one);

        // Polynomial approximation for atan(x) on [0, 1]
        // Using Horner's method
        auto x2 = x_reduced * x_reduced;

        auto result = P(0.0028662257f);
        result = result * x2 - P(0.0161657367f);
        result = result * x2 + P(0.0429096138f);
        result = result * x2 - P(0.0752896400f);
        result = result * x2 + P(0.1065626393f);
        result = result * x2 - P(0.1420889944f);
        result = result * x2 + P(0.1999355085f);
        result = result * x2 - P(0.3333314528f);
        result = result * x2 * x_reduced + x_reduced;

        // If |x| > 1, apply correction: π/2 - atan(1/x)
        result = blend(result, P(math_constants::PI_2_F) - result, gt_one);

        // Restore sign: if x < 0, negate result
        auto negative = cmp_lt(x, P(0.0f));
        result = blend(result, -result, negative);

        return result;
    }

    // -------------------------------------------------------------------------
    // pack<float, 8> - AVX
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<float, 8> atan(const pack<float, 8> &x) noexcept {
        using P = pack<float, 8>;

        auto abs_x = x.abs();

        auto gt_one = cmp_gt(abs_x, P(1.0f));
        auto x_reduced = blend(abs_x, P(1.0f) / abs_x, gt_one);

        auto x2 = x_reduced * x_reduced;

        auto result = P(0.0028662257f);
        result = result * x2 - P(0.0161657367f);
        result = result * x2 + P(0.0429096138f);
        result = result * x2 - P(0.0752896400f);
        result = result * x2 + P(0.1065626393f);
        result = result * x2 - P(0.1420889944f);
        result = result * x2 + P(0.1999355085f);
        result = result * x2 - P(0.3333314528f);
        result = result * x2 * x_reduced + x_reduced;

        result = blend(result, P(math_constants::PI_2_F) - result, gt_one);

        auto negative = cmp_lt(x, P(0.0f));
        result = blend(result, -result, negative);

        return result;
    }

    // -------------------------------------------------------------------------
    // pack<double, 2> - SSE (double precision)
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<double, 2> atan(const pack<double, 2> &x) noexcept {
        using P = pack<double, 2>;

        auto abs_x = x.abs();

        auto gt_one = cmp_gt(abs_x, P(1.0));
        auto x_reduced = blend(abs_x, P(1.0) / abs_x, gt_one);

        auto x2 = x_reduced * x_reduced;

        // Higher precision coefficients for double
        auto result = P(0.00282363896258175);
        result = result * x2 - P(0.0159569028764963);
        result = result * x2 + P(0.0425049886107612);
        result = result * x2 - P(0.0748900772278524);
        result = result * x2 + P(0.1065626393994362);
        result = result * x2 - P(0.1420889944832810);
        result = result * x2 + P(0.1999355085506763);
        result = result * x2 - P(0.3333314528138536);
        result = result * x2 * x_reduced + x_reduced;

        result = blend(result, P(math_constants::PI_2_D) - result, gt_one);

        auto negative = cmp_lt(x, P(0.0));
        result = blend(result, -result, negative);

        return result;
    }

    // -------------------------------------------------------------------------
    // pack<double, 4> - AVX (double precision)
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<double, 4> atan(const pack<double, 4> &x) noexcept {
        using P = pack<double, 4>;

        auto abs_x = x.abs();

        auto gt_one = cmp_gt(abs_x, P(1.0));
        auto x_reduced = blend(abs_x, P(1.0) / abs_x, gt_one);

        auto x2 = x_reduced * x_reduced;

        auto result = P(0.00282363896258175);
        result = result * x2 - P(0.0159569028764963);
        result = result * x2 + P(0.0425049886107612);
        result = result * x2 - P(0.0748900772278524);
        result = result * x2 + P(0.1065626393994362);
        result = result * x2 - P(0.1420889944832810);
        result = result * x2 + P(0.1999355085506763);
        result = result * x2 - P(0.3333314528138536);
        result = result * x2 * x_reduced + x_reduced;

        result = blend(result, P(math_constants::PI_2_D) - result, gt_one);

        auto negative = cmp_lt(x, P(0.0));
        result = blend(result, -result, negative);

        return result;
    }

} // namespace optinum::simd
