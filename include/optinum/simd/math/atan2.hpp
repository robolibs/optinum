#pragma once

// =============================================================================
// optinum/simd/math/atan2.hpp
// Two-argument arc tangent: atan2(y, x)
// Returns the angle in radians between the positive x-axis and the point (x,y)
// =============================================================================

#include <optinum/simd/mask.hpp>
#include <optinum/simd/math/atan.hpp>
#include <optinum/simd/math/detail/constants.hpp>
#include <optinum/simd/pack/pack.hpp>

namespace optinum::simd {

    // =============================================================================
    // atan2(y, x) - Two-argument arc tangent
    //
    // Returns the angle θ such that x = r*cos(θ) and y = r*sin(θ)
    // Range: [-π, π]
    //
    // Handles all quadrants correctly:
    //   Quadrant I   (x > 0, y > 0): atan(y/x)
    //   Quadrant II  (x < 0, y > 0): atan(y/x) + π
    //   Quadrant III (x < 0, y < 0): atan(y/x) - π
    //   Quadrant IV  (x > 0, y < 0): atan(y/x)
    // =============================================================================

    // -------------------------------------------------------------------------
    // Generic scalar fallback for any pack size (including pack<T, 1>)
    // -------------------------------------------------------------------------
    template <typename T, std::size_t W>
    OPTINUM_INLINE pack<T, W> atan2(const pack<T, W> &y, const pack<T, W> &x) noexcept {
        pack<T, W> result;
        for (std::size_t i = 0; i < W; ++i) {
            result.data_[i] = std::atan2(y.data_[i], x.data_[i]);
        }
        return result;
    }

    // -------------------------------------------------------------------------
    // pack<float, 4> - SSE
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<float, 4> atan2(const pack<float, 4> &y, const pack<float, 4> &x) noexcept {
        using P = pack<float, 4>;

        // Compute atan(y/x)
        auto ratio = y / x;
        auto result = atan(ratio);

        // Adjust for quadrants
        // If x < 0 and y >= 0: add π
        auto x_neg = cmp_lt(x, P(0.0f));
        auto y_nonneg = cmp_ge(y, P(0.0f));
        auto add_pi = x_neg & y_nonneg;
        result = blend(result, result + P(math_constants::PI_F), add_pi);

        // If x < 0 and y < 0: subtract π
        auto y_neg = cmp_lt(y, P(0.0f));
        auto sub_pi = x_neg & y_neg;
        result = blend(result, result - P(math_constants::PI_F), sub_pi);

        // Handle x == 0 cases
        auto x_zero = cmp_eq(x, P(0.0f));
        auto y_pos = cmp_gt(y, P(0.0f));
        auto y_neg_strict = cmp_lt(y, P(0.0f));

        // If x == 0 and y > 0: π/2
        auto set_pi2 = x_zero & y_pos;
        result = blend(result, P(math_constants::PI_2_F), set_pi2);

        // If x == 0 and y < 0: -π/2
        auto set_neg_pi2 = x_zero & y_neg_strict;
        result = blend(result, P(-math_constants::PI_2_F), set_neg_pi2);

        // If x == 0 and y == 0: return 0 (undefined, but this is a common convention)
        auto both_zero = x_zero & cmp_eq(y, P(0.0f));
        result = blend(result, P(0.0f), both_zero);

        return result;
    }

    // -------------------------------------------------------------------------
    // pack<float, 8> - AVX
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<float, 8> atan2(const pack<float, 8> &y, const pack<float, 8> &x) noexcept {
        using P = pack<float, 8>;

        auto ratio = y / x;
        auto result = atan(ratio);

        auto x_neg = cmp_lt(x, P(0.0f));
        auto y_nonneg = cmp_ge(y, P(0.0f));
        auto add_pi = x_neg & y_nonneg;
        result = blend(result, result + P(math_constants::PI_F), add_pi);

        auto y_neg = cmp_lt(y, P(0.0f));
        auto sub_pi = x_neg & y_neg;
        result = blend(result, result - P(math_constants::PI_F), sub_pi);

        auto x_zero = cmp_eq(x, P(0.0f));
        auto y_pos = cmp_gt(y, P(0.0f));
        auto y_neg_strict = cmp_lt(y, P(0.0f));

        auto set_pi2 = x_zero & y_pos;
        result = blend(result, P(math_constants::PI_2_F), set_pi2);

        auto set_neg_pi2 = x_zero & y_neg_strict;
        result = blend(result, P(-math_constants::PI_2_F), set_neg_pi2);

        auto both_zero = x_zero & cmp_eq(y, P(0.0f));
        result = blend(result, P(0.0f), both_zero);

        return result;
    }

    // -------------------------------------------------------------------------
    // pack<double, 2> - SSE (double precision)
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<double, 2> atan2(const pack<double, 2> &y, const pack<double, 2> &x) noexcept {
        using P = pack<double, 2>;

        auto ratio = y / x;
        auto result = atan(ratio);

        auto x_neg = cmp_lt(x, P(0.0));
        auto y_nonneg = cmp_ge(y, P(0.0));
        auto add_pi = x_neg & y_nonneg;
        result = blend(result, result + P(math_constants::PI_D), add_pi);

        auto y_neg = cmp_lt(y, P(0.0));
        auto sub_pi = x_neg & y_neg;
        result = blend(result, result - P(math_constants::PI_D), sub_pi);

        auto x_zero = cmp_eq(x, P(0.0));
        auto y_pos = cmp_gt(y, P(0.0));
        auto y_neg_strict = cmp_lt(y, P(0.0));

        auto set_pi2 = x_zero & y_pos;
        result = blend(result, P(math_constants::PI_2_D), set_pi2);

        auto set_neg_pi2 = x_zero & y_neg_strict;
        result = blend(result, P(-math_constants::PI_2_D), set_neg_pi2);

        auto both_zero = x_zero & cmp_eq(y, P(0.0));
        result = blend(result, P(0.0), both_zero);

        return result;
    }

    // -------------------------------------------------------------------------
    // pack<double, 4> - AVX (double precision)
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<double, 4> atan2(const pack<double, 4> &y, const pack<double, 4> &x) noexcept {
        using P = pack<double, 4>;

        auto ratio = y / x;
        auto result = atan(ratio);

        auto x_neg = cmp_lt(x, P(0.0));
        auto y_nonneg = cmp_ge(y, P(0.0));
        auto add_pi = x_neg & y_nonneg;
        result = blend(result, result + P(math_constants::PI_D), add_pi);

        auto y_neg = cmp_lt(y, P(0.0));
        auto sub_pi = x_neg & y_neg;
        result = blend(result, result - P(math_constants::PI_D), sub_pi);

        auto x_zero = cmp_eq(x, P(0.0));
        auto y_pos = cmp_gt(y, P(0.0));
        auto y_neg_strict = cmp_lt(y, P(0.0));

        auto set_pi2 = x_zero & y_pos;
        result = blend(result, P(math_constants::PI_2_D), set_pi2);

        auto set_neg_pi2 = x_zero & y_neg_strict;
        result = blend(result, P(-math_constants::PI_2_D), set_neg_pi2);

        auto both_zero = x_zero & cmp_eq(y, P(0.0));
        result = blend(result, P(0.0), both_zero);

        return result;
    }

} // namespace optinum::simd
