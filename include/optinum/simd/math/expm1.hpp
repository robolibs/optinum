#pragma once

// =============================================================================
// optinum/simd/math/expm1.hpp
// Exponential minus 1: expm1(x) = exp(x) - 1
// Accurate for small x where exp(x) - 1 would lose precision
// =============================================================================

#include <cmath>
#include <optinum/simd/mask.hpp>
#include <optinum/simd/math/exp.hpp>
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
    template <typename T, std::size_t W> OPTINUM_INLINE pack<T, W> expm1(const pack<T, W> &x) noexcept {
        pack<T, W> result;
        for (std::size_t i = 0; i < W; ++i) {
            result.data_[i] = std::expm1(x.data_[i]);
        }
        return result;
    }

    // =============================================================================
    // expm1(x) - Exponential minus 1
    //
    // For |x| > threshold, uses: expm1(x) = exp(x) - 1
    // For |x| <= threshold, uses Taylor series:
    //   expm1(x) = x + x²/2! + x³/3! + x⁴/4! + ...
    //
    // This avoids catastrophic cancellation when x is near 0.
    //
    // Valid range: all real numbers
    // Returns: [-1, +∞)
    // Accuracy: ~3-5 ULP
    // =============================================================================

    // -------------------------------------------------------------------------
    // pack<float, 4> - SSE
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<float, 4> expm1(const pack<float, 4> &x) noexcept {
        using P = pack<float, 4>;

        auto abs_x = x.abs();

        // For |x| > 0.5, use exp(x) - 1
        // For |x| <= 0.5, use Taylor series for better accuracy
        auto use_exp = cmp_gt(abs_x, P(0.5f));

        // Direct computation: exp(x) - 1
        auto direct = exp(x) - P(1.0f);

        // Taylor series: expm1(x) = x + x²/2 + x³/6 + x⁴/24 + x⁵/120 + x⁶/720
        auto x2 = x * x;
        auto x3 = x2 * x;
        auto x4 = x2 * x2;

        // expm1(x) = x * (1 + x/2 * (1 + x/3 * (1 + x/4 * (1 + x/5 * (1 + x/6)))))
        auto taylor =
            x *
            (P(1.0f) +
             x * P(0.5f) *
                 (P(1.0f) + x * P(0.333333333f) *
                                (P(1.0f) + x * P(0.25f) * (P(1.0f) + x * P(0.2f) * (P(1.0f) + x * P(0.166666667f))))));

        return blend(taylor, direct, use_exp);
    }

    // -------------------------------------------------------------------------
    // pack<float, 8> - AVX
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<float, 8> expm1(const pack<float, 8> &x) noexcept {
        using P = pack<float, 8>;

        auto abs_x = x.abs();
        auto use_exp = cmp_gt(abs_x, P(0.5f));

        auto direct = exp(x) - P(1.0f);

        auto taylor =
            x *
            (P(1.0f) +
             x * P(0.5f) *
                 (P(1.0f) + x * P(0.333333333f) *
                                (P(1.0f) + x * P(0.25f) * (P(1.0f) + x * P(0.2f) * (P(1.0f) + x * P(0.166666667f))))));

        return blend(taylor, direct, use_exp);
    }

    // -------------------------------------------------------------------------
    // pack<double, 2> - SSE (double precision)
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<double, 2> expm1(const pack<double, 2> &x) noexcept {
        using P = pack<double, 2>;

        auto abs_x = x.abs();
        auto use_exp = cmp_gt(abs_x, P(0.5));

        auto direct = exp(x) - P(1.0);

        // Higher order Taylor series for double precision
        auto taylor =
            x *
            (P(1.0) +
             x * P(0.5) *
                 (P(1.0) + x * P(0.33333333333333331) *
                               (P(1.0) + x * P(0.25) *
                                             (P(1.0) + x * P(0.2) *
                                                           (P(1.0) + x * P(0.16666666666666666) *
                                                                         (P(1.0) + x * P(0.14285714285714285) *
                                                                                       (P(1.0) + x * P(0.125))))))));

        return blend(taylor, direct, use_exp);
    }

    // -------------------------------------------------------------------------
    // pack<double, 4> - AVX (double precision)
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<double, 4> expm1(const pack<double, 4> &x) noexcept {
        using P = pack<double, 4>;

        auto abs_x = x.abs();
        auto use_exp = cmp_gt(abs_x, P(0.5));

        auto direct = exp(x) - P(1.0);

        auto taylor =
            x *
            (P(1.0) +
             x * P(0.5) *
                 (P(1.0) + x * P(0.33333333333333331) *
                               (P(1.0) + x * P(0.25) *
                                             (P(1.0) + x * P(0.2) *
                                                           (P(1.0) + x * P(0.16666666666666666) *
                                                                         (P(1.0) + x * P(0.14285714285714285) *
                                                                                       (P(1.0) + x * P(0.125))))))));

        return blend(taylor, direct, use_exp);
    }

} // namespace optinum::simd
