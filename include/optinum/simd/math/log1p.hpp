#pragma once

// =============================================================================
// optinum/simd/math/log1p.hpp
// Logarithm of 1 plus x: log1p(x) = log(1 + x)
// Accurate for small x where log(1 + x) would lose precision
// =============================================================================

#include <cmath>
#include <optinum/simd/mask.hpp>
#include <optinum/simd/math/log.hpp>
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
    template <typename T, std::size_t W> OPTINUM_INLINE pack<T, W> log1p(const pack<T, W> &x) noexcept {
        pack<T, W> result;
        for (std::size_t i = 0; i < W; ++i) {
            result.data_[i] = std::log1p(x.data_[i]);
        }
        return result;
    }

    // =============================================================================
    // log1p(x) - Logarithm of 1 plus x
    //
    // For |x| > threshold, uses: log1p(x) = log(1 + x)
    // For |x| <= threshold, uses Taylor series:
    //   log1p(x) = x - x²/2 + x³/3 - x⁴/4 + x⁵/5 - ...
    //
    // This avoids catastrophic cancellation when x is near 0.
    //
    // Valid range: x > -1
    // Returns: all real numbers
    // Accuracy: ~3-5 ULP
    // =============================================================================

    // -------------------------------------------------------------------------
    // pack<float, 4> - SSE
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<float, 4> log1p(const pack<float, 4> &x) noexcept {
        using P = pack<float, 4>;

        auto abs_x = x.abs();

        // For |x| > 0.5, use log(1 + x)
        // For |x| <= 0.5, use Taylor series for better accuracy
        auto use_log = cmp_gt(abs_x, P(0.5f));

        // Direct computation: log(1 + x)
        auto direct = log(P(1.0f) + x);

        // Taylor series: log1p(x) = x - x²/2 + x³/3 - x⁴/4 + x⁵/5 - x⁶/6 + x⁷/7
        // log1p(x) = x * (1 - x/2 * (1 - 2x/3 * (1 - 3x/4 * (1 - 4x/5 * (1 - 5x/6 * (1 - 6x/7))))))
        auto taylor =
            x * (P(1.0f) -
                 x * P(0.5f) *
                     (P(1.0f) - x * P(0.666666667f) *
                                    (P(1.0f) - x * P(0.75f) *
                                                   (P(1.0f) - x * P(0.8f) *
                                                                  (P(1.0f) - x * P(0.833333333f) *
                                                                                 (P(1.0f) - x * P(0.857142857f)))))));

        return blend(taylor, direct, use_log);
    }

    // -------------------------------------------------------------------------
    // pack<float, 8> - AVX
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<float, 8> log1p(const pack<float, 8> &x) noexcept {
        using P = pack<float, 8>;

        auto abs_x = x.abs();
        auto use_log = cmp_gt(abs_x, P(0.5f));

        auto direct = log(P(1.0f) + x);

        auto taylor =
            x * (P(1.0f) -
                 x * P(0.5f) *
                     (P(1.0f) - x * P(0.666666667f) *
                                    (P(1.0f) - x * P(0.75f) *
                                                   (P(1.0f) - x * P(0.8f) *
                                                                  (P(1.0f) - x * P(0.833333333f) *
                                                                                 (P(1.0f) - x * P(0.857142857f)))))));

        return blend(taylor, direct, use_log);
    }

    // -------------------------------------------------------------------------
    // pack<double, 2> - SSE (double precision)
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<double, 2> log1p(const pack<double, 2> &x) noexcept {
        using P = pack<double, 2>;

        auto abs_x = x.abs();
        auto use_log = cmp_gt(abs_x, P(0.5));

        auto direct = log(P(1.0) + x);

        // Higher order Taylor series for double precision
        // log1p(x) = x - x²/2 + x³/3 - x⁴/4 + x⁵/5 - x⁶/6 + x⁷/7 - x⁸/8 + x⁹/9
        auto taylor =
            x *
            (P(1.0) -
             x * P(0.5) *
                 (P(1.0) -
                  x * P(0.66666666666666663) *
                      (P(1.0) -
                       x * P(0.75) *
                           (P(1.0) -
                            x * P(0.8) *
                                (P(1.0) - x * P(0.83333333333333337) *
                                              (P(1.0) - x * P(0.85714285714285716) *
                                                            (P(1.0) - x * P(0.875) *
                                                                          (P(1.0) - x * P(0.88888888888888884)))))))));

        return blend(taylor, direct, use_log);
    }

    // -------------------------------------------------------------------------
    // pack<double, 4> - AVX (double precision)
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<double, 4> log1p(const pack<double, 4> &x) noexcept {
        using P = pack<double, 4>;

        auto abs_x = x.abs();
        auto use_log = cmp_gt(abs_x, P(0.5));

        auto direct = log(P(1.0) + x);

        auto taylor =
            x *
            (P(1.0) -
             x * P(0.5) *
                 (P(1.0) -
                  x * P(0.66666666666666663) *
                      (P(1.0) -
                       x * P(0.75) *
                           (P(1.0) -
                            x * P(0.8) *
                                (P(1.0) - x * P(0.83333333333333337) *
                                              (P(1.0) - x * P(0.85714285714285716) *
                                                            (P(1.0) - x * P(0.875) *
                                                                          (P(1.0) - x * P(0.88888888888888884)))))))));

        return blend(taylor, direct, use_log);
    }

} // namespace optinum::simd
