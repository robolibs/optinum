#pragma once

// =============================================================================
// optinum/simd/math/acosh.hpp
// Inverse hyperbolic cosine: acosh(x) = ln(x + sqrt(x² - 1))
// Valid for x >= 1
// =============================================================================

#include <cmath>
#include <optinum/simd/mask.hpp>
#include <optinum/simd/math/log.hpp>
#include <optinum/simd/math/sqrt.hpp>
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
    template <typename T, std::size_t W> OPTINUM_INLINE pack<T, W> acosh(const pack<T, W> &x) noexcept {
        pack<T, W> result;
        for (std::size_t i = 0; i < W; ++i) {
            result.data_[i] = std::acosh(x.data_[i]);
        }
        return result;
    }

    // =============================================================================
    // acosh(x) - Inverse hyperbolic cosine
    //
    // Formula: acosh(x) = ln(x + sqrt(x² - 1))
    //
    // For large x, uses: acosh(x) = ln(2x)
    // to avoid overflow in x² - 1
    //
    // Valid range: x >= 1
    // Returns: [0, +∞)
    // Accuracy: ~3-5 ULP
    //
    // Note: For x < 1, the result is undefined (NaN in IEEE arithmetic)
    // =============================================================================

    // -------------------------------------------------------------------------
    // pack<float, 4> - SSE
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<float, 4> acosh(const pack<float, 4> &x) noexcept {
        using P = pack<float, 4>;

        // For x > 1e8, use approximation: acosh(x) ≈ ln(2x)
        // This avoids overflow in x² - 1
        auto use_approx = cmp_gt(x, P(1e8f));

        // Standard formula: acosh(x) = ln(x + sqrt(x² - 1))
        auto x2 = x * x;
        auto standard = log(x + sqrt(x2 - P(1.0f)));

        // Approximation for large x: acosh(x) = ln(2x)
        auto approx = log(P(2.0f) * x);

        return blend(standard, approx, use_approx);
    }

    // -------------------------------------------------------------------------
    // pack<float, 8> - AVX
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<float, 8> acosh(const pack<float, 8> &x) noexcept {
        using P = pack<float, 8>;

        auto use_approx = cmp_gt(x, P(1e8f));

        auto x2 = x * x;
        auto standard = log(x + sqrt(x2 - P(1.0f)));

        auto approx = log(P(2.0f) * x);

        return blend(standard, approx, use_approx);
    }

    // -------------------------------------------------------------------------
    // pack<double, 2> - SSE (double precision)
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<double, 2> acosh(const pack<double, 2> &x) noexcept {
        using P = pack<double, 2>;

        auto use_approx = cmp_gt(x, P(1e154));

        auto x2 = x * x;
        auto standard = log(x + sqrt(x2 - P(1.0)));

        auto approx = log(P(2.0) * x);

        return blend(standard, approx, use_approx);
    }

    // -------------------------------------------------------------------------
    // pack<double, 4> - AVX (double precision)
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<double, 4> acosh(const pack<double, 4> &x) noexcept {
        using P = pack<double, 4>;

        auto use_approx = cmp_gt(x, P(1e154));

        auto x2 = x * x;
        auto standard = log(x + sqrt(x2 - P(1.0)));

        auto approx = log(P(2.0) * x);

        return blend(standard, approx, use_approx);
    }

} // namespace optinum::simd
