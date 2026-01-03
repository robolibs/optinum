#pragma once

// =============================================================================
// optinum/simd/math/atanh.hpp
// Inverse hyperbolic tangent: atanh(x) = 0.5 * ln((1+x)/(1-x))
// Valid for |x| < 1
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
    template <typename T, std::size_t W> OPTINUM_INLINE pack<T, W> atanh(const pack<T, W> &x) noexcept {
        pack<T, W> result;
        for (std::size_t i = 0; i < W; ++i) {
            result.data_[i] = std::atanh(x.data_[i]);
        }
        return result;
    }

    // =============================================================================
    // atanh(x) - Inverse hyperbolic tangent
    //
    // Formula: atanh(x) = 0.5 * ln((1+x)/(1-x))
    //
    // Valid range: -1 < x < 1
    // Returns: all real numbers
    // Accuracy: ~3-5 ULP
    //
    // Note: For |x| >= 1, the result is undefined (±∞ or NaN in IEEE arithmetic)
    // =============================================================================

    // -------------------------------------------------------------------------
    // pack<float, 4> - SSE
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<float, 4> atanh(const pack<float, 4> &x) noexcept {
        using P = pack<float, 4>;

        // atanh(x) = 0.5 * ln((1+x)/(1-x))
        auto one = P(1.0f);
        auto numerator = one + x;
        auto denominator = one - x;

        return P(0.5f) * log(numerator / denominator);
    }

    // -------------------------------------------------------------------------
    // pack<float, 8> - AVX
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<float, 8> atanh(const pack<float, 8> &x) noexcept {
        using P = pack<float, 8>;

        auto one = P(1.0f);
        auto numerator = one + x;
        auto denominator = one - x;

        return P(0.5f) * log(numerator / denominator);
    }

    // -------------------------------------------------------------------------
    // pack<double, 2> - SSE (double precision)
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<double, 2> atanh(const pack<double, 2> &x) noexcept {
        using P = pack<double, 2>;

        auto one = P(1.0);
        auto numerator = one + x;
        auto denominator = one - x;

        return P(0.5) * log(numerator / denominator);
    }

    // -------------------------------------------------------------------------
    // pack<double, 4> - AVX (double precision)
    // -------------------------------------------------------------------------
    OPTINUM_INLINE pack<double, 4> atanh(const pack<double, 4> &x) noexcept {
        using P = pack<double, 4>;

        auto one = P(1.0);
        auto numerator = one + x;
        auto denominator = one - x;

        return P(0.5) * log(numerator / denominator);
    }

} // namespace optinum::simd
