#pragma once

// =============================================================================
// optinum/simd/math/atanh.hpp
// Inverse hyperbolic tangent: atanh(x) = 0.5 * ln((1+x)/(1-x))
// Valid for |x| < 1
// =============================================================================

#include <optinum/simd/mask.hpp>
#include <optinum/simd/math/log.hpp>
#include <optinum/simd/pack/avx.hpp>
#include <optinum/simd/pack/pack.hpp>
#include <optinum/simd/pack/sse.hpp>

namespace optinum::simd {

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
